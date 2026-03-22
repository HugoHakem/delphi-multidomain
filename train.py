# %%
import os, sys
from pathlib import Path
import yaml
from dataclasses import dataclass, asdict
from pprint import pformat    
import warnings
import pandas as pd
from auc.aucs import evaluate_aucs
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
import mlflow

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if ( DELPHI_DIR := Path(__file__).resolve().parent ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from data.dataset import (
    DelphiDataset,
    DelphiCollateFn,
    DelphiBatch,
    AgeSampler,
)

from delphi.optim import (
    OptimConfig, 
    configure_optimizers
)

from delphi.model import (
    Delphi,
    DelphiConfig,
)

from utils.trainer import (
    MLFlowLogger,
    Trainer,
    clone_run_to_new_experiment,
)

from utils.cv_utils import get_data_partitions
from utils.utils import load_domain_config

root_path = DELPHI_DIR / "data" / "transforms"
ATTENTION_SCHEMES = yaml.safe_load( (DELPHI_DIR / "config" / "attention_schemes.yaml").read_text() )
AUTO_BLOCK_SIZE = 512   # cache upper bound when block_size="auto"

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

USE_TQDM = sys.stdout.isatty()

def parse_attention_scheme(attention_scheme, as_list=True):
    
    if isinstance(attention_scheme, str):
        if attention_scheme in ATTENTION_SCHEMES.keys():
            attention_scheme = ATTENTION_SCHEMES[attention_scheme]["scheme"]
        if as_list:
            attention_scheme = [attention_scheme]
        return attention_scheme
    elif isinstance(attention_scheme, list):
        return [parse_attention_scheme(scheme, as_list=False) for scheme in attention_scheme]


# ——————————————— CLI ———————————————————————————————————————————————————————————————

def get_cli_args():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--domain_config_yaml", default="config/domain_config_default.yaml")
    parser.add_argument("--domains",            default="diseases,death,cv_drugs,ns_drugs,lifestyle,hla_alleles,sex")
    parser.add_argument("--attention_scheme",   default="[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death,padding]:causal(mask_ties=True)", nargs="+")
    parser.add_argument("--n_layer",            default=12,   type=int)
    parser.add_argument("--n_head",             default=6,   type=int)
    parser.add_argument("--n_embd",             default=120,  type=int)
    parser.add_argument("--block_size",         default="auto",
                        type=lambda v: v if v == "auto" else int(v),
                        help="Max tokens per subject in the cache. 'auto' (default) uses a "
                             "generous upper bound; the effective per-batch length is always "
                             "trimmed to the longest sequence in that batch.")
    parser.add_argument("--batch_size",         default=32, type=int)
    parser.add_argument("--num_workers",        default=4, type=int)
    parser.add_argument("--token_dropout",      default=0.1, type=float)
    parser.add_argument("--no-compile",         default=False, action='store_true')
    parser.add_argument("--learning_rate", "--lr", dest="lr", default=1e-4, type=float)

    # ── Optimizer / LR schedule ───────────────────────────────────────────────
    parser.add_argument("--min_lr",         default=None,  type=float,
                        help="Minimum LR at end of cosine decay (default: lr/10)")
    parser.add_argument("--weight_decay",   default=1e-1,  type=float)
    parser.add_argument("--beta1",          default=0.9,   type=float)
    parser.add_argument("--beta2",          default=0.95,  type=float)
    parser.add_argument("--grad_clip",      default=1.0,   type=float,
                        help="Gradient clipping norm (0 = disabled)")
    parser.add_argument("--schedule",       default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--warmup_iters",   default=2000,  type=int)
    parser.add_argument("--lr_decay_iters", default=10000, type=int)
    parser.add_argument("--test_fold",          default=1,    type=int)
    parser.add_argument("--subjects",           default=None, type=str)
    parser.add_argument("--seed",               default=142, type=int)
    parser.add_argument("--compute_aucs",           default=False, action="store_true")
    parser.add_argument("--log_loss_per_disease",   default=False, action="store_true",
                        help="Log per-disease CE loss breakdown as a CSV artifact each validation epoch")
    parser.add_argument("--checkpoint_every",       default=None, type=int,
                        help="Save a periodic checkpoint every N epochs (in addition to best-model checkpoints)")
    
    parser.add_argument("--no_event_token_rate",           default=2, type=float)
    parser.add_argument("--no_event_token_insertion_mode", default="random", type=str)

    parser.add_argument("--no-warnings", "--no_warnings", dest="no_warnings", default=False, action="store_true")
    parser.add_argument("--use_amp", "--amp", dest="use_amp", default=False, action="store_true",
                    help="Enable mixed precision training (float16)")

    parser.add_argument("--experiment_name", "--experiment-name", "--exp_name", "--exp-name", "-x", dest="experiment_name", required=True, default=None)
    parser.add_argument("--run_name",         default=None)
    parser.add_argument("--resume_run_id", type=str, default=None,
                    help="Resume training from the latest checkpoint of this MLflow run")

    parser.add_argument("--dryrun", "--dry-run", "--dry_run", dest="dry_run", action="store_true", default=False)

    parser.add_argument("--no_rich", "--no-rich", dest="no_rich", action="store_true", default=False,
                    help="Disable rich display (use tqdm instead, e.g. for cluster log files)")

    args = parser.parse_args()
    return args


# ——————————————— Data loading ——————————————————————————————————————————————————————

def get_continuous_domains(domain_cfg):
    return {
        dname: cfg.n_latent_tokens or 1
        for dname, cfg in domain_cfg.items()
        if cfg.type == "continuous"
    }


def get_dataloaders(
    domain_cfg, 
    model,
    test_fold, 
    block_size,
    batch_size,
    num_workers=4,
    no_event_token_rate=2.0,
    no_event_insertion_mode="random",
    seed=42,
    subjects_include_list=None,
):
    train_ids, val_ids, test_ids = get_data_partitions(
        "./data/transforms/subject_lists", fold=test_fold
    )
    
    if subjects_include_list is not None:
        subject_ids = pd.read_csv(subjects_include_list, header=None)[0].tolist()
        train_ids = list(set(train_ids) & set(subject_ids))
        val_ids   = list(set(val_ids)   & set(subject_ids))
        test_ids  = list(set(test_ids)  & set(subject_ids))

    continuous_domains = get_continuous_domains(domain_cfg)

    # Dataset always needs a fixed integer cache size; collate receives the
    # original value ("auto" or int) to decide per-batch trimming behaviour.
    cache_block_size = AUTO_BLOCK_SIZE if block_size == "auto" else block_size

    dataset_kwargs = dict(
        root=root_path,
        domains_cfg=domain_cfg,
        domain_to_int=model.domain_to_int,
        block_size=cache_block_size,
        exclusions=[],
        required_domains=["diseases"],
        no_event_token_rate=no_event_token_rate,
        no_event_insertion_mode=no_event_insertion_mode,
        continuous_domains=continuous_domains,
        age_domains=["diseases", "death"],
    )

    train_dataset = DelphiDataset(subjects=train_ids, **dataset_kwargs)
    valid_dataset = DelphiDataset(subjects=val_ids,   **dataset_kwargs)
    test_dataset  = DelphiDataset(subjects=test_ids,  **dataset_kwargs)

    # Collate function (shared by all loaders)
    age_sampler = AgeSampler(
        insertion_mode=no_event_insertion_mode,
        token_rate=no_event_token_rate,
        seed=seed,
    )

    collate = DelphiCollateFn(
        age_sampler=age_sampler,
        block_size=block_size,      # "auto" or int
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_token_id=1,
        continuous_domains=continuous_domains,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return [train_loader, valid_loader, test_loader]


# ——————————————————————————————————————————————————————————————————————————————————

if __name__ == "__main__":    
  
    args = get_cli_args()

    cache_block_size = AUTO_BLOCK_SIZE if args.block_size == "auto" else args.block_size
    logging.info(
        "block_size=%s  cache_block_size=%d", args.block_size, cache_block_size
    )

    if (train_from_scratch := not args.resume_run_id):
  
        args.attention_scheme = parse_attention_scheme(args.attention_scheme)
        domains = [d for d in args.domains.split(",") if d != "padding"]
        domain_config_yaml = DELPHI_DIR / args.domain_config_yaml
        default_cfg_per_domain = load_domain_config(domain_config_yaml, root_path / 'tokens')
        domain_cfg = {k: v for k, v in default_cfg_per_domain.items() if k in domains or k == "padding"}

        assert all([k in default_cfg_per_domain for k in domains])
        assert len(args.attention_scheme) in {1, args.n_layer}, \
            f"len of --attention_scheme should be 1 or n_layer (={args.n_layer})"
        
        if len(args.attention_scheme) == 1:
            attention_scheme = args.n_layer * args.attention_scheme
        elif args.n_layer == len(args.attention_scheme):
            attention_scheme = args.attention_scheme
        
        # ── Model ─────────────────────────────────────────────────────────
        delphi_config = DelphiConfig(
            n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
            domains=domain_cfg, attention_scheme=attention_scheme,
            token_dropout=args.token_dropout,
            block_size=cache_block_size,
            no_event_token_rate=args.no_event_token_rate, 
            no_event_token_insertion_mode=args.no_event_token_insertion_mode,
            seed=args.seed
        )
    
        logging.info("Config:\n%s", pformat(asdict(delphi_config), sort_dicts=False))
        model = Delphi(delphi_config).to(DEVICE)
        if not args.no_compile:
            logging.info("Compiling model with torch.compile (first batch will be slower)...")
        model = torch.compile(model, disable=args.no_compile)

        # ── Data ──────────────────────────────────────────────────────────
        dataloaders = get_dataloaders(
            domain_cfg,
            model=model,
            test_fold=args.test_fold,
            block_size=args.block_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            no_event_token_rate=args.no_event_token_rate,
            no_event_insertion_mode=args.no_event_token_insertion_mode,
            seed=args.seed,
            subjects_include_list=args.subjects,
        )

        # ── Optimizer ─────────────────────────────────────────────────────
        optim_config = OptimConfig(
            learning_rate  = args.lr,
            min_lr         = args.min_lr if args.min_lr is not None else args.lr / 10,
            weight_decay   = args.weight_decay,
            beta1          = args.beta1,
            beta2          = args.beta2,
            grad_clip      = args.grad_clip,
            schedule       = args.schedule,
            warmup_iters   = args.warmup_iters,
            lr_decay_iters = args.lr_decay_iters,
        )
        logging.info(f"Optimizer configuration: \n%s", pformat(asdict(optim_config), sort_dicts=False))
        
        optimizer, scheduler = configure_optimizers(model=model, cfg=optim_config, device_type=DEVICE)  
        
        logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=args.run_name)
    
        mlflow.log_artifact(domain_config_yaml)
    
        logged_params = { 
            "test_fold": args.test_fold, 
            "batch_size": args.batch_size, 
            "learning_rate": args.lr,
            "seed": args.seed,
            "optim_config": optim_config            
        }
     
    else:
  
        ################################ FROM PREVIOUS RUN ################################
        from utils.utils import config_from_runid 
        model, \
        dataloaders, \
        optim_config, optimizer_state, scheduler_state, \
        start_epoch, \
        logged_params, previous_run_name = config_from_runid(args.resume_run_id)

        model = model.to(DEVICE)
        if not args.no_compile:
            logging.info("Compiling model with torch.compile (first batch will be slower)...")
        model = torch.compile(model, disable=args.no_compile)
        optimizer, scheduler = configure_optimizers(model=model, cfg=optim_config, device_type=DEVICE)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        new_run_id = clone_run_to_new_experiment(args.resume_run_id, args.experiment_name)
        logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=previous_run_name, autostart=False)
        logger.start(resume_run_id=new_run_id)
    
        print(f"Resuming from MLflow run {args.resume_run_id} ...")

# —————————————————————————————————————————————————————————————————————————————————————

    # ── Run metadata ──────────────────────────────────────────────────────────────────
    logger.log_run_metadata(cwd=DELPHI_DIR)

    n_params = sum(p.numel() for p in model.parameters())
    logged_params["n_params"] = n_params

    train_loader, val_loader, test_loader = dataloaders
    logged_params["n_train"] = len(train_loader.dataset)
    logged_params["n_val"]   = len(val_loader.dataset)
    logged_params["n_test"]  = len(test_loader.dataset)

    if args.no_warnings:
        warnings.filterwarnings("ignore")
  
    trainer = Trainer(
        model, dataloaders,
        optimizer, scheduler,
        log_loss_per_disease=args.log_loss_per_disease,
        checkpoint_every=args.checkpoint_every,
        logger=logger,
        mlflow_params=logged_params,
        use_tqdm=args.no_rich,
        use_amp=args.use_amp,
        use_rich=not args.no_rich,
        optim_config=optim_config,
    )

    trainer.train(max_epochs=1000, patience=10)

    if args.compute_aucs:
        
        from auc.aucs import evaluate_aucs
        model.eval()
        test_loader = dataloaders[2]    
        del dataloaders
    
        auc_df = evaluate_aucs(
            model,
            test_loader,
            block_size=cache_block_size,
            run_id=logger.active_run.info.run_id,
            n_jobs=8,
            logger=logger,
        )
        logging.info("AUCs:\n%s", pformat(auc_df, sort_dicts=False))
