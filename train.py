# %%
import os
import sys
from pathlib import Path
import yaml
from dataclasses import asdict
from pprint import pformat
import warnings
import pandas as pd
from auc.aucs import evaluate_aucs
import torch
from data.dataset import FlexibleDataLoader, BatchSizeScheduler, DataModule
import mlflow

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if ( DELPHI_DIR := Path(__file__).resolve().parent ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from data.dataset import (
    DelphiDataset,
    DelphiCollateFn,
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
from utils import load_domain_config, setup_mlflow, AUTO_BLOCK_SIZE # cache upper bound when block_size="auto"

setup_mlflow()

class _MLflowWarningFilter(logging.Filter):
    _SUPPRESS = ("malformed experiment", "malformed run")

    def filter(self, record):
        if record.levelno == logging.WARNING:
            msg = record.getMessage().lower()
            if any(kw in msg for kw in self._SUPPRESS):
                return False
        return True

_mlflow_filter = _MLflowWarningFilter()
for _handler in logging.root.handlers:
    _handler.addFilter(_mlflow_filter)

root_path = DELPHI_DIR / "data" / "transforms"
ATTENTION_SCHEMES = yaml.safe_load((DELPHI_DIR / "config" / "attention_schemes.yaml").read_text())

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

USE_TQDM = sys.stdout.isatty()

def _fmt(v):
    if v is True:  return "✓"
    if v is False: return "✗"
    if v is None:  return "—"
    return str(v)


def format_delphi_config(cfg) -> str:
    """Compact human-readable summary of a DelphiConfig."""
    d = asdict(cfg)
    domains = d.pop("domains", {})

    # ── Scalar params ──────────────────────────────────────────────────
    skip = {"path", "n_layers", "n_hidden", "input_size", "pretrained_path",
            "subdomain", "group", "n_latent_tokens"}
    lines = ["Model config:"]
    for k, v in d.items():
        lines.append(f"  {k}: {_fmt(v)}")

    # ── Domains table ──────────────────────────────────────────────────
    bool_cols   = ["predict", "at_birth", "age_jitter", "freeze"]
    str_cols    = ["projector", "type", "dropout_mode", "dropout_rate"]
    cols        = bool_cols + str_cols
    col_widths  = {c: max(len(c), max(len(_fmt(d.get(c))) for d in domains.values()) if domains else 0)
                   for c in cols}
    dom_width   = max((len(n) for n in domains), default=6)

    header = f"  {'domain':<{dom_width}}  " + "  ".join(f"{c:>{col_widths[c]}}" for c in cols)
    sep    = "  " + "-" * (len(header) - 2)
    lines += ["", "Domains:", header, sep]
    for name, dc in domains.items():
        row = f"  {name:<{dom_width}}  " + "  ".join(
            f"{_fmt(dc.get(c)):>{col_widths[c]}}" for c in cols
        )
        lines.append(row)

    return "\n".join(lines)


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

    parser.add_argument("--domain_config_yaml", "--domain-config-yaml", dest="domain_config_yaml", default="config/domain_config_default.yaml")
    parser.add_argument("--domains",            default="diseases,death,cv_drugs,ns_drugs,lifestyle,hla_alleles,sex")
    parser.add_argument("--attention_scheme", "--attention-scheme", dest="attention_scheme",
                        default="[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death,padding]:causal(mask_ties=True)", nargs="+")
    parser.add_argument("--n_layer", "--n-layer",       dest="n_layer", default=12,  type=int)
    parser.add_argument("--n_head",  "--n-head",        dest="n_head",  default=6,   type=int)
    parser.add_argument("--n_embd",  "--n-embd",        dest="n_embd",  default=120, type=int)
    parser.add_argument("--block_size", "--block-size", dest="block_size", default="auto",
                        type=lambda v: v if v == "auto" else int(v),
                        help="Max tokens per subject in the cache. 'auto' (default) uses a "
                             "generous upper bound; the effective per-batch length is always "
                             "trimmed to the longest sequence in that batch.")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", default=32, type=int)
    parser.add_argument("--batch_size_schedule", "--batch-size-schedule", dest="batch_size_schedule",
                        default=None, type=BatchSizeScheduler.from_string,
                        help=(
                            "Batch size schedule. Format: comma-separated stages, each "
                            "'n_epochs:batch_size' or 'n_epochs:batch_sizexgrad_accum'. "
                            "Use '*' as n_epochs for the last (open-ended) stage. "
                            "Overrides --batch_size. "
                            "Examples: "
                            "'20:32,20:128,*:256' (no accumulation); "
                            "'20:32,20:128,10:256,*:256x4' (last stage: effective batch=1024)."
                        ))
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", default=4, type=int)
    parser.add_argument("--token_dropout", "--token-dropout", dest="token_dropout", default=0.1, type=float)
    parser.add_argument("--no-compile", "--no_compile", dest="no_compile", default=False, action='store_true')
    parser.add_argument("--learning_rate", "--learning-rate", "--lr", dest="lr", default=None, type=float,
                        help="Peak learning rate. For fresh runs defaults to 1e-4. "
                             "On resume, if provided, overrides the stored LR and resets the scheduler.")

    # ── Optimizer / LR schedule ───────────────────────────────────────────────
    parser.add_argument("--min_lr", "--min-lr",             dest="min_lr",         default=None,  type=float,
                        help="Minimum LR at end of cosine decay (default: lr/10)")
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay",   default=1e-1,  type=float)
    parser.add_argument("--beta1",                                                  default=0.9,   type=float)
    parser.add_argument("--beta2",                                                  default=0.95,  type=float)
    parser.add_argument("--grad_clip", "--grad-clip",       dest="grad_clip",      default=1.0,   type=float,
                        help="Gradient clipping norm (0 = disabled)")
    parser.add_argument("--schedule",                                               default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--warmup_iters", "--warmup-iters", dest="warmup_iters",   default=2000,  type=int)
    parser.add_argument("--lr_decay_iters", "--lr-decay-iters", dest="lr_decay_iters", default=10000, type=int)
    parser.add_argument("--test_fold", "--test-fold",       dest="test_fold",      default=1,     type=int)
    parser.add_argument("--subjects",                                               default=None,  type=str)
    parser.add_argument("--date_cutoff", "--date-cutoff",   dest="date_cutoff",    default=None,  type=str,
                        help="ISO date (YYYY-MM-DD). Tokens after this date are marked via eval_mask.")
    parser.add_argument("--birth_dates_file", "--birth-dates-file", dest="birth_dates_file", default=None, type=str,
                        help="Path to TSV with columns eid, year, month (used with --date_cutoff).")
    parser.add_argument("--seed",                                                   default=142,   type=int)
    parser.add_argument("--max_epochs", "--max-epochs", dest="max_epochs", default=1000, type=int)
    parser.add_argument("--min_epochs", "--min-epochs", dest="min_epochs", default=0,    type=int)
    parser.add_argument("--patience",                                       default=20,   type=int)
    parser.add_argument("--compute_aucs", "--compute-aucs", dest="compute_aucs",   default=False, action="store_true")
    parser.add_argument("--log_loss_per_disease", "--log-loss-per-disease", dest="log_loss_per_disease",
                        default=False, action="store_true",
                        help="Log per-disease CE loss breakdown as a CSV artifact each validation epoch")
    parser.add_argument("--checkpoint_every", "--checkpoint-every", dest="checkpoint_every", default=None, type=int,
                        help="Save a periodic checkpoint every N epochs (in addition to best-model checkpoints)")

    parser.add_argument("--no_event_token_rate", "--no-event-token-rate", dest="no_event_token_rate", default=2, type=float)
    parser.add_argument("--no_event_token_insertion_mode", "--no-event-token-insertion-mode",
                        dest="no_event_token_insertion_mode", default="random", type=str)

    parser.add_argument("--no-warnings", "--no_warnings", dest="no_warnings", default=False, action="store_true")
    parser.add_argument("--use_amp", "--use-amp", "--amp", dest="use_amp", default=False, action="store_true",
                        help="Enable mixed precision training (float16)")

    parser.add_argument("--experiment_name", "--experiment-name", "--exp_name", "--exp-name", "-x", dest="experiment_name", default=None)
    parser.add_argument("--run_name", "--run-name", dest="run_name", default=None)
    parser.add_argument("--resume_run_id", "--resume-run-id", "--resume_runid", "--resume-runid",
                        dest="resume_run_id", type=str, default=None,
                        help="Resume training from the latest checkpoint of this MLflow run")

    parser.add_argument("--interactive", "-i", dest="interactive", action="store_true", default=False,
                        help="Run in interactive mode (prompts user for input at key points)")
    parser.add_argument("--resume_from_previous", "--resume-from-previous", "-r", dest="resume_from_previous",
                        action="store_true", default=False,
                        help="Resume from a previous run. With --interactive (-i), prompts to select experiment and run.")

    parser.add_argument("--dryrun", "--dry-run", "--dry_run", dest="dry_run", action="store_true", default=False)

    parser.add_argument("--no_rich", "--no-rich", dest="no_rich", action="store_true", default=False,
                    help="Disable rich display (use tqdm instead, e.g. for cluster log files)")

    args = parser.parse_args()

    if args.experiment_name is None and not args.resume_run_id and not args.resume_from_previous:
        parser.error("--experiment_name / -x is required unless --resume_run_id or --resume_from_previous is set.")

    return args


def _interactive_select_run() -> tuple[str, str]:
    """Prompt the user to pick an experiment and a run.

    Returns (run_id, target_experiment_name) where the target experiment
    is either the original one or a different one chosen by the user.
    """
    experiments = mlflow.search_experiments(order_by=["last_update_time DESC"])
    if not experiments:
        raise RuntimeError("No MLflow experiments found.")

    print("\nAvailable experiments:")
    for i, exp in enumerate(experiments):
        print(f"  [{i}] {exp.name}")
    idx = int(input("Select experiment [0]: ").strip() or "0")
    source_experiment = experiments[idx]

    runs_df = mlflow.search_runs(
        experiment_ids=[source_experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=20,
    )
    if runs_df.empty:
        raise RuntimeError(f"No runs found in experiment '{source_experiment.name}'.")

    cols = ["run_id", "tags.mlflow.runName", "start_time", "status"]
    cols = [c for c in cols if c in runs_df.columns]
    print(f"\nRuns in '{source_experiment.name}' (most recent first):")
    for i, row in runs_df[cols].iterrows():
        name = row.get("tags.mlflow.runName", "")
        print(f"  [{i}] {row['run_id'][:8]}…  {name:<30}  {row['status']}  {row['start_time']}")
    run_idx = int(input("Select run [0]: ").strip() or "0")
    run_id = runs_df.iloc[run_idx]["run_id"]

    answer = input(f"\nKeep original experiment '{source_experiment.name}'? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        target_experiment = source_experiment.name
    else:
        print("\nAvailable experiments:")
        for i, exp in enumerate(experiments):
            print(f"  [{i}] {exp.name}")
        target_idx = int(input("Select target experiment [0]: ").strip() or "0")
        target_experiment = experiments[target_idx].name

    return run_id, target_experiment


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
    use_compile=True,
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
        date_cutoff=args.date_cutoff,
        birth_dates_file=args.birth_dates_file,
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

    domain_dropout = {
        model.domain_to_int[dname]: (cfg.dropout_mode, cfg.dropout_rate)
        for dname, cfg in domain_cfg.items()
        if cfg.dropout_mode is not None and cfg.dropout_rate > 0
    }

    age_jitter = {
        model.domain_to_int[dname]: (cfg.age_jitter_min, cfg.age_jitter_max)
        for dname, cfg in domain_cfg.items()
        if cfg.age_jitter and dname in model.domain_to_int
    }
    if age_jitter:
        logging.info(
            "Age jitter enabled for: %s",
            {dname: (cfg.age_jitter_min, cfg.age_jitter_max)
             for dname, cfg in domain_cfg.items() if cfg.age_jitter},
        )

    collate_kwargs = dict(
        age_sampler=age_sampler,
        block_size=block_size,      # "auto" or int
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_token_id=1,
        continuous_domains=continuous_domains,
        domain_dropout=domain_dropout,
    )
    train_collate = DelphiCollateFn(**collate_kwargs, age_jitter=age_jitter, training=True)
    eval_collate  = DelphiCollateFn(**collate_kwargs, training=False)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if num_workers > 0 and use_compile:
        # Fix: 
        # Workers are re-forked at the end of each epoch (iterator GC). On exit,
        # forked workers trigger LLVM thread cleanup which fails with pthread_join
        # errors because torch.compile's LLVM thread pool is invalid in child processes.
        # persistent_workers keeps workers alive across epochs so they never exit
        # mid-training; the harmless LLVM error at program end is after training completes.
        loader_kwargs["persistent_workers"] = True

    train_loader = FlexibleDataLoader(train_dataset, shuffle=True,  collate_fn=train_collate, **loader_kwargs)
    valid_loader = FlexibleDataLoader(valid_dataset, shuffle=False, collate_fn=eval_collate,  **loader_kwargs)
    test_loader  = FlexibleDataLoader(test_dataset,  shuffle=False, collate_fn=eval_collate,  **loader_kwargs)

    return DataModule(train_loader, valid_loader, test_loader)


# ——————————————————————————————————————————————————————————————————————————————————

if __name__ == "__main__":    
  
    args = get_cli_args()

    if args.resume_from_previous and args.interactive:
        args.resume_run_id, args.experiment_name = _interactive_select_run()
    elif args.resume_from_previous and not args.resume_run_id:
        raise ValueError("--resume_from_previous requires --interactive or --resume_run_id.")

    cache_block_size = AUTO_BLOCK_SIZE if args.block_size == "auto" else args.block_size
    logging.info(
        "block_size=%s  cache_block_size=%d", args.block_size, cache_block_size
    )

    bs_scheduler = None
    start_epoch = 0

    if (train_from_scratch := not args.resume_run_id):
  
        args.attention_scheme = parse_attention_scheme(args.attention_scheme)
        domains = [d for d in args.domains.split(",") if d != "padding"]
        domain_config_yaml = DELPHI_DIR / args.domain_config_yaml
        default_cfg_per_domain = load_domain_config(domain_config_yaml, root_path / 'tokens')
        domain_cfg = {k: v for k, v in default_cfg_per_domain.items() if k in domains or k == "padding"}

        assert all([k in default_cfg_per_domain for k in domains])
        assert len(args.attention_scheme) in {1, args.n_layer}, \
            f"len of --attention_scheme should be 1 or n_layer (={args.n_layer})"

        # Pass as-is: single string → same scheme for all layers,
        # list of n_layer strings → per-layer schemes. The model expands internally.
        attention_scheme = (
            args.attention_scheme[0] if len(args.attention_scheme) == 1
            else args.attention_scheme
        )
        
        # ── Model ─────────────────────────────────────────────────────────
        delphi_config = DelphiConfig(
            n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
            domains=domain_cfg, attention_scheme=attention_scheme,
            token_dropout=args.token_dropout,
            # block_size=cache_block_size,
            block_size=128,
            no_event_token_rate=args.no_event_token_rate, 
            no_event_token_insertion_mode=args.no_event_token_insertion_mode,
            seed=args.seed
        )
    
        logging.info("\n%s", format_delphi_config(delphi_config))
        model = Delphi(delphi_config).to(DEVICE)
        if not args.no_compile:
            logging.info("Compiling model with torch.compile (first batch will be slower)...")
        model = torch.compile(model, disable=args.no_compile)

        # ── Data ──────────────────────────────────────────────────────────
        bs_scheduler = args.batch_size_schedule
        initial_batch_size = bs_scheduler.step(0).batch_size if bs_scheduler is not None else args.batch_size
        dataloaders = get_dataloaders(
            domain_cfg,
            model=model,
            test_fold=args.test_fold,
            block_size=args.block_size,
            batch_size=initial_batch_size,
            num_workers=args.num_workers,
            no_event_token_rate=args.no_event_token_rate,
            no_event_insertion_mode=args.no_event_token_insertion_mode,
            seed=args.seed,
            subjects_include_list=args.subjects,
            use_compile=not args.no_compile,
        )

        # ── Optimizer ─────────────────────────────────────────────────────
        lr = args.lr if args.lr is not None else 1e-4
        optim_config = OptimConfig(
            learning_rate  = lr,
            min_lr         = args.min_lr if args.min_lr is not None else lr / 10,
            weight_decay   = args.weight_decay,
            beta1          = args.beta1,
            beta2          = args.beta2,
            grad_clip      = args.grad_clip,
            schedule       = args.schedule,
            warmup_iters   = args.warmup_iters,
            lr_decay_iters = args.lr_decay_iters,
        )
        logging.info("Optimizer configuration: \n%s", pformat(asdict(optim_config), sort_dicts=False))
        
        optimizer, scheduler = configure_optimizers(model=model, cfg=optim_config, device_type=DEVICE)  
        
        logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=args.run_name)
    
        mlflow.log_artifact(domain_config_yaml)
    
        logged_params = {
            "test_fold": args.test_fold,
            "batch_size": args.batch_size,
            "batch_size_schedule": args.batch_size_schedule,
            "learning_rate": args.lr,
            "seed": args.seed,
            "optim_config": optim_config,
            "max_epochs": args.max_epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
        }
     
    else:

        ################################ FROM PREVIOUS RUN ################################
        from utils.run_loader import config_from_runid
        model, \
        dataloaders, \
        optim_config, optimizer_state, scheduler_state, \
        start_epoch, \
        logged_params, previous_run_name = config_from_runid(args.resume_run_id)

        bs_scheduler = logged_params.pop("batch_size_scheduler", None)
        if args.batch_size_schedule is not None:
            bs_scheduler = args.batch_size_schedule
            new_bs = bs_scheduler.step(start_epoch).batch_size
            dataloaders.train.set_batch_size(new_bs)
            logging.info("batch_size_schedule overridden from CLI: %s (batch_size at epoch %d: %d)", bs_scheduler, start_epoch, new_bs)
        elif bs_scheduler is not None:
            logging.info("batch_size_schedule restored from run params: %s", bs_scheduler)
        else:
            logging.info("No batch_size_schedule — using fixed batch_size=%d", logged_params.get("batch_size", "?"))

        if args.lr is not None:
            optim_config.learning_rate = args.lr
            optim_config.min_lr = args.min_lr if args.min_lr is not None else args.lr / 10

        model = model.to(DEVICE)
        if not args.no_compile:
            logging.info("Compiling model with torch.compile (first batch will be slower)...")
        model = torch.compile(model, disable=args.no_compile)
        optimizer, scheduler = configure_optimizers(model=model, cfg=optim_config, device_type=DEVICE)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        if args.lr is not None:
            # Keep position in the cosine schedule (last_epoch) but rescale to new peak LR.
            # scheduler.base_lrs was restored from checkpoint; override it and recompute current LR.
            old_base_lr = scheduler.base_lrs[0]
            scheduler.base_lrs = [args.lr] * len(scheduler.base_lrs)
            current_lrs = scheduler.get_lr()
            for pg, lr in zip(optimizer.param_groups, current_lrs):
                pg['lr'] = lr
            logging.info(
                "Learning rate peak changed: %.2e → %.2e; "
                "current LR at schedule step %d: %.2e (min_lr=%.2e)",
                old_base_lr, args.lr, scheduler.last_epoch, current_lrs[0], optim_config.min_lr,
            )

        if args.experiment_name is None:
            run_info = mlflow.get_run(args.resume_run_id)
            args.experiment_name = mlflow.get_experiment(run_info.info.experiment_id).name

        new_run_id = clone_run_to_new_experiment(args.resume_run_id, args.experiment_name)
        logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=previous_run_name, autostart=False)
        logger.start(resume_run_id=new_run_id)
    
        print(f"Resuming from MLflow run {args.resume_run_id} ...")

# —————————————————————————————————————————————————————————————————————————————————————

    # ── Run metadata ──────────────────────────────────────────────────────────────────
    logger.log_run_metadata(cwd=DELPHI_DIR)

    n_params = sum(p.numel() for p in model.parameters())
    logged_params["n_params"] = n_params

    dataloaders.log_info()
    logged_params["n_train"] = dataloaders.n_train
    logged_params["n_val"]   = dataloaders.n_val
    logged_params["n_test"]  = dataloaders.n_test

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
        batch_size_scheduler=bs_scheduler,
        start_epoch=start_epoch,
    )

    trainer.train(max_epochs=args.max_epochs, min_epochs=args.min_epochs, patience=args.patience)

    if args.compute_aucs:

        from auc.aucs import evaluate_aucs
        import copy
        model.eval()

        # evaluate_aucs requires fixed T across all batches; swap collate to use
        # block_size=128 instead of "auto" so torch.cat on embeddings doesn't fail.
        auc_collate = copy.copy(dataloaders[2]._collate_fn)
        auc_collate.block_size = 128
        test_loader = DataLoader(
            dataloaders[2].dataset,
            batch_size=dataloaders[2].batch_size,
            shuffle=False,
            num_workers=dataloaders[2].num_workers,
            pin_memory=True,
            collate_fn=auc_collate,
        )
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
