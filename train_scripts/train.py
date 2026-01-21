# %%
import os, sys
from pathlib import Path
import yaml
from dataclasses import dataclass, asdict
from pprint import pformat    
import warnings
import pandas as pd
import torch
from easydict import EasyDict
import mlflow

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if ( DELPHI_DIR := Path(__file__).resolve().parent.parent ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DELPHI_DIR / "mlruns" )

from data.dataset import DelphiDataset, DelphiDataloader
from utils.cv_utils import get_data_partitions
from utils.utils import load_embed_config
from delphi.optim import OptimConfig, configure_optimizers
from delphi.model.transformer import ( 
    Delphi,
    DelphiConfig,
)
from trainer import (    
    MLFlowLogger,
    Trainer,
    clone_run_to_new_experiment,
)

root_path = DELPHI_DIR / "data" / "transforms"
ATTENTION_SCHEMES = yaml.safe_load( (DELPHI_DIR / "config" / "attention_schemes.yaml").read_text() )

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

USE_TQDM = sys.stdout.isatty()

# ——————————————— CONFIG ———————————————————————————————————————————————————————————————

def get_cli_args():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_scheme", default="[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death,padding]:causal(mask_ties=True)", nargs="+")
    parser.add_argument("--n_layer",          default=12,   type=int)
    parser.add_argument("--n_head",           default=10,   type=int)
    parser.add_argument("--n_embd",           default=120,  type=int)
    parser.add_argument("--block_size",       default=96,   type=int)
    parser.add_argument("--test_fold",        default=1,    type=int)
    parser.add_argument("--subjects",         default=None, type=str)
    parser.add_argument("--domain_config_yaml", default="config/domain_config_default.yaml")
    parser.add_argument("--domains",          default="diseases,death,cv_drugs,ns_drugs,lifestyle,hla_alleles,sex,padding")
    parser.add_argument("--experiment_name",  default="drugs-predicted")
    parser.add_argument("--run_name",         default=None)
    parser.add_argument("--batch_size",       default=32, type=int)
    parser.add_argument("--learning_rate", "--lr", dest="lr", default=1e-4, type=float)
    parser.add_argument("--no-warnings", "--no_warnings", dest="no_warnings", default=False, action="store_true")
    parser.add_argument("--resume_run_id", type=str, default=None,
                    help="Resume training from the latest checkpoint of this MLflow run")
    parser.add_argument("--dryrun", "--dry-run", "--dry_run", dest="dry_run", action="store_true", default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":    
    args =  get_cli_args()

else:
    args = DEFAULT_ARGS = EasyDict({
        "attention_scheme": ["[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death,hla_alleles]:causal(mask_ties=True)"],
        "n_layer": 12,
        "test_fold": 3,
        "patience": 2,
        "batch_size": 4,
        "lr": 1e-4,
        "run_name": os.getenv("RUN_NAME", "default"),
    })

def parse_attention_scheme(attention_scheme, as_list=True):
    
    if isinstance(attention_scheme, str):
        if attention_scheme in ATTENTION_SCHEMES.keys():
            attention_scheme = ATTENTION_SCHEMES[attention_scheme]["scheme"]
        if as_list:
            attention_scheme = [attention_scheme]
        return attention_scheme
    elif isinstance(attention_scheme, list):
        return [parse_attention_scheme(scheme, as_list=False) for scheme in attention_scheme]



@dataclass
class ProcessedArgs:
    domains: list[str]
    domain_cfg: dict
    attention_scheme: list[str]
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]
    dataset_config: dict
    domain_config_yaml: Path


# Still not in use
def process_args(args, root_path) -> ProcessedArgs:

    # ---- attention scheme ----
    attention_scheme = parse_attention_scheme(args.attention_scheme)
    assert len(attention_scheme) in {1, args.n_layer}, (
        f"--attention_scheme must have length 1 or n_layer={args.n_layer}"
    )

    if len(attention_scheme) == 1:
        attention_scheme = attention_scheme * args.n_layer

    # ---- domains ----
    domains = args.domains.split(",")

    domain_config_yaml = DELPHI_DIR / args.domain_config_yaml
    default_cfg_per_domain = load_embed_config(
        domain_config_yaml,
        root_path / "tokens",
    )

    assert all(d in default_cfg_per_domain for d in domains), (
        f"Some domains not found in {domain_config_yaml}"
    )

    domain_cfg = {d: default_cfg_per_domain[d] for d in domains}

    # ---- subject splits ----
    train_ids, val_ids, test_ids = get_data_partitions(
        "../data/transforms/subject_lists",
        fold=args.test_fold,
    )

    if args.subjects is not None:
        subject_ids = set(pd.read_csv(args.subjects, header=None)[0])
        train_ids = list(set(train_ids) & subject_ids)
        val_ids   = list(set(val_ids)   & subject_ids)
        test_ids  = list(set(test_ids)  & subject_ids)

    # ---- dataset config ----
    dataset_config = dict(
        root=root_path,
        domains=domain_cfg,
        exclusions=[],
        required_domains=["diseases"],
    )

    return ProcessedArgs(
        domains=domains,
        domain_cfg=domain_cfg,
        attention_scheme=attention_scheme,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        dataset_config=dataset_config,
        domain_config_yaml=domain_config_yaml,
    )


# %%

if __name__ == "__main__":

  if (train_from_scratch := not args.resume_run_id):

    args.attention_scheme = parse_attention_scheme(args.attention_scheme)
    domains = args.domains.split(",")
    domain_config_yaml = DELPHI_DIR / args.domain_config_yaml
    default_cfg_per_domain = load_embed_config(domain_config_yaml, root_path / 'tokens')
    domain_cfg = { k: v for k, v in default_cfg_per_domain.items() if k in domains }
    
    assert all([k in default_cfg_per_domain for k in domains])
    assert len(args.attention_scheme) in {1, args.n_layer}, f"len of the --attention_scheme argument should be either 1 or args.n_layer (={args.n_layer})"
    
    if len(args.attention_scheme) == 1:
        attention_scheme = args.n_layer * args.attention_scheme
    elif args.n_layer == len(args.attention_scheme):
        attention_scheme = args.attention_scheme
    
    # —————————————————————————————————————————————————————————————————————————————————————————————————————————
    train_ids, val_ids, test_ids = get_data_partitions("../data/transforms/subject_lists", fold=args.test_fold)
    
    if args.subjects is not None:
        subject_ids = pd.read_csv(args.subjects, header=None)[0].tolist()
        train_ids   = list( set(train_ids) & set(subject_ids) )
        val_ids     = list( set(val_ids)   & set(subject_ids) )
        test_ids    = list( set(test_ids)  & set(subject_ids) )

    dataset_config = dict(root=root_path, domains=domain_cfg, exclusions=[], required_domains=["diseases"])
    
    train_dataset = DelphiDataset(subjects=train_ids,   **dataset_config).to(DEVICE)
    valid_dataset = DelphiDataset(subjects=val_ids,   **dataset_config).to(DEVICE)
    test_dataset  = DelphiDataset(subjects=test_ids,   **dataset_config).to(DEVICE)

    dataloaders = [ DelphiDataloader(d, batch_size=[args.batch_size, args.batch_size, args.batch_size][i]) for i, d in enumerate([train_dataset, valid_dataset, test_dataset]) ]
    # —————————————————————————————————————————————————————————————————————————————————————————————————————————
    
    config = DelphiConfig( 
        n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, block_size=args.block_size,
        token_dropout=0.1, domains=domain_cfg, attention_scheme=attention_scheme
    )

    logging.info("Config:\n%s", pformat(asdict(config), sort_dicts=False))
    torch.compile(model := Delphi(config).to(DEVICE))

    optim_config = OptimConfig(learning_rate=args.lr, min_lr=args.lr/10)
    logging.info(f"Optimizer configuration: \n%s", pformat(asdict(optim_config), sort_dicts=False))
    
    optimizer, scheduler = configure_optimizers(model=model, cfg=optim_config, device_type=DEVICE)  
    
    logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=args.run_name)

    mlflow.log_artifact(domain_config_yaml)

    logged_params = { "test_fold": args.test_fold, "batch_size": args.batch_size, "learning_rate": args.lr }
   
  else:

    ################################ FROM PREVIOUS RUN ################################

    model, dataloaders, optimizer, scheduler, logged_params, previous_run_name = config_from_runid(args.resume_run_id)
        
    new_run_id = clone_run_to_new_experiment(args.resume_run_id, args.experiment_name)
    logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=previous_run_name, autostart=False)
    logger.start(resume_run_id=new_run_id)

    #TODO: Add possibility to change some parameters, e.g. attention scheme, or add domains (e.g. genetic PCs and HLA alleles)

    print(f"Resuming from MLflow run {args.resume_run_id} ...")

# —————————————————————————————————————————————————————————————————————————————————————————————————————————

  if args.no_warnings:
      warnings.filterwarnings("ignore")

  trainer = Trainer( model, dataloaders, optimizer, scheduler, logger=logger, mlflow_params=logged_params, use_tqdm=USE_TQDM ) 
  trainer.train(max_epochs=1000)

# %%
