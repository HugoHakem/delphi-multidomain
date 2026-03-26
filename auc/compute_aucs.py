"""
Standalone AUC evaluation script.

Loads a trained model from an MLflow run_id (using best_model.pt),
reconstructs the test dataloader, and computes AUCs.

Usage:
    python compute_aucs.py --runid <mlflow_run_id> [--block_size 128] [--batch_size 512] [--n_jobs 8]
"""

import os, sys, ast, re
from pathlib import Path
from pprint import pformat
import argparse
import logging

import torch
import pandas as pd
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

if (DELPHI_DIR := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from delphi.model import Delphi, DelphiConfig, DomainConfig
from data.dataset import (
    DelphiDataset,
    DelphiCollateFn,
    AgeSampler,
)
from torch.utils.data import DataLoader
from auc.aucs import evaluate_aucs
from utils.utils import load_domain_config, setup_mlflow

root_path = DELPHI_DIR / "data" / "transforms"
AUTO_BLOCK_SIZE = 512

setup_mlflow()

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True


# ═══════════════════════════════════════════════════════════════════════════════
#  MLflow helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_checkpoint_path(run_id):
    """
    Get the path to best_model.pt for a given run.
    Falls back to the latest epoch checkpoint if best_model.pt doesn't exist.
    """
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    ckpt_dir = Path(re.sub(r"^file://", "", artifact_uri)) / "checkpoints"

    best = ckpt_dir / "best_model.pt"
    if best.exists():
        logging.info(f"Using best_model.pt → {best.resolve().name}")
        return best

    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    epoch_re = re.compile(r"epoch(\d+)", re.IGNORECASE)
    best_ckpt, best_epoch = None, -1
    for ck in ckpts:
        m = epoch_re.search(ck.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch, best_ckpt = epoch, ck

    best_ckpt = best_ckpt or ckpts[-1]
    logging.info(f"Using checkpoint: {best_ckpt.name}")
    return best_ckpt


def load_run_params(run_id):
    """Load and parse params from an MLflow run."""
    run = mlflow.get_run(run_id)
    params = dict(run.data.params)
    if "attention_scheme" in params:
        try:
            params["attention_scheme"] = ast.literal_eval(params["attention_scheme"])
        except (ValueError, SyntaxError):
            # Stored as a plain scheme string, not a Python repr — wrap in list
            params["attention_scheme"] = [params["attention_scheme"]]
    return params


def parse_domains_param(domains_str):
    """
    Parse the domains param stored in MLflow (may contain PosixPath references).
    Matches the parsing used in config_from_runid.
    """
    s_clean = re.sub(r"PosixPath\(([^)]+)\)", r"\1", domains_str)
    domains_dict = ast.literal_eval(s_clean)
    return {k: DomainConfig(**v) for k, v in domains_dict.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  Reconstruct model and dataloader
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_from_run(run_id, block_size=None, batch_size=512, num_workers=4):
    """
    Reconstruct model and test dataloader from an MLflow run.

    Returns: model, test_loader, run_params
    """
    params = load_run_params(run_id)

    ckpt_path = get_checkpoint_path(run_id)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    metadata = ckpt.get("metadata", {})
    test_ids = metadata.get("test_ids")
    if test_ids is None:
        raise ValueError("Checkpoint does not contain test_ids in metadata.")

    domain_cfg = parse_domains_param(params["domains"])

    # block_size must be a fixed integer here: the AUC index arithmetic
    # (cases // block_size, cases % block_size) assumes every batch has
    # the same sequence length. "auto" trimming is incompatible with this.
    stored_block_size = params.get("block_size", "96")
    if block_size is not None:
        bs = block_size
    elif stored_block_size == "auto":
        logging.warning(
            "Run was trained with block_size='auto'; using AUTO_BLOCK_SIZE=%d "
            "as the fixed block size for AUC evaluation.", AUTO_BLOCK_SIZE
        )
        bs = AUTO_BLOCK_SIZE
    else:
        bs = int(stored_block_size)

    attn_scheme = params.get("attention_scheme", ["all:causal(mask_ties=True)"])
    n_layer = int(params.get("n_layer", 12))
    if isinstance(attn_scheme, str):
        attn_scheme = [attn_scheme]
    if len(attn_scheme) == 1:
        attn_scheme = n_layer * attn_scheme

    delphi_config = DelphiConfig(
        n_embd=int(params.get("n_embd", 120)),
        n_layer=n_layer,
        n_head=int(params.get("n_head", 6)),
        domains=domain_cfg,
        attention_scheme=attn_scheme,
        block_size=bs,
        token_dropout=float(params.get("token_dropout", 0.1)),
        no_event_token_rate=float(params.get("no_event_token_rate", 2.0)),
        no_event_token_insertion_mode=params.get("no_event_token_insertion_mode", "random"),
        seed=int(params.get("seed", 42)),
    )

    model = Delphi(delphi_config)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(DEVICE)
    model.eval()

    logging.info(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

    continuous_domains = {
        dname: cfg.n_latent_tokens or 1
        for dname, cfg in domain_cfg.items()
        if cfg.type == "continuous"
    }

    test_dataset = DelphiDataset(
        root=root_path,
        domains_cfg=domain_cfg,
        domain_to_int=model.domain_to_int,
        block_size=bs,
        subjects=test_ids,
        exclusions=[],
        required_domains=["diseases"],
        no_event_token_rate=delphi_config.no_event_token_rate,
        no_event_insertion_mode=delphi_config.no_event_token_insertion_mode,
        continuous_domains=continuous_domains,
        age_domains=["diseases", "death"],
    )

    age_sampler = AgeSampler(
        insertion_mode=delphi_config.no_event_token_insertion_mode,
        token_rate=delphi_config.no_event_token_rate,
        seed=delphi_config.seed,
    )

    collate = DelphiCollateFn(
        age_sampler=age_sampler,
        block_size=bs,
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_token_id=1,
        continuous_domains=continuous_domains,
        domain_dropout={},
        training=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    return model, test_loader, params


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compute AUCs for a trained Delphi model")
    parser.add_argument("--runid", required=True, help="MLflow run ID")
    parser.add_argument("--block_size", type=int, default=128, help="Fixed block size for AUC evaluation (must be an integer; 'auto' is not supported)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_jobs", type=int, default=8, help="Parallel jobs for AUC computation")
    parser.add_argument("--output_file", type=str, default="aucs.csv")
    args = parser.parse_args()

    model, test_loader, run_params = reconstruct_from_run(
        args.runid,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    from utils.trainer import MLFlowLogger

    run = mlflow.get_run(args.runid)
    experiment_name = mlflow.get_experiment(run.info.experiment_id).name

    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=None,
        autostart=False,
    )
    logger.start(resume_run_id=args.runid)

    auc_df = evaluate_aucs(
        model,
        test_loader,
        block_size=model.block_size,
        run_id=args.runid,
        n_jobs=args.n_jobs,
        logger=logger,
        output_file=args.output_file,
    )

    logger.end()
    logging.info(f"Done. {len(auc_df)} AUC rows computed.")


if __name__ == "__main__":
    main()
