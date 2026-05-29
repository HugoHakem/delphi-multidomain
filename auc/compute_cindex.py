"""
Standalone C-index evaluation script.

Loads a trained Delphi model from an MLflow run_id and computes
Harrell's C-index per disease per sex on the held-out test fold.

Usage:
    python auc/compute_cindex.py --runid <mlflow_run_id>
    python auc/compute_cindex.py --runid <id> --block_size 128 --max_gap 5 --output cindex.csv
"""

import argparse
import logging
import os
from pathlib import Path

import torch

from auc.cindex import evaluate_cindex
from utils.mlflow_utils import setup_mlflow
from utils.run_loader import reconstruct_from_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

setup_mlflow()


def main():
    parser = argparse.ArgumentParser(description="Compute C-index for a Delphi model.")
    parser.add_argument("--runid", type=str, required=True, help="MLflow run ID.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Override block size. Required when model was trained with block_size='auto'.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--min_time_gap",
        type=float,
        default=0.0,
        help="Min gap in years between query time and case event (default: 0, query_time == event_time).",
    )
    parser.add_argument(
        "--max_gap",
        type=float,
        default=5.0,
        help="Max gap in years between query time and control score timestep (default: 5).",
    )
    parser.add_argument("--output", type=str, default="cindex.csv", help="Output CSV filename (default: cindex.csv).")
    args = parser.parse_args()

    model, loaders, _ = reconstruct_from_run(
        run_id=args.runid,
        split="test",
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = loaders["test"]

    cindex_df = evaluate_cindex(
        model=model,
        test_loader=test_loader,
        block_size=args.block_size,
        min_time_gap=args.min_time_gap,
        max_gap=args.max_gap,
        run_id=args.runid,
        output_file=Path(args.output).name,
    )

    out_path = Path(args.output)
    cindex_df.to_csv(out_path, index=False)
    logging.info(f"Saved C-index to {out_path.resolve()}")
    print(cindex_df.to_string(index=False))


if __name__ == "__main__":
    main()
