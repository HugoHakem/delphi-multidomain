#!/usr/bin/env bash

#SBATCH --job-name=auc_array
#SBATCH --output=logs/auc/auc_%A_%a.out
#SBATCH --error=logs/auc/auc_%A_%a.err
#SBATCH --array=0-21
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

# -----------------------------
# Required environment variables
# -----------------------------
: "${RUNIDS_FILE:?Environment variable RUNIDS_FILE is not set}"

if [[ ! -f "$RUNIDS_FILE" ]]; then
    echo "ERROR: RUNIDS_FILE does not exist: $RUNIDS_FILE" >&2
    exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "SLURM_ARRAY_TASK_ID = $TASK_ID"

# -----------------------------
# Resolve RUNID
# -----------------------------
RUNID=$(sed -n "$((TASK_ID + 1))p" "$RUNIDS_FILE")

if [[ -z "$RUNID" ]]; then
    echo "ERROR: No RUNID found for TASK_ID=$TASK_ID in $RUNIDS_FILE" >&2
    exit 1
fi

echo "Processing RUNID = $RUNID"
echo

# -----------------------------
# Run pipeline
# -----------------------------
python 03_logits_to_parquet.py --runid "$RUNID"
python 04_compute_auc.py       --runid "$RUNID"
