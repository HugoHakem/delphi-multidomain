#!/usr/bin/env bash

#SBATCH -t 00:10:00
#SBATCH -J indices
#SBATCH --mem=16g
#SBATCH -o logs/indices/indices_%A_%a.out

set -euo pipefail

# -----------------------------
# Required environment variables
# -----------------------------
: "${RUNIDS_FILE:?Environment variable RUNIDS_FILE is not set}"

if [[ ! -f "$RUNIDS_FILE" ]]; then
    echo "ERROR: RUNIDS_FILE does not exist: $RUNIDS_FILE" >&2
    exit 1
fi

# -----------------------------
# Config
# -----------------------------
DRY_RUN=${DRY_RUN:-0}   # default: no dry run

N_MODELS=100
N_CHUNKS=50
N_DCHUNKS=10

TASK_ID=${SLURM_ARRAY_TASK_ID}

TOTAL_TASKS=$((N_MODELS * N_CHUNKS * N_DCHUNKS))

# -----------------------------
# Index decomposition
# -----------------------------
RUN_IDX=$(( TASK_ID / (N_CHUNKS * N_DCHUNKS) ))
REST=$(( TASK_ID % (N_CHUNKS * N_DCHUNKS) ))
CHUNK_IDX=$(( REST / N_DCHUNKS ))
DCHUNK_IDX=$(( REST % N_DCHUNKS ))

# -----------------------------
# Resolve RUNID
# -----------------------------
RUNID=$(sed -n "$((RUN_IDX + 1))p" "$RUNIDS_FILE")

if [[ -z "$RUNID" ]]; then
    echo "ERROR: No RUNID found for RUN_IDX=$RUN_IDX in $RUNIDS_FILE" >&2
    exit 1
fi

# -----------------------------
# Command
# -----------------------------
CMD="python 02_compute_case_ctrl_indices.py \
    --runid $RUNID \
    --chunk_index $CHUNK_IDX \
    --dchunk $DCHUNK_IDX \
    --n_dchunks $N_DCHUNKS"

# -----------------------------
# Logging
# -----------------------------
echo "RUNID        = $RUNID"
echo "RUN_IDX      = $RUN_IDX"
echo "CHUNK_IDX    = $CHUNK_IDX"
echo "DCHUNK_IDX   = $DCHUNK_IDX"
echo "TASK_ID      = $TASK_ID"
echo "TOTAL_TASKS  = $TOTAL_TASKS"
echo

# -----------------------------
# Execution / Dry run
# -----------------------------
if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$TASK_ID" -lt 3 || "$TASK_ID" -eq $((TOTAL_TASKS / 2)) || "$TASK_ID" -ge $((TOTAL_TASKS - 3)) ]]; then
        echo "[DRY-RUN]"
        echo "$CMD"
        echo
    fi
else
    echo "[RUN]"
    echo "$CMD"
    eval "$CMD"
fi
