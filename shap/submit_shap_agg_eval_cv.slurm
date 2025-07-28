#!/bin/bash
#SBATCH --job-name=shap-agg-eval
#SBATCH --output=logs/shap-agg-eval/%A_%a.out
#SBATCH --error=logs/shap-agg-eval/%A_%a.err
#SBATCH --array=1-1000%300
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

export CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}
bash shap-agg-eval_cv.sh --device=cpu
