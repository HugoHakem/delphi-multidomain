#!/bin/bash
#SBATCH --job-name=merge_shap
#SBATCH --output=logs/merge_shap/merge_shap_%A_%a.out
#SBATCH --error=logs/merge_shap/merge_shap_%A_%a.err
#SBATCH --array=0-0
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

python merge_shap_chunks.py --run-idx $SLURM_ARRAY_TASK_ID
