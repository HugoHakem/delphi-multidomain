#!/bin/bash
#SBATCH --job-name=logits_array
#SBATCH --output=logs_logits/logits_%A_%a.out
#SBATCH --error=logs_logits/logits_%A_%a.err
#SBATCH --array=0-15
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
RUNID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ../runids.txt)
echo "Processing RUNID = $RUNID"
python get_logits_to_parquet.py --runid "$RUNID"

