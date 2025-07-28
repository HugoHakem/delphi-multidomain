#!/bin/bash
#SBATCH --job-name=summarize_gwas
#SBATCH --output=logs/gwas_summarize/summarize_%A_%a.out
#SBATCH --error=logs/gwas_summarize/summarize_%A_%a.err
#SBATCH --array=0-720   # <-- Reemplazá <N-1> con el número de archivos - 1
#SBATCH --time=00:10:00
#SBATCH --mem=8G

INPUT_DIR=gwas_outputs_60
FILES=($(ls ${INPUT_DIR}/embedding_*.assoc.linear))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "Processing $FILE"
python gwas_summarize_one.py "$FILE"

