#!/bin/bash
#SBATCH --job-name=opt_parquet
#SBATCH --output=logs/opt/parquetoptimize_%A_%a.out
#SBATCH --error=logs/opt_parquet/optimize_%A_%a.err
#SBATCH --array=0-15
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

# Cargar entorno si es necesario
# module load python
# source activate tu_entorno

FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" parquet_list.txt)

echo "🧠 Procesando archivo: $FILE"
python optimize_parquet.py "$FILE"

