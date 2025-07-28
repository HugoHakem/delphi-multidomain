#!/bin/bash
#SBATCH --job-name=generate_embeddings
#SBATCH --output=logs/generate_embeddings/job_%A_%a.out
#SBATCH --error=logs/generate_embeddings/job_%A_%a.err
#SBATCH --array=0-6  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G

# Define las edades (indexadas por SLURM_ARRAY_TASK_ID)
AGES=(20 30 40 50 60 70 80)

AGE=${AGES[$SLURM_ARRAY_TASK_ID]}
RUNID="5e79356b24de4d7badd9ae7e2ec55da8"
RUNID="28cb9dfb2c5446eabc66424919b13349"
echo "Running for age: $AGE"

python generate_embeddings.py \
    --age "$AGE" \
    --output "embeddings_${AGE}.csv" \
    --run_id ${RUNID}
