#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --job-name=logits_by_chunks
#SBATCH -o logs/compute_logits_by_chunk/auc_%A_%a.out

RUNIDS_FILE="runids_drugs.txt"   # lista de runids del experimento
N_CHUNKS=50                # cuántos chunks por modelo
TOTAL_MODELS=$(wc -l < $RUNIDS_FILE)
TOTAL_TASKS=$((TOTAL_MODELS * N_CHUNKS))

echo "Total models: $TOTAL_MODELS"
echo "Total tasks : $TOTAL_TASKS"

# SLURM_ARRAY_TASK_ID va de 0 a TOTAL_TASKS-1
TID=${SLURM_ARRAY_TASK_ID}

# Derivar modelo y chunk
MODEL_INDEX=$(( TID / N_CHUNKS ))
CHUNK_INDEX=$(( TID % N_CHUNKS ))

RUNID=$(sed -n "$((MODEL_INDEX+1))p" $RUNIDS_FILE)

echo "-----------------------------------"
echo " SLURM task ID : $TID"
echo " MODEL_INDEX   : $MODEL_INDEX"
echo " CHUNK_INDEX   : $CHUNK_INDEX"
echo " RUNID         : $RUNID"
echo "-----------------------------------"

EXPID="352591210095838422"

python auc_chunk.py \
    --experiment_id $EXPID \
    --runid $RUNID \
    --chunk_index $CHUNK_INDEX \
    --n_chunks $N_CHUNKS \
    --output_dir /hps/nobackup/birney/users/bonazzola/auc/full_logits
