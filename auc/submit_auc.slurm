#!/bin/bash
#SBATCH --job-name=auc_eval
#SBATCH --output=logs/%x/auc_%j.out
#SBATCH --error=logs/%x/auc_%j.err
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

echo "Running AUC evaluation for experiment ${EXP_ID}, run ${RUN_ID}, step ${STEP}, with checkpoint ${CKPT_FILE}, val data ${VAL_FILENAME}"

DELPHI_LABELS="delphi_labels_chapters_colours_icd_with_hla4d.csv"
# DELPHI_LABELS="delphi_labels_chapters_colours_icd.csv"

python evaluate_auc.py \
  --input_path . \
  --data_filename ${VAL_FILENAME} \
  --output_path mlruns/${EXP_ID}/${RUN_ID}/artifacts/auc \
  --delphi_labels $DELPHI_LABELS \
  --model_ckpt_path mlruns/${EXP_ID}/${RUN_ID}/artifacts/checkpoints/${CKPT_FILE} \
  --no_event_token_rate 5 \
  --health_token_replacement_prob 0.0 \
  --dataset_subset_size -1 \
  --n_bootstrap 100 \
  --filter_min_total 20 \
  --disease_chunk_size 200
