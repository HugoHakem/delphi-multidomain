#!/bin/bash
#SBATCH --job-name=auc_eval
#SBATCH --output=logs/auc_%j.out
#SBATCH --error=logs/auc_%j.err
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

echo "Running AUC evaluation for experiment ${EXP_ID}, run ${RUN_ID}, step ${STEP}"

python evaluate_auc.py \
  --input_path data/ukb_real_data/ \
  --data_file_prefix "ukb_real_" \
  --output_path mlruns/${EXP_ID}/${RUN_ID}/artifacts/auc \
  --model_ckpt_path mlruns/${EXP_ID}/${RUN_ID}/artifacts/checkpoints/best_ckpt__${RUN_ID}__${STEP}.pt \
  --no_event_token_rate 5 \
  --health_token_replacement_prob 0.0 \
  --dataset_subset_size -1 \
  --n_bootstrap 100 \
  --filter_min_total 100 \
  --disease_chunk_size 200
