#!/bin/bash

# 1. Create parent run and capture ID
RUN_NAME="cross_validation_parent"
PARENT_RUN_ID=$(python -c "import mlflow; run = mlflow.start_run(run_name=${RUN_NAME}); print(run.info.run_id); mlflow.end_run()")

echo "Parent run: $PARENT_RUN_ID"

# 2. Submit job array inline, exporting parent run id
sbatch \
  --export=PARENT_RUN_ID=$PARENT_RUN_ID \
  --job-name=crossval \
  --gpus=h200:1 \
  --array=1-5 \
  --time=2:00:00 \
  --cpus-per-task=4 \
  --mem=32G \
  --output=logs/%A_%a.out \
  --wrap="python train_fold.py --fold \${SLURM_ARRAY_TASK_ID} --parent_run \${PARENT_RUN_ID}"
