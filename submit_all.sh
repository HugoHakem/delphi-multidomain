#!/bin/bash

while read EXP_ID RUN_ID VAL_LOSS STEP; do
  OUTPUT_FILE="mlruns/${EXP_ID}/${RUN_ID}/artifacts/auc"

  if [[ -f "$OUTPUT_FILE" ]]; then
    echo "⏭️  Skip: ${RUN_ID} ya tiene resultado"
  else
    echo "🚀 Enviando job para run ${RUN_ID} (step ${STEP})"
    sbatch --export=ALL,EXP_ID="$EXP_ID",RUN_ID="$RUN_ID",STEP="$STEP" submit_auc.slurm
  fi
done < auc_runs.txt
