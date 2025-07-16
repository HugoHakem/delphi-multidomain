
DELPHI_LABELS=${DELPHI_LABELS:="delphi_labels_chapters_colours_icd_with_hla4d.csv"}
LABELS=${LABELS:="./data/ukb_real_5_folds_4digit/labels.csv"}

RUNID="f945b0c1f1b84294bf29a6d39e6ef831"

N_CHUNKS=${N_CHUNKS:=1000}
CKPT_PATH=${CKPT_PATH:="./Delphi-hla/ckpt__${RUNID}__100000.pt"}
OUTPUT_PKL=shap_values_chunk${CHUNK_INDEX}of${N_CHUNKS}_${RUNID}.pkl

python shap-agg-parallel.py \
  --delphi_labels $DELPHI_LABELS \
  --labels $LABELS \
  --ckpt_path $CKPT_PATH \
  --data_root ./data/ukb_real_5_folds_4digit \
  --output_pickle $OUTPUT_PKL \
  --num_chunks $N_CHUNKS \
  --chunk_idx $CHUNK_INDEX \
  $@
