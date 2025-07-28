EXPID=437945341875567335

# EXPID=278131880607437980
N_CHUNKS=${N_CHUNKS:=1000}

for FOLD in `seq 1 5`; do
  python shap_agg_parallel_cv.py --experiment_id $EXPID --val_fold $FOLD --device=cpu --num_chunks $N_CHUNKS --chunk_idx $CHUNK_INDEX $@
done
