python evaluate_auc.py \
  --input_path data/ukb_real_data/ \
  --data_file_prefix "ukb_real_hla_" \
  --output_path auc_ukb_real_hla_f945b0c1f1b84294bf29a6d39e6ef831 \
  --model_ckpt_path ./Delphi-hla/ckpt__f945b0c1f1b84294bf29a6d39e6ef831__100000.pt \
  --no_event_token_rate 5 \
  --health_token_replacement_prob 0.0 \
  --dataset_subset_size -1 \
  --n_bootstrap 100 \
  --filter_min_total 100 \
  --disease_chunk_size 200

# --model_ckpt_path ./Delphi/ckpt__f5bdc1ea47bc48eeab170bf54d52b3ec__100000.pt \
# ./Delphi/ckpt__f5bdc1ea47bc48eeab170bf54d52b3ec__100000.pt