# python shap-agg-eval-hla_FIX.py \
python shap-agg-eval.py \
  --delphi_labels delphi_labels_chapters_colours_icd_with_hla4d.csv \
  --labels ./data/ukb_real_data_4digit/labels.csv \
  --ckpt_path ./Delphi-hla/ckpt__f945b0c1f1b84294bf29a6d39e6ef831__100000.pt \
  --data_root ./data/ukb_real_data_4digit \
  --output_pickle shap_values_tenth_f945b0c1f1b84294bf29a6d39e6ef831.pkl

