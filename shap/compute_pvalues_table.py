"""
Compute Wilcoxon signed-rank p-values from delta-logit pkl files produced by
custom_hla_shap.py, and compile results into a table as a function of n_counterfactuals.

Output: shap/delta_logits_OnlyWhite/pvalues_table.csv
"""

import pickle
from pathlib import Path
import re
import numpy as np
import pandas as pd
import yaml
from scipy import stats

DELPHI_DIR = Path("/nfs/research/birney/users/bonazzola/repos/delphis/delphi-refactor")
BASE_DIR = DELPHI_DIR / "shap/delta_logits_OnlyWhite"

hla_tokenizer = yaml.safe_load(
    open(DELPHI_DIR / "data/transforms/tokens/hla_alleles/tokenizer.yaml")
)
disease_tokenizer = list(yaml.safe_load(
    open(DELPHI_DIR / "data/transforms/tokens/diseases/tokenizer.yaml")
))

rows = []
pkl_files = list(BASE_DIR.glob("n_counterfactuals_*/*.pkl"))
print(f"Found {len(pkl_files)} pkl files")

for pkl_path in pkl_files:
    # Parse n_counterfactuals from parent dir name
    m_ncf = re.match(r"n_counterfactuals_(\d+)", pkl_path.parent.name)
    if not m_ncf:
        continue
    n_cf = int(m_ncf.group(1))

    # Parse disease_id and allele_id from filename
    m_ids = re.match(r"(\d+)_(\d+)\.pkl", pkl_path.name)
    if not m_ids:
        continue
    disease_id = int(m_ids.group(1))
    allele_id  = int(m_ids.group(2))

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  SKIP (load error): {pkl_path} — {e}")
        continue

    delta = np.asarray(data["delta"]).ravel()

    if len(delta) < 10:
        stat, p = np.nan, np.nan
    else:
        try:
            stat, p = stats.wilcoxon(delta)
        except Exception:
            stat, p = np.nan, np.nan

    disease_name = disease_tokenizer[disease_id] if disease_id < len(disease_tokenizer) else str(disease_id)
    allele_name  = hla_tokenizer[allele_id]       if allele_id  < len(hla_tokenizer)  else str(allele_id)

    rows.append({
        "disease_id":       disease_id,
        "disease_name":     disease_name,
        "allele_id":        allele_id,
        "allele_name":      allele_name,
        "n_counterfactuals": n_cf,
        "n_subjects":       len(delta),
        "mean_delta":       float(np.mean(delta)),
        "median_delta":     float(np.median(delta)),
        "wilcoxon_stat":    float(stat) if not np.isnan(stat) else np.nan,
        "p_value":          float(p)    if not np.isnan(p)    else np.nan,
    })

df = pd.DataFrame(rows)
df = df.sort_values(["disease_id", "allele_id", "n_counterfactuals"]).reset_index(drop=True)

out_path = BASE_DIR / "pvalues_table.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.head(20).to_string(index=False))
