# HLA SHAP analysis

Counterfactual analysis to estimate the effect of HLA alleles on disease risk, using the Delphi model.

## Method

For each subject carrying a given HLA allele **and** a given disease, we compare the model's predicted logit for that disease against the logit obtained after replacing their HLA block with that of a randomly chosen non-carrier (donor). The difference:

```
Δlogit = logit_original − logit_counterfactual
```

is computed across all folds and all disease-preceding positions. A positive mean Δlogit indicates that carrying the allele increases predicted disease risk. Statistical significance is assessed with a two-sided Wilcoxon signed-rank test.

## Script: `custom_hla_shap.py`

Run from `DELPHI_DIR`:

```bash
python shap/custom_hla_shap.py [options]
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `--disease` | str | Disease name (fuzzy matched against tokenizer) |
| `--disease_id` | int | Disease token ID (takes precedence over `--disease`) |
| `--hla_allele` | str | HLA allele prefix (e.g. `HLA-C*06`); all matching alleles are grouped |
| `--allele_id` | int | HLA allele token ID (takes precedence over `--hla_allele`) |
| `--n_counterfactuals` | int | Number of donor draws per receptor batch to average (default: 1) |
| `--subjects` | str | Path to file of subject IDs to intersect with the test set |
| `--output` | str | Output file path (default: `shap/output_delta_logit/{disease_id}__{allele_id}.pkl`). Supports `{disease_id}` and `{allele_id}` placeholders. |

Either `--disease` or `--disease_id` must be provided. Same for `--hla_allele` / `--allele_id`.

### Examples

```bash
# By name (fuzzy match)
python shap/custom_hla_shap.py \
    --disease "psoriasis" \
    --hla_allele "HLA-C*06"

# By numeric ID
python shap/custom_hla_shap.py \
    --disease_id 713 \
    --allele_id 194

# With multiple counterfactual draws (less noisy estimate)
python shap/custom_hla_shap.py \
    --disease "psoriasis" \
    --hla_allele "HLA-C*06" \
    --n_counterfactuals 5

# Restrict to a subset of subjects
python shap/custom_hla_shap.py \
    --disease_id 713 \
    --allele_id 194 \
    --subjects /path/to/subject_ids.txt

# Custom output path
python shap/custom_hla_shap.py \
    --disease_id 713 \
    --allele_id 194 \
    --output /hps/nobackup/birney/users/bonazzola/delphi/output/delta_{disease_id}__{allele_id}.pkl
```

### Subjects file format

A plain text file, one ID per line, with or without header. Only the first column is used. Example:

```
1234567
2345678
3456789
```

### Output

A pickle file containing a 1D numpy array of Δlogit values (one per disease-preceding position, aggregated across all CV folds).

Stdout also reports:
```
fold 1 n=...
fold 2 n=...
mean Δlogit = 0.XXXX
Wilcoxon two-sided p = X.XXe-XX
Saved to <output_file>
```

## SLURM array job: `custom_shap_hla.sh`

Runs the script as a SLURM array job, one job per allele ID (0–359), for a fixed disease:

```bash
sbatch shap/custom_shap_hla.sh
```

Edit the `--disease` argument inside the script to change the target disease.
