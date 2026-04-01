> ⚠️ Stability notice
>
> The codebase is evolving and interfaces are not yet stable. Changes may affect the CLI, model architecture, configuration schema, default parameter values and output formats.

# Multi-domain Delphi

This repository extends the **Delphi** core codebase to support **multi-domain longitudinal data**, beyond standard diagnosis codes.

It focuses on:
- Adding heterogeneous domains (e.g. diseases, drugs, lifestyle, HLA alleles, rare variants).
- Defining **custom attention policies** within and across domains.
- Scaling experiments via **Slurm job arrays** while tracking models with MLflow.

## Table of contents

- [Multi-domain Delphi](#multi-domain-delphi)
  - [Table of contents](#table-of-contents)
  - [Software environment](#software-environment)
  - [Training](#training)
    - [Preparing the data for each domain](#preparing-the-data-for-each-domain)
    - [Specifying the attention scheme](#specifying-the-attention-scheme)
      - [Example 1: Fully causal attention (with tie-masking, i.e. no same-time attention)](#example-1-fully-causal-attention-with-tie-masking-ie-no-same-time-attention)
      - [Example 2 — Bidirectional HLA block + causal domains](#example-2--bidirectional-hla-block--causal-domains)
    - [Exemplar command](#exemplar-command)
    - [Mixed precision (`--use_amp`)](#mixed-precision---use_amp)
    - [AUC computation (`--compute_aucs`)](#auc-computation---compute_aucs)
    - [Model tracking with MLflow](#model-tracking-with-mlflow)
  - [Model explainability](#model-explainability)
  - [Submitting a hyperparameter search as Slurm job array](#submitting-a-hyperparameter-search-as-slurm-job-array)
  - [Tips for querying MLflow runs](#tips-for-querying-mlflow-runs)
  - [Notes for developers](#notes-for-developers)

## Software environment

_To be completed_
The code has been tested with the following versions:
- `numpy=1.26.4`
- `pandas=2.2.3`
- `mlflow=2.22.0`
- `torch=2.3.0`

## Training

_To be completed_

### Preparing the data for each domain

Domains are configured via a YAML file (see `config/domain_config_default.yaml` for a reference).
Each entry defines one domain and its properties:

```yaml
diseases:
  projector: embed
  path: diseases
  predict: true
  no_repeat: true   # each disease is recorded only at first diagnosis

lifestyle:
  projector: embed
  path: lifestyle
  age_jitter: true

sex:
  projector: embed
  path: sex
  at_birth: true

genetic_pcs:
  type: continuous
  projector: linear
  path: genetic_pcs
  at_birth: true
  input_size: 40
  n_latent_tokens: 5
```

**`no_repeat: true`** should be set for domains where each token appears at most once per subject by construction (e.g. diseases recorded only at first diagnosis). When enabled, the logits for already-seen tokens are set to `-inf` after each forward pass, before the loss is computed. This prevents the model from learning the spurious pattern of suppressing a disease's logit once it has appeared - an artefact of the data, not a biological signal.

Pass the config file via `--domain_config_yaml` and select which domains to activate with `--domains`:

```bash
python train.py --domain_config_yaml config/domain_config_default.yaml --domains diseases,lifestyle,sex ...
```

The `padding` domain is injected automatically — do not add it to the YAML or `--domains`.

For each domain, create a folder under `data/transforms/tokens/<domain_name>/` with two files:
- `tokens.csv`: columns `subject_id`, `age` (in days), `token_id` — one row per token event, all subjects together.
- `tokenizer.yaml`: list of token names in order; position determines `token_id` (zero-based).

### Specifying the attention scheme

The attention scheme within and across domains is specified via `--attention_scheme`. You can either pass a scheme string directly or use a named alias defined in `config/attention_schemes.yaml`:

```yaml
# config/attention_schemes.yaml
hla_bidir:
  description: HLA bidirectional with sex, rest causal
  scheme: "[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"

hla_causal:
  description: fully causal, no bidirectional HLA
  scheme: "all:causal(mask_ties=True)"
```

```bash
python train.py --attention_scheme hla_bidir ...
```

Add your own aliases to that file to avoid repeating long scheme strings across runs.

#### Example 1: Fully causal attention (with tie-masking, i.e. no same-time attention)

For instance:
`"[sex,diseases,lifestyle,death,hla_alleles,rare_variants]:causal(mask_ties=True)"`

The corresponding attention matrix:

| From \ To    | HLA | sex | diseases | lifestyle | death |
|--------------|-----|-----|----------|-----------|-------|
| **HLA**      | ·   | ·   | ·        | ·         | ·     |
| **sex**      | ←   | ·   | ·        | ·         | ·     |
| **diseases** | ←   | ←   | ←        | ·         | ·     |
| **lifestyle**| ←   | ←   | ←        | ←         | ·     |
| **death**    | ←   | ←   | ←        | ←         | ←     |

In this configuration, all domains follow a strictly causal structure.
Each domain may attend to **past tokens of itself and previous domains**, but never to the future
nor to same-time tokens (`mask_ties=True`).

Two built-in aliases are available for use in any domain group:

| Alias | Expands to |
|-------|-----------|
| `all` | Every domain present in the model (equivalent to listing them all explicitly) |
| `at_birth` | Every domain whose config entry has `at_birth: true` (resolved at model construction time from `config.domains`) |

For example:
```
"all:causal(mask_ties=True)"
"[at_birth]:bidirectional,all:causal(mask_ties=True)"
```

The second scheme gives every at-birth domain (e.g. `sex`, `hla_alleles`, `genetic_pcs`) a bidirectional block and makes all remaining domains causal — without having to enumerate them by name. If you later add or remove a domain from the config, the scheme automatically reflects the change.

#### Example 2 — Bidirectional HLA block + causal domains

On the other hand:
`"[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"`

| From \ To    | HLA | sex | diseases | lifestyle | death |
|--------------|-----|-----|----------|-----------|-------|
| **HLA**      | ↔   | ↔   | ·        | ·         | ·     |
| **sex**      | ↔   | ↔   | ·        | ·         | ·     |
| **diseases** | ←   | ←   | ←        | ·         | ·     |
| **lifestyle**| ←   | ←   | ←        | ←         | ·     |
| **death**    | ←   | ←   | ←        | ←         | ←     |

Here, the HLA allele and sex domains form a **bidirectional static block**, allowing mutual
contextualization of at-birth attributes. All downstream domains follow a causal structure,
ensuring temporal consistency while allowing conditioning on static information.

### Exemplar command

This is an exemplar training command, training with the usual domains (`diseases,lifestyle,sex,death`) plus the `rare_variants` domain:

```bash
python train.py \
  --domains diseases,death,lifestyle,sex,rare_variants \
  --attention_scheme "[sex,diseases,lifestyle,death,rare_variants]:causal(mask_ties=True)" \
  --n_layer 12 \
  --n_embd 240 \
  --experiment_name rare_variants
```

Note: `padding` is added automatically and does not need to be listed in `--domains` or `--attention_scheme`.

### Mixed precision (`--use_amp`)

Enables automatic mixed precision using **bfloat16**, which reduces memory usage and speeds up training on supported GPUs (Ampere and newer):

```bash
python train.py --use_amp ...
```

bfloat16 has the same exponent range as float32, so gradient scaling is not required. If the GPU does not support bfloat16, the flag has no effect.

### AUC computation (`--compute_aucs`)

If passed, AUCs are computed at the end of training and saved as a CSV file under the `aucs/` subdirectory of the run's MLflow artifact directory.

### Model tracking with MLflow

You can specify a custom MLflow location by setting the `MLFLOW_TRACKING_URI` environment variable, otherwise it's the `mlruns` folder within this repo's root directory.
The previous command will create an MLflow experiment called `rare_variants`.
Instructions are provided later on how to query the information logged by MLflow.

## Model explainability

_To be completed_

This section will contain details on how to perform SHAP calculation using Nextflow.

## Submitting a hyperparameter search as Slurm job array

_To be completed_

This section will provide tips to explore different combinations of hyperparameters by using Slurm's job array feature.
It requires generating a tabular file, where columns are command-line arguments of the `train.py` script, and the cells contain their values. Each row is a different run.
Then a Slurm scripts reads this file line by line, building the command based on the configuration given by the row, and submitting it to a different GPU node.

## Tips for querying MLflow runs

```bash
export MLFLOW_TRACKING_URI=$HOME/...

# Examine your available experiments (try to use representative names when you create them)
mlflow experiments search

# Get runs for a given experiment
export EXP_NAME=attention_schemes # an example
export EXP_ID=$(mlflow experiments search | grep -w $EXP_NAME | awk '{print $1}')

# or directly EXP_ID=... 
mlflow runs list --exp-id $EXP_ID
```

## Notes for developers

> **Note on Jupytext usage**
>
> This repository makes extensive use of Jupyter notebooks in `.py` format via **Jupytext**.
> These files can be identified by `# %%` cell separators.
>
> This choice allows the same files to be run both as notebooks and as regular Python scripts, and improves readability and version control compared to `.ipynb` notebooks.
>
> Install:
>
> ```bash
> pip install jupytext
> ```
>
> Convert to `.ipynb`:
>
> ```bash
> jupytext --to ipynb PATH_TO_FILE.py
> ```

_To be completed_

This section will contain:
- Notes on how to extend this codebase.
- Tips on unit tests.
