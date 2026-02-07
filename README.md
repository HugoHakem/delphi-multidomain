# Multi-domain Delphi

This repository extends the **Delphi** core codebase to support **multi-domain longitudinal data**, beyond standard diagnosis codes.

It focuses on:
- Adding heterogeneous domains (e.g. diseases, drugs, lifestyle, HLA alleles, rare variants).
- Defining **custom attention policies** within and across domains.
- Scaling experiments via **Slurm job arrays** while tracking models with MLflow.

## Table of contents

- [Overview](#multi-domain-delphi)
- [Training](#training)
  - [Preparing the data for each domain](#preparing-the-data-for-each-domain)
  - [Specifying the attention scheme](#specifying-the-attention-scheme)
  - [Exemplar command](#exemplar-command)
  - [Model tracking with MLflow](#model-tracking-with-mlflow)
- [Evaluation](#evaluation)
- [Model explainability](#model-explainability)
- [Hyperparameter search with Slurm](#submitting-a-hyperparameter-search-as-slurm-job-array)
- [Querying MLflow runs](#tips-for-querying-mlflow-runs)
- [Developer notes](#notes-for-developers)

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
```python
from pathlib import Path
tokens_path = Path("./data/tokens")
domain_config = {
   'diseases':      DomainConfig(projector="embed", path=tokens_path / 'diseases',       predict=True), # default is predict=False 
   'death':         DomainConfig(projector="embed", path=tokens_path / 'death',          predict=True),
   'drugs':         DomainConfig(projector="embed", path=tokens_path / 'drugs',          predict=True),
   'lifestyle':     DomainConfig(projector="embed", path=tokens_path / 'lifestyle',      age_jitter=True),  
   "hla_alleles":   DomainConfig(projector="embed", path=tokens_path / 'hla_alleles',    at_birth=True),
   "sex":           DomainConfig(projector="embed", path=tokens_path / 'sex',            at_birth=True),
   "padding":       DomainConfig(projector="embed")
   "rare_variants": DomainConfig(projector="embed", path=tokens_path / 'rare_variants',  at_birth=True),
}
```
Then you need to create a folder for the domain, e.g. `./data/tokens/rare_variants` with two files, named `tokens.csv` and `tokenizer.yaml`.
- `tokens.csv` contains `subject_id`, `age` (in days) and `token_id`, one row per token (all subjects together).
- `tokenizer.yaml` contains each token in order, and from this order the mapping to `token_id` is established. Note that the token indexing is zero-based, meaning that the first element of `tokenizer.yaml` gets assigned index `0` in `tokens.csv`.

### Specifying the attention scheme
The attention scheme within and across domains is specified via a string command-line argument:

#### Example 1: Fully causal attention (with tie-masking, i.e. no same-time attention)
For instance:
`"[sex,diseases,lifestyle,death,padding,hla_alleles,rare_variants]:causal(mask_ties=True)"`

The corresponding 
| From \ To        | HLA | sex | diseases | lifestyle | death | padding |
|------------------|-------------|-----|----------|-----------|-------|---------|
| **HLA**  | · | · | · | · | · | · |
| **sex**          | ← | · | · | · | · | · |
| **diseases**     | ← | ← | ← | · | · | · |
| **lifestyle**    | ← | ← | ← | ← | · | · |
| **death**        | ← | ← | ← | ← | ← | · |
| **padding**      | ← | ← | ← | ← | ← | ← |

In this configuration, all domains follow a strictly causal structure.
Each domain may attend to **past tokens of itself and previous domains**, but never to the future
nor to same-time tokens (`mask_ties=True`).

#### Example 2 — Bidirectional HLA block + causal domains
On the other hand:
`"[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death,hla_alleles,padding]:causal(mask_ties=True)"`

| From \ To        | HLA | sex | diseases | lifestyle | death | padding |
|------------------|-------------|-----|----------|-----------|-------|---------|
| **HLA**  | ↔ | ↔ | · | · | · | · |
| **sex**          | ↔ | ↔ | · | · | · | · |
| **diseases**     | ← | ← | ← | · | · | · |
| **lifestyle**    | ← | ← | ← | ← | · | · |
| **death**        | ← | ← | ← | ← | ← | · |
| **padding**      | ← | ← | ← | ← | ← | ← |

Here, the HLA allele and sex domains form a **bidirectional static block**, allowing mutual
contextualization of at-birth attributes. All downstream domains follow a causal structure,
ensuring temporal consistency while allowing conditioning on static information.

### Exemplar command
This is an exemplar training command, training with the usual domains (`diseases,lifestyle,sex,death,padding`) plus the `rare_variants` domain:
```
python train.py \
  --domains diseases,death,lifestyle,sex,rare_variants,padding \
  --attention_scheme "[sex,diseases,lifestyle,death,padding,rare_variants]:causal(mask_ties=True)" \
  --n_layer 12 \
  --n_embd 240 \
  --experiment_name rare_variants
```

### Model tracking with MLflow
You can specify a custom MLflow location by setting the `MLFLOW_TRACKING_URI` environment variable, otherwise it's the `mlruns` folder within this repo's root directory.
The previous command will create an MLflow experiment called `rare_variants`. 
Instructions are provided later on how to query the information logged by MLflow.

## Evaluation
A Nextflow pipeline is available to compute AUCs on a Slurm cluster. The objective is to parallelize the logit computation across many CPUs.
Note that it generates bulky intermediate logit files.

You simply need to generate a file called `runs.csv` with the `runid` header and a set of MLflow run IDs, one per line. Place it in the `auc/scripts` folder and run the following.

I recommend setting the `MLFLOW_TRACKING_URI` environment variable in your `~/.bashrc`

```
module load nextflow

cd auc/scripts
nextflow run auc-calculation.nf -profile slurm
```

This will produce a set of AUC files, split by chunks of diseases.


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
> ```bash
> pip install jupytext
> ```
> 
> Convert to `.ipynb`:
> ```bash
> jupytext --to ipynb PATH_TO_FILE.py
> ```

_To be completed_

This section will contain:
- Notes on how to extend this codebase. 
- Tips on unit tests.
