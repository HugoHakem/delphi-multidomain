## Multi-domain Delphi
This repository contains an extension of the Delphi core codebase, allowing to easily add multi-domain data (apart from the usual diagnosis codes).
It allows to define custom attention masking schemes within each domain and across domains.

> [!NOTE]
> This repository makes extensive use of Jupyter notebooks in `.py` format via **Jupytext**.
> These files can be identified by `# %%` cell separators.
>
> This choice allows the same files to be run both as notebooks and as regular Python scripts, and improves readability and version control compared to `.ipynb` notebooks.
>  
> To install Jupytext:
> ```bash
> pip install jupytext
> ```
>
> To convert a file to `.ipynb`:
> ```bash
> jupytext --to ipynb PATH_TO_FILE.py
> ```



## Training (_to be completed_)

### Preparing the data for each domain
```python
from pathlib import Path
tokens_path = Path("./data/tokens")
domain_config = {
   'diseases':      EmbedConfig(projector="embed", path=tokens_path / 'diseases',       predict=True), # default is predict=False 
   'death':         EmbedConfig(projector="embed", path=tokens_path / 'death',          predict=True),
   'drugs':         EmbedConfig(projector="embed", path=tokens_path / 'drugs',          predict=True),
   'lifestyle':     EmbedConfig(projector="embed", path=tokens_path / 'lifestyle',      age_jitter=True),  
   "hla_alleles":   EmbedConfig(projector="embed", path=tokens_path / 'hla_alleles',    at_birth=True),
   "sex":           EmbedConfig(projector="embed", path=tokens_path / 'sex',            at_birth=True),
   "padding":       EmbedConfig(projector="embed")
   "rare_variants": EmbedConfig(projector="embed", path=tokens_path / 'rare_variants',  at_birth=True),
}
```
Then you need to create a folder for the domain, e.g. `./data/tokens/rare_variants` with two files, named `tokens.csv` and `tokenizer.yaml`.
- `tokens.csv` contains `subject_id`, `age` (in days) and `token_id`, one row per token (all subjects together).
- `tokenizer.yaml` contains each token in order, and from this order the mapping to `token_id` is established.

### Specifying the attention scheme
The attention scheme within and across domains is specified via a string command-line argument:

For instance:
`"[sex,diseases,lifestyle,death,padding,hla_alleles,rare_variants]:causal(mask_ties=True)"`

specifies that all tokens can only attend to tokens that are _strictly_ in their past (`mask_ties=True`).

On the other hand:
`"[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death,hla_alleles,padding]:causal(mask_ties=True)"`
will allow bidirectional attention within the HLA allele and sex domains, however the rest of the tokens will be able to have causal attention with respect to the previous and also themselves.

### Exemplar command
This is an exemplar training command, training with the usual domains (`diseases,lifestyle,sex,death`) plus the `rare_variants` domain:
```
python train.py \
  --domains diseases,death,lifestyle,sex,rare_variants \
  --attention_scheme "[sex,diseases,lifestyle,death,padding,rare_variants]:causal(mask_ties=True)" \
  --n_layer 12 \
  --n_embd 240 \
  --experiment_name rare_variants
```

### Model tracking with MLflow
You can specify a custom MLflow location by setting the MLFLOW_URI environment variable, otherwise it's the `mlruns` folder within this repo's root directory.
The previous command will create an MLflow experiment called `rare_variants`. 


## Evaluation


## Model explainability


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
_To be completed_
This section will contain:
- Notes on how to extend this codebase. 
- Tips on unit tests.
