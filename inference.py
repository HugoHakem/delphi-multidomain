# %%
import torch
import os, sys

import mlflow
import re
import numpy as np
import pandas as pd
from typing import List, Dict
from itertools import pairwise
import importlib
from copy import deepcopy
from collections import defaultdict

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "./data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "./output/checkpoints")
from pathlib import Path
root_path = Path("./data/transforms")

from train_scripts.train import Trainer

import data
data = importlib.reload(data)
import data.event_set
EventSet = data.event_set.EventSet

from data.dataset import (
    get_tensors_from_batch,
    truncate_subjects_by_age,
    pad_subjects,
    adjust_to_seqlen,
    DelphiDataset, DelphiDataloader
)

from utils.cv_utils import get_data_partitions
from utils.utils import get_p2i, get_batch

from delphi.model.transformer import (
    Delphi,
    EmbedConfig,
    DelphiConfig,
)

torch.set_grad_enabled(False)

from collections import OrderedDict
import re

def load_legacy_weights_into_delphi(
    model_new,
    state_dict_old,
    skip_prefixes=("transformer.wte", "embedding"),
    verbose=True
):
    """
    Load legacy GPT-style weights into the new Delphi model.

    The function:
    - maps old keys -> new keys using regex rules,
    - skips embeddings (handled separately),
    - checks shape compatibility,
    - loads only the weights that match cleanly,
    - uses strict=False to ignore uninitialized parts gracefully.
    """

    new_sd = model_new.state_dict()
    loaded = {}
    skipped = []

    # ------------------------------------------------------------------
    # Mapping rules: old_key_pattern -> new_key_pattern
    # These patterns correspond to the legacy GPT-like architecture.
    # They must be updated if the legacy checkpoint structure changes.
    # ------------------------------------------------------------------
    KEYMAP_RULES = [
        # LayerNorm
        (r"transformer\.h\.(\d+)\.ln_1\.weight",
         r"transformer.h.\1.ln_1.weight"),
        (r"transformer\.h\.(\d+)\.ln_2\.weight",
         r"transformer.h.\1.ln_2.weight"),

        # Attention projections
        (r"transformer\.h\.(\d+)\.attn\.c_attn\.weight",
         r"transformer.h.\1.attn.c_attn.weight"),
        (r"transformer\.h\.(\d+)\.attn\.c_attn\.bias",
         r"transformer.h.\1.attn.c_attn.bias"),

        (r"transformer\.h\.(\d+)\.attn\.c_proj\.weight",
         r"transformer.h.\1.attn.c_proj.weight"),
        (r"transformer\.h\.(\d+)\.attn\.c_proj\.bias",
         r"transformer.h.\1.attn.c_proj.bias"),

        # MLP
        (r"transformer\.h\.(\d+)\.mlp\.c_fc\.weight",
         r"transformer.h.\1.mlp.c_fc.weight"),
        (r"transformer\.h\.(\d+)\.mlp\.c_fc\.bias",
         r"transformer.h.\1.mlp.c_fc.bias"),

        (r"transformer\.h\.(\d+)\.mlp\.c_proj\.weight",
         r"transformer.h.\1.mlp.c_proj.weight"),
        (r"transformer\.h\.(\d+)\.mlp\.c_proj\.bias",
         r"transformer.h.\1.mlp.c_proj.bias"),

        # Final layer norm
        (r"transformer\.ln_f\.weight",
         r"transformer.ln_f.weight"),
        (r"transformer\.ln_f\.bias",
         r"transformer.ln_f.bias"),

        # lm_head (best-effort; may not always match)
        (r"lm_head\.weight",
         r"embedding_to_logits.weight")
    ]

    # Apply the rules
    def map_key(old_key):
        for patt, repl in KEYMAP_RULES:
            if re.fullmatch(patt, old_key):
                return re.sub(patt, repl, old_key)
        return None

    # ------------------------------------------------------------------
    # Main loop over old weights
    # ------------------------------------------------------------------
    for k_old, v_old in state_dict_old.items():

        # Skip legacy embeddings (already handled)
        if any(k_old.startswith(pref) for pref in skip_prefixes):
            skipped.append(k_old)
            continue

        # Map old -> new
        k_new = map_key(k_old)
        if k_new is None:
            skipped.append(k_old)
            continue

        # Check existence in new model
        if k_new not in new_sd:
            skipped.append(k_old)
            continue

        # Check shape compatibility
        if new_sd[k_new].shape != v_old.shape:
            if verbose:
                print(f"[SKIP: shape mismatch] {k_old} -> {k_new} "
                      f"{v_old.shape} != {new_sd[k_new].shape}")
            skipped.append(k_old)
            continue

        loaded[k_new] = v_old

    # ------------------------------------------------------------------
    # Merge into the new model's state_dict
    # ------------------------------------------------------------------
    merged_state = OrderedDict(new_sd)
    merged_state.update(loaded)

    model_new.load_state_dict(merged_state, strict=False)

    if verbose:
        print("\n=== LEGACY WEIGHT LOADING SUMMARY ===")
        print(f"Loaded {len(loaded)} tensors:")
        for k in loaded:
            print("  ✔", k)
        print(f"\nSkipped {len(skipped)} tensors:")
        for k in skipped:
            print("  ✘", k)

    return model_new


def load_fold_ids(fold_dir: str, num_folds: int = 10) -> List[List[str]]:
    """
    Load subject IDs from predefined fold files.
    Each file must be named `fold_i_of_num.txt` (0-indexed).
    """
    folds = []
    for i in range(1, num_folds+1):
        fname = os.path.join(fold_dir, f"subset{i}of{num_folds}.csv")
        with open(fname) as f:
            ids = [line.strip() for line in f if line.strip()]
            folds.append(ids)
    return folds

def generate_splits(
    folds: List[List[str]],
    n_train_folds: int,
    n_val_folds: int,
    n_test_folds: int,
    val_as_last: bool = True,
) -> List[Dict[str, List[str]]]:
    """
    Generate splits (train/valid/test) given predefined folds.

    Parameters
    ----------
    folds : list of list
        List of folds, each containing subject IDs.
    n_train_folds : int
        Number of folds to use for training.
    n_val_folds : int
        Number of folds to use for validation.
    n_test_folds : int
        Number of folds to use for testing.
    val_as_last : bool
        If True, take validation folds as the last `n_val_folds` among the remaining.
        If False, take the first `n_val_folds`.

    Returns
    -------
    splits : list of dict
        Each dict has keys "train", "valid", "test".
    """
    num_folds = len(folds)
    window_size = n_train_folds + n_val_folds + n_test_folds
    if window_size > num_folds:
        raise ValueError("Not enough folds for requested split sizes.")

    splits = []
    for start in range(0, num_folds, n_test_folds):
        test_idx = list(range(start, start + n_test_folds))
        remaining = [i for i in range(num_folds) if i not in test_idx]
        
        if val_as_last:
            val_idx = remaining[-n_val_folds:]
            train_idx = remaining[:-n_val_folds]
        else:
            val_idx = remaining[:n_val_folds]
            train_idx = remaining[n_val_folds:n_val_folds + n_train_folds]
    
        split = {
            "train": sum([folds[i] for i in train_idx], []),
            "valid": sum([folds[i] for i in val_idx],   []),
            "test":  sum([folds[i] for i in test_idx],  []),
        }       
        splits.append(split)
    
    return splits


def old_get_data_partitions(datafile, fold):
    
    '''
    datafile: numpy file containing three columns (subject_id, time, token_id)
    '''

    load_data_from_bin = lambda file: np.fromfile(file, dtype=np.uint32).reshape(-1, 3)
    
    data = load_data_from_bin(datafile)
    fold_ids = load_fold_ids("./data/transforms/subject_lists", num_folds=10)

    splits = generate_splits(fold_ids, n_train_folds=7, n_val_folds=1, n_test_folds=2, val_as_last=True)
    split_idx = fold - 1
    
    train_ids  = splits[split_idx]["train"]
    val_ids    = splits[split_idx]["valid"]
    test_ids   = splits[split_idx]["test"]
        
    train_data = data[np.isin(data[:,0], train_ids)]
    val_data   = data[np.isin(data[:,0], val_ids)]
    test_data  = data[np.isin(data[:,0], test_ids)]
    
    train_p2i  = get_p2i(train_data)
    val_p2i    = get_p2i(val_data)
    test_p2i   = get_p2i(test_data)
    
    return (train_data, train_p2i, train_ids), \
           (val_data, val_p2i, val_ids), \
           (test_data, test_p2i, test_ids)


MLFLOW_URI = "./output/mlruns"
MLFLOW_URI = "./mlruns"
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

tokens_path = root_path / 'tokens'

default_cfg_per_domain = {
    # 'genetic_pcs': EmbedConfig(projector="linear", path=tokens_path / 'genetic_pcs', type='continuous', at_birth=True),
    'diseases':    EmbedConfig(projector="embed", path=tokens_path / 'diseases',    predict=True),
    'death':       EmbedConfig(projector="embed", path=tokens_path / 'death',       predict=True),
    'lifestyle':   EmbedConfig(projector="embed", path=tokens_path / 'lifestyle',   age_jitter=True),  
    "hla_alleles": EmbedConfig(projector="embed", path=tokens_path / 'hla_alleles', at_birth=True),
    "sex":         EmbedConfig(projector="embed", path=tokens_path / 'sex',         at_birth=True),
    "padding":     EmbedConfig(projector="embed")    
}

domain_cfg = default_cfg_per_domain

attention_scheme = "[diseases, death, lifestyle, sex, padding]:causal(mask_ties=True)"

config = DelphiConfig(
    n_layer=6, 
    n_embd=240,
    token_dropout=0.1, 
    domains=domain_cfg, 
    attention_scheme=attention_scheme
)
    
model  = Delphi(config).to(DEVICE)

# %%
mlflow.set_tracking_uri(MLFLOW_URI)

hla_exp   = "278131880607437980"
def fix_artifact_uri(path):
    return re.sub(".*mlruns", "/homes/bonazzola/repos/delphi/mlruns", path)

runs_hla = mlflow.search_runs(experiment_ids=hla_exp)
runs_hla.artifact_uri = runs_hla.artifact_uri.apply(fix_artifact_uri)
runid = runs_hla.run_id[0]

display(runs_hla)

run = mlflow.get_run(runid)
run_artifact_uri = fix_artifact_uri(run.info.artifact_uri)

ckpt_path  = list((Path(run_artifact_uri) / "checkpoints").glob("*pt"))[-1]
checkpoint = torch.load(ckpt_path, map_location=DEVICE)        
state_dict = { k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items() }
wte_weights = state_dict['transformer.wte.weight']

domains = ['padding','sex','lifestyle','hla_alleles','diseases','death']
kk = np.array([0] + [ model.transformer.embed.domain_embed[d].projector.num_embeddings for d in domains ])

for i, (start, end) in enumerate(pairwise(kk.cumsum())):
    domain = domains[i]
    model.transformer.embed.domain_embed[domain].parameters = wte_weights[start:end]

model = load_legacy_weights_into_delphi(model, state_dict)

# NEW DATA
# train_ids, val_ids, test_ids = get_data_partitions("./data/transforms/subject_lists", fold=0)

# OLD DATA
train, valid, test = old_get_data_partitions("./data/transforms/ukb_real_5_folds_4digit/all.bin", 1)
train_data, train_p2i, train_ids = train
val_data, val_p2i, val_ids       = valid
test_data, test_p2i, val_ids     = test

# %%
# val_dataset      = DelphiDataset(domains=domain_cfg, root="./data/transforms", subjects=val_ids).to('cuda')
# test_dataset     = DelphiDataset(domains=domain_cfg, root="./data/transforms", subjects=test_ids).to('cuda')
# val_dataloader   = DelphiDataloader(val_dataset,   batch_size=16)
# test_dataloader  = DelphiDataloader(test_dataset,  batch_size=16)
# dataloaders      = [train_dataloader, val_dataloader, test_dataloader]

# dataset = DelphiDataset(domains=domain_cfg, root="./data/transforms", subjects=train_ids).to('cuda')
# dataloader = train_dataloader

tokenizer = pd.read_csv("data/delphi_labels_chapters_colours_icd_with_hla4d.csv")["name"].to_dict()

x, a, y, b, subject_ids = get_batch(range(256), train_data, train_p2i, select='left', return_subject_ids=True, block_size=128)
subject_ids = subject_ids.tolist()

event_set = EventSet.from_batch((x, a, y, b), subject_ids=subject_ids, tokenizer=tokenizer)
event_set.X_tokens, event_set.X_ages

dataset    = DelphiDataset(domains=domain_cfg, root="./data/transforms", subjects=subject_ids).to('cuda')
dataloader = DelphiDataloader(dataset, batch_size=256)  
batch = next(iter(dataloader))

# trainer = Trainer(model, dataloaders, optimizer:=None, scheduler:=None, logger:=None, mlflow_params:=None)

# pd.concat(
#     list(map(lambda x: pd.DataFrame(x.cpu().numpy(), columns=["subject_id", "age", "token"]), dataset[2005166].values()))
# ).\
# astype({"token": int, "subject_id": int}).\
# assign(age=lambda df: (df.age / 365.25).round(2)).\
# sort_values(["age", "token"])

model.to('cuda')
x, ages, subject_ids = get_tensors_from_batch(batch)
max_ages             = model.get_max_ages_per_subject(ages, subject_ids)
x, ages, subject_ids = model.insert_no_event_tokens(x, ages, subject_ids)
x, ages, subject_ids = model.mask_tokens_after_age (x, ages, subject_ids, max_ages)
x, ages, subject_ids = adjust_to_seqlen(x, ages, subject_ids, seqlen:=128, verbose="debug")       
logits, att = model(x, ages, subject_ids)

# %%
embeddings = model.transformer.embed(x)
# x_tensor, ages_tensor, embeddings_tensor, uniq_subjs, domains = model.to_tensor(x, ages, embeddings, subject_ids)

embeddings['padding']
# %%
# Okay, now let's try to load the weights using the old model code.
from old_model import Delphi as OldDelphi

old_model = OldDelphi.from_checkpoint(ckpt_path)
old_logits, loss, _, _ = old_model(*event_set.to_tensor().to('cuda').as_tuple())
old_logits = old_logits.cpu()[:, :, 372:]


def attach_tracer(model, prefix=""):
    trace = {}
    hooks = []

    def register(module, name):
        def hook(_, inp, out):
            trace[name] = out.detach().cpu()
        return module.register_forward_hook(hook)

    for name, module in model.named_modules():
        # filtramos cosas que no tienen datos (e.g. container modules)
        if not list(module.children()):  
            h = register(module, prefix + name)
            hooks.append(h)

    return trace, hooks

# %%

trace_old, hooks_old = attach_tracer(old_model, prefix="old/")
trace_new, hooks_new = attach_tracer(model, prefix="new/")

# Hacemos forward
_ = old_model(*event_set.to_tensor().to('cuda').as_tuple())
_ = model(x, ages, subject_ids)

# Luego removemos los hooks para no ensuciar nada
for h in hooks_old: h.remove()
for h in hooks_new: h.remove()
# %%
trace_old['old/transformer.wte'][0]
# %%
trace_new['new/transformer.embed.domain_embed.padding.projector'][:, 0]
# %%
list(old_model.transformer.wte.parameters())[0].cpu().numpy()
# %%
x['padding']
# %%
