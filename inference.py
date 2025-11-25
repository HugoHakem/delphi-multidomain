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

import data
from data.dataset import DelphiDataset, DelphiDataloader
from utils.cv_utils import get_data_partitions
from utils.utils import get_p2i, get_batch

from delphi.model.transformer import (
    Delphi,
    EmbedConfig,
    DelphiConfig,
)

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
    n_layer=12, 
    n_embd=240,
    token_dropout=0.1, 
    domains=domain_cfg, 
    attention_scheme=attention_scheme
)
    
model  = Delphi(config).to(DEVICE)

# mlflow.set_tracking_uri("./output/mlruns")
mlflow.set_tracking_uri("./mlruns")

hla_exp   = "278131880607437980"
def fix_artifact_uri(path):
    return re.sub(".*mlruns", "/homes/bonazzola/repos/delphi/mlruns", path)

runs_hla = mlflow.search_runs(experiment_ids=hla_exp)
runs_hla.artifact_uri = runs_hla.artifact_uri.apply(fix_artifact_uri)
runid = runs_hla.run_id[0]

display(runs_hla)

run = mlflow.get_run(runid)
run_artifact_uri = fix_artifact_uri(run.info.artifact_uri)

ckpt_path = list((Path(run_artifact_uri) / "checkpoints").glob("*pt"))[-1]
checkpoint = torch.load(ckpt_path, map_location=DEVICE)        
state_dict = checkpoint['model']
state_dict = { k.replace("_orig_mod.", ""): v for k, v in state_dict.items() }
wte_weights = state_dict['transformer.wte.weight']

domains = ['padding','sex','lifestyle','hla_alleles','diseases','death']
kk = np.array([0] + [ model.transformer.embed.domain_embed[d].projector.num_embeddings for d in domains ])

for i, (start, end) in enumerate(pairwise(kk.cumsum())):
    domain = domains[i]
    model.transformer.embed.domain_embed[domain].parameters = wte_weights[start:end]

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

# NEW DATA
train_ids, val_ids, test_ids = get_data_partitions("./data/transforms/subject_lists", fold=0)
train_dataset    = DelphiDataset(domains=domain_cfg, root="./data/transforms", subjects=train_ids)
val_dataset      = DelphiDataset(domains=domain_cfg, root="./data/transforms", subjects=val_ids)
test_dataset     = DelphiDataset(domains=domain_cfg, root="./data/transforms", subjects=test_ids)

train_dataloader = DelphiDataloader(train_dataset, batch_size=16)
val_dataloader   = DelphiDataloader(val_dataset, batch_size=16)
test_dataloader  = DelphiDataloader(test_dataset, batch_size=16)
dataloaders      = [train_dataloader, val_dataloader, test_dataloader]

dataset = train_dataset
dataloader = train_dataloader

# OLD DATA
train, valid, test = old_get_data_partitions("./data/transforms/tokens/drugs/all.bin", 1)
train_data, train_p2i, train_ids = train
val_data, val_p2i, val_ids       = valid
test_data, test_p2i, val_ids     = test

## %%
tokenizer = pd.read_csv("./data/transforms/tokens/drugs/labels.csv").assign(token=lambda df: df.token.astype(int)+1).set_index("token").iloc[:,0].to_dict()

data = importlib.reload(data)
import data.event_set
EventSet = data.event_set.EventSet

# %%
x, a, y, b, subject_ids = get_batch(range(100), train_data, train_p2i, select='left', return_subject_ids=True, block_size=96)
event_set = EventSet.from_batch((x, a, y, b), subject_ids=subject_ids, tokenizer=tokenizer)
event_set[2005166].as_dataframe()#.query("token != 'Healthy'")

# %%
pd.concat(
    list(map(lambda x: pd.DataFrame(x.cpu().numpy(), columns=["subject_id", "age", "token"]), dataset[2005166].values()))
).\
astype({"token": int, "subject_id": int}).\
assign(age=lambda df: (df.age / 365.25).round(2)).\
sort_values(["age", "token"])

# %%
batch = next(iter(dataloader))
batch
# %%
from train_scripts.train import Trainer

trainer = Trainer(model, dataloaders, optimizer:=None, scheduler:=None, logger:=None, mlflow_params:=None)

# %%

def get_tensors_from_batch(batch):
    from easydict import EasyDict
        
    tokens, ages, subject_ids = EasyDict(), EasyDict(), EasyDict()
    
    for dname in batch:               
        SUBJECT_ID_COLUMN, AGE_COLUMN, TOKEN_COLUMN = 0, 1, 2
        domain_data = batch.get(dname, [])  
        if len(domain_data) == 0:
            # Create empty tensor without device assignment (always on CPU)
            if isinstance(domain_data, torch.Tensor):
                # If it's already a tensor with device, create new one without device
                domain_data = torch.empty(0, 3, dtype=domain_data.dtype)
            else:
                # If it's a list, create tensor on CPU (no device)
                domain_data = torch.empty(0, 3, dtype=torch.float32)
        if dname == "genetic_pcs":
            tokens[dname] = domain_data[:, 1:-1].float()
            ages[dname] = domain_data[:, -1]
            subject_ids[dname] = domain_data[:, 0]
        else:    
            tokens[dname] = domain_data[:, TOKEN_COLUMN].int()
            ages[dname] = domain_data[:, AGE_COLUMN]
            subject_ids[dname] = domain_data[:, SUBJECT_ID_COLUMN]

    return tokens, ages, subject_ids

def truncate_subjects_by_age(
        x: dict[str, torch.Tensor],
        ages: dict[str, torch.Tensor],
        subject_ids: dict[str, torch.Tensor],
        total_per_subject: dict[int, int],
        seqlen: int,
        trim_domains: set[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Truncate subjects with more than `seqlen` tokens, globally across trim_domains.
    
        Drops the most recent (highest age) events until each subject has exactly `seqlen` tokens total.
        Returns updated dicts with tokens removed.
        """
        device = next(iter(subject_ids.values())).device
    
        # Accumulate indices to drop per domain
        to_drop_by_domain: dict[str, list[int]] = defaultdict(list)
    
        for subj_id, tot in total_per_subject.items():
            diff = seqlen - tot
            if diff >= 0:
                continue  # nothing to drop
    
            R = -diff  # number of tokens to remove
    
            # Collect all candidate (age, domain, idx) from trim_domains
            candidates = []
            for d in trim_domains:
                if d not in subject_ids:
                    continue
                sids_d, ages_d = subject_ids[d], ages[d]
                idx = (sids_d == subj_id).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                for j in idx.tolist():
                    candidates.append((float(ages_d[j].item()), d, j))
    
            if not candidates:
                # No eligible domains for trimming
                continue
    
            # Sort by age (ascending) and remove the R most recent
            candidates.sort(key=lambda t: t[0])
            drop = candidates[-min(R, len(candidates)):]
            for _, d, j in drop:
                to_drop_by_domain[d].append(j)
    
        # Apply drops per domain
        for d, drop_list in to_drop_by_domain.items():
            if not drop_list:
                continue
            mask = torch.ones(len(subject_ids[d]), dtype=torch.bool, device=device)
            mask[torch.tensor(sorted(set(drop_list)), device=device)] = False
            x[d] = x[d][mask]
            ages[d] = ages[d][mask]
            subject_ids[d] = subject_ids[d][mask]
    
        return x, ages, subject_ids


def pad_subjects(
        x: dict[str, torch.Tensor],
        ages: dict[str, torch.Tensor],
        subject_ids: dict[str, torch.Tensor],
        total_per_subject: dict[int, int],
        seqlen: int,
        pad_domain: str,
        PADDING_TOKEN: int,
        PAD_AGE: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Pad subjects with fewer than `seqlen` tokens by appending padding tokens
        in the specified `pad_domain`.
        """
        device = next(iter(subject_ids.values())).device
        dtype_x = next(iter(x.values())).dtype
        dtype_age = next(iter(ages.values())).dtype
        dtype_sid = next(iter(subject_ids.values())).dtype
    
        pad_x, pad_a, pad_sid = [], [], []
        
        for subj_id, tot in total_per_subject.items():            
            diff = seqlen - tot
            if diff <= 0:
                continue
            pad_x.append(torch.full((diff,), PADDING_TOKEN, device=device, dtype=dtype_x))
            pad_a.append(torch.full((diff,), PAD_AGE, device=device, dtype=dtype_age))
            pad_sid.append(torch.full((diff,), subj_id, device=device, dtype=dtype_sid))
    
        if pad_x:
            x[pad_domain] = torch.cat([x[pad_domain], torch.cat(pad_x)])
            ages[pad_domain] = torch.cat([ages[pad_domain], torch.cat(pad_a)])
            subject_ids[pad_domain] = torch.cat([subject_ids[pad_domain], torch.cat(pad_sid)])
    
        return x, ages, subject_ids


def adjust_to_seqlen(
    x: dict[str, torch.Tensor],
    ages: dict[str, torch.Tensor],
    subject_ids: dict[str, torch.Tensor],
    seqlen: int,
    *,
    pad_domain: str = "padding",
    trim_domains: set[str] = frozenset({"diseases"}),
    PADDING_TOKEN: int = 0,
    PAD_AGE: float = -10000.0,
    verbose=False,
    debug_max_subjects: int = 10,
):
    """
    Main orchestrator: ensures all subjects have exactly `seqlen` tokens in total,
    truncating by age when too long, padding otherwise.

    Verbosity:
        verbose=False        → no logging
        verbose=True/'info'  → aggregate info per subject
        verbose='debug'      → extra stats and per-domain details (capped)
    """

    # Normalize verbosity mode
    if verbose is True:
        verbose_mode = "info"
    elif verbose in ("info", "debug"):
        verbose_mode = verbose
    else:
        verbose_mode = None

    x, ages, subject_ids = deepcopy(x), deepcopy(ages), deepcopy(subject_ids)

    # Domains to consider for counting (including padding, for diagnostics)
    domains = [d for d in subject_ids.keys()]

    # Helper: compute per-subject total tokens
    def compute_totals(subject_ids_dict):
        if not subject_ids_dict:
            return {}
        all_subj_ids = torch.cat([subject_ids_dict[d] for d in subject_ids_dict])
        subj_uniq, counts = np.unique(
            all_subj_ids.cpu().int().numpy(), return_counts=True
        )
        return {int(k): int(v) for k, v in zip(subj_uniq, counts)}

    total_per_subject = compute_totals(subject_ids)

    # Debug: global stats before
    if verbose_mode:
        counts = np.array(list(total_per_subject.values())) if total_per_subject else np.array([0])
        print(f"[adjust_to_seqlen] seqlen={seqlen}, n_subjects={len(total_per_subject)}")
        print(
            f"[adjust_to_seqlen] tokens per subject (before): "
            f"min={counts.min()}, p50={np.percentile(counts,50):.1f}, "
            f"p90={np.percentile(counts,90):.1f}, p95={np.percentile(counts,95):.1f}, "
            f"max={counts.max()}"
        )

        if verbose_mode == "debug":
            # Show a few extreme subjects
            sorted_subj = sorted(total_per_subject.items(), key=lambda kv: kv[1], reverse=True)
            print(f"[adjust_to_seqlen][debug] Top {min(debug_max_subjects, len(sorted_subj))} subjects by length (before):")
            for sid, tot in sorted_subj[:debug_max_subjects]:
                print(f"  • Subject {sid}: {tot} tokens total")

    # Keep copies for debug comparison
    if verbose_mode == "debug":
        old_x = deepcopy(x)
        old_ages = deepcopy(ages)
        old_sids = deepcopy(subject_ids)

    # Step 1: truncate subjects with too many tokens
    if verbose_mode:
        print(f"[adjust_to_seqlen] Step 1: truncating subjects > {seqlen} tokens...")

    x, ages, subject_ids = truncate_subjects_by_age(
        x, ages, subject_ids, total_per_subject, seqlen, trim_domains
    )

    # Recompute totals after truncation
    total_per_subject_after_trunc = compute_totals(subject_ids)

    if verbose_mode:
        # Basic per-subject info
        n_truncated = sum(
            1 for sid in total_per_subject
            if total_per_subject_after_trunc.get(sid, 0) < total_per_subject[sid]
        )
        print(f"[adjust_to_seqlen] Subjects truncated: {n_truncated} / {len(total_per_subject)}")

        if verbose_mode == "debug" and n_truncated > 0:
            print(f"[adjust_to_seqlen][debug] Example truncated subjects (up to {debug_max_subjects}):")
            shown = 0
            for sid, before in sorted(total_per_subject.items(), key=lambda kv: kv[1], reverse=True):
                after = total_per_subject_after_trunc.get(sid, 0)
                if after < before:
                    print(f"  • Subject {sid}: {before} → {after} tokens (truncated {before - after})")
                    # Per-domain breakdown before/after (debug)
                    per_domain_before = {}
                    per_domain_after = {}
                    for d in domains:
                        per_domain_before[d] = int((old_sids[d] == sid).sum().item())
                        per_domain_after[d] = int((subject_ids[d] == sid).sum().item())
                    print(f"    by domain before: {per_domain_before}")
                    print(f"    by domain after : {per_domain_after}")
                    shown += 1
                    if shown >= debug_max_subjects:
                        break

    # Step 2: recompute totals (after truncation, before padding)
    total_per_subject = compute_totals(subject_ids)

    # Step 3: pad subjects that are still short
    if verbose_mode:
        print(f"[adjust_to_seqlen] Step 3: padding subjects < {seqlen} tokens in domain '{pad_domain}'...")

    x, ages, subject_ids = pad_subjects(
        x, ages, subject_ids, total_per_subject, seqlen, pad_domain, PADDING_TOKEN, PAD_AGE
    )

    # Final totals
    final_totals = compute_totals(subject_ids)

    if verbose_mode:
        final_counts = np.array(list(final_totals.values())) if final_totals else np.array([0])
        print(
            f"[adjust_to_seqlen] tokens per subject (final): "
            f"min={final_counts.min()}, p50={np.percentile(final_counts,50):.1f}, "
            f"p90={np.percentile(final_counts,90):.1f}, p95={np.percentile(final_counts,95):.1f}, "
            f"max={final_counts.max()}"
        )

        bad_lt = [sid for sid, tot in final_totals.items() if tot < seqlen]
        bad_gt = [sid for sid, tot in final_totals.items() if tot > seqlen]

        if bad_lt:
            print(f"[adjust_to_seqlen][WARN] {len(bad_lt)} subject(s) ended with < seqlen tokens.")
            if verbose_mode == "debug":
                print(f"  Example: {bad_lt[:debug_max_subjects]}")
        if bad_gt:
            print(f"[adjust_to_seqlen][ERROR] {len(bad_gt)} subject(s) ended with > seqlen tokens (this should not happen).")
            if verbose_mode == "debug":
                print(f"  Example: {bad_gt[:debug_max_subjects]}")

        if verbose_mode == "debug":
            # Show some subjects with big padding
            print(f"[adjust_to_seqlen][debug] Example subjects after padding (up to {debug_max_subjects}):")
            for sid, tot in list(final_totals.items())[:debug_max_subjects]:
                per_domain = {}
                for d in subject_ids:
                    per_domain[d] = int((subject_ids[d] == sid).sum().item())
                print(f"  • Subject {sid}: {tot} tokens, by domain: {per_domain}")

        print("[adjust_to_seqlen] Done.\n")

    return x, ages, subject_ids

# %%
dataset[0]
# %%
next(iter(dataloader))

# %%
model.to('cpu')
x, ages, subject_ids = get_tensors_from_batch(batch)
max_ages             = model.get_max_ages_per_subject(ages, subject_ids)
x, ages, subject_ids = model.insert_no_event_tokens(x, ages, subject_ids)
x, ages, subject_ids = model.mask_tokens_after_age (x, ages, subject_ids, max_ages)
x, ages, subject_ids = adjust_to_seqlen(x, ages, subject_ids, seqlen:=96, verbose="debug")       

# %%
embeddings = model.transformer.embed(x)
x_tensor, ages_tensor, embeddings_tensor, uniq_subjs, domains = model.to_tensor(x, ages, embeddings, subject_ids)

# %%
logits, att = model(x, ages, subject_ids)
logits['diseases'].shape
# %%
x['diseases'].shape
# %%
