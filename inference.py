# %%
import torch
import os, sys
import copy

import mlflow
import re
import numpy as np
import pandas as pd
from typing import List, Dict
from itertools import pairwise
import importlib
from copy import deepcopy
from collections import defaultdict

import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import interact

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

from old_model import Delphi as OldDelphi

import traceback
torch.set_grad_enabled(False)

from collections import OrderedDict
import re

def load_legacy_weights_into_new_delphi(
    model_new,
    state_dict_old,
    domains,
    skip_prefixes=("transformer.wte", "embedding"),
    verbose=False
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

    wte_weights = state_dict['transformer.wte.weight']

    vocab_lens = np.array([0] + [ model.transformer.embed.domain_embed[d].projector.num_embeddings for d in domains ])
    
    with torch.no_grad():
        for i, (start, end) in enumerate(pairwise(vocab_lens.cumsum())):
            domain = domains[i]
            model_new.transformer.embed.domain_embed[domain].weight.copy_(wte_weights[start:end])

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


def debug_run_forward(model, *args, **kwargs):
    """
    Ejecuta el forward módulo por módulo con hooks,
    capturando exactamente dónde ocurre un IndexError
    y mostrando los inputs/outputs de cada módulo.
    """
    model = model.cuda()

    traces = {}

    # Hook para registrar inputs/outputs de cada submódulo
    def hook_fn(name):
        def hook(module, inp, out):
            traces[name] = {"input": inp, "output": out}
        return hook

    hooks = []
    for name, module in model.named_modules():
        if not list(module.children()):  # solo leaf modules
            hooks.append(module.register_forward_hook(hook_fn(name)))

    try:
        out = model(*args, **kwargs)
        print("Forward OK sin errores.")
        return out

    except IndexError as e:
        print("INDEX ERROR DETECTADO")
        print("Mensaje:", e)
        print("-------------------------------\n")

        print("➡ Buscando en qué módulo se produjo...\n")

        # Mostramos los últimos módulos que ejecutaron correctamente
        for name in reversed(traces.keys()):
            print(f"Último módulo ejecutado OK: {name}")
            inp = traces[name]["input"]
            out = traces[name]["output"]

            print(f"  - Input shapes: {[x.shape if torch.is_tensor(x) else x for x in inp]}")
            if torch.is_tensor(out):
                print(f"  - Output shape: {out.shape}")
            else:
                print(f"  - Output: {out}")

            print("  - Valores max/min de input:")
            for i, x in enumerate(inp):
                if torch.is_tensor(x):
                    print(f"    input[{i}] max={x.max().item()}, min={x.min().item()}")

            print("\n")
            break  # solo mostramos el último módulo correcto

        print("Stack trace completo:")
        traceback.print_exc()

    finally:
        for h in hooks:
            h.remove()


def filter_subjects(batch, subject_ids):
    
    batch = copy.deepcopy(batch)
    subject_ids = torch.tensor(subject_ids)

    for dname in batch:
        batch[dname] = batch[dname][torch.isin(batch[dname][:,0].cpu(), subject_ids)]

    return batch


def fix_artifact_uri(path):
    return re.sub(".*mlruns", f"{HOME}/repos/delphi/{MLFLOW_URI}", path)


def get_ckpt_path_from_runid(runid):

    run = mlflow.get_run(runid)
    run_artifact_uri = fix_artifact_uri(run.info.artifact_uri)
    ckpt_path  = list((Path(run_artifact_uri) / "checkpoints").glob("*pt"))[-1]
    return ckpt_path


def get_state_dict(ckpt_path):

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)        
    state_dict = { k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items() }
    return state_dict

#————————————————————————————————————————————————————————————————————————————————————————————————

# MLFLOW_URI = "./output/mlruns"
MLFLOW_URI = "./mlruns"
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HOME = os.environ['HOME']
TOKENS_PATH = root_path / 'tokens'

HLA4D_EXPERIMENT   = "278131880607437980"
DATA_FILE = "./data/transforms/ukb_real_5_folds_4digit/all.bin"
LABELS_FILE = "data/delphi_labels_chapters_colours_icd_with_hla4d.csv"
DOMAINS = ['padding','sex','lifestyle','hla_alleles','diseases','death']
tokenizer = pd.read_csv(LABELS_FILE)["name"].to_dict()

mlflow.set_tracking_uri(MLFLOW_URI)

DEFAULT_CFG_PER_DOMAIN = {
    # 'genetic_pcs': EmbedConfig(projector="linear", path=tokens_path / 'genetic_pcs', type='continuous', at_birth=True),
    'diseases':    EmbedConfig(projector="embed", path=TOKENS_PATH / 'diseases',    predict=True),
    'death':       EmbedConfig(projector="embed", path=TOKENS_PATH / 'death',       predict=True),
    'lifestyle':   EmbedConfig(projector="embed", path=TOKENS_PATH / 'lifestyle',   age_jitter=True),  
    "hla_alleles": EmbedConfig(projector="embed", path=TOKENS_PATH / 'hla_alleles', at_birth=True),
    "sex":         EmbedConfig(projector="embed", path=TOKENS_PATH / 'sex',         at_birth=True),
    "padding":     EmbedConfig(projector="embed")    
}

SEQLEN = 128

#————————————————————————————————————————————————————————————————————————————————————————————————

attention_scheme = "[hla_alleles, sex]:bidirectional,[diseases, death, lifestyle, sex, padding]:causal(mask_ties=True)"
# attention_scheme = "[diseases, death, lifestyle, sex, hla_alleles, padding]:causal(mask_ties=True)"

config = DelphiConfig(
    n_layer=6, 
    n_embd=240,
    token_dropout=0.1, 
    domains=DEFAULT_CFG_PER_DOMAIN, 
    attention_scheme=attention_scheme
)
    
model  = Delphi(config).to(DEVICE)

runid = mlflow.search_runs(experiment_ids=HLA4D_EXPERIMENT).run_id[0]
ckpt_path = get_ckpt_path_from_runid(runid)
state_dict = get_state_dict(ckpt_path)

model   = load_legacy_weights_into_new_delphi(model, state_dict, domains=DOMAINS)

# ——————————————————————————————————————————————————————————————————————————————————————————————————
# OLD DATA
train, valid, test = old_get_data_partitions(DATA_FILE, fold=1)
train_data, train_p2i, train_ids = train

n_subjects = len(train_ids)
n_subjects = 256
x, a, y, b, subject_ids = get_batch(range(n_subjects), train_data, train_p2i, select='left', return_subject_ids=True, block_size=128)
event_set  = EventSet.from_batch((x, a, y, b), subject_ids=subject_ids, tokenizer=tokenizer)
x_old, x_ages_old, y_old, y_ages_old = event_set.to_tensor().to('cuda').as_tuple()

# NEW DATA
dataset    = DelphiDataset(domains=DEFAULT_CFG_PER_DOMAIN, root="./data/transforms", subjects=subject_ids, required_domains=["diseases", "hla_alleles"]).to("cuda")
dataloader = DelphiDataloader(dataset, batch_size=16)  
# batch      = next(iter(dataloader))

# batch = filter_subjects(batch, [subject_ids[48], subject_ids[77], subject_ids[122]])

pd.concat(
     list(map(lambda x: pd.DataFrame(x.cpu().numpy(), columns=["subject_id", "age", "token"]), dataset[2005166].values()))
).\
astype({"token": int, "subject_id": int}).\
assign(age=lambda df: (df.age / 365.25).round(2)).\
sort_values(["age", "token"])#.pipe(print)

# debug_run_forward(old_model, *event_set.to_tensor().to(DEVICE).as_tuple())

# %%
for i, batch in enumerate(dataloader):

    x, ages, subject_ids = get_tensors_from_batch(batch, device=DEVICE)
    max_ages             = model.get_max_ages_per_subject(ages, subject_ids)
    x, ages, subject_ids = model.insert_no_event_tokens(x, ages, subject_ids)
    x, ages, subject_ids = model.mask_tokens_after_age (x, ages, subject_ids, max_ages)
    x, ages, subject_ids = adjust_to_seqlen(x, ages, subject_ids, seqlen=SEQLEN, verbose=False)
    
    logits, _ = model(x, ages, subject_ids)
    
    for dname in logits:
        logits[dname] = logits[dname].cpu()
    
    kk = logits['diseases'].isnan().sum().cpu().item()
    
    # print(kk)

    if kk != 0:
        print(i)
        
    # print({ i: logits['diseases'][i].isnan().sum().cpu().item() for i in range(len(logits['diseases'])) })

# %%
embeddings = model.transformer.embed(x)
x_tensor, ages_tensor, embeddings_tensor, uniq_subjs, domains = model.to_tensor(x, ages, embeddings, subject_ids)

# %%
# OLD MODEL
old_model = OldDelphi.from_checkpoint(ckpt_path).to(DEVICE)
# old_logits, loss, _, _ = old_model(*event_set.to_tensor().to('cuda').as_tuple())

# %%
def attach_tracer(model, prefix=""):

    trace, hooks = {}, []

    def register(module, name):
        def hook(_, inp, out):
            trace[name] = out.detach().cpu()
        return module.register_forward_hook(hook)

    for name, module in model.named_modules():
        if not list(module.children()):  
            h = register(module, prefix + name)
            hooks.append(h)

    return trace, hooks

# debug_run_forward(old_model, *event_set.to_tensor().to('cuda').as_tuple())

trace_old, hooks_old = attach_tracer(old_model, prefix="old/")
trace_new, hooks_new = attach_tracer(model, prefix="new/")

_ = old_model(*event_set.to_tensor().to('cuda').as_tuple())
_ = model(x, ages, subject_ids)

for h in hooks_old: h.remove()
for h in hooks_new: h.remove()

# %%

DISEASE = 434
trace_old['old/transformer.wte'].cpu()[x_old.cpu() == 372+DISEASE]
trace_new['new/transformer.embed.domain_embed.diseases.projector'][x['diseases'].cpu() == DISEASE]

# %%
single_mask = model.transformer.attn_mask_builder[0][0].build(
    ages_tensor, domains, model.domain_to_int
)  # (B, L-1, L-1)

attn_mask = single_mask.unsqueeze(1).unsqueeze(1).\
    expand(-1, model.config.n_layer, model.config.n_head, -1, -1).\
    permute(1, 0, 2, 3, 4)        
        
attn_mask
# %%
@interact
def show_attn(i=widgets.IntSlider(min=0, max=128)):
    plt.imshow(attn_mask[0][i][0][-128:,-128:].cpu().numpy())

# %%
attn_masks = {}

def hook_attn_mask(module, inputs, outputs):
    # inputs: (x_norm, attn_mask)
    _, attn_mask = inputs
    attn_masks[id(module)] = attn_mask.detach().cpu()

# attach hooks to all attention modules
for i, block in enumerate(old_model.transformer.h):
    print(block.attn)
    block.attn.register_forward_hook(hook_attn_mask)


old_logits, loss, _, _ = old_model(*event_set.to_tensor().to('cuda').as_tuple())
attn_masks

plt.imshow(list(attn_masks.values())[0][0,0].numpy())
plt.show()

# %%
int_to_domain = { v: k for k, v in model.domain_to_int.items() }

def show_attention_map(att, domains, int_to_domain, title=None):
    """
    att: tensor 2D o 3D
        - [seq, seq]                  -> 1 head
        - [heads, seq, seq]           -> varias heads
        - [batch, heads, seq, seq]    -> batch + heads
    """

    # Normalizar dimensiones
    if att.dim() == 2:
        att = att.unsqueeze(0).unsqueeze(0)    # [1,1,seq,seq]
    elif att.dim() == 3:
        att = att.unsqueeze(0)                 # [1,heads,seq,seq]
    elif att.dim() != 4:
        raise ValueError("Formato no soportado.")

    b, h, seq, _ = att.shape

    # Preparo labels
    domain_labels = [int_to_domain[int(d.item())] for d in domains.squeeze(0)]

    # Grilla
    fig, axes = plt.subplots(
        nrows=b,
        ncols=h,
        figsize=(16*h, 16*b),
        squeeze=False
    )

    for bi in range(b):
        for hi in range(h):

            ax = axes[bi][hi]
            ax.imshow(att[bi, hi].detach().cpu(), aspect='auto')
            ax.set_title(f"Batch {bi}, Head {hi}", fontsize=10)

            ax.set_xticks(range(seq))
            ax.set_xticklabels(domain_labels, rotation=90)

            ax.set_yticks(range(seq))
            ax.set_yticklabels(domain_labels)

            ax.set_xlabel("Key positions")
            ax.set_ylabel("Query positions")

            ax.grid(which="major", color="black", linewidth=0.4)


    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()

# %%
subject_idx = 4
kk = attn_mask[0][subject_idx][0].cpu().numpy() # attn_mask[0][i][0][-128:,-128:].cpu().numpy()

n = 64
show_attention_map(torch.tensor(kk)[-n:,-n:], domains[subject_idx][-n:], int_to_domain)
# %%
kk.shape
# %%
