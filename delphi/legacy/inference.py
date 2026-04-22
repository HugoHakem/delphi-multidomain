# %%
import torch
import os, sys

import mlflow
import re
from easydict import EasyDict

import numpy as np
import pandas as pd
from typing import List, Dict
from itertools import pairwise
import importlib
from copy import deepcopy
from collections import defaultdict, OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

import torch
import traceback

sys.path.insert(0, DELPHI_DIR := Path(__file__).parent.resolve())

from train_scripts.trainer import Trainer

import data
data = importlib.reload(data)
import data.event_set
EventSet = data.event_set.EventSet
EventSetLegacy = data.event_set.EventSetLegacy

from data.dataset import DelphiDataset, DelphiDataloader

from utils.cv_utils import get_data_partitions

from viz.styles import color_by_domain
from viz.trajectories import create_friendly_view

from delphi.model import (
    Delphi,
    DomainConfig,
    DelphiConfig,
)

from legacy.model import Delphi as OldDelphi

torch.set_grad_enabled(False)

MLFLOW_URI = str( Path(DELPHI_DIR) / "legacy" / "mlruns" )
mlflow.set_tracking_uri(MLFLOW_URI)
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from legacy.utils import (
    get_batch,
    get_p2i,
    generate_splits
)

# ————————————————————————————————————————————————————————————————————————————————————————————————

def load_legacy_weights_into_delphi(
    model_new,
    state_dict_old,
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


class DelphiDebug(Delphi):

    def __init__(self, *args, **kwargs):
        super().__init(*args, **kwargs)


def debug_run_forward(model, *args, **kwargs):
    """
    Ejecuta el forward módulo por módulo con hooks,
    capturando exactamente dónde ocurre un IndexError
    y mostrando los inputs/outputs de cada módulo.
    """
    model = model.to(DEVICE)

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
        print("\n Index error detected ")
        print("Message:", e)
        print("-------------------------------\n")

        print("➡ Searching which module produced the error...\n")
        
        for name in reversed(traces.keys()):
            print(f"Last module correctly executed: {name}")
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

        print("Stack trace completed:")
        traceback.print_exc()

    finally:
        for h in hooks:
            h.remove()


root_path = Path(DELPHI_DIR) / "data/transforms"
tokens_path = root_path / 'tokens'

default_cfg_per_domain = {
    # 'genetic_pcs': DomainConfig(projector="linear", path=tokens_path / 'genetic_pcs', type='continuous', at_birth=True),
    'diseases':    DomainConfig(projector="embed", path=tokens_path / 'diseases',    predict=True),
    'death':       DomainConfig(projector="embed", path=tokens_path / 'death',       predict=True),
    'lifestyle':   DomainConfig(projector="embed", path=tokens_path / 'lifestyle',   age_jitter=True),  
    "hla_alleles": DomainConfig(projector="embed", path=tokens_path / 'hla_alleles', at_birth=True),
    "sex":         DomainConfig(projector="embed", path=tokens_path / 'sex',         at_birth=True),
    "padding":     DomainConfig(projector="embed")    
}

domain_cfg = default_cfg_per_domain

# attention_scheme = "[hla_alleles]:bidirectional,[diseases, death, lifestyle, sex, padding]:causal(mask_ties=True)"
attention_scheme = "[diseases, death, lifestyle, sex, hla_alleles, padding]:causal(mask_ties=True)"
# attention_scheme = "[hla_alleles, sex]:bidirectional, [diseases, death, lifestyle, sex, hla_alleles, padding]:causal(mask_ties=True)"

config = DelphiConfig(
    n_layer=6, 
    n_embd=240,
    token_dropout=0.1, 
    domains=domain_cfg, 
    attention_scheme=attention_scheme
)
    
model  = Delphi(config).to(DEVICE)

hla_exp   = "278131880607437980"

def fix_artifact_uri(path):
    return re.sub(".*mlruns", MLFLOW_URI, path)

runs_hla = mlflow.search_runs(experiment_ids=hla_exp)
runs_hla.artifact_uri = runs_hla.artifact_uri.apply(fix_artifact_uri)

runid = runs_hla.run_id[0]

run = mlflow.get_run(runid)
run_artifact_uri = fix_artifact_uri(run.info.artifact_uri)

#——————————————————————————————————————————————————————————————————

ckpt_path  = list((Path(run_artifact_uri) / "checkpoints").glob("*pt"))[-1]
checkpoint = torch.load(ckpt_path, map_location=DEVICE)        
state_dict = { k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items() }

model = load_legacy_weights_into_delphi(model, state_dict)

wte_weights = state_dict['transformer.wte.weight']

domains = ['padding','sex','lifestyle','hla_alleles','diseases','death']
kk = np.array([0] + [ model.transformer.embed.domain_embed[d].projector.num_embeddings for d in domains ])

with torch.no_grad():
    for i, (start, end) in enumerate(pairwise(kk.cumsum())):
        domain = domains[i]
        model.transformer.embed.domain_embed[domain].weight.copy_(wte_weights[start:end])

model.to(DEVICE)

#——————————————————————————————————————————————————————————————————

old_model = OldDelphi.from_checkpoint(ckpt_path).to(DEVICE)

# NEW DATA
# train_ids, val_ids, test_ids = get_data_partitions("./data/transforms/subject_lists", fold=0)

# OLD DATA
train, valid, test = old_get_data_partitions("./data/transforms/deprecated/ukb_real_5_folds_4digit/all.bin", 1)

train_data, train_p2i, train_ids = train
# val_data, val_p2i, val_ids       = valid
# test_data, test_p2i, val_ids     = test

tokenizer = pd.read_csv("data/delphi_labels_chapters_colours_icd_with_hla4d.csv")["name"].to_dict()

x, a, y, b, subject_ids = get_batch(
    range(256), 
    train_data, train_p2i, select='left', block_size=128, return_subject_ids=True
)

event_set = EventSetLegacy.from_batch((x, a, y, b), subject_ids=subject_ids, tokenizer=tokenizer)

dataset    = DelphiDataset(domains=domain_cfg, root="./data/transforms", subjects=subject_ids).to(DEVICE)
dataloader = DelphiDataloader(dataset, batch_size=256)  
batch      = next(iter(dataloader))

model.set_block_size(128)
x, ages, subject_ids = model.prepare_input(batch)
logits, att          = model(x, ages, subject_ids, return_attention=True)

embeddings = model.transformer.embed(x)

x_tensor, ages_tensor, embeddings_tensor, \
uniq_subjs, domains = model.to_tensor(x, ages, embeddings, subject_ids)

single_mask = model.transformer.attn_mask_builder[0][0].build(
    domains=domains, local_token_ids=x_tensor, ages=ages_tensor
) # (B, L-1, L-1)

attn_mask = single_mask.unsqueeze(1).unsqueeze(1).\
    expand(-1, model.config.n_layer, model.config.n_head, -1, -1).\
    permute(0, 1, 2, 3, 4)            

@interact
def show_attn(i=widgets.IntSlider(min=0, max=128)):    
    plt.imshow(attn_mask[i][0][0].cpu().numpy())

# %%
df, styled_df = create_friendly_view(x_tensor, ages_tensor, domains, uniq_subjs, model=model, dataset=dataset)

def show_subject(df):

    @interact(
        subject=widgets.IntSlider(min=0, max=df.subject_id.nunique() - 1),
        hide_padding=True,
        hide_no_events=True,
        hide_hla_alleles=True,
    )
    def _show(subject, hide_padding, hide_no_events, hide_hla_alleles):

        d = df.copy()
        d = list(d.groupby("subject_id"))[subject][1]
        d = d.reset_index(drop=True)

        if hide_padding:
            d = d.query('token_name != "unknown_0"')
        if hide_no_events:
            d = d.query('token_name != "unknown_1"')
        if hide_hla_alleles:
            d = d.query("domain_name != 'hla_alleles'")

        display( 
            d.style.apply(color_by_domain, axis=1) 
        )


def show_subject_from_event_set(event_set):
    
    x, a, _, _ = event_set.to_tensor().to(DEVICE).as_tuple()

    @interact(
        subject=widgets.IntSlider(min=0, max=255)
    )
    def _show(subject):
        old_token_df = pd.DataFrame(
            [ (k.item(), tokenizer[k.item()], a[subject][i].numpy()/365.25) for i, k in enumerate(x[subject]) ],
            columns=["token_id", "token_name", "age"]
        ).sort_values(["age", "token_name"])
    
        display( 
            old_token_df.drop_duplicates() 
        )

show_subject(df)
show_subject_from_event_set(event_set)

# %%
old_logits, loss, _, _ = old_model(*event_set.to_tensor().to(DEVICE).as_tuple())

# %%
order = ["hla_alleles", "sex", "lifestyle", "diseases", "death"]
logits_c = torch.cat([logits[k] for k in order], dim=2)
logits_c.shape

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

trace_old, hooks_old = attach_tracer(old_model, prefix="old/")
trace_new, hooks_new = attach_tracer(model,     prefix="new/")

_ = old_model(*event_set.to_tensor().to(DEVICE).as_tuple())
_ = model(x, ages, subject_ids)

for h in hooks_old: h.remove()
for h in hooks_new: h.remove()

# %%
# DISEASE_ID = 434
# trace_old['old/transformer.wte'][x_old == 372+DISEASE_ID]

# %%
# trace_new['new/transformer.embed.domain_embed.diseases.projector'][x['diseases'].cpu() == kk].shape

# Examine attentions
attn_masks = {}

def hook_attn_mask(module, inputs, outputs):
    # inputs: (x_norm, attn_mask)
    _, attn_mask = inputs
    attn_masks[id(module)] = attn_mask.detach().cpu()

# attach hooks to all attention modules
for i, block in enumerate(old_model.transformer.h):
    block.attn.register_forward_hook(hook_attn_mask)

# %%
x_old, x_ages_old, y_old, y_ages_old = event_set.to_tensor().to(DEVICE).as_tuple()


@interact(subject_id=widgets.IntSlider(min=0, max=100))
def show_old_attention_mask(subject_id):
    plt.imshow(
        # old_model.build_attention_mask(x_old, x_ages_old, y_old, y_ages_old, mask_ties=True)[subject_id,0].int().numpy()
        old_model.build_attention_mask(x_old, x_ages_old, y_old, y_ages_old, mask_ties=True)[subject_id,0].int().numpy()
    )

# %%



# @interact(subject_id=widgets.IntSlider(min=0, max=100))
# def show_new_attention_mask(subject_id):
    # plt.imshow(
        # old_model.build_attention_mask(x_old, x_ages_old, y_old, y_ages_old, mask_ties=True)[subject_id,0].int().numpy()
        # old_model.build_attention_mask(x_old, x_ages_old, y_old, y_ages_old, mask_ties=True)[subject_id,0].int().numpy()
    # )
# %%

from torch import nn

class AttentionMaskBuilder(nn.Module):
    """
    Parses and builds attention masks from a string like:
        [hla_alleles,sex]:bidirectional,[disease,lifestyle,sex,death]:causal(mask_ties=True)
            which is the same as NoAttention([hla_alleles, sex]:bidirectional, [disease,lifestyle,sex,death]:causal(mask_ties=True))
        
    """

    def __init__(self, scheme_str: str, domain2id: dict):
        
        super().__init__()
        
        self.scheme = self._parse_scheme(scheme_str)

        self.domain2id = domain2id

        self.ignore_token = 0


    # —-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—
    def _split_top_level(self, s: str, sep: str = ","):
        """
        Splits a string by sep but ignores separators inside [] or ().
        """
        parts, buf, depth_brack, depth_paren = [], "", 0, 0
        for ch in s:
            if ch == "[":   depth_brack += 1
            elif ch == "]": depth_brack -= 1
            elif ch == "(": depth_paren += 1
            elif ch == ")": depth_paren -= 1

            if ch == sep and depth_brack == 0 and depth_paren == 0:
                parts.append(buf.strip())
                buf = ""
            else:
                buf += ch

        if buf.strip():
            parts.append(buf.strip())

        return parts

    # —-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—
    def _parse_scheme(self, scheme_str: str):
        scheme = {}
        parts = self._split_top_level(scheme_str)

        for part in parts:
            if ":" not in part:
                raise ValueError(f"Invalid rule fragment: {part}")
            domain_part, rule_part = part.split(":", 1)
            domain_part, rule_part = domain_part.strip(), rule_part.strip()

            # domains
            if domain_part.startswith("[") and domain_part.endswith("]"):
                domains = [d.strip() for d in domain_part[1:-1].split(",")]
            else:
                domains = [domain_part]

            # rule type
            if rule_part.startswith("causal"):
                rule_type = "causal"
                mask_ties = "mask_ties=True" in rule_part
            elif rule_part.startswith("bidirectional"):
                rule_type = "bidirectional"
                mask_ties = False
            else:
                raise ValueError(f"Unknown rule type: {rule_part}")

            scheme[tuple(domains)] = {"type": rule_type, "mask_ties": mask_ties}

        return scheme


    def build(self, domains: torch.Tensor, local_token_ids: torch.Tensor, ages: torch.Tensor):
        """
        Build a [B, L, L] attention mask from a declarative DSL.
        1 = attention allowed
        0 = attention blocked
        """
        B, L = ages.shape
        dd = { 'device': ages.device }

        # Start with everything blocked
        mask = torch.zeros(B, L, L, **dd)

        # Always allow self-attention
        # diag = torch.arange(L, **dd)
        # mask[..., diag, diag] = 1        

        # Precompute expanded views for "causal" tests
        age_row  = ages.unsqueeze(2)  # [B, L, 1]
        age_col  = ages.unsqueeze(1)  # [B, 1, L]

        for dom_names, cfg in self.scheme.items():
            # Gather domain ids
            dom_ids = torch.tensor([self.domain2id[d] for d in dom_names], **dd)

            # Boolean mask for tokens belonging to this rule
            dom_mask = torch.isin(domains, dom_ids)   # [B, L]

            # Pairs of tokens both belonging to allowed domains in this rule
            pair_mask = dom_mask.unsqueeze(2) & dom_mask.unsqueeze(1)   # [B, L, L]

            if cfg["type"] == "bidirectional":
                # Allow everything inside the domain pair
                mask[pair_mask] = 1

            elif cfg["type"] == "causal":
                if cfg.get("mask_ties", False):
                    # strict causal: do not allow ties
                    causal = age_row > age_col
                else:
                    # allow ties
                    causal = age_row >= age_col

                # Combine with the domain pair mask
                final = pair_mask & causal
                mask[final] = 1

            else:
                raise ValueError(f"Unknown attention type: {cfg['type']}")

        mask = mask.bool()
        is_padding = (domains == self.domain2id["padding"]) & (local_token_ids == 0)
        not_padding = ~is_padding            # [B, L]

        # mask &= ~ ( 
            # is_padding.unsqueeze(2) |       # query not padding
            # is_padding.unsqueeze(1)         # key   not padding
        # )
# 
        # mask = mask.int()
# 
        #--- fallback self-attention to avoid NaNs ---
        #If a row is entirely zero, allow self-attention
        # row_has_any = mask.any(dim=-1)        # [B, L]
        # needs_fallback = ~row_has_any         # [B, L]
        # 
        # b_idx, i_idx = needs_fallback.nonzero(as_tuple=True)
        # mask[b_idx, i_idx, i_idx] = 1

        return mask
    

    def forward(self, domains, local_token_ids, ages):
        return self.build(domains, local_token_ids, ages)
    
attn_mask_builder = AttentionMaskBuilder(attention_scheme, model.domain_to_int)

attn_mask_builder.scheme

# %%
plt.imshow(
    attn_mask_builder(domains, ages_tensor, x_tensor)[0].numpy()
)
# %%
domains
# %%
model.domain_to_int
# %%
dataset.domains.keys()
# %%
