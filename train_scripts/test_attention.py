# %%
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")
from pathlib import Path
root_path = Path("../data/transforms")

import torch
from torch.utils.data import Dataset, random_split, DataLoader

from ast import literal_eval
import importlib

from easydict import EasyDict

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

from dataclasses import asdict, dataclass, field

from omegaconf import OmegaConf
import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import matplotlib.pyplot as plt

import delphi
from delphi.optim import OptimConfig, configure_optimizers
from delphi.model.transformer import (
    Delphi,
    EmbedConfig,
    DelphiConfig,
)

# from sklearn.model_selection import train_test_split
# from utils.utils import get_p2i, get_batch

import data.dataset

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

import torch
import random

B = 8; L = 20
domains_list = ["a", "b", "c"]
domain2id = {d:i for i,d in enumerate(domains_list)}

torch.manual_seed(0)
random.seed(0)

ages = torch.zeros(B, L)
domains = torch.zeros(B, L, dtype=torch.long)

for b in range(B):
    counts = torch.multinomial(torch.ones(len(domains_list)), len(domains_list), replacement=True)
    counts = torch.randint(L//4, L//2, (len(domains_list),))
    counts = counts * L // counts.sum()  # ajusta para sumar aprox L
    counts[-1] = L - counts[:-1].sum()   # corrige total

    all_tokens = []
    for dom_idx, n in enumerate(counts):
        domain_ages = torch.sort(torch.randint(0, 100, (n,)).float()).values
        domain_ids = torch.full((n,), dom_idx)
        all_tokens.append((domain_ages, domain_ids))

    seq_ages  = torch.cat([a for a, _ in all_tokens])
    seq_doms  = torch.cat([d for _, d in all_tokens])

    ages[b] = seq_ages
    domains[b] = seq_doms

scheme = {
    ("a",): {"type": "causal", "mask_ties": True},
    ("b",): {"type": "bidirectional", "mask_ties": False},
    ("c",): {"type": "causal", "mask_ties": False},
}

print("ages:", ages.shape, "domains:", domains.shape)

# %%
delphi = importlib.reload(delphi)
AttentionMaskBuilder = delphi.model.transformer.AttentionMaskBuilder

scheme = "[a,b]:causal(mask_ties=True),c:bidirectional(mask_ties=False)"
scheme = "[a,c]:bidirectional,[b,c]:causal(mask_ties=True)"

mask_builder = AttentionMaskBuilder(scheme)

mask_slow = mask_builder.build_slow(ages, domains, domain2id)
mask_fast = mask_builder.build(ages, domains, domain2id)

print("¿Son iguales?", torch.allclose(mask_slow, mask_fast))

import ipywidgets
from ipywidgets import interact

@interact
def show_attention_matrices(i=ipywidgets.IntSlider(min=0,max=B-1)):
    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    axs[0].imshow(mask_slow[i], cmap="coolwarm", interpolation="none")
    axs[0].set_title("Slow")
    axs[1].imshow(mask_fast[i], cmap="coolwarm", interpolation="none")
    axs[1].set_title("Fast")
    for ax in axs: ax.set_xlabel("j"); ax.set_ylabel("i")
    plt.tight_layout()
    plt.show()
# %%