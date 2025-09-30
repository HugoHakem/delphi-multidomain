# %%
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")

import time
import math
import pickle as pkl
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader

from ast import literal_eval
import importlib

from pprint import pprint
from collections import defaultdict

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional, List, Dict, Set

from omegaconf import OmegaConf
import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import yaml
import warnings

import delphi
from delphi.model.transformer import Delphi
from delphi.optim import OptimConfig, configure_optimizers
from delphi.model.components import (
  EmbedConfig,
  DelphiConfig,
)

from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.utils import get_p2i, get_batch

import data.dataset

root_path = Path("../data/transforms")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ————————————————————————————————————————————————————————————————————————————————————————————

@dataclass
class TrainBaseConfig:
    ckpt_dir: str = "."
    eval_interval: int = 2000
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    init_from: str = "scratch"

    seed: int = 42
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes

    # if gradient_accumulation_steps > 1, this is the micro-batch size
    batch_size: int = 128

    # system
    device: str = DEVICE
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float32"
    # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

    train_data: dict = field(default_factory=dict)
    val_data: dict = field(default_factory=dict)

    model: dict = field(default_factory=dict)
    optim: OptimConfig = field(default_factory=OptimConfig)
    # log: TrainLogConfig = field(default_factory=TrainLogConfig)

# ————————————————————————————————————————————————————————————————————————————————————————————

# class DelphiTokenizer():
# 
#     def __init__(self, mapping):
#         self.mapping = mapping
# 
#     def tokens_to_ids(self, tokens):
#         return [self.token_to_id[t] for t in tokens]
# 
#     def ids_to_tokens(self, ids):
#         return [self.id_to_token[int(id_)] for id_ in ids]


def batch_to_tensors(df, block_size):
    
    df = df.sort_values(["subject_id", "age"])
    grouped = df.groupby("subject_id")

    tokens, ages, mask = [], [], []

    for _, g in grouped:
        g = g.head(block_size).copy()  # or pad if there are less
        if len(g) < block_size:
            pad_len = block_size - len(g)
            g = pd.concat([g, pd.DataFrame({
                "token_id": [0]*pad_len,
                "age": [0]*pad_len,
                "predict": [False]*pad_len
            })], ignore_index=True)

        tokens.append(torch.tensor(g["token_id"].values, dtype=torch.long))
        ages.append(torch.tensor(g["age"].values, dtype=torch.float32))
        masks.append(torch.tensor(g["predict"].astype(int).values, dtype=torch.bool))
    
    ts = torch.stack
    tokens, ages, masks = ts(tokens), ts(ages), ts(masks)
    
    return tokens, ages, masks

# ————————————————————————————————————————————————————————————————————————————————————————————

class Trainer():

    def __init__(self, model, training_loader, valid_loader, test_loader, optimizer):

        self.model           = model
        self.training_loader = training_loader
        self.valid_loader    = valid_loader
        self.test_loader     = test_loader
        self.optimizer   = optimizer

    # ——————————————————————————————————————————————————————————————————————————————
    def train(self):

        for batch in self.training_loader:
            import ipdb; ipdb.set_trace()
            
            tokens, ages, masks = batch_to_tensors(batch, block_size=128)
            logits, _, _ = model(tokens, ages, validation_loss_mode=True)

            # logits, _, _, _ = model(tokens, ages, Y, B, validation_loss_mode=True)


    def train_step(self):
        pass


    # ——————————————————————————————————————————————————————————————————————————————    
    def evaluate(self):

        out = {}
        model.eval()
        
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters, 2)
            data = self.train_dataloader if split == 'train' else self.val_dataloader
            p2i = train_p2i if split == 'train' else val_p2i

            for k in range(eval_iters):
                ix = torch.randint(len(p2i), (batch_size,))
                X, A, Y, B = get_batch(ix, data, p2i, block_size=block_size,
                                       device=device, select='left', lifestyle_augmentations=True,
                                       no_event_token_rate=no_event_token_rate, 
                                       cut_batch=True)
                with ctx:
                    logits, loss, _, _ = model(X, A, Y, B, validation_loss_mode=True)

                losses[k] = torch.stack([loss['loss_ce'], loss['loss_dt']])

            out[split] = losses.mean(0)

        model.train()   
        return out


domain_config = {

    'diseases': EmbedConfig(
       projector="embed",
       input_size=None,
       path=root_path / 'diseases',
       predict=True,
    #    mask_ties=True
    ),
    'death': EmbedConfig(
        projector="embed",
        input_size=None,
        path=root_path / 'death',
        predict=True,
        # mask_ties=True
    ),
    'lifestyle': EmbedConfig(
        projector="embed",
        input_size=None,
        path=root_path / 'lifestyle',
        age_jitter=True,
        # mask_ties=False
    ),
    "hla_alleles": EmbedConfig(
        projector="embed",
        input_size=None,
        path=root_path / 'hla_alleles',
        # mask_ties=False
    ),
    "sex": EmbedConfig(
        projector="embed",
        input_size=None,
        path=root_path / 'sex',
        # mask_ties=False
    )    
}

cfg = DelphiConfig(    
    token_dropout=0.1,
    domains=domain_config,
)

# %%

data.dataset = importlib.reload(data.dataset)
DelphiDataset = data.dataset.DelphiDataset
DelphiDataloader = data.dataset.DelphiDataloader

# %%
folds = [ f"subject_lists/subset{i}of5.csv" for i in range(1, 6) ]
test_fold, dev_folds = [folds.pop(0)], folds

dataset_config = dict(root="../data/transforms", domains=domain_config, exclusions=[]) # "subject_lists/genetic_white_ids.txt"])

t0 = time.perf_counter()
dev_dataset  = DelphiDataset(subjects=dev_folds, **dataset_config, n_samples=200)
print(len(dev_dataset))

logging.info(f"DelphiDataset(dev) built in {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()

test_dataset = DelphiDataset(subjects=test_fold, **dataset_config, n_samples=100)

logging.info(f"DelphiDataset(test) built in {time.perf_counter()-t0:.2f}s")

# t0 = time.perf_counter()
# dev_dataset  = DelphiBatchDataset(dev_dataset)
# test_dataset = DelphiBatchDataset(test_dataset)

# logging.info(f"BatchDatasets built in {time.perf_counter()-t0:.2f}s")

n_valid      = len(dev_dataset) - (n_train := int(0.8*len(dev_dataset)))
t0 = time.perf_counter()

train_dataset, valid_dataset = random_split(
    # dev_dataset, [ n_train, n_valid ],
    dev_dataset, [ 100, 100 ],
    generator=torch.Generator().manual_seed(42)
)
logging.info(f"Random split done in {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
dataloaders = [ DelphiDataloader(d, batch_size=32) for d in [train_dataset, valid_dataset, test_dataset] ]
logging.info(f"Dataloaders built in {time.perf_counter()-t0:.2f}s")

config = DelphiConfig(vocab_size=1270, n_embd=120, domains=domain_config)


# %%
import data.dataset
data.dataset = importlib.reload(data.dataset)
DelphiDataset = data.dataset.DelphiDataset
DelphiDataloader = data.dataset.DelphiDataloader

dataset  = DelphiDataset(subjects=folds, **dataset_config, n_samples=None, required_domains=['diseases', 'lifestyle', 'sex'])
dataloader = DelphiDataloader(dataset, batch_size=64, num_workers=16)

# %%
# trainer = Trainer(model, *dataloaders, optimizer=optimizer)

# t0 = time.perf_counter()
# trainer.train()
# logging.info(f"Training finished in {time.perf_counter()-t0:.2f}s")


# %%

delphi = importlib.reload(delphi)
Delphi = delphi.model.transformer.Delphi
model  = Delphi(config).to(DEVICE)
optimizer, scheduler = configure_optimizers(model=model, cfg=OptimConfig(), device_type=DEVICE)

kk = next(iter(dataloader))

from easydict import EasyDict

pp = EasyDict()
ages = EasyDict()

for dname in model.transformer.embed.domain_embed:
    
    x    = torch.tensor(kk[dname].token_id.cat.codes.values).type(torch.int32)
    ages[dname] = torch.tensor(kk[dname].age.values).type(torch.int32)

    x_embed = model.transformer.embed.domain_embed[dname].projector(x)
    print(f"{dname}: {x_embed.shape}")
    age_embed = model.transformer.embed.age_encoding(ages[dname].unsqueeze(1))
    print(f"{dname}: {age_embed.shape}")

    pp[dname] = x_embed + age_embed


# %%

"hla_alleles"
"sex"
"lifestyle"
"diseases"
"death"

pp.diseases.shape
ages['diseases']

# %%
dataset[9]['diseases']
# %%
