# %%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")

import time
import math
import pickle as pkl
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

from ast import literal_eval

from torch.utils.data import Dataset, random_split, DataLoader

from pprint import pprint
from collections import defaultdict

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional

from omegaconf import OmegaConf

import yaml
import warnings
from typing import List, Dict, Set

from data.dataset import DelphiDataset, DelphiBatchDataset, DelphiDataloader
from delphi.model.transformer import Delphi

from delphi.model.components import (
    EmbedConfig,
    DelphiConfig,
    DomainEmbedding,
    DelphiEmbedding,
    CrossEntropyHead,
    CompetingExpHead,
    causal_attention_mask,
    target_mask,
    ties_adjusted_delta_t,
)

from sklearn.model_selection import train_test_split

from utils.utils import get_p2i, get_batch
from delphi.model.transformer import Delphi
from delphi.optim import OptimConfig, configure_optimizers

DEVICE = "cuda"

# %%
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


# %%
class DelphiTokenizer():

    def __init__(self, mapping):
        self.mapping = mapping

    def tokens_to_ids(self, tokens):
        return [self.token_to_id[t] for t in tokens]

    def ids_to_tokens(self, ids):
        return [self.id_to_token[int(id_)] for id_ in ids]


def batch_to_tensors(df, block_size):
    # aseguramos orden por subject_id y edad
    df = df.sort_values(["subject_id", "age"])

    # agrupamos por sujeto
    grouped = df.groupby("subject_id")

    tokens = []
    ages = []
    masks = []

    for _, g in grouped:
        # cortar o rellenar al block_size
        g = g.head(block_size).copy()  # o pad si hay menos
        if len(g) < block_size:
            pad_len = block_size - len(g)
            g = pd.concat([
                g,
                pd.DataFrame({
                    "token_id": [0]*pad_len,
                    "age": [0]*pad_len,
                    "predict": [False]*pad_len
                })
            ], ignore_index=True)

        tokens.append(torch.tensor(g["token_id"].values, dtype=torch.long))
        ages.append(torch.tensor(g["age"].values, dtype=torch.float32))
        masks.append(torch.tensor(g["predict"].astype(int).values, dtype=torch.bool))
    
    tokens = torch.stack(tokens)
    ages = torch.stack(ages)
    masks = torch.stack(masks)

    return tokens, ages, masks


# %%
class Trainer():

    def __init__(self, model, training_loader, valid_loader, test_loader, optimizer):

        self.model           = model
        self.training_loader = training_loader
        self.valid_loader    = valid_loader
        self.test_loader     = test_loader
        self.optimizer   = optimizer


    def train(self):

        for batch in self.training_loader:
            import ipdb; ipdb.set_trace()
            tokens, ages, masks = batch_to_tensors(batch, block_size=128)

            # apply dropout
            # 
            
            # logits, _, _, _ = model(tokens, ages, validation_loss_mode=True)
            logits, _, _ = model(tokens, ages, validation_loss_mode=True)

            # logits, _, _, _ = model(tokens, ages, Y, B, validation_loss_mode=True)


    def train_step(self):
        pass


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


# %%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delphi.model.components import (
    EmbedConfig,
    DelphiConfig,
    DomainEmbedding,
    DelphiEmbedding,
)

root_path = "../data/transforms"

domain_config = {

    # 'diseases': EmbedConfig(
    #    projector="embed",
    #    input_size=None,
    #    path=os.path.join(root_path, 'diseases'),
    #    predict=True
    #),
    'death': EmbedConfig(
        projector="embed",
        input_size=None,
        path=os.path.join(root_path, 'death'),
        predict=True
    ),
    'lifestyle': EmbedConfig(
        projector="embed",
        input_size=None,
        path=os.path.join(root_path, 'lifestyle'),
        age_jitter=True
    ),
    "hla_alleles": EmbedConfig(
        projector="embed",
        input_size=None,
        path=os.path.join(root_path, 'hla_alleles'),
    ),
    "sex": EmbedConfig(
        projector="embed",
        input_size=None,
        path=os.path.join(root_path, 'sex'),
    )    
}

cfg = DelphiConfig(    
    token_dropout=0.1,
    domains=domain_config,
)

import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# %%
folds = [ f"subject_lists/subset{i}of5.csv" for i in range(1, 6) ]
test_fold, dev_folds = [folds.pop(0)], folds

dataset_config = dict(root="../data/transforms", domains=domain_config, exclusions=["subject_lists/genetic_white_ids.txt"])

t0 = time.perf_counter()
dev_dataset  = DelphiDataset(subjects=dev_folds, **dataset_config)

from torch.utils.data import Subset
dev_dataset = Subset(dev_dataset, range(200))  # primeras 200 muestras

logging.info(f"DelphiDataset(dev) built in {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()

test_dataset = DelphiDataset(subjects=test_fold, **dataset_config)
test_dataset = Subset(test_dataset, range(100))  # primeras 200 muestras

logging.info(f"DelphiDataset(test) built in {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
dev_dataset  = DelphiBatchDataset(dev_dataset)
test_dataset = DelphiBatchDataset(test_dataset)

logging.info(f"BatchDatasets built in {time.perf_counter()-t0:.2f}s")

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

t0 = time.perf_counter()
model  = Delphi(config).to(DEVICE)
logging.info(f"Model built in {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
optimizer, scheduler = configure_optimizers(model=model, cfg=OptimConfig(), device_type=DEVICE)
logging.info(f"Optimizers configured in {time.perf_counter()-t0:.2f}s")

trainer = Trainer(model, *dataloaders, optimizer=optimizer)

t0 = time.perf_counter()
trainer.train()
logging.info(f"Training finished in {time.perf_counter()-t0:.2f}s")
