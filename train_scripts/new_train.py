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

pprint(EmbedConfig)
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


# @dataclass
# class TrainConfig(TrainBaseConfig):
# 
#     # finetune
#     resume_from: Optional[str] = None
# 
#     # data
#     data_fraction: float = 1.0
#     memmap: bool = False
#     train_data: UKBDataConfig = field(default_factory=UKBDataConfig)
#     infer_train_biomarkers: bool = True
#     val_data: UKBDataConfig = field(default_factory=UKBDataConfig)
# 
#     infer_val_biomarkers: bool = True
#     infer_val_expansion_packs: bool = True
#     infer_val_transforms: bool = True
#     infer_val_subject_filters: bool = True
# 
#     model: DelphiConfig = field(default_factory=DelphiConfig)
#     ignore_expansion_tokens: bool = True

# %%
class DelphiTokenizer():

    def __init__(self, mapping):
        self.mapping = mapping

    def tokens_to_ids(self, tokens):
        return [self.token_to_id[t] for t in tokens]

    def ids_to_tokens(self, ids):
        return [self.id_to_token[int(id_)] for id_ in ids]


#  class TokenDomainManager:
     # def __init__(self):
        #  Dictionary of domains -> per-domain vocab {token: local_id}
         # self.domains: Dict[str, Dict[str, int]] = {}
#  
     # def add_tokens(self, domain: str, tokens: List[str]):
         # """
         # Add new tokens to a given domain.
         # If the domain does not exist, create it.
         # Tokens already present will be ignored.
         # """
         # if domain not in self.domains:
             # self.domains[domain] = {}
         # d = self.domains[domain]
         # for tok in tokens:
             # if tok not in d:
                 # d[tok] = len(d)
#  
#  
     # def flatten(self, domains: List[str] = None) -> Dict[str, int]:
         # """
         # Return a flat vocabulary combining one or multiple domains.
         # Token IDs are assigned contiguously, domain by domain.
         # Keys in the flat vocab are namespaced as "domain:token".
         # """
         # flat, offset = {}, 0
         # if domains is None:
             # domains = list(self.domains.keys())
         # for dom in domains:
             # for tok, idx in self.domains[dom].items():
                 # flat[f"{dom}:{tok}"] = offset + idx
             # offset += len(self.domains[dom])
         # return flat
#  
#  
     # def shared_tokens(self, domains: List[str]) -> Set[str]:
         # """
         # Return the set of tokens that are shared across all given domains.
         # Comparison is done on raw token strings (not prefixed).
         # """
         # sets = [set(self.domains[d].keys()) for d in domains]
         # return set.intersection(*sets)
#  
#  
     # def get_domain_tokens(self, domain: str) -> List[str]:
         # """Return the list of tokens in a given domain."""
         # return list(self.domains.get(domain, {}).keys())


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
            print(batch)
            X, A, Y, B = batch['x'], batch['a'], batch['y'], batch['b']
            logits, _, _, _ = model(X, A, Y, B, validation_loss_mode=True)


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
    'diseases': EmbedConfig(
        projector="embed",
        input_size=None,
        path=os.path.join(root_path, 'diseases'),
        predict=True
    ),
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

# domains = {
#     'diseases': {
#         'predict': True
#     },
#     'lifestyle': {
#         'predict': False
#     },
#     "hla_alleles": {
#         'predict': True
#     },
#     'sex': {
#         'predict': False
#     }, 
#     'death': {
#         'predict': True
#     }
# }
# 
# domain_config = {}
# for d in domains:
#     domain_config[d] = EmbedConfig(
#         projector="embed",
#         input_size=None,
#         path=os.path.join(root_path, d),
#         predict=domains[d]['predict']
#     )

cfg = DelphiConfig(    
    # n_embd=n_embd,
    token_dropout=0.1,
    domains=domain_config,
)

model = DelphiEmbedding(cfg)

# %%
folds = [ f"subject_lists/subset{i}of5.csv" for i in range(1, 6) ]
test_fold, dev_folds = [folds.pop(0)], folds

dataset_config = dict(root="../data/transforms", domains=domain_config, exclusions=["subject_lists/genetic_white_ids.txt"])

dev_dataset  = DelphiDataset(subjects=dev_folds, **dataset_config)
test_dataset = DelphiDataset(subjects=test_fold, **dataset_config)

dev_dataset  = DelphiBatchDataset(dev_dataset)
test_dataset = DelphiBatchDataset(test_dataset)
n_valid      = len(dev_dataset) - (n_train := int(0.8*len(dev_dataset)))
train_dataset, valid_dataset = random_split(
    dev_dataset, [ n_train, n_valid ],
    generator=torch.Generator().manual_seed(42)
)

dataloaders = [ DelphiDataloader(d, batch_size=32) for d in [train_dataset, valid_dataset, test_dataset] ]

config = DelphiConfig(vocab_size=1270, n_embd=120, domains=domain_config)
model  = Delphi(config).to(DEVICE)
optimizer, scheduler = configure_optimizers(model=model, cfg=OptimConfig(), device_type=DEVICE)

trainer = Trainer(model, *dataloaders, optimizer=optimizer)
trainer.train()

# ds = DelphiDataset(root_path, domains=domains, subjects=dev_folds)
# batch_ds = DelphiBatchDataset(ds)
# 
# item = batch_ds[0]
# 
# print("Subject ID:", item["subject_id"])
# print("Dominios disponibles:", list(item["domains"].keys()))
# 
# for dname, arr in item["domains"].items():
#     print(f"\nDominio: {dname}")
#     print("Shape:", arr.shape)
#     print("Primeras filas:\n", arr[:20])