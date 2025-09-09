#%%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")

import time
import math
import pickle as pkl
from contextlib import nullcontext
import ipdb

import numpy as np
import pandas as pd
import torch

from ast import literal_eval

from torch.utils.data import Dataset

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
from delphi.data.ukb import UKBDataConfig, UKBDataset

# %%
from delphi.data.multimodal import (
    UKBDataConfig,
    load_sequences,
)

# from model import Delphi, DelphiConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import get_p2i, get_batch

# import hla_genes
# from hla_genes import get_hla_protein_sequences

# from delphi.data.utils import train_iter
# from delphi.env import DELPHI_CKPT_DIR
from delphi.log import TrainLogConfig, TrainLogger
from delphi.model.config import (
    DelphiConfig,
    # parse_token_list,
    # validate_model_config,
    # validate_model_config_for_finetuning,
)
from delphi.model.transformer import Delphi
from delphi.optim import OptimConfig, configure_optimizers

# %%
class TokenDomain:
    def __init__(self, path: str):
        """
        Load a single domain: tokenizer.yaml + tokens.csv
        """
        self.path = path
        self.tokenizer = self._load_tokenizer(os.path.join(path, "tokenizer.yaml"))
        self.tokens = self._load_tokens(os.path.join(path, "tokens.csv"))

    def _load_tokenizer(self, path: str) -> Dict:
        with open(path, "r") as f:
            tok = yaml.safe_load(f)
        return tok

    def _load_tokens(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            assert False, f"Tokens file {path} does not exist"
            return pd.DataFrame(columns=["subject_id", "token_id", "age"])
        df = pd.read_csv(path)
        # enforce schema
        if "subject_id" not in df.columns or "token_id" not in df.columns:
            raise ValueError(f"Invalid tokens file {path}, must contain subject_id and token_id")
        if "age" not in df.columns:
            df["age"] = None
        return df
    
# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————    


class DelphiDataset:

    def __init__(self, root: str, domains: List[str], subjects: str = None, exclusions: List[str] = []):
        """
        Args:
            root: base data directory
            domains: list of domain names (e.g. ["diagnosis", "lifestyle", "sex", "death"])
            fold: if specified, restrict subjects to that fold
            exclusions: list of exclusion list filenames under exclusion_lists/
        """
        self.root = root
        self.domains = {d: TokenDomain(os.path.join(root, d)) for d in domains}
        self.subjects = pd.read_csv(os.path.join(root, subjects), names=["subject_id"])
        
        self.excluded_subjects = set()
        for excl in exclusions:
            excl_path = os.path.join(root, excl)
            if os.path.exists(excl_path):
                ids = open(excl_path).read().strip().splitlines()
                self.excluded_subjects |= set(map(str, ids))

        self.subjects = self.subjects[~self.subjects["subject_id"].astype(str).isin(self.excluded_subjects)]        


    def get_subject_events(self, subject_id: str) -> Dict[str, pd.DataFrame]:
        """
        Return all events for a subject, per domain.
        """
        subject_events = {}
        for dname, domain in self.domains.items():
            tokens = domain.tokens
            tokens_subj = tokens[tokens["subject_id"].astype(str) == str(subject_id)]
            subject_events[dname] = tokens_subj
        return subject_events


    def iter_subjects(self):
        """Iterate over subject IDs in dataset"""
        for sid in self.subjects["subject_id"].astype(str).tolist():
            yield sid, self.get_subject_events(sid)

    def validate(self):
        """Check domain-specific constraints, e.g. age presence"""
        for dname, domain in self.domains.items():
            if dname == "diagnosis":
                missing_age = domain.events["age"].isna().sum()
                if missing_age > 0:
                    raise ValueError(f"{dname} domain has {missing_age} missing ages")
            if dname == "genetics":
                with_age = domain.events["age"].notna().sum()
                if with_age > 0:
                    warnings.warn(f"{dname} domain has {with_age} rows with age provided (should not).")


    @staticmethod
    def get_p2i(data):
        patient_ids = data[:, 0].astype(int)
        _, idx_start, counts = np.unique(patient_ids, return_index=True, return_counts=True)
        return np.stack([idx_start, counts], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.subjects.iloc[index]["subject_id"]
        return self.get_subject_events(index)

# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

class DelphiDataloader():
    
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            yield [self.dataset[i] for i in batch_indices]


# %%
folds = [ f"subject_lists/fold{i}.csv" for i in range(1, 6) ]

test_fold, dev_folds = folds.pop(0), folds

test_dataset = DelphiDataset(
    root="../data/transforms",
    domains=["diseases", "lifestyle", "hla_alleles", 'sex'],
    subjects=test_fold,
    exclusions=["subject_lists/genetic_white_ids.txt"]
)

dev_dataset = DelphiDataset(
    root="../data/transforms",
    domains=["diseases", "lifestyle", "hla_alleles", 'sex'],
    subjects=dev_folds,
    exclusions=["subject_lists/genetic_white_ids.txt"]
)

# %%
from delphi.model.transformer import Delphi, DelphiConfig
config = DelphiConfig(vocab_size=1270, n_embd=120)

model = Delphi(config=config)
# dataset.get_subject_events("1000015")

# %%
model(
  torch.randint(0, 1270, (2, 10)), 
  torch.tensor([[0,10],[0,10]])
)