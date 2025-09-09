# %%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")

import time
import math
import pickle as pkl
from contextlib import nullcontext
# import ipdb

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
@dataclass
class TrainBaseConfig:
    ckpt_dir: str = "."
    eval_interval: int = 2000
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    init_from: str = "scratch"

    seed: int = 42
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes
    batch_size: int = 128
    # if gradient_accumulation_steps > 1, this is the micro-batch size

    # system
    device: str = "cpu"
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float32"
    # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

    train_data: dict = field(default_factory=dict)
    val_data: dict = field(default_factory=dict)

    model: dict = field(default_factory=dict)

    optim: OptimConfig = field(default_factory=OptimConfig)

    log: TrainLogConfig = field(default_factory=TrainLogConfig)


@dataclass
class TrainConfig(TrainBaseConfig):

    # finetune
    resume_from: Optional[str] = None

    # data
    data_fraction: float = 1.0
    memmap: bool = False
    train_data: UKBDataConfig = field(default_factory=UKBDataConfig)
    infer_train_biomarkers: bool = True
    val_data: UKBDataConfig = field(default_factory=UKBDataConfig)

    infer_val_biomarkers: bool = True
    infer_val_expansion_packs: bool = True
    infer_val_transforms: bool = True
    infer_val_subject_filters: bool = True

    model: DelphiConfig = field(default_factory=DelphiConfig)
    ignore_expansion_tokens: bool = True

# %%
class DelphiTokenizer():

    def __init__(self, mapping):
        self.mapping = mapping

    def tokens_to_ids(self, tokens):
        return [self.token_to_id[t] for t in tokens]

    def ids_to_tokens(self, ids):
        return [self.id_to_token[int(id_)] for id_ in ids]

# %%


class TokenDomainManager:
    def __init__(self):
        # Dictionary of domains -> per-domain vocab {token: local_id}
        self.domains: Dict[str, Dict[str, int]] = {}

    def add_tokens(self, domain: str, tokens: List[str]):
        """
        Add new tokens to a given domain.
        If the domain does not exist, create it.
        Tokens already present will be ignored.
        """
        if domain not in self.domains:
            self.domains[domain] = {}
        d = self.domains[domain]
        for tok in tokens:
            if tok not in d:
                d[tok] = len(d)

    def flatten(self, domains: List[str] = None) -> Dict[str, int]:
        """
        Return a flat vocabulary combining one or multiple domains.
        Token IDs are assigned contiguously, domain by domain.
        Keys in the flat vocab are namespaced as "domain:token".
        """
        flat, offset = {}, 0
        if domains is None:
            domains = list(self.domains.keys())
        for dom in domains:
            for tok, idx in self.domains[dom].items():
                flat[f"{dom}:{tok}"] = offset + idx
            offset += len(self.domains[dom])
        return flat

    def shared_tokens(self, domains: List[str]) -> Set[str]:
        """
        Return the set of tokens that are shared across all given domains.
        Comparison is done on raw token strings (not prefixed).
        """
        sets = [set(self.domains[d].keys()) for d in domains]
        return set.intersection(*sets)

    def get_domain_tokens(self, domain: str) -> List[str]:
        """Return the list of tokens in a given domain."""
        return list(self.domains.get(domain, {}).keys())

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
            return pd.DataFrame(columns=["subject_id", "token_id", "age"])
        df = pd.read_csv(path)
        # enforce schema
        if "subject_id" not in df.columns or "token_id" not in df.columns:
            raise ValueError(f"Invalid tokens file {path}, must contain subject_id and token_id")
        if "age" not in df.columns:
            df["age"] = None
        return df


class DelphiDataset:
    def __init__(self, root: str, domains: List[str], fold: int = None, exclusions: List[str] = []):
        """
        Args:
            root: base data directory
            domains: list of domain names (e.g. ["diagnosis", "lifestyle", "sex", "death"])
            fold: if specified, restrict subjects to that fold
            exclusions: list of exclusion list filenames under exclusion_lists/
        """
        self.root = root
        self.domains = {d: TokenDomain(os.path.join(root, "domains", d)) for d in domains}
        self.subjects = pd.read_csv(os.path.join(root, "subjects.csv"))
        
        # apply exclusions
        self.excluded_subjects = set()
        for excl in exclusions:
            excl_path = os.path.join(root, "exclusion_lists", excl)
            if os.path.exists(excl_path):
                ids = open(excl_path).read().strip().splitlines()
                self.excluded_subjects |= set(map(str, ids))

        if fold is not None:
            fold_path = os.path.join(root, "folds", f"fold{fold}.txt")
            with open(fold_path) as f:
                fold_ids = set(f.read().strip().splitlines())
            self.subjects = self.subjects[self.subjects["subject_id"].astype(str).isin(fold_ids)]

        # apply exclusion lists
        self.subjects = self.subjects[~self.subjects["subject_id"].astype(str).isin(self.excluded_subjects)]

    def get_subject_events(self, subject_id: str) -> Dict[str, pd.DataFrame]:
        """
        Return all events for a subject, per domain.
        """
        subject_events = {}
        for dname, domain in self.domains.items():
            ev = domain.events
            ev_sub = ev[ev["subject_id"].astype(str) == str(subject_id)]
            subject_events[dname] = ev_sub
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
        return self.data[index]


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


def get_batch(
    ix, data, p2i, 
    select='center', index='patient', padding='regular',
    block_size=48, device='cpu', lifestyle_augmentations=False, 
    no_event_token_rate=5, cut_batch=False, return_subject_ids=False
    ):
    
    MASKING_TOKEN, MASKING_AGE = -1, -10000    
    
    # TODO: move this to tokenizer
    LIFESTYLE_MIN_INDEX, LIFESTYLE_MAX_INDEX = 3, 11
    # LIFESTYLE_MIN_INDEX, LIFESTYLE_MAX_INDEX = 3+359, 11+359

    # Define the columns of the data array    
    SUBJECT_ID_COLUMN, AGE_COLUMN, TOKEN_COLUMN = 0, 1, 2
    DAYS_PER_YEAR = 365.25

    subject_start_and_count = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    if return_subject_ids:
        subject_ids = torch.tensor(np.array([data[int(subject_index[0]), SUBJECT_ID_COLUMN] for subject_index in subject_start_and_count]))        
        
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())  # we want some things be random, but also deterministic

    if index == 'patient':
        if select == 'left':
            traj_start_idx = subject_start_and_count[:, 0]
        elif select == 'right':
            traj_start_idx = torch.clamp(subject_start_and_count[:, 0] + subject_start_and_count[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == 'random':
            traj_start_idx = subject_start_and_count[:, 0] + (torch.randint(2**63-1, (len(ix),), generator=gen) % torch.clamp(subject_start_and_count[:, 1] - block_size, 1))
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    mask = torch.from_numpy(data[:, SUBJECT_ID_COLUMN][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.numpy()][:, SUBJECT_ID_COLUMN], SUBJECT_ID_COLUMN][:, None].astype(np.int64)).to(mask.dtype)

    tokens = torch.from_numpy(data[:, TOKEN_COLUMN][batch_idx].astype(np.int64))
    ages   = torch.from_numpy(data[:, AGE_COLUMN][batch_idx].astype(np.float32))

    # augment lifestyle tokens to avoid immortality bias
    if lifestyle_augmentations:
        lifestyle_idx = (tokens >= LIFESTYLE_MIN_INDEX) * (tokens <= LIFESTYLE_MAX_INDEX)
        n_lifestyles_tokens = lifestyle_idx.sum()
        if n_lifestyles_tokens:
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (n_lifestyles_tokens,), generator=gen).float()

    tokens = tokens.masked_fill(~mask, MASKING_TOKEN)
    ages   = ages.masked_fill(~mask, MASKING_AGE)
    
    # insert a "no event" token every 5 years on average
    if (padding.lower() == 'none' or padding is None or no_event_token_rate == 0 or no_event_token_rate is None):
        pad = torch.ones(len(ix), 0)
    elif padding == 'regular':
        pad = torch.arange(0, 100 * DAYS_PER_YEAR, DAYS_PER_YEAR * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == 'random':
        pad = torch.randint(1, 100 * DAYS_PER_YEAR, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError
    
    m = ages.max(1, keepdim=True).values

    # stack "no event" tokens with real tokens
    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages = torch.hstack([ages, pad])

    # mask out "no event" tokens that are too far in the future (i.e. after the last real token)
    tokens = tokens.masked_fill(ages > m, MASKING_TOKEN)
    ages = ages.masked_fill(ages > m, MASKING_AGE)

    # sort everything so that things are correctly ordered about stacking
    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)

    # a technical detail: the token 0 is reserved for padding, so we shift all tokens by one
    tokens = tokens + 1

    # cut the padded tokens if possible
    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # cut to maintain the block size
    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # shift by one to generate targets
    x, y = tokens[:, :-1], tokens[:, 1:]
    a, b = ages[:, :-1]  , ages[:, 1:]

    # if the first token is a "no event" token, mask it and the corresponding target
    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, MASKING_AGE)

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a, y, b = [i.pin_memory().to(device, non_blocking=True) for i in [x, a, y, b]]
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)

    if return_subject_ids:
        return x, a, y, b, subject_ids
    else:           
        return x, a, y, b


# %%
class Trainer():

    def __init__(self, model, training_loader, valid_loader, test_loader, optimizer):

        self.model           = model
        
        self.training_loader = training_loader
        self.valid_loader    = valid_loader
        self.test_loader     = test_loader

        self.optimizer   = optimizer


    def train(self):
        while True:
            self.train_step()


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

