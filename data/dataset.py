#%%
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from torch.utils.data import DataLoader

# from delphi.data.ukb import UKBDataConfig, UKBDataset
DEVICE='cuda'

# %%
# from delphi.data.multimodal import (
    # UKBDataConfig,
    # load_sequences,
# )

# from model import Delphi, DelphiConfig

from utils.utils import get_p2i, get_batch

# import hla_genes
# from hla_genes import get_hla_protein_sequences

# from delphi.data.utils import train_iter
# from delphi.env import DELPHI_CKPT_DIR
# from delphi.log import TrainLogConfig, TrainLogger

from delphi.model.components import (
    DelphiConfig,
    # parse_token_list,
    # validate_model_config,
    # validate_model_config_for_finetuning,
)
from delphi.model.transformer import Delphi
from delphi.optim import OptimConfig, configure_optimizers

# %%
class TokenDomain:
    def __init__(self, path: str, predict: bool, age_jitter: bool):
        """
        Load a single domain: tokenizer.yaml + tokens.csv
        """
        self.path = path
        self.predict = predict
        self.age_jitter = age_jitter
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

    def __init__(self, root: str, domains: dict, subjects: List[str] = None, exclusions: List[str] = [], n_samples=None):
        """
        Args:
            root: base data directory
            domains: list of domain names (e.g. ["diagnosis", "lifestyle", "sex", "death"])
            fold: if specified, restrict subjects to that fold
            exclusions: list of exclusion list filenames under exclusion_lists/
        """
        self.root = root
        self.domains = {
            dname: TokenDomain(os.path.join(root, dname), predict=dinfo.predict, age_jitter=dinfo.age_jitter) 
            for dname, dinfo in domains.items()
        }
        
        # print(subjects)
        self.subjects = pd.concat([
            pd.read_csv(os.path.join(root, subj_file), names=["subject_id"]) for subj_file in subjects
        ])

        if n_samples is not None:
           self.subjects = self.subjects.sample(n_samples)
        
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
        return len(self.subjects)


    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.subjects.iloc[index]["subject_id"]
        return self.get_subject_events(index)


    def get_subject_array(self, subject_id: str) -> np.ndarray:
                
        arrays = []
        for dname, domain in self.domains.items():
            df = domain.tokens
            tokens_subj = df[df["subject_id"].astype(str) == str(subject_id)]
            if tokens_subj.empty:
                continue
            print(tokens_subj)
            arr = tokens_subj[["subject_id", "age", "token_id"]].to_numpy()
            arrays.append(arr)
        if not arrays:
            return np.zeros((0, 3))
        return np.vstack(arrays)



class DelphiBatchDataset(torch.utils.data.Dataset):

    def __init__(self, delphi_dataset):
        self.ds = delphi_dataset
        
        self.base = delphi_dataset
        while isinstance(self.base, torch.utils.data.Subset):
            self.base = self.base.dataset

    @property
    def subjects(self):
        return self.base.subjects

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        subj_id = self.ds.subjects.iloc[idx]["subject_id"]
        subject_events = self.ds.get_subject_events(subj_id)

        domains_dict = {}
        for dname, df in subject_events.items():
            arr = df[["subject_id", "age", "token_id"]].to_numpy()
            domains_dict[dname] = arr

        return {
          "subject_id": subj_id,
          "domains": domains_dict,
          "domains_metadata": self.ds.domains
        }
# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————


def get_batch(
    ix, data, p2i, 
    select='left', index='patient', padding='regular',
    block_size=48, device='', lifestyle_augmentations=False, 
    no_event_token_rate=5, cut_batch=False, return_subject_ids=False
    ):
    
    MASKING_TOKEN, MASKING_AGE = -1, -10000    
    
    # TODO: move this to tokenizer
    LIFESTYLE_MIN_INDEX, LIFESTYLE_MAX_INDEX = 3, 11

    # Define the columns of the data array    
    SUBJECT_ID_COLUMN, AGE_COLUMN, TOKEN_COLUMN = 0, 1, 2
    DAYS_PER_YEAR = 365.25

    subject_start_and_count = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    if return_subject_ids:
        subject_ids = torch.tensor(np.array([data[int(subject_index[0]), SUBJECT_ID_COLUMN] for subject_index in subject_start_and_count]))        
        
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='')
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

def collate_fn(batch, block_size=48, device=""):
    """
    batch: list of items returned by DelphiBatchDataset
           each item is { "subject_id": ..., "domains": {dname: arr} }
    """
    data_list, subject_ids = [], []
    for item in batch:
        subj_id = item["subject_id"]
        subject_ids.append(subj_id)

        # collect all domains into a single array per subject
        arrays = [arr for arr in item["domains"].values() if arr.size > 0]
        if arrays:
            data_list.append(np.vstack(arrays))

    # concatenate arrays from all subjects in the batch
    data = np.vstack(data_list)

    # map patient_id -> (start_index, count) pairs
    p2i_map = DelphiDataset.get_p2i(data)

    # use all subjects in the batch
    ix = list(range(len(subject_ids)))

    # call the batching function that prepares tensors
    x, a, y, b = get_batch(
        ix=ix,
        data=data,
        p2i=p2i_map,
        block_size=block_size,
        device=device,
    )

    return {
        "x": x, "a": a, "y": y, "b": b,
        "subject_ids": subject_ids
    }


def select_window(subject_start_and_count, block_size, data, mode="left", gen=None):
    """Choose start index of trajectory window for each subject."""
    if mode == "left":
        traj_start_idx = subject_start_and_count[:, 0]
    elif mode == "right":
        traj_start_idx = torch.clamp(
            subject_start_and_count[:, 0] + subject_start_and_count[:, 1] - block_size - 1,
            0, data.shape[0]
        )
    elif mode == "random":
        traj_start_idx = subject_start_and_count[:, 0] + (
            torch.randint(2**63-1, (len(subject_start_and_count),), generator=gen) %
            torch.clamp(subject_start_and_count[:, 1] - block_size, 1)
        )
        traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
    else:
        raise NotImplementedError(f"Unknown mode {mode}")
    return traj_start_idx.numpy()


def build_batch_indices(traj_start_idx, block_size):
    """Return indices into data for each subject's window."""
    return np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

# block_size=48, select="left", padding="regular"
# no_event_token_rate=5

def process_domain(data, domain_id, p2i,  device="", age_jitter=False, predict=True, gen=None):

    """General pipeline for one domain."""
    # subject_start_and_count = torch.tensor(np.array([p2i[int(i)] for i in range(len(p2i))]))
    # traj_start_idx = select_window(p2i, block_size, data, mode=select, gen=gen)
    # print("traj_start_idx.max():", traj_start_idx.max())
    # print("traj_start_idx.min():", traj_start_idx.min())
    
    # batch_idx = build_batch_indices(traj_start_idx, block_size)

    # print("data.shape:", data.shape)
    # print("batch_idx.max():", batch_idx.max())
    # print("batch_idx.shape:", batch_idx.shape)

    import ipdb; ipdb.set_trace()
    # tokens  = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64))
    # ages    = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32))
    
    if age_jitter:
        tokens, ages = age_jitter_tokens(tokens, ages, generator=gen)

    data = pd.DataFrame(data).assign(domain_id=domain_id, predict=predict)

    return data

    # domain_column  = torch.from_numpy([domain_id]*len(data)).astype(np.int64)
    # predict_column = torch.from_numpy([predict]  *len(data)).astype(np.int64)

    return ages, domains, tokens, predict_column
        
    # max_age = ages.max(1, keepdim=True).values
    # tokens, ages = insert_no_event_tokens(tokens, ages, max_age, no_event_token_rate=no_event_token_rate, padding=padding, gen=gen)
    # tokens, ages = sort_by_age(tokens, ages)

    # if predict:
        # x, a, y, b = make_targets(tokens, ages, block_size, device=device)
        # return {"x": x, "a": a, "y": y, "b": b, "predict": True, "d": domains.to(device)}
    # else:
        # no target: just return inputs
        # return {"x": tokens.to(device), "a": ages.to(device), "y": None, "b": None, "predict": False, "d": domains.to(device)}



def age_jitter_tokens(tokens, ages, generator=None):
    """
    Apply random age jitter to all tokens in the sequence.
    By default assumes all tokens are from a modality like lifestyle
    where fixed-age bias exists.

    tokens: torch.Tensor [batch, seq_len]
    ages:   torch.Tensor [batch, seq_len]
    generator: optional torch.Generator for reproducibility
    """
    DAYS_PER_YEAR = 365.25
    n_tokens = tokens.numel()

    if n_tokens > 0:
        age_jitter = torch.randint(
            low=int(-20 * DAYS_PER_YEAR),
            high=int(40 * DAYS_PER_YEAR),
            size=(n_tokens,),
            generator=generator
        ).float().view_as(ages)
        ages = ages + age_jitter

    return tokens, ages


def insert_no_event_tokens(tokens, ages, max_age, no_event_token_rate=5, padding="regular", gen=None):
    """Insert synthetic 'no event' tokens at regular or random intervals."""
    DAYS_PER_YEAR = 365.25
    if padding in [None, "none"] or no_event_token_rate in [0, None]:
        pad = torch.ones(tokens.shape[0], 0)
    elif padding == "regular":
        pad = torch.arange(0, 100 * DAYS_PER_YEAR, DAYS_PER_YEAR * no_event_token_rate) * torch.ones(tokens.shape[0], 1) + 1
    elif padding == "random":
        pad = torch.randint(1, 100 * DAYS_PER_YEAR, (tokens.shape[0], int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError(f"Unknown padding {padding}")

    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages   = torch.hstack([ages, pad])

    tokens = tokens.masked_fill(ages > max_age, -1)       # MASKING_TOKEN
    ages   = ages.masked_fill(ages > max_age, -10000)     # MASKING_AGE

    return tokens, ages


def sort_by_age(tokens, ages):
    """Ensure chronological order."""
    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages   = torch.gather(ages, 1, s)
    return tokens, ages


def make_targets(tokens, ages, block_size, device=DEVICE, cut_batch=False):
    """Shift sequences to make (x,a) → (y,b)."""
    # cut if needed
    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages   = ages[:, cut_margin:]

    # shift
    x, y = tokens[:, :-1], tokens[:, 1:]
    a, b = ages[:, :-1], ages[:, 1:]

    # mask if first token is "no event"
    x = x.masked_fill((x == 0) & (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, -10000)

    return x.to(device), a.to(device), y.to(device), b.to(device)


def get_domain_id(domain_name):

    if isinstance(domain_name, str):
        domain_id = abs(hash(domain_name)) % (10**6)
    else:
        domain_id = int(domain_name)

    return domain_id


def collate_fn_domains(batch, block_size=48, device=DEVICE):

    """
    Collate function that processes each domain separately, using domain metadata.
    Each item in batch is expected to be:
      {
        "subject_id": ...,
        "domains": {dname: arr},
        "domains_metadata": {dname: TokenDomain}  # must expose .predict and .jitter
      }
    """

    subject_ids = [item["subject_id"] for item in batch]
    out = {"subject_ids": subject_ids, "domains": {}}        

    # Group arrays per domain
    domains_dict = {}
    for item in batch:
        for dname, arr in item["domains"].items():
            if dname not in domains_dict:
                domains_dict[dname] = []
            domains_dict[dname].append(arr)        
   
    domain_ids = { dname: get_domain_id(dname) for dname in domains_dict }

    all_data = []
    for dname, arrs in domains_dict.items():

        data = np.concatenate(arrs)
        p2i_map = DelphiDataset.get_p2i(data)
        domain_id = get_domain_id(dname)
        # grab meta info from the first item in the batch
        domain_meta = batch[0]["domains_metadata"][dname]
        
        domain_data = pd.DataFrame(data, columns=["subject_id", "age", "token_id"]).assign(
            domain_id=pd.Categorical([dname]*len(data), categories=domain_ids.keys()), 
            predict=domain_meta.predict
        )

        all_data.append(domain_data)
    
    return pd.concat(all_data, axis=0).sort_values(["subject_id", "age"])

    # out["domains"][dname] = process_domain(
        # data,
        # domain_id,
        # p2i=p2i_map,
        # block_size=block_size,
        # device=device,
        # age_jitter=domain_meta.age_jitter,
        # predict=domain_meta.predict
    # )

    # import ipdb; ipdb.set_trace()
    # enmascaramos aca? creo que es un buen lugar para hacerlo
    # concatenamos dominios


class DelphiDataloader(DataLoader):
    def __init__(self, dataset, block_size=48, device="", return_dictionary=True, **kwargs):
        
        if return_dictionary:
            collate_fn = collate_fn_domains
        else:
            collate_fn = lambda b: collate_fn(b, block_size=block_size, device=device),
        
        super().__init__(
            dataset,
            collate_fn=collate_fn,
            **kwargs
        )
