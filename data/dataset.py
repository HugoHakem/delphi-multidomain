#%%
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")

import time
from contextlib import nullcontext

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from ast import literal_eval

from torch.utils.data import Dataset

from pprint import pprint
from collections import defaultdict

from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional

from omegaconf import OmegaConf

import yaml
import warnings
from typing import List, Dict, Set
from torch.utils.data import DataLoader

import logging

# from delphi.data.ukb import UKBDataConfig, UKBDataset
DEVICE='cpu'

from utils.utils import get_p2i, get_batch

from delphi.model.components import DelphiConfig
from delphi.model.transformer import Delphi
from delphi.optim import OptimConfig, configure_optimizers
from typing import Union

from easydict import EasyDict

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————

class TokenDomain:
    
    '''
    '''

    def __init__(self, path: str, predict: bool, age_jitter: bool, subjects: Union[None, set]=None):
        """
        Load a single domain: tokenizer.yaml + tokens.csv
        """
        self.path = path
        self.predict = predict
        self.age_jitter = age_jitter
        self.tokenizer = self._load_tokenizer(os.path.join(path, "tokenizer.yaml"))
        
        self.tokens = self._load_tokens(os.path.join(path, "tokens.csv"))

        if subjects is not None:
            self.tokens = self.tokens.query("subject_id in @subjects")

        self.tokens["token_id"] = pd.Categorical(
            self.tokens["token_id"].map(self.tokenizer),
            categories=self.tokenizer.values()
            # ordered=True
        )

    def _load_tokenizer(self, path: str) -> Dict:
        with open(path, "r") as f:
            tokenizer = yaml.safe_load(f)        
        return { idx: token for idx, token in enumerate(tokenizer) }  

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
    
    
    def filter_subjects(self, subjects: set):

        from copy import deepcopy
        filtered_domain = deepcopy(self)
        filtered_domain.tokens = filtered_domain.tokens.query("subject_id in @subjects")
        return filtered_domain


    def __repr__(self):

        return repr( 
            self.tokens.\
            assign( **{"age (years)": (self.tokens.age / 363.25).round(2)} ).\
            drop("age", axis=1)#.set_index("subject_id") 
        )
    

    def _repr_html_(self):
        return self.tokens.\
            assign( **{"age (years)": (self.tokens.age / 363.25).round(2)} ).\
            drop("age", axis=1)._repr_html_()
            # set_index("subject_id").\
             
    
    def __getitem__(self, subject_id):

        # corner case

        return self.tokens.set_index("subject_id").loc[[subject_id]].\
            assign( **{"age (years)": lambda df: (df.age / 363.25).round(2)} ).\
            drop("age", axis=1)
            # set_index("subject_id")

# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————    

class DelphiDataset:

    def __init__(self, root: str, domains: dict, subjects: List[str] = None, exclusions: List[str] = [], n_samples=None, required_domains=['sex', 'diseases'], merge_namespaces=False):
        """
        Args:
            root: base data directory
            domains: list of domain names (e.g. ["diagnosis", "lifestyle", "sex", "death"])
            fold: if specified, restrict subjects to that fold
            exclusions: list of exclusion list filenames under exclusion_lists/
        """

        self.root = Path(root)

        # Load the data        
        self.domains = EasyDict()
        for dname, dinfo in domains.items():
            datafile = self.root / dname
            self.domains[dname] = TokenDomain(datafile, predict=dinfo.predict, age_jitter=dinfo.age_jitter) 

        # ——————————— DEFINE ALLOWED SUBJECTS —————————————————————————————————
        self.included_subjects = pd.concat([
            pd.read_csv(self.root / subj_file, names=["subject_id"]) for subj_file in subjects
        ])
                
        self.excluded_subjects = self.get_excluded_subjects(exclusion_files=exclusions)

        self._subjects = self.included_subjects[~self.included_subjects["subject_id"].astype(str).isin(self.excluded_subjects)]
        self._subjects = self.filter_subj_for_required_domains(self._subjects, required_domains)        

        if n_samples is not None:
            self._subjects = self._subjects.sample(n_samples)
        
        self._subjects = set(self._subjects.subject_id.to_list())

        # —————————————————————————————————————————————————————————————————————
        
        # Now we filter each domain for the final list of allowed subjects
        for dname in self.domains:
            self.domains[dname] = self.domains[dname].filter_subjects(self.subjects)

        # "re-index" categories (not the default behaviour, 
        # which consists in keeping the tokens for each namespace separate):
        if merge_namespaces:
            all_cats = pd.Index([])
            for dname, domain in self.domains.items():
                all_cats = all_cats.union(domain.tokens['token_id'].cat.categories)        
            for dname in self.domains:
                self.domains[dname].tokens['token_id'] = self.domains[dname].tokens['token_id'].cat.set_categories(all_cats)                    
            self.categories = all_cats.tolist()

        for dname, domain in self.domains.items():
            self.domains[dname].tokens = self.domains[dname].tokens.set_index("subject_id")

        # of internal use, for faster slicing
        self._subject_indices = self._precompute_subject_indices_per_domain()
           

    @property
    def subjects(self):
        return list(self._subjects)


    # This returns all data (all subjects and domains) merged into a single dataframe
    def merge_data(self):
        return pd.concat([ self.domains[dname].tokens for dname in self.domains ]).\
            sort_index().\
            sort_values(['age'])
            
            # set_index('subject_id')


    def list_domains(self):
        return list(self.domains.keys())


    def filter_subj_for_required_domains(self, subjects: pd.DataFrame, required_domains: List[str]) -> pd.DataFrame:
        
        """
        Keep only subjects that have at least one token in all required domains.
        """
        
        for dname in required_domains:
            if dname not in self.domains:
                logging.debug(f"Required domain '{dname}' not found in self.domains. Returning empty DataFrame.")
                return subjects.iloc[0:0]

            # Get set of subjects present in this domain
            domain_subjects = self.domains[dname].tokens["subject_id"].astype(str).unique()
            logging.debug(f"Domain '{dname}': {len(domain_subjects)} unique subjects.")

            # Intersect with current subject list
            before_count = len(subjects)
            subjects = subjects[subjects["subject_id"].astype(str).isin(domain_subjects)]
            after_count = len(subjects)
            logging.debug(
                f"Filtered by domain '{dname}': {before_count} → {after_count} subjects remaining."
            )

        return subjects


    def get_excluded_subjects(self, exclusion_files):
        
        self.excluded_subjects = set()
        for excl in exclusion_files:
            excl_path = os.path.join(root, excl)
            if os.path.exists(excl_path):
                ids = open(excl_path).read().strip().splitlines()
                self.excluded_subjects |= set(map(str, ids))        
        return self.excluded_subjects
 

    def get_subject_events(self, subject_id: str) -> Dict[str, pd.DataFrame]:
        
        """
        Return all events for a subject, per domain.
        """        
        subject_events = {}
        for dname, domain in self.domains.items():            
            # tokens_subj = tokens[tokens["subject_id"].astype(str) == str(subject_id)]            
            try:
                # indexing with .loc is a bit slower than .iloc (but not too much)
                # tokens_subj = domain.tokens.loc[[subject_id]]

                start, count = self._subject_indices[dname][subject_id]
                tokens_subj  = domain.tokens.iloc[start:(start+count)]
            except KeyError:
                tokens_subj = pd.DataFrame()
            # tokens_subj = domain.tokens.loc[subject_id] # [tokens["subject_id"].astype(str) == str(subject_id)]
            subject_events[dname] = tokens_subj         

        return subject_events


    @staticmethod
    def is_ukb_id(index):
        return index > 1000000


    def __len__(self):
        return len(self.subjects)        


    def _precompute_subject_indices_per_domain(self):

        '''
        pre-computes indices for each subject in each domain for fast slicing
        returns something like:
          { 'domain_1' : {
                id_1: (0, 18), <- (start_index, token_count)
                id_2: (18, 17), ... }
            'domain_2' : {
                id_1: (0, 4),
                id_2: (4, 6), ... }
            }
        '''

        subject_indices = {}
        for domain in self.list_domains():
            ids, counts = np.unique(self.domains[domain].tokens.index.values, return_counts=True)
            counts = np.array([0] + counts.tolist())
            pp = [ (int(x), int(y)) for x, y in zip(np.cumsum(counts)[:-1], counts[1:])]
            subject_indices[domain] = { int(id): pp[i] for i, id in enumerate(ids) }
        
        self._subject_indices = subject_indices
        return self._subject_indices


    def __getitem__(self, index):

        if not self.is_ukb_id(index) and isinstance(index, int):
            index = self.subjects[index]

        out = self.get_subject_events(index)
        return out


    # def iter_subjects(self):
    #     """Iterate over subject IDs in dataset"""
    #     for sid in self.subjects["subject_id"].astype(str).tolist():
    #         yield sid, self.get_subject_events(sid)


    # def validate(self):
    #     """Check domain-specific constraints, e.g. age presence"""
    #     for dname, domain in self.domains.items():
    #         if dname == "diagnosis":
    #             missing_age = domain.events["age"].isna().sum()
    #             if missing_age > 0:
    #                 raise ValueError(f"{dname} domain has {missing_age} missing ages")
    #         if dname == "genetics":
    #             with_age = domain.events["age"].notna().sum()
    #             if with_age > 0:
    #                 warnings.warn(f"{dname} domain has {with_age} rows with age provided (should not).")


    # @staticmethod
    # def get_p2i(data):
    #     patient_ids = data[:, 0].astype(int)
    #     _, idx_start, counts = np.unique(patient_ids, return_index=True, return_counts=True)
    #     return np.stack([idx_start, counts], axis=1)


    # def get_subject_array(self, subject_id: str) -> np.ndarray:                
    #     arrays = []
    #     for dname, domain in self.domains.items():
    #         df = domain.tokens
    #         tokens_subj = df[df["subject_id"].astype(str) == str(subject_id)]
    #         if tokens_subj.empty:
    #             continue
    #         print(tokens_subj)
    #         arr = tokens_subj[["subject_id", "age", "token_id"]].to_numpy()
    #         arrays.append(arr)
    #     if not arrays:
    #         return np.zeros((0, 3))
    #     return np.vstack(arrays)

  
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

# def collate_fn(batch, block_size=48, device=""):
#     """
#     batch: list of items returned by DelphiBatchDataset
#            each item is { "subject_id": ..., "domains": {dname: arr} }
#     """
#     data_list, subject_ids = [], []
#     for item in batch:
#         subj_id = item["subject_id"]
#         subject_ids.append(subj_id)

#         # collect all domains into a single array per subject
#         arrays = [arr for arr in item["domains"].values() if arr.size > 0]
#         if arrays:
#             data_list.append(np.vstack(arrays))

#     # concatenate arrays from all subjects in the batch
#     data = np.vstack(data_list)

#     # map patient_id -> (start_index, count) pairs
#     p2i_map = DelphiDataset.get_p2i(data)

#     # use all subjects in the batch
#     ix = list(range(len(subject_ids)))

#     # call the batching function that prepares tensors
#     x, a, y, b = get_batch(
#         ix=ix,
#         data=data,
#         p2i=p2i_map,
#         block_size=block_size,
#         device=device,
#     )

#     return {
#         "x": x, "a": a, "y": y, "b": b,
#         "subject_ids": subject_ids
#     }


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


# def build_batch_indices(traj_start_idx, block_size):
#     """Return indices into data for each subject's window."""
#     return np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]


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

    # tokens  = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64))
    # ages    = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32))
    
    if age_jitter:
        tokens, ages = age_jitter_tokens(tokens, ages, generator=gen)

    data = pd.DataFrame(data).assign(domain_id=domain_id, predict=predict)

    return data

    # domain_column  = torch.from_numpy([domain_id]*len(data)).astype(np.int64)
    # predict_column = torch.from_numpy([predict]  *len(data)).astype(np.int64)

    # return ages, domains, tokens, predict_column
        
    # max_age = ages.max(1, keepdim=True).values
    # tokens, ages = insert_no_event_tokens(tokens, ages, max_age, no_event_token_rate=no_event_token_rate, padding=padding, gen=gen)
    # tokens, ages = sort_by_age(tokens, ages)

    # if predict:
        # x, a, y, b = make_targets(tokens, ages, block_size, device=device)
        # return {"x": x, "a": a, "y": y, "b": b, "predict": True, "d": domains.to(device)}
    # else:
        # no target: just return inputs
        # return {"x": tokens.to(device), "a": ages.to(device), "y": None, "b": None, "predict": False, "d": domains.to(device)}

# block_size=48, select="left", padding="regular"
# no_event_token_rate=5

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


# def sort_by_age(tokens, ages):
#     """Ensure chronological order."""
#     s = torch.argsort(ages, 1)
#     tokens = torch.gather(tokens, 1, s)
#     ages   = torch.gather(ages, 1, s)
#     return tokens, ages


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

# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————   

# class Subject:

#     def __init__(self, data_domains):
        
#         self.data = pd.concat([ data_domains[dname].tokens for dname in data_domains ]).\
#             sort_values(['subject_id', 'age']).\
#             set_index('subject_id')


#     def _repr_html_(self):
#         pass

# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————   


# class DelphiBatchDataset(torch.utils.data.Dataset):

#     def __init__(self, delphi_dataset):
#         self.ds = delphi_dataset
        
#         self.base = delphi_dataset

#     @property
#     def subjects(self):
#         return self.base.subjects

#     def __len__(self):
#         return len(self.ds)

#     def __getitem__(self, idx):
#         t0 = time.perf_counter()
#         subj_id = self.ds.subjects[idx]
#         # subj_id = self.ds.subjects.iloc[idx]["subject_id"]
    
#         # medir get_subject_events
#         t1 = time.perf_counter()
#         subject_events = self.ds.get_subject_events(subj_id)
#         t2 = time.perf_counter()
    
#         domains_dict = {}
#         for dname, df in subject_events.items():
#             arr = df[["subject_id", "age", "token_id"]].to_numpy()
#             domains_dict[dname] = arr
#         t3 = time.perf_counter()
    
#         print(f"[getitem] total={t3-t0:.3f}s, get_events={t2-t1:.3f}s, df->numpy={t3-t2:.3f}s")
    
#         return {
#           "subject_id": subj_id,
#           "domains": domains_dict,
#           "domains_metadata": self.ds.domains
#     }

# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def collate_fn_domains(batch, device=DEVICE):

    """
    Collate function that processes each domain separately, using domain metadata.
    Each item in batch is expected to be:
      {
        "subject_id": ...,
        "domains": {dname: arr},
        "domains_metadata": {dname: TokenDomain}  # must expose .predict and .jitter
      }
    """

    domains_dict = {}
    for item in batch:
        for dname, arr in item.items():
            if len(arr) == 0:
                continue
            arr.token_id = arr.token_id.cat.codes
            values = torch.tensor(arr.values).to(device)
            domains_dict.setdefault(dname, []).append(values)

    return domains_dict # { dname: pd.DataFrame(np.stack(domain, axis=0)) for dname, domain in domains_dict.items()}
    # return { dname: pd.DataFrame(np.stack(domain, axis=0)) for dname, domain in domains_dict.items()}
    # return { dname: pd.concat(domain, axis=0) for dname, domain in domains_dict.items()}
    
    all_data = []
    for dname, arrs in domains_dict.items():
        t_dom0 = time.perf_counter()
        domain_data = pd.concat(arrs)
        t_dom1 = time.perf_counter()
        # domain_data = pd.DataFrame(data, columns=["subject_id","age","token_id"])
        all_data.append(domain_data)
        t_dom2 = time.perf_counter()
        # print(f"[collate] domain {dname}: concat={t_dom1-t_dom0:.3f}s, df={t_dom2-t_dom1:.3f}s")

    merged = pd.concat(all_data, axis=0)
    t2 = time.perf_counter()
    merged = merged.reset_index().sort_values(["subject_id","age"])
    t3 = time.perf_counter()

    # print(f"[collate] group+concat={t1-t0:.3f}s, pd.concat={t2-t1:.3f}s, sort={t3-t2:.3f}s, total={t3-t0:.3f}s")
    return merged


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
# %%
