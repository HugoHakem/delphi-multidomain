#%%
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")

import time
from contextlib import nullcontext
from pathlib import Path
from ast import literal_eval
from pprint import pprint
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional, List, Dict, Set
from easydict import EasyDict
import yaml
import warnings
from omegaconf import OmegaConf
import logging

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from delphi.model.components import DelphiConfig
from delphi.model.transformer import Delphi
from delphi.optim import OptimConfig, configure_optimizers
from typing import Union

DAYS_PER_YEAR = 365.25
DEVICE = os.getenv("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu')

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
        
        self.tokens = self._load_tokens(os.path.join(path, "tokens.csv"), subjects=subjects)                

    def _load_tokenizer(self, path: str) -> Dict:
        with open(path, "r") as f:
            tokenizer = yaml.safe_load(f)        
        return { idx: token for idx, token in enumerate(tokenizer) }  

    def _load_tokens(self, path: str, subjects) -> pd.DataFrame:
        if not os.path.exists(path):
            assert False, f"Tokens file {path} does not exist"
            return pd.DataFrame(columns=["subject_id", "token_id", "age"])
        df = pd.read_csv(path)
        
        # enforce schema
        if "subject_id" not in df.columns or "token_id" not in df.columns:
            raise ValueError(f"Invalid tokens file {path}, must contain subject_id and token_id")
        if "age" not in df.columns:
            df["age"] = None

        if subjects is not None:
            df = df.query("subject_id in @subjects")
        # see if I need to return also the offsets for each subject

        self._as_dataframe = df 

        return torch.tensor(df.values) 
    
    
    def filter_subjects(self, subjects: set):

        from copy import deepcopy
        filtered_domain = deepcopy(self._as_dataframe)
        isin_subset = filtered_domain.subject_id.isin(subjects).values
        self.tokens = self.tokens[isin_subset]
        self._as_dataframe = self._as_dataframe[isin_subset]
        return self


    @property
    def subject_ids(self):
        return self.tokens[:, 0].cpu().numpy()


    def as_dataframe(self):

        return self._as_data_frame.assign(
            token_id=lambda df: pd.Categorical(df["token_id"].map(self.tokenizer), categories=self.tokenizer.values())
        )


    def _repr_html_(self):
        return self.tokens.\
            assign( **{"age (years)": (self.tokens.age / 363.25).round(2)} ).\
            drop("age", axis=1)._repr_html_()
             
    
    def __getitem__(self, subject_id):

        return self.tokens.set_index("subject_id").loc[[subject_id]].\
            assign( **{"age (years)": lambda df: (df.age / DAYS_PER_YEAR).round(2)} ).\
            drop("age", axis=1)

# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————    

class DelphiDataset:

    def __init__(self, root: str, domains: dict, subjects: List[str] = None, exclusions: List[str] = [], n_samples=None, required_domains=['sex', 'diseases'], merge_namespaces=False, device=DEVICE):
        """
        Args:
            root: base data directory
            domains: list of domain names (e.g. ["diagnosis", "lifestyle", "sex", "death"])
            fold: if specified, restrict subjects to that fold
            exclusions: list of exclusion list filenames under exclusion_lists/
        """

        self.root = Path(root)
        self.device = device

        # Load the data        
        self.domains = EasyDict()
        for dname, dinfo in domains.items():
            datafile = self.root / dname
            if dname == "padding":
                continue
            self.domains[dname] = TokenDomain(datafile, predict=dinfo.predict, age_jitter=dinfo.age_jitter) 

        # ——————————————————— DEFINE ALLOWED SUBJECTS —————————————————————————————————
        self.included_subjects = pd.concat([
            pd.read_csv(self.root / subj_file, names=["subject_id"]) for subj_file in subjects
        ])
                
        self.excluded_subjects = self.get_excluded_subjects(exclusion_files=exclusions)

        self._subjects = self.included_subjects[~self.included_subjects["subject_id"].astype(str).isin(self.excluded_subjects)]
        self._subjects = self.filter_subj_for_required_domains(self._subjects, required_domains)        

        if n_samples is not None:
            self._subjects = self._subjects.sample(n_samples)
        
        self._subjects = set(self._subjects.subject_id.to_list())

        # —————————————————————————————————————————————————————————————————————————————
        
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

        # for dname, domain in self.domains.items():
            # self.domains[dname].tokens = self.domains[dname].tokens.set_index("subject_id")

        # of internal use, for faster slicing
        self._subject_indices = self._precompute_subject_indices_per_domain()
           

    @property
    def subjects(self):
        if not hasattr(self, "_subjects_as_list"):
            self._subjects_as_list = list(self._subjects)
        return self._subjects_as_list


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
            # domain_subjects = self.domains[dname].tokens["subject_id"].astype(str).unique()
            domain_subjects = self.domains[dname]._as_dataframe["subject_id"].astype(str).unique()
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
            try:
                start, count = self._subject_indices[dname][subject_id]
                tokens_subj  = domain.tokens[start:(start+count)]
            except KeyError as e:
                tokens_subj = torch.empty(0, 3, dtype=torch.float32, device=self.device)
            subject_events[dname] = tokens_subj         

        return subject_events


    @staticmethod
    def is_ukb_id(index):
        return index > 1000000


    def __len__(self):
        return len(self.subjects)        


    def to(self, device):
        for dname in self.domains:
            self.domains[dname].tokens = self.domains[dname].tokens.to(device)
        return self


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
            ids, counts = np.unique(self.domains[domain].subject_ids, return_counts=True)
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


def insert_no_event_tokens(tokens, ages, subject_ids, max_age, no_event_token_rate=5, padding="regular", gen=None):

    """Insert synthetic 'no event' tokens at regular or random intervals."""
    
    if padding == "random" and gen is None:
        gen = torch.Generator(device='cpu')
        gen.manual_seed(tokens.sum().item())

    if padding in [None, "none"] or no_event_token_rate in [0, None]:
        pad = torch.ones(tokens.shape[0], 0)
    elif padding == "regular":
        pad = torch.arange(0, 100 * DAYS_PER_YEAR, DAYS_PER_YEAR * no_event_token_rate) * torch.ones(len(subject_ids), 1) + 1
    elif padding == "random":
        pad = torch.randint(1, 100 * DAYS_PER_YEAR, (tokens.shape[0], int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError(f"Unknown padding {padding}")

    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages   = torch.hstack([ages, pad])

    tokens = tokens.masked_fill(ages > max_age, -1)       # MASKING_TOKEN
    ages   = ages.masked_fill(ages > max_age, -10000)     # MASKING_AGE

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

# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def collate_fn_domains(batch):

    domains_dict = {}
    for item in batch:
        for dname, arr in item.items():
            domains_dict.setdefault(dname, []).append(arr)
    out = { d: torch.concat(lst, dim=0) for d, lst in domains_dict.items() }
    
    # infer device from any existing tensor in the batch
    if len(out) > 0:
        sample_device = next(iter(out.values())).device
    else:
        sample_device = torch.device("cpu")

    # This is for compatibility with Delphi.forward, since it expects that the data 
    # has exactly the same keys as the embedding module, which has a 'padding' domain (for regular GPT padding tokens and no-event tokens) 
    if "padding" not in out:
        out["padding"] = torch.empty((0, 3), dtype=torch.long, device=sample_device)
    
    return out


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
