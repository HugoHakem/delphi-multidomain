#%%
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from contextlib import nullcontext
from pathlib import Path
from ast import literal_eval
from pprint import pprint
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional, List, Dict, Set, Any, Tuple
from easydict import EasyDict
import yaml

import warnings
import logging

import numpy as np
import pandas as pd

from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

from delphi.optim import OptimConfig, configure_optimizers
from typing import Union, Callable

DAYS_PER_YEAR = 365.25
DEVICE = os.getenv("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————

class TokenDomain:
    
    """
    Representation of a single token domain in the Delphi-style event model.

    A TokenDomain encapsulates all information associated with one domain
    (e.g. diseases, drugs, labs, demographics), including:
    - the tokenizer definition (tokenizer.yaml),
    - the per-subject token occurrences (tokens.csv),
    - metadata controlling how tokens are interpreted in time and during training.

    The domain can be either categorical (discrete token IDs) or continuous
    (real-valued observations), and may optionally be restricted to a subset
    of subjects.

    Parameters
    ----------
    path : str
        Path to the domain directory. Must contain:
        - tokenizer.yaml : list-like definition of tokens
        - tokens.csv     : per-subject token occurrences
    predict : bool
        Whether this domain is a prediction target.
    age_jitter : bool
        Whether age jittering is applied to this domain.
    type : {"categorical", "continuous"}
        Domain type. Determines the expected schema of tokens.csv.
    subjects : set or None, optional
        Optional subset of subject IDs to keep. If None, all subjects are loaded.
    at_birth : bool, default False
        If True, all token ages are forced to zero. A warning is raised if
        non-zero ages are found.
    aggregation_strategy : callable or None, optional
        Placeholder for future aggregation logic. Currently not implemented.

    Notes
    -----
    - Internally, tokens are stored both as a pandas DataFrame (for inspection)
      and as a torch.Tensor (for efficient downstream processing).
    - The tensor representation follows the column order of tokens.csv.
    - Subject filtering can be applied at load time or later via `filter_subjects`.
    - Aggregation strategies are explicitly not implemented yet.

    Raises
    ------
    AssertionError
        If `type` is not one of {"categorical", "continuous"}.
    ValueError
        If required columns are missing from tokens.csv.
    NotImplementedError
        If `aggregation_strategy` is provided.
    """

    REQUIRED_FILES = {
        "tokens": "tokens.csv",
        "tokenizer": "tokenizer.yaml",
    }

    BASE_COLUMNS = {"subject_id"}
    CATEGORICAL_COLUMNS = {"token_id"}
    CONTINUOUS_COLUMNS = {"values"}
    AGE_COLUMN = "age"

    def __init__(self,
            name: str,
            path: Union[str, Path],
            predict: bool, 
            age_jitter: bool, 
            type: str, 
            subjects: Union[None, set]=None, 
            at_birth=False, 
            aggregation_strategy: Union[None, Callable]=None
        ):
        
        """
        Load a single domain: tokenizer.yaml + tokens.csv
        """

        self.path = path
        self.predict = predict
        self.age_jitter = age_jitter        
        self.at_birth = at_birth
        self.type = type
        self.name = name

        assert type in ["categorical", "continuous"], f"Domain type must be either 'categorical' or 'continuous', got {type}"
       
        self.tokens    = self._load_tokens(os.path.join(path, "tokens.csv"), subjects=subjects)
        self.tokenizer = self._load_tokenizer(os.path.join(path, "tokenizer.yaml"))        
        self.aggregation_strategy = aggregation_strategy

        if aggregation_strategy is not None:
            raise NotImplementedError

    @property
    def tokens_path(self) -> Path:
        return self.path / self.REQUIRED_FILES["tokens"]

    @property
    def tokenizer_path(self) -> Path:
        return self.path / self.REQUIRED_FILES["tokenizer"]

    @property
    def subject_ids(self):
        return self.tokens[:, 0].cpu().numpy()

    def _check_required_files(self) -> None:
        missing = [
            fname
            for fname in self.REQUIRED_FILES.values()
            if not (self.path / fname).exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing required files in domain '{self.path}': {missing}"
            )


    def _expected_columns(self) -> set:
        cols = set(self.BASE_COLUMNS)

        if self.type == "categorical":
            cols |= self.CATEGORICAL_COLUMNS
        else:
            cols |= self.CONTINUOUS_COLUMNS

        cols.add(self.AGE_COLUMN)
        return cols


    def _validate_schema(self, df: pd.DataFrame, path: Path) -> None:
        expected = self._expected_columns()
        missing = expected - set(df.columns)

        if missing:
            raise ValueError(
                f"Invalid tokens file {path}. "
                f"Missing required columns: {sorted(missing)}"
            )


    def _load_tokenizer(self, path: str) -> Dict:
        with open(path, "r") as f:
            tokenizer = yaml.safe_load(f)        
        return { idx: token for idx, token in enumerate(tokenizer) }  


    def _load_tokens(self, path: str, subjects) -> pd.DataFrame:
                
        if not os.path.exists(path):
            assert False, f"Tokens file {path} does not exist"
            if self.type == "categorical":
                return pd.DataFrame(columns=["subject_id", "token_id", "age"])
            elif self.type == "continuous":
                return pd.DataFrame(columns=["subject_id", "values", "age"])
        
        df = pd.read_csv(path)

        # enforce schema
        if "subject_id" not in df.columns:
            raise ValueError(f"Invalid tokens file {path}, must contain subject_id")
        if self.type == "categorical" and "token_id" not in df.columns:
            raise ValueError(f"Invalid tokens file {path}, must contain token_id since type==categorical")
            

        df = df.sort_values("subject_id")
        
        if self.at_birth:
            if "age" in df.columns:
                if (df["age"] != 0).any():
                    warnings.warn(
                        "at_birth=True but non-zero ages found in dataframe. "
                        "All ages will be overwritten to 0.",
                        UserWarning
                    )                    
                    df["age"] = 0

        if subjects is not None:
            df = df.query("subject_id in @subjects")

        self._as_dataframe = df

        return torch.tensor(df.values) 
    
    
    def filter_subjects(self, subjects: set):
        
        filtered_domain = deepcopy(self._as_dataframe)
        isin_subset = filtered_domain.subject_id.isin(subjects).values
        self.tokens = self.tokens[isin_subset]
        self._as_dataframe = self._as_dataframe[isin_subset]
        return self


    def __len__(self):
        return len(self._as_dataframe)


    def __repr__(self):
        return str(self._as_dataframe)


    def _repr_html_(self):
        return self.tokens.\
            assign( **{"age (years)": (self.tokens.age / DAYS_PER_YEAR).round(2)} ).\
            drop("age", axis=1)._repr_html_()
             
    
    def __getitem__(self, subject_id):

        return self.tokens.set_index("subject_id").loc[[subject_id]].\
            assign( **{"age (years)": lambda df: (df.age / DAYS_PER_YEAR).round(2)} ).\
            drop("age", axis=1)

    
    def as_dataframe(self):

        return self._as_data_frame.assign(
            token_id=lambda df: pd.Categorical(df["token_id"].map(self.tokenizer), categories=self.tokenizer.values())
        )

# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————    

class DelphiDataset:

    def __init__(self, 
        root: str, 
        domains: dict, 
        subjects: Union[str, List[str]] = None, 
        exclusions: List[str] = [], 
        n_samples=None, 
        required_domains=['sex', 'diseases'], 
        device=DEVICE):

        """
        Args:
            root: base data directory
            domains: list of domain names (e.g. ["diagnosis", "lifestyle", "sex", "death"])
            fold: if specified, restrict subjects to that fold
            exclusions: list of exclusion filenames under exclusion_lists/
        """

        self.root = Path(root)
        self._device = device

        # Load the data        
        self.domains = EasyDict()
        for dname, dinfo in domains.items():
            
            if dname == "padding":
                continue

            datafile = self.root / "tokens" / dname
            self.domains[dname] = TokenDomain(
                dname,
                datafile, 
                predict=dinfo.predict, 
                age_jitter=dinfo.age_jitter, 
                type=dinfo.type, 
                at_birth=dinfo.at_birth
            )

        logger.info(
            "Initializing DelphiDataset | root=%s | domains=%s | required_domains=%s",
            str(self.root),
            list(domains.keys()),
            required_domains
        )

        # ——————————————————— DEFINE ALLOWED SUBJECTS —————————————————————————————————
        if subjects is None:
            raise ValueError("You must provide either subject IDs or paths to subject lists.")

        # detect if subjects are files or IDs
        if isinstance(subjects, (str, Path)):
            subjects = [subjects]

        if all(isinstance(s, (str, Path)) and os.path.exists(s) for s in subjects):
            # subjects are file paths
            included_subjects = pd.concat([
                pd.read_csv(self.root / subj_file, names=["subject_id"]) for subj_file in subjects
            ])
        else:
            # assume it's a list/array of IDs
            included_subjects = pd.DataFrame({"subject_id": subjects})

        self.included_subjects = included_subjects
        logger.info("Included subjects: %d", len(self.included_subjects))
    
        self.excluded_subjects = self.get_excluded_subjects(exclusion_files=exclusions)
        if exclusions:
            logger.info(
                "Excluding %d subjects using files: %s",
                len(self.excluded_subjects),
                exclusions
            )

        self._subjects = self.included_subjects[~self.included_subjects["subject_id"].astype(str).isin(self.excluded_subjects)]
        self._subjects = self.filter_subj_for_required_domains(self._subjects, required_domains)        

        logger.info(
            "Subjects after required domain filtering (%s): %d",
            required_domains,
            len(self._subjects)
         )
 
        if n_samples is not None:
            logger.info("Subsampling subjects to n_samples=%d", n_samples)
            self._subjects = self._subjects.sample(n_samples)
        
        self._subjects = set(self._subjects.subject_id.to_list())

        # —————————————————————————————————————————————————————————————————————————————

        # Now we filter each domain for the final list of allowed subjects
        for dname in self.domains:
            filtered_domain = self.domains[dname].filter_subjects(self.subjects)            
            
            assert len(filtered_domain) != 0, f"""\n{'—'*80}
Error: Domain '{dname}' has no tokens. 
Here are a few original subjects and a few allowed subjects to help you troubleshoot:
{self.subjects[:10]}
and
{self.domains[dname]}
{'—'*80}
"""
            self.domains[dname] = filtered_domain

        # —————————————————————————————————————————————————————————————————————————————

        # of internal use, for faster slicing
        self._subject_indices = self._precompute_subject_indices_per_domain()
           

    @property
    def device(self):
        device = [ self.domains[dname].tokens.device for dname in self.domains ][0]
        return device

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

    domains = defaultdict(list)
    for item in batch:
        for d, arr in item.items():
            domains[d].append(arr)

    sample_device = next(iter(batch[0].values())).device
    out = {d: torch.cat(lst, dim=0) for d, lst in domains.items()}

    if "padding" not in out:
        out["padding"] = torch.empty((0, 3), dtype=torch.long, device=sample_device)

    return out


class FlexibleDataLoader(DataLoader):
    
    def set_batch_size(self, new_batch_size):
        return self.__class__(
            dataset=self.dataset,
            batch_size=new_batch_size,
            shuffle = isinstance(self.sampler, torch.utils.data.RandomSampler),
            sampler=self.sampler,
            batch_sampler=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
            multiprocessing_context=self.multiprocessing_context,
            generator=self.generator,
            prefetch_factor=getattr(self, "prefetch_factor", 2),
            persistent_workers=getattr(self, "persistent_workers", False)
        )


class DelphiDataloader(FlexibleDataLoader):
    
    def __init__(self, dataset, block_size=48, device=None, return_dictionary=True, **kwargs):

        self.device = dataset.device if device is None else device
        #if return_dictionary:
        collate_fn = collate_fn_domains
        # collate_fn = lambda b: collate_fn(b, block_size=block_size, device=self.device)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
        # else:
        
    
    def get_tensors_from_batch(self, batch, device=None):
        
        tokens, ages, subject_ids = EasyDict(), EasyDict(), EasyDict()
        
        for dname in batch:               
            SUBJECT_ID_COLUMN, AGE_COLUMN, TOKEN_COLUMN = 0, 1, 2
            domain_data = batch.get(dname, [])  
            if len(domain_data) == 0:
                domain_data = domain_data.view(0, 3)
            if dname == "genetic_pcs":
                tokens[dname] = domain_data[:, 1:-1].float()
                ages[dname] = domain_data[:, -1]
                subject_ids[dname] = domain_data[:, 0]
            else:    
                tokens[dname] = domain_data[:, TOKEN_COLUMN].int()
                ages[dname] = domain_data[:, AGE_COLUMN]
                subject_ids[dname] = domain_data[:, SUBJECT_ID_COLUMN]

        if device is not None:
            for dname in x:
                x[dname] = x[dname].to(device)
                ages[dname] = ages[dname].to(device)
                subject_ids[dname] = subject_ids[dname].to(device)

        return tokens, ages, subject_ids
    
    
    def truncate_subjects_by_age(self,
        x: dict[str, torch.Tensor],
        ages: dict[str, torch.Tensor],
        subject_ids: dict[str, torch.Tensor],
        total_per_subject: dict[int, int],
        seqlen: int,
        trim_domains: set[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Truncate subjects with more than `seqlen` tokens, globally across trim_domains.
    
        Drops the most recent (highest age) events until each subject has exactly `seqlen` tokens total.
        Returns updated dicts with tokens removed.
        """
        device = next(iter(subject_ids.values())).device
    
        # Accumulate indices to drop per domain
        to_drop_by_domain: dict[str, list[int]] = defaultdict(list)
    
        for subj_id, tot in total_per_subject.items():
            diff = seqlen - tot
            if diff >= 0:
                continue  # nothing to drop
    
            R = -diff  # number of tokens to remove
    
            # Collect all candidate (age, domain, idx) from trim_domains
            candidates = []
            for d in trim_domains:
                if d not in subject_ids:
                    continue
                sids_d, ages_d = subject_ids[d], ages[d]
                idx = (sids_d == subj_id).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                for j in idx.tolist():
                    candidates.append((float(ages_d[j].item()), d, j))
    
            if not candidates:
                # No eligible domains for trimming
                continue
    
            # Sort by age (ascending) and remove the R most recent
            candidates.sort(key=lambda t: t[0])
            drop = candidates[-min(R, len(candidates)):]
            for _, d, j in drop:
                to_drop_by_domain[d].append(j)
    
        # Apply drops per domain
        for d, drop_list in to_drop_by_domain.items():
            if not drop_list:
                continue
            mask = torch.ones(len(subject_ids[d]), dtype=torch.bool, device=device)
            mask[torch.tensor(sorted(set(drop_list)), device=device)] = False
            x[d] = x[d][mask]
            ages[d] = ages[d][mask]
            subject_ids[d] = subject_ids[d][mask]
    
        return x, ages, subject_ids


    def pad_subjects(self,
        x: dict[str, torch.Tensor],
        ages: dict[str, torch.Tensor],
        subject_ids: dict[str, torch.Tensor],
        total_per_subject: dict[int, int],
        seqlen: int,
        pad_domain: str,
        PADDING_TOKEN: int,
        PAD_AGE: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Pad subjects with fewer than `seqlen` tokens by appending padding tokens
        in the specified `pad_domain`.
        """
        device = next(iter(subject_ids.values())).device
        dtype_x = next(iter(x.values())).dtype
        dtype_age = next(iter(ages.values())).dtype
        dtype_sid = next(iter(subject_ids.values())).dtype
    
        pad_x, pad_a, pad_sid = [], [], []
    
        for subj_id, tot in total_per_subject.items():
            diff = seqlen - tot
            if diff <= 0:
                continue
            pad_x.append(torch.full((diff,), PADDING_TOKEN, device=device, dtype=dtype_x))
            pad_a.append(torch.full((diff,), PAD_AGE, device=device, dtype=dtype_age))
            pad_sid.append(torch.full((diff,), subj_id, device=device, dtype=dtype_sid))
    
        if pad_x:
            x[pad_domain] = torch.cat([x[pad_domain], torch.cat(pad_x)])
            ages[pad_domain] = torch.cat([ages[pad_domain], torch.cat(pad_a)])
            subject_ids[pad_domain] = torch.cat([subject_ids[pad_domain], torch.cat(pad_sid)])
    
        return x, ages, subject_ids


    def adjust_to_seqlen(self,
        x: dict[str, torch.Tensor],
        ages: dict[str, torch.Tensor],
        subject_ids: dict[str, torch.Tensor],
        seqlen: int,
        *,
        pad_domain: str = "padding",
        trim_domains: set[str] = frozenset({"diseases"}),
        PADDING_TOKEN: int = 0,
        PAD_AGE: float = -10000.0,
        ):
        
        """
        Main orchestrator: ensures all subjects have exactly `seqlen` tokens in total,
        truncating by age when too long, padding otherwise.
        """
                
        x  = deepcopy(x)
        ages  = deepcopy(ages)
        subject_ids  = deepcopy(subject_ids)
     
        # Domains to consider for counting (exclude pad domain)
        domains = [d for d in subject_ids.keys() if d != pad_domain]
    
        # Count total tokens per subject (excluding padding)
        if domains:
            all_sids = torch.cat([subject_ids[d] for d in domains])
            subj_uniq, counts = np.unique(all_sids.cpu().int().numpy(), return_counts=True)
            total_per_subject = {int(k): int(v) for k, v in zip(subj_uniq, counts)}
        else:
            total_per_subject = {}
    
        # Step 1: truncate subjects with too many tokens
        x, ages, subject_ids = self.truncate_subjects_by_age(
            x, ages, subject_ids, total_per_subject, seqlen, trim_domains
        )
    
        # Step 2: recompute totals (after truncation)
        if domains:
            all_sids = torch.cat([subject_ids[d] for d in domains])
            subj_uniq, counts = np.unique(all_sids.cpu().int().numpy(), return_counts=True)
            total_per_subject = {int(k): int(v) for k, v in zip(subj_uniq, counts)}
    
        # Step 3: pad subjects that are still short
        x, ages, subject_ids = self.pad_subjects(
            x, ages, subject_ids, total_per_subject, seqlen, pad_domain, PADDING_TOKEN, PAD_AGE
        )
    
        return x, ages, subject_ids
# %%

def get_tensors_from_batch(batch, device=None):
    from easydict import EasyDict
        
    tokens, ages, subject_ids = EasyDict(), EasyDict(), EasyDict()
    
    for dname in batch:               
        SUBJECT_ID_COLUMN, AGE_COLUMN, TOKEN_COLUMN = 0, 1, 2
        domain_data = batch.get(dname, [])  
        if len(domain_data) == 0:
            # Create empty tensor without device assignment (always on CPU)
            if isinstance(domain_data, torch.Tensor):
                # If it's already a tensor with device, create new one without device
                domain_data = torch.empty(0, 3, dtype=domain_data.dtype)
            else:
                # If it's a list, create tensor on CPU (no device)
                domain_data = torch.empty(0, 3, dtype=torch.float32)
        if dname == "genetic_pcs":
            tokens[dname] = domain_data[:, 1:-1].float()
            ages[dname] = domain_data[:, -1]
            subject_ids[dname] = domain_data[:, 0]
        else:    
            tokens[dname] = domain_data[:, TOKEN_COLUMN].int()
            ages[dname] = domain_data[:, AGE_COLUMN]
            subject_ids[dname] = domain_data[:, SUBJECT_ID_COLUMN]

    if device is not None:
        for dname in tokens:
            tokens[dname] = tokens[dname].to(device)
            ages[dname] = ages[dname].to(device)
            subject_ids[dname] = subject_ids[dname].to(device)

    return tokens, ages, subject_ids

def truncate_subjects_by_age(
        x: dict[str, torch.Tensor],
        ages: dict[str, torch.Tensor],
        subject_ids: dict[str, torch.Tensor],
        total_per_subject: dict[int, int],
        seqlen: int,
        trim_domains: set[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Truncate subjects with more than `seqlen` tokens, globally across trim_domains.
    
        Drops the most recent (highest age) events until each subject has exactly `seqlen` tokens total.
        Returns updated dicts with tokens removed.
        """
        device = next(iter(subject_ids.values())).device
    
        # Accumulate indices to drop per domain
        to_drop_by_domain: dict[str, list[int]] = defaultdict(list)
    
        for subj_id, tot in total_per_subject.items():
            diff = seqlen - tot
            if diff >= 0:
                continue  # nothing to drop
    
            R = -diff  # number of tokens to remove
    
            # Collect all candidate (age, domain, idx) from trim_domains
            candidates = []
            for d in trim_domains:
                if d not in subject_ids:
                    continue
                sids_d, ages_d = subject_ids[d], ages[d]
                idx = (sids_d == subj_id).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                for j in idx.tolist():
                    candidates.append((float(ages_d[j].item()), d, j))
    
            if not candidates:
                # No eligible domains for trimming
                continue
    
            # Sort by age (ascending) and remove the R most recent
            candidates.sort(key=lambda t: t[0])
            drop = candidates[-min(R, len(candidates)):]
            for _, d, j in drop:
                to_drop_by_domain[d].append(j)
    
        # Apply drops per domain
        for d, drop_list in to_drop_by_domain.items():
            if not drop_list:
                continue
            mask = torch.ones(len(subject_ids[d]), dtype=torch.bool, device=device)
            mask[torch.tensor(sorted(set(drop_list)), device=device)] = False
            x[d] = x[d][mask]
            ages[d] = ages[d][mask]
            subject_ids[d] = subject_ids[d][mask]
    
        return x, ages, subject_ids


def pad_subjects(
        x: dict[str, torch.Tensor],
        ages: dict[str, torch.Tensor],
        subject_ids: dict[str, torch.Tensor],
        total_per_subject: dict[int, int],
        seqlen: int,
        pad_domain: str,
        PADDING_TOKEN: int,
        PAD_AGE: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Pad subjects with fewer than `seqlen` tokens by appending padding tokens
        in the specified `pad_domain`.
        """
        device = next(iter(subject_ids.values())).device
        dtype_x = next(iter(x.values())).dtype
        dtype_age = next(iter(ages.values())).dtype
        dtype_sid = next(iter(subject_ids.values())).dtype
    
        pad_x, pad_a, pad_sid = [], [], []
        
        for subj_id, tot in total_per_subject.items():            
            diff = seqlen - tot
            if diff <= 0:
                continue
            pad_x.append(torch.full((diff,), PADDING_TOKEN, device=device, dtype=dtype_x))
            pad_a.append(torch.full((diff,), PAD_AGE, device=device, dtype=dtype_age))
            pad_sid.append(torch.full((diff,), subj_id, device=device, dtype=dtype_sid))
    
        if pad_x:
            x[pad_domain] = torch.cat([x[pad_domain], torch.cat(pad_x)])
            ages[pad_domain] = torch.cat([ages[pad_domain], torch.cat(pad_a)])
            subject_ids[pad_domain] = torch.cat([subject_ids[pad_domain], torch.cat(pad_sid)])
    
        return x, ages, subject_ids


def adjust_to_seqlen(
    x: dict[str, torch.Tensor],
    ages: dict[str, torch.Tensor],
    subject_ids: dict[str, torch.Tensor],
    seqlen: int,
    *,
    pad_domain: str = "padding",
    trim_domains: set[str] = frozenset({"diseases"}),
    PADDING_TOKEN: int = 0,
    PAD_AGE: float = -10000.0,
    verbose=False,
    debug_max_subjects: int = 10,
):
    """
    Main orchestrator: ensures all subjects have exactly `seqlen` tokens in total,
    truncating by age when too long, padding otherwise.

    Verbosity:
        verbose=False        → no logging
        verbose=True/'info'  → aggregate info per subject
        verbose='debug'      → extra stats and per-domain details (capped)
    """

    # Normalize verbosity mode
    if verbose is True:
        verbose_mode = "info"
    elif verbose in ("info", "debug"):
        verbose_mode = verbose
    else:
        verbose_mode = None

    x, ages, subject_ids = deepcopy(x), deepcopy(ages), deepcopy(subject_ids)

    # Domains to consider for counting (including padding, for diagnostics)
    domains = [d for d in subject_ids.keys()]

    # Helper: compute per-subject total tokens
    def compute_totals(subject_ids_dict):
        if not subject_ids_dict:
            return {}
        all_subj_ids = torch.cat([subject_ids_dict[d] for d in subject_ids_dict])
        subj_uniq, counts = np.unique(
            all_subj_ids.cpu().int().numpy(), return_counts=True
        )
        return {int(k): int(v) for k, v in zip(subj_uniq, counts)}

    total_per_subject = compute_totals(subject_ids)

    # Debug: global stats before
    if verbose_mode:
        counts = np.array(list(total_per_subject.values())) if total_per_subject else np.array([0])
        print(f"[adjust_to_seqlen] seqlen={seqlen}, n_subjects={len(total_per_subject)}")
        print(
            f"[adjust_to_seqlen] tokens per subject (before): "
            f"min={counts.min()}, p50={np.percentile(counts,50):.1f}, "
            f"p90={np.percentile(counts,90):.1f}, p95={np.percentile(counts,95):.1f}, "
            f"max={counts.max()}"
        )

        if verbose_mode == "debug":
            # Show a few extreme subjects
            sorted_subj = sorted(total_per_subject.items(), key=lambda kv: kv[1], reverse=True)
            print(f"[adjust_to_seqlen][debug] Top {min(debug_max_subjects, len(sorted_subj))} subjects by length (before):")
            for sid, tot in sorted_subj[:debug_max_subjects]:
                print(f"  • Subject {sid}: {tot} tokens total")

    # Keep copies for debug comparison
    if verbose_mode == "debug":
        old_x = deepcopy(x)
        old_ages = deepcopy(ages)
        old_sids = deepcopy(subject_ids)

    # Step 1: truncate subjects with too many tokens
    if verbose_mode:
        print(f"[adjust_to_seqlen] Step 1: truncating subjects > {seqlen} tokens...")

    x, ages, subject_ids = truncate_subjects_by_age(
        x, ages, subject_ids, total_per_subject, seqlen, trim_domains
    )

    # Recompute totals after truncation
    total_per_subject_after_trunc = compute_totals(subject_ids)

    if verbose_mode:
        # Basic per-subject info
        n_truncated = sum(
            1 for sid in total_per_subject
            if total_per_subject_after_trunc.get(sid, 0) < total_per_subject[sid]
        )
        print(f"[adjust_to_seqlen] Subjects truncated: {n_truncated} / {len(total_per_subject)}")

        if verbose_mode == "debug" and n_truncated > 0:
            print(f"[adjust_to_seqlen][debug] Example truncated subjects (up to {debug_max_subjects}):")
            shown = 0
            for sid, before in sorted(total_per_subject.items(), key=lambda kv: kv[1], reverse=True):
                after = total_per_subject_after_trunc.get(sid, 0)
                if after < before:
                    print(f"  • Subject {sid}: {before} → {after} tokens (truncated {before - after})")
                    # Per-domain breakdown before/after (debug)
                    per_domain_before = {}
                    per_domain_after = {}
                    for d in domains:
                        per_domain_before[d] = int((old_sids[d] == sid).sum().item())
                        per_domain_after[d] = int((subject_ids[d] == sid).sum().item())
                    print(f"    by domain before: {per_domain_before}")
                    print(f"    by domain after : {per_domain_after}")
                    shown += 1
                    if shown >= debug_max_subjects:
                        break

    # Step 2: recompute totals (after truncation, before padding)
    total_per_subject = compute_totals(subject_ids)

    # Step 3: pad subjects that are still short
    if verbose_mode:
        print(f"[adjust_to_seqlen] Step 3: padding subjects < {seqlen} tokens in domain '{pad_domain}'...")

    x, ages, subject_ids = pad_subjects(
        x, ages, subject_ids, total_per_subject, seqlen, pad_domain, PADDING_TOKEN, PAD_AGE
    )

    # Final totals
    final_totals = compute_totals(subject_ids)

    if verbose_mode:
        final_counts = np.array(list(final_totals.values())) if final_totals else np.array([0])
        print(
            f"[adjust_to_seqlen] tokens per subject (final): "
            f"min={final_counts.min()}, p50={np.percentile(final_counts,50):.1f}, "
            f"p90={np.percentile(final_counts,90):.1f}, p95={np.percentile(final_counts,95):.1f}, "
            f"max={final_counts.max()}"
        )

        bad_lt = [sid for sid, tot in final_totals.items() if tot < seqlen]
        bad_gt = [sid for sid, tot in final_totals.items() if tot > seqlen]

        if bad_lt:
            print(f"[adjust_to_seqlen][WARN] {len(bad_lt)} subject(s) ended with < seqlen tokens.")
            if verbose_mode == "debug":
                print(f"  Example: {bad_lt[:debug_max_subjects]}")
        if bad_gt:
            print(f"[adjust_to_seqlen][ERROR] {len(bad_gt)} subject(s) ended with > seqlen tokens (this should not happen).")
            if verbose_mode == "debug":
                print(f"  Example: {bad_gt[:debug_max_subjects]}")

        if verbose_mode == "debug":
            # Show some subjects with big padding
            print(f"[adjust_to_seqlen][debug] Example subjects after padding (up to {debug_max_subjects}):")
            for sid, tot in list(final_totals.items())[:debug_max_subjects]:
                per_domain = {}
                for d in subject_ids:
                    per_domain[d] = int((subject_ids[d] == sid).sum().item())
                print(f"  • Subject {sid}: {tot} tokens, by domain: {per_domain}")

        print("[adjust_to_seqlen] Done.\n")

    return x, ages, subject_ids


@dataclass
class PatientTrajectory:
    subject_id: int
    token_ids: torch.Tensor        # shape: [L] (local token ids)
    domain_ids: torch.Tensor       # shape: [L] (domain index per token)
    token_ages: torch.Tensor       # shape: [L] (float ages)
    metadata: Dict[str, Any]       # arbitrary extra info (sex, cohort, etc.)
    tokenizer: Optional[Any] = None