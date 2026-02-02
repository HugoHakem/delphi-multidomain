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

@dataclass
class SinglePatientTrajectory:
    subject_id: int
    token_ids: torch.Tensor        # shape: [L] (local token ids)
    domain_ids: torch.Tensor       # shape: [L] (domain index per token)
    token_ages: torch.Tensor       # shape: [L] (float ages)
    metadata: Dict[str, Any]       # arbitrary extra info (sex, cohort, etc.)
    tokenizer: Optional[Any] = None

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
        collate_fn = collate_fn_domains
        # collate_fn = lambda b: collate_fn(b, block_size=block_size, device=self.device)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
        
