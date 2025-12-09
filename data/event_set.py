from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable

import numpy as np
import torch
import weakref

from copy import deepcopy
from easydict import EasyDict


@dataclass
class EventSet:
    
    """
    Represents a collection of temporal token sequences, one per subject.
    Each subject has a stream of (token, age) pairs, optionally with other metadata.
    This class provides high-level, semantic operations on those sequences.
    """

    X_tokens: np.ndarray           # shape: [n_subjects, seq_len]
    X_ages:   np.ndarray           # shape: [n_subjects, seq_len]
    Y_tokens: Optional[np.ndarray] = None           # shape: [n_subjects, seq_len]
    Y_ages:   Optional[np.ndarray] = None           # shape: [n_subjects, seq_len]

    meta: Dict[str, Any] = field(default_factory=dict)

    subject_ids:  Optional[np.ndarray] = None  # [n_subjects]
    tokenizer: Optional[dict] = None
    
    # Optional metadata
    # domain_info:  Optional[Dict[str, Any]] = None            # e.g. model.transformer.embed.domain_embed
    # token_maps:   Optional[Dict[str, Dict[int, str]]] = None  # {domain: {id: name}}
    # model_config: Optional[Any] = None                      # could be DelphiConfig

    cache: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_batch(cls, batch_tuple, subject_ids=None, tokenizer=None):
        """
        Build an EventSet from a 4-tuple of arrays/tensors:
        (X_tokens, X_ages, Y_tokens, Y_ages).
        """
        X_tokens, X_ages, Y_tokens, Y_ages = batch_tuple
        if isinstance(X_tokens, torch.Tensor):
            X_tokens, X_ages, Y_tokens, Y_ages = [
                x.detach().cpu().numpy() for x in batch_tuple
            ]
        return cls(
            X_tokens=X_tokens, 
            X_ages=X_ages, 
            Y_tokens=Y_tokens, 
            Y_ages=Y_ages, 
            subject_ids=np.array(subject_ids).astype(int), tokenizer=tokenizer
        )

    
    # ─────────────────────────── Properties ───────────────────────────
    @property
    def n_subjects(self) -> int:
        return self.X_tokens.shape[0]

    @property
    def seq_len(self) -> int:
        return self.X_tokens.shape[1]

    @property
    def block_size(self) -> int:
        """ Just an alias """
        return self.seq_len

    @property
    def has_targets(self) -> bool:
        return self.Y_tokens is not None and self.Y_ages is not None

    @property
    def shape(self):
        return self.X_tokens.shape
    
    # --------------------------------------------------------------------------------
    # Core utilities
    # --------------------------------------------------------------------------------
    def for_token(self, token_id: int, return_type: str = "EventSet", on_targets=True):
        """
        Return a new subject mask/token mask/EventSet view focused on a specific token ID.
        if return_type == "EventSet", it keeps only subjects that have at least one token 
        equal to token_id in Y_tokens (i.e. they experienced this event/have this feature).
        """
      
        if on_targets:
            is_token  = (self.Y_tokens == token_id)
        else:
            is_token  = (self.X_tokens == token_id)
            
        has_token = is_token.any(axis=1)
  
        if return_type == "token_mask":
            return is_token
        if return_type == "subject_mask":
            # Boolean mask of shape [n_subjects]
            return has_token
        else:
            # Filter all arrays by subject
            X_tokens_f = self.X_tokens[has_token]
            X_ages_f   = self.X_ages[has_token]
            Y_tokens_f = self.Y_tokens[has_token]
            Y_ages_f   = self.Y_ages[has_token]
  
            if self.subject_ids is not None:
                subj_ids_f = self.subject_ids[has_token]
            else:
                subj_ids_f = None
   
            # Shallow copy of metadata
            meta = dict(self.meta)
            meta["focus_token"] = token_id
  
            # Return a new view
            return EventSet(
                X_tokens_f,
                X_ages_f,
                Y_tokens_f,
                Y_ages_f,
                subject_ids=subj_ids_f,
                meta=meta
            )

    # def cut_to_age(self, max_age_years: float):
    #     """Truncate sequences to a maximum age (in years)."""
    #     max_age_days = max_age_years * 365.25
    #     mask = self.X_ages <= max_age_days
    #     X_tokens_cut = np.where(mask, self.X_tokens, 0)
    #     X_ages_cut = np.where(mask, self.X_ages, -10_000.0)
    #     return EventSet(X_tokens_cut, X_ages_cut, self.Y_tokens, self.Y_ages, self.subject_ids)
    

    def filter_by_age(self, min_age: float, max_age: float, return_type="mask"):
        """
        Return a new EventSet view including only tokens within [min_age, max_age) years.
        Tokens outside this range are masked out (set to 0).
        """
        # Convert from days to years if necessary
        z_years = self.X_ages / 365.25
    
        # Boolean mask of valid ages
        age_mask = (z_years >= min_age) & (z_years < max_age)
    
        if return_type == "mask":
            return age_mask
        
        # Apply the same mask to all arrays        
        meta = dict(self.meta)
        meta["age_range"] = (min_age, max_age)
    
        return EventSet(
            X_tokens = np.where(age_mask, self.X_tokens, 0),
            X_ages   = np.where(age_mask, self.X_ages, 0),
            Y_tokens = np.where(age_mask, self.Y_tokens, 0),
            Y_ages   = np.where(age_mask, self.Y_ages, 0),
            subject_ids=self.subject_ids,
            meta=self.meta | {"age_range": (min_age, max_age)},
        )

    def count_tokens(self, token_filter=None):
        """
        Count token occurrences in X_tokens.
        Args:
            token_filter (callable or set): Function or set of allowed token_ids.
        """
        flat = self.X_tokens.flatten()
        if token_filter is not None:
            if callable(token_filter):
                flat = flat[np.vectorize(token_filter)(flat)]
            else:
                flat = flat[np.isin(flat, list(token_filter))]
        tokens, counts = np.unique(flat, return_counts=True)
        return dict(zip(tokens, counts))


    def exclude_token(self, token_id: int, return_type: str = "EventSet"):
        """
        Return a new EventSet view excluding all subjects that ever had the given token.
        This is typically used to select control subjects for a case-control comparison.
        """
        # Find subjects that never had this token
        never_had = ~(self.Y_tokens == token_id).any(axis=1)
    
        if return_type == "mask":
            return never_had
        elif return_type == "EventSet":
            meta = dict(self.meta)
            meta["excluded_token"] = token_id
            return EventSet(
                X_tokens = self.X_tokens[never_had],
                X_ages   = self.X_ages[never_had],
                Y_tokens = self.Y_tokens[never_had],
                Y_ages   = self.Y_ages[never_had],
                # subject_ids = self.subject_ids[never_had],
                meta=meta
            )


    def get_prediction_context(self, offset: float):
        """
        For each target token, find the last input token occurring at least
        `offset` days (or years) in the past.
        Returns an array of indices with shape [n_subjects, seq_len].
        """
        raise NotImplementedError

    
    def sample_n_subjects(self, n):
        sampled_subjects = np.random.choice(np.arange(self.X_tokens.shape[0]), n, replace=False)
        return EventSet(
            self.X_tokens[sampled_subjects],
            self.X_ages[sampled_subjects],
            self.Y_tokens[sampled_subjects],
            self.Y_ages[sampled_subjects],
            # self.subject_ids[sampled_subjects]
            # meta=meta
        )

    
    def sample_one_token_per_subject(
        self,
        condition: Optional[np.ndarray] = None,
        allowed_token_ids: Optional[np.ndarray] = None,
        use_output: bool = False,
        seed: Optional[int] = None,
        return_type: str = "mask",
        ):

        """
        Select exactly one token per subject that satisfies the given constraints.
    
        Args:
            condition (np.ndarray, optional):
                Boolean mask [n_subjects, seq_len]. If provided, sampling is
                restricted to True positions.
            allowed_token_ids (array-like, optional):
                1D array/list of token ids that are eligible for sampling.
                Positions whose token is not in this set are ignored.
            use_output (bool):
                If True, operate on Y_tokens/Y_ages instead of X_tokens/X_ages.
            seed (int, optional):
                Random seed for reproducibility.
            return_type (str):
                One of {"mask", "indices", "values"}.
    
        Returns:
            If return_type == "mask":
                np.ndarray[bool] of shape [n_subjects, seq_len].
            If return_type == "indices":
                (subject_idx, token_idx) as two 1D arrays.
            If return_type == "values":
                dict with:
                    - subject_idx
                    - token_idx
                    - subject_ids
                    - tokens
                    - ages
        """
        assert return_type in {'mask', 'indices', 'values'}, "'return_type' must be 'mask', 'indices', or 'values'"

        tokens_arr = self.Y_tokens if use_output else self.X_tokens
        ages_arr   = self.Y_ages if use_output else self.X_ages
    
        n_subj, seq_len = tokens_arr.shape
        rng = np.random.default_rng(seed)
    
        # Base "everything is allowed"
        valid = np.ones_like(tokens_arr, dtype=bool)
    
        # Restrict by condition mask if given
        if condition is not None:
            valid &= condition
    
        # Restrict by allowed token ids if given
        if allowed_token_ids is not None:
            allowed_token_ids = np.array(allowed_token_ids)
            valid &= np.isin(tokens_arr, allowed_token_ids)
    
        selected = np.zeros_like(valid, dtype=bool)
        subj_idx = []
        tok_idx = []
    
        for i in range(n_subj):
            valid_idx = np.flatnonzero(valid[i])
            if len(valid_idx) == 0:
                continue
            chosen = rng.choice(valid_idx)
            selected[i, chosen] = True
            subj_idx.append(i)
            tok_idx.append(chosen)
    
        subj_idx = np.array(subj_idx, dtype=int)
        tok_idx  = np.array(tok_idx, dtype=int)
    
        if return_type == "mask":
            return selected
        elif return_type == "indices":
            return subj_idx, tok_idx
        elif return_type == "values":
            return {
                "subject_idx": subj_idx,
                "token_idx": tok_idx,
                "subject_ids": self.subject_ids[subj_idx] if self.subject_ids is not None else None,
                "tokens": tokens_arr[subj_idx, tok_idx],
                "ages": ages_arr[subj_idx, tok_idx],
            }        

    def get_logits(self, model, device="cpu", token_range=None, detach=True):
        """Compute model logits for each sequence and return as np.ndarray."""
        X = torch.tensor(self.X_tokens, device=device)
        A = torch.tensor(self.X_ages, device=device)
        logits, *_ = model(X, A)
        if detach:
            logits = logits.detach().cpu().numpy()
        if token_range is not None:
            logits = logits[:, :, slice(*token_range)]
        return logits
    
    
    def case_mask(self, token_id: int):
        """
        Return a boolean mask selecting the token event token (case) for each subject.
    
        Args:
            token_id (int): token ID of interest.
    
        Returns:
            np.ndarray: Boolean mask [n_subjects, seq_len] where True marks the token event.
        """
        # One-hot where token appears in the output tokens
        if isinstance(self.Y_tokens, torch.Tensor):
            Y_tokens = self.Y_tokens.cpu().numpy()
        else:
            Y_tokens = self.Y_tokens

        mask = (Y_tokens == token_id)
    
        # Keep only the *first occurrence* per subject (if multiple)
        first_occurrence = np.zeros_like(mask, dtype=bool)
        for i in range(mask.shape[0]):
            idx = np.flatnonzero(mask[i])
            if len(idx):
                first_occurrence[i, idx[0]] = True
    
        return first_occurrence


    def summary(self, verbose=False):
        """Quick overview of the internal structures."""
        n_subj, seq_len = self.X_tokens.shape
        print(f"EventSet | subjects={n_subj}, seq_len={seq_len}")
        print(f"  dtype: {self.X_tokens.dtype}, device: {self.meta.get('device')}")
        if verbose:
            print("  X_tokens:", self.X_tokens[:2])
            print("  X_ages:", self.X_ages[:2])
            print("  Y_tokens:", self.Y_tokens[:2])
            print("  Y_ages:", self.Y_ages[:2])
            # print("  subject_ids:", "None" or self.subject_ids[:5])
  

    def as_tuple(self):
        return self.X_tokens, self.X_ages, self.Y_tokens, self.Y_ages 
    

    def cut_sequence_after_age(
        self,
        max_age_years: float,
        cut_on: str = "inputs",  # "inputs" or "targets"
        drop_targets: bool = False,
        ):
        """
        Truncate sequences consistently up to a given age.
    
        Args:
            max_age_years (float): cutoff age in years.
            cut_on (str): whether the cutoff applies to "inputs" (X) or "targets" (Y).
            drop_targets (bool): if True, drop Y entirely after truncation.
    
        Rules:
            - If cut_on == "inputs", keep only X events <= cutoff; 
              Y is filtered to remain consistent (no events before last X).
            - If cut_on == "targets", keep only Y events <= cutoff;
              X is filtered to remain consistent (no events after first kept Y).
            - If drop_targets=True, Y is dropped regardless.
        """
        assert cut_on in {"inputs", "targets"}, "cut_on must be 'inputs' or 'targets'"
        max_age_days = max_age_years * 365.25
    
        if cut_on == "inputs":
            mask_X = self.X_ages <= max_age_days
            # define last valid X per subject
            last_age = np.array([a[m].max() if m.any() else -np.inf for a, m in zip(self.X_ages, mask_X)])
            mask_Y = np.array([(y_age > last_age[i]) for i, y_age in enumerate(self.Y_ages)]) if self.Y_ages is not None else None
    
        elif cut_on == "targets":
            mask_Y = self.Y_ages <= max_age_days if self.Y_ages is not None else None
            # define earliest valid Y per subject
            first_age = np.array([a[m].min() if m.any() else np.inf for a, m in zip(self.Y_ages, mask_Y)]) if mask_Y is not None else np.full(len(self.X_ages), np.inf)
            mask_X = np.array([(x_age < first_age[i]) for i, x_age in enumerate(self.X_ages)])
    
        # apply masks
        X_tokens_new = [x[m] for x, m in zip(self.X_tokens, mask_X)]
        X_ages_new   = [a[m] for a, m in zip(self.X_ages, mask_X)]
    
        if drop_targets or self.Y_tokens is None or self.Y_ages is None:
            Y_tokens_new, Y_ages_new = None, None
        else:
            Y_tokens_new = [y[m] for y, m in zip(self.Y_tokens, mask_Y)]
            Y_ages_new   = [a[m] for a, m in zip(self.Y_ages, mask_Y)]
    
        # padding for consistency
        def pad_to_rectangular(lst, fill):

            if lst is None:
                return None
            maxlen = max(len(x) for x in lst) if lst else 0
            dtype = float if isinstance(fill, float) else int
            return np.array([
                np.pad(x, (0, maxlen - len(x)), constant_values=fill).astype(dtype)
                for x in lst
            ])
    
        X_tokens_new = pad_to_rectangular(X_tokens_new, fill=0)
        X_ages_new   = pad_to_rectangular(X_ages_new, fill=-10000.0)
        if Y_tokens_new is not None:
            Y_tokens_new = pad_to_rectangular(Y_tokens_new, fill=0)
            Y_ages_new   = pad_to_rectangular(Y_ages_new, fill=-10000.0)
    
        return EventSet(
            X_tokens=X_tokens_new,
            X_ages=X_ages_new,
            Y_tokens=Y_tokens_new,
            Y_ages=Y_ages_new,
            subject_ids=self.subject_ids,
            meta={**self.meta, "cut_to_age": max_age_years, "cut_on": cut_on}
        )

    def subject_age_max(self):
        """Return maximum age per subject (in years), as a numpy array."""
        ages = self.X_ages
        if isinstance(ages, torch.Tensor):
            # torch.max devuelve (values, indices)
            max_age = ages.max(dim=1).values.detach().cpu().numpy()
        else:
            max_age = np.max(ages, axis=1)
        return max_age / 365.25


    def mask_subjects(self, condition: np.ndarray):
        """Create a mask bound to this EventSet."""
        return EventSetMask(_mask=condition, _eventset_ref=self)


    def mask_by_age(self, min_age=None, max_age=None):
        """Return mask of subjects within an age range."""
        max_per_subj = self.X_ages.max(axis=1) / 365.25
        cond = np.ones_like(max_per_subj, dtype=bool)
        if min_age is not None:
            cond &= max_per_subj >= min_age
        if max_age is not None:
            cond &= max_per_subj <= max_age
        return EventSetMask(_mask=cond, _eventset_ref=self)

    
    def insert_tokens(self, tokens, ages, inplace=False):

        """
        Insert one or more tokens at given ages for each subject.
        Keeps events sorted by age per subject.
    
        Accepts the following shapes:
            - scalar token, scalar age: same token and age for all subjects
            - scalar token, vector age: same token, per-subject age
            - vector token, vector age: per-subject token/age pair
            - matrix token, matrix age: per-subject, per-token sequences
    
        Args:
            tokens: int, np.ndarray or torch.Tensor
            ages: float, np.ndarray or torch.Tensor (in years)
            inplace (bool): if True, modify current EventSet instead of returning new.
    
        Returns:
            EventSet (if inplace=False), else modifies self.
        """
        # Convert to numpy if tensor
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        elif isinstance(tokens, list) or isinstance(tokens, tuple):
            tokens = np.array(tokens)
        if isinstance(ages, torch.Tensor):
            ages = ages.detach().cpu().numpy()
        elif isinstance(ages, list) or isinstance(tokens, tuple):
            ages = np.array(ages)
        
        # Convert ages to days if in years
        if np.nanmax(ages) < 300:  # crude heuristic
            ages_days = np.array(ages, dtype=float) * 365.25
        else:
            ages_days = np.array(ages, dtype=float)
    
        n = self.n_subjects
        X_tokens_new = []
        X_ages_new = []
    
        for i in range(n):
            x_tok = self.X_tokens[i]#.cpu().numpy()
            x_age = self.X_ages[i]#.cpu().numpy()
    
            # Drop padding
            valid = x_age > 0
            x_tok = x_tok[valid]
            x_age = x_age[valid]
    
            # Select what to insert for this subject
            if np.ndim(tokens) == 0 and np.ndim(ages_days) == 0:
                tok_i, age_i = tokens, ages_days
            elif np.ndim(tokens) == 0 and np.ndim(ages_days) == 1:
                tok_i, age_i = tokens, ages_days[i]
            elif np.ndim(tokens) == 1 and np.ndim(ages_days) == 1:
                tok_i, age_i = tokens[i], ages_days[i]
            elif np.ndim(tokens) == 2 and np.ndim(ages_days) == 2:
                tok_i, age_i = tokens[i], ages_days[i]
            else:
                raise ValueError("tokens and ages must have compatible shapes.")
    
            # Ensure iterable
            tok_i = np.atleast_1d(tok_i)
            age_i = np.atleast_1d(age_i)
    
            # Append and re-sort by age
            new_tok = np.concatenate([x_tok, tok_i])
            new_age = np.concatenate([x_age, age_i])
            order = np.argsort(new_age)
            X_tokens_new.append(new_tok[order])
            X_ages_new.append(new_age[order])
    
        # Pad to rectangular arrays
        maxlen = max(len(x) for x in X_tokens_new)
        X_tokens_pad = np.zeros((n, maxlen), dtype=int)
        X_ages_pad = np.full((n, maxlen), -10000.0)
    
        for i in range(n):
            L = len(X_tokens_new[i])
            X_tokens_pad[i, :L] = X_tokens_new[i]
            X_ages_pad[i, :L] = X_ages_new[i]
            order = np.argsort(X_ages_pad[i], kind="stable")
            X_tokens_pad[i] = X_tokens_pad[i, order]
            X_ages_pad[i] = X_ages_pad[i, order]
    
        if inplace:
            self.X_tokens = X_tokens_pad
            self.X_ages = X_ages_pad
            return self
    
        return EventSet(
            X_tokens=X_tokens_pad,
            X_ages=X_ages_pad,
            Y_tokens=self.Y_tokens,
            Y_ages=self.Y_ages,
            subject_ids=self.subject_ids,
            meta={**self.meta, "inserted": True},
        )

    def to_tensor(self):

        self.X_tokens = torch.tensor(self.X_tokens)
        self.X_ages   = torch.tensor(self.X_ages)
        self.Y_tokens = torch.tensor(self.Y_tokens)
        self.Y_ages   = torch.tensor(self.Y_ages)

        return self


    def pipe(self, func, *args, batch_size=None, device="cpu", progress_bar=True, **kwargs):
        """
        Apply a function or model to the EventSet.
    
        If `batch_size` is set, data are split into mini-batches automatically.
        The function should accept (X_tokens, X_ages).
        """
        if batch_size is None:
            return func(self.X_tokens, self.X_ages, *args, **kwargs)
    
        n = self.n_subjects
        outputs = []
        
        from tqdm import tqdm
        pbar = tqdm(range(0, n, batch_size)) if progress_bar else range(0, n, batch_size)

        for i in pbar:
            Xb = self.X_tokens[i:i+batch_size]
            Ab = self.X_ages[i:i+batch_size]
    
            # Optional: move to device if tensors
            if isinstance(Xb, torch.Tensor):
                Xb = Xb.to(device)
                Ab = Ab.to(device)
    
            outputs.append(func(Xb, Ab, *args, **kwargs))
    
        # Try concatenating outputs if compatible
        if all(isinstance(o, np.ndarray) for o in outputs):
            return np.concatenate(outputs, axis=0)
        elif all(isinstance(o, torch.Tensor) for o in outputs):
            return torch.cat(outputs, dim=0)
        return outputs

    def to(self, device):
        self.X_tokens = self.X_tokens.to(device)
        self.X_ages   = self.X_ages.to(device)
        self.Y_tokens = self.Y_tokens.to(device)
        self.Y_ages   = self.Y_ages.to(device)
        return self


    def __getitem__(self, index):
        
        if isinstance(index, int) and index > 1_000_000:
            return self.mask_subjects(self.subject_ids == index).apply_mask()        
        if isinstance(index, str):
            return self[int(index)]
        if index.shape[0] == self.n_subjects:
            subject_mask = index 
            return self.mask_subjects(subject_mask).apply_mask()

        else:
            raise ValueError("Qué hacés?")

    def as_dataframe(self):
        import pandas as pd
        
        df = pd.DataFrame(
            np.concatenate([self.X_tokens.flatten()[:, np.newaxis], self.X_ages.flatten()[:, np.newaxis] / 365.25], axis=1),
            columns=["token", "age"]
          ).\
          astype({"token": int}).\
          query("token != 0").\
          assign( **{"age (years)": lambda df: df.age.round(2) }).\
          drop("age", axis=1)
        
        return df.assign(token=lambda df: pd.Categorical(df["token"].map(self.tokenizer)))


@dataclass
class EventSetMask:
    """
    Boolean mask view over an EventSet.
    Can be combined with logical operations (&, |, ~)
    and applied later to produce a filtered EventSet.
    """
    _mask: Optional[np.ndarray] = None
    _op: Optional[Callable] = None
    _parents: Optional[tuple] = None
    _eventset_ref: Optional["EventSet"] = None

    def __post_init__(self):
        if self._eventset_ref is not None and not isinstance(self._eventset_ref, weakref.ReferenceType):
            self._eventset_ref = weakref.ref(self._eventset_ref)

    @property
    def eventset(self):
        if self._eventset_ref is None:
            raise ValueError("No EventSet reference stored.")
        e = self._eventset_ref()
        if e is None:
            raise ReferenceError("Referenced EventSet no longer exists.")
        return e

    # ————————————————————————————————————————————————————————
    # Lazy logical operators
    # ————————————————————————————————————————————————————————
    def __and__(self, other: "EventSetMask"):
        return EventSetMask(_op=np.logical_and, _parents=(self, other), _eventset_ref=self._eventset_ref)

    def __or__(self, other: "EventSetMask"):
        return EventSetMask(_op=np.logical_or, _parents=(self, other), _eventset_ref=self._eventset_ref)

    def __invert__(self):
        return EventSetMask(_op=np.logical_not, _parents=(self,), _eventset_ref=self._eventset_ref)

    # ————————————————————————————————————————————————————————
    # Evaluation
    # ————————————————————————————————————————————————————————
    def evaluate(self) -> np.ndarray:
        """Recursively evaluate the boolean mask as a numpy array."""
        if self._mask is not None:
            return self._mask
        elif self._op is not None and self._parents:
            evaluated = [p.evaluate() for p in self._parents]
            return self._op(*evaluated)
        else:
            raise ValueError("Invalid EventSetMask: no base mask or operation defined.")

    # ————————————————————————————————————————————————————————
    # Application to EventSet
    # ————————————————————————————————————————————————————————
    def apply_mask(self) -> "EventSet":
        """Apply the evaluated mask to its parent EventSet."""
        e = self.eventset
        mask_eval = self.evaluate()

        # Subject-level mask (if token-level, collapse to subject-level)
        subj_mask = mask_eval.any(axis=1) if mask_eval.ndim == 2 else mask_eval

        return EventSet(
            X_tokens=e.X_tokens[subj_mask],
            X_ages=e.X_ages[subj_mask],
            Y_tokens=e.Y_tokens[subj_mask],
            Y_ages=e.Y_ages[subj_mask],
            subject_ids=e.subject_ids[subj_mask] if e.subject_ids is not None else None,
            meta={**e.meta, "mask_applied": True},
            tokenizer=e.tokenizer
        )    
    


class EventSetV2:
    """
    Multi-domain event set.
    Each domain is a tensor of shape [N_events, 3]:
        column 0 → subject_id
        column 1 → age (days)
        column 2 → token_id inside that domain
    """

    def __init__(
        self,
        domains: dict[str, torch.Tensor],
        domain_to_int: dict[str, int] = None,
        meta: dict = None,
    ):
        """
        domains: dict[str, torch.Tensor] 
            Each tensor must be shape [N_events, 3] = [subject_id, age_days, token_id]
    
        domain_to_int: optional mapping {domain_name -> integer_id}.
            If None, a stable mapping is generated automatically using sorted(domains.keys()).
    
        meta: optional metadata dict.
        """
    
        # --- Core structure (your original behavior) ---
        self.domains = {d: arr.clone() for d, arr in domains.items()}
        self.meta = meta or {}
    
        # Validate minimal structure
        for dname, arr in self.domains.items():
            # TODO: full validation of dtype/range if needed
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"Domain '{dname}' must be tensor [N,3], got {arr.shape}")
    
        # --- Establish subject_ids ---
        self._update_subject_ids()   # same behavior as before
    
        # --- Domain ID mapping ---
        if domain_to_int is None:
            # stable deterministic mapping (important for reproducibility)
            domain_names = sorted(self.domains.keys())
            self.domain_to_int = {name: i for i, name in enumerate(domain_names)}
        else:
            self.domain_to_int = dict(domain_to_int)  # copy to avoid side-effects
    
        # inverse mapping
        self.int_to_domain = {i: d for d, i in self.domain_to_int.items()}
    

    # ----------------------------------------------------------------------
    # INTERNAL UTILITIES
    # ----------------------------------------------------------------------

    def _update_subject_ids(self):
        """Compute the full set of subjects across all domains."""
        # TODO: Handle case of empty domains without raising ValueError
        sids = []
        for arr in self.domains.values():
            if len(arr) > 0:
                sids.append(arr[:, 0].cpu().numpy())
        self.subject_ids = np.unique(np.concatenate(sids)) if sids else np.array([])

    def _subset_by_subjects(self, subjects):
        """Return new domains containing only rows whose subject_id is in `subjects`."""
        # TODO: Performance concern: np.isin on large arrays
        subjects = set(int(s) for s in subjects)
        new = {}
        for d, arr in self.domains.items():
            if len(arr) == 0:
                new[d] = arr.clone()
                continue
            mask = np.isin(arr[:, 0].cpu().numpy(), list(subjects))
            # TODO: Could use boolean mask directly on GPU without round-trip to numpy
            new[d] = arr[torch.tensor(mask, device=arr.device)]
        return new

    # ----------------------------------------------------------------------
    # FILTERING BY SUBJECTS
    # ----------------------------------------------------------------------

    def filter_subjects(self, subjects):
        """Keep only these subjects."""
        # TODO: Should we preserve domains exactly or drop empty ones?
        new_domains = self._subset_by_subjects(subjects)
        return EventSetV2(new_domains)

    # ----------------------------------------------------------------------
    # FILTERING BY TOKENS (domain-aware)
    # ----------------------------------------------------------------------

    def for_token(self, domain: str, token_id: int):
        """Keep only subjects that contain this (domain, token_id)."""
        if domain not in self.domains:
            raise ValueError(f"Domain '{domain}' not found.")

        arr = self.domains[domain]
        if len(arr) == 0:
            # TODO: Should empty domain mean empty EventSet or copy of current?
            return EventSetV2({d: a.clone() for d, a in self.domains.items()})

        sids = arr[arr[:, 2] == token_id][:, 0].cpu().numpy()
        # TODO: preserve ordering of subject_ids?
        return self.filter_subjects(sids)

    def exclude_token(self, domain: str, token_id: int):
        """Remove all subjects that contain this (domain, token_id)."""
        if domain not in self.domains:
            raise ValueError(f"Domain '{domain}' not found.")

        arr = self.domains[domain]
        if len(arr) == 0:
            # TODO: same logic as for_token: what do we expect?
            return EventSetV2({d: a.clone() for d, a in self.domains.items()})

        bad = set(arr[arr[:, 2] == token_id][:, 0].cpu().numpy())
        keep = [sid for sid in self.subject_ids if sid not in bad]
        return self.filter_subjects(keep)

    # ----------------------------------------------------------------------
    # DATA MERGING
    # ----------------------------------------------------------------------

    def merge_data(self, as_dataframe=False):
        """
        Merge all domains into a single tensor or pandas DataFrame.
    
        Tensor form (default) returns:
            [subject_id, age_days, token_id, domain_id]
    
        DataFrame form (as_dataframe=True) returns:
            subject_id | age_days | token_id | domain | domain_id
            padded events removed, sorted by (subject_id, age_days)
        """
    
        rows = []
        for domain_name, arr in self.domains.items():
            # arr shape = [N, 3]: sid, age_days, token_id
            t = arr
    
            # Add domain_id column
            dom_id = self.domain_to_int[domain_name]
            dom_col = torch.full((t.shape[0], 1), dom_id, device=t.device, dtype=torch.int64)
    
            rows.append(torch.cat([t, dom_col], dim=1))
    
        merged = torch.cat(rows, dim=0)
    
        # If DataFrame is not requested → return the raw tensor
        if not as_dataframe:
            return merged
    
        # Otherwise build a clean dataframe
        import pandas as pd
    
        df = pd.DataFrame({
            "subject_id": merged[:, 0].cpu().numpy().astype(int),
            "age_days":   merged[:, 1].cpu().numpy().astype(float),
            "token_id":   merged[:, 2].cpu().numpy().astype(int),
            "domain_id":  merged[:, 3].cpu().numpy().astype(int),
        })
    
        # Map domain names
        inv_domain_map = {v: k for k, v in self.domain_to_int.items()}
        df["domain"] = df["domain_id"].map(inv_domain_map)
    
        # Remove padding (token_id == 0 or age_days <= 0)
        df = df[(df.age_days > 0) & (df.token_id > 0)]
    
        # Sort for chronological access
        df = df.sort_values(["subject_id", "age_days"], ignore_index=True)
    
        return df


    # ----------------------------------------------------------------------
    # PIPE (batching!)
    # ----------------------------------------------------------------------

    def pipe(self, func, batch_size=None, device=None, *args, **kwargs):
        """
        Apply func(domains) to either:
            - full set if batch_size=None
            - batches of subjects otherwise
        func receives a dict[str, Tensor] for each batch.
        """
        # TODO: func return type may vary — consider contract
        # TODO: Should func get EventSetV2 object instead of dict?

        if batch_size is None:
            return func(self.domains, *args, **kwargs)

        outputs = []
        sids = self.subject_ids
        n = len(sids)

        for i in range(0, n, batch_size):
            batch_sids = sids[i : i + batch_size]
            sub = EventSetV2(self._subset_by_subjects(batch_sids))

            if device:
                # TODO: verify device exists (cpu/cuda)
                sub_dev = {
                    d: arr.to(device) for d, arr in sub.domains.items()
                }
                out = func(sub_dev, *args, **kwargs)
            else:
                out = func(sub.domains, *args, **kwargs)

            outputs.append(out)

        # Try concat
        if all(isinstance(o, torch.Tensor) for o in outputs):
            try:
                return torch.cat(outputs, dim=0)
            except Exception:
                # TODO: inconsistent shapes → return list?
                return outputs

        if all(isinstance(o, np.ndarray) for o in outputs):
            try:
                return np.concatenate(outputs, axis=0)
            except Exception:
                return outputs

        return outputs

    # ----------------------------------------------------------------------
    # INSERTION (subject-wise)
    # ----------------------------------------------------------------------

    def insert_tokens(self, domain: str, tokens, ages, inplace=False):
        """
        Insert events into one domain. tokens/ages follow the same broadcasting rules
        as your previous implementation.
        """

        if domain not in self.domains:
            raise ValueError(f"Domain '{domain}' not found.")

        # TODO: Could support specifying new domain automatically
        # TODO: Should tokens be validated for integers?

        # Convert to numpy
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        if isinstance(ages, torch.Tensor):
            ages = ages.cpu().numpy()

        # Convert ages to days if needed
        # TODO: better heuristic for days / years
        if np.nanmax(ages) < 300:
            ages_days = np.array(ages) * 365.25
        else:
            ages_days = np.array(ages, dtype=float)

        arr = self.domains[domain].cpu().numpy()
        out_rows = []

        # TODO: performance: loop over subjects instead of events is fine, but for large datasets?
        for sid in self.subject_ids:
            mask = arr[:, 0] == sid
            subj_tok = arr[mask][:, 2]
            subj_age = arr[mask][:, 1]

            tok_i, age_i = self._broadcast_insert(tokens, ages_days, sid)

            # Append + sort
            new_tok = np.concatenate([subj_tok, np.atleast_1d(tok_i)])
            new_age = np.concatenate([subj_age, np.atleast_1d(age_i)])
            # TODO: age ties? deterministic order?
            order = np.argsort(new_age)

            new_tok = new_tok[order]
            new_age = new_age[order]
            new_sid = np.full_like(new_tok, fill_value=sid)

            out_rows.append(
                np.stack([new_sid, new_age, new_tok], axis=1)
            )

        # TODO: dtype inference might be inconsistent if ages float32 vs float64
        new_arr = torch.tensor(np.concatenate(out_rows), dtype=self.domains[domain].dtype)

        if inplace:
            self.domains[domain] = new_arr
            self._update_subject_ids()
            return self

        new = deepcopy(self)
        new.domains[domain] = new_arr
        new._update_subject_ids()
        return new

    def _broadcast_insert(self, tokens, ages, sid):
        """Internal helper replicating previous broadcast rules."""
        # TODO: sid indexing assumes tokens/ages are indexed by subject order
        # TODO: better error messages for shape mismatch

        if np.ndim(tokens) == 0 and np.ndim(ages) == 0:
            return tokens, ages
        if np.ndim(tokens) == 0 and np.ndim(ages) == 1:
            return tokens, ages[sid]  # TODO: sid may not index ages array literally
        if np.ndim(tokens) == 1 and np.ndim(ages) == 1:
            return tokens[sid], ages[sid]
        if np.ndim(tokens) == 2 and np.ndim(ages) == 2:
            return tokens[sid], ages[sid]
        raise ValueError("Incompatible token/age shapes")

    # ----------------------------------------------------------------------
    # UTILS
    # ----------------------------------------------------------------------

    def summary(self):
        print("EventSetV2 Summary:")
        print(f"  Domains: {list(self.domains.keys())}")
        print(f"  Subjects: {len(self.subject_ids)}")
        for d, arr in self.domains.items():
            print(f"    {d}: {len(arr)} events")
            # TODO: maybe show min/max age or token distribution

    def __len__(self):
        return len(self.subject_ids)
    

    def compute_max_ages(self):
        """
        Compute max age per subject across ALL domains.
        Stores result as self.meta['max_age_per_subject'].
        Returns a NEW EventSet (pipeable).
        """
    
        # recolectar todos los ages y subject_ids
        ages_all = []
        sids_all = []
    
        for dname, arr in self.domains.items():
            if arr.numel() == 0:
                continue
            sids_all.append(arr[:, 0].cpu().numpy())
            ages_all.append(arr[:, 1].cpu().numpy())
    
        sids = np.concatenate(sids_all)
        ages = np.concatenate(ages_all)
    
        # max age per subject
        df = pd.DataFrame({"sid": sids, "age": ages})
        max_age = df.groupby("sid")["age"].max().to_dict()
    
        new = deepcopy(self)
        new.meta["max_age_per_subject"] = max_age
        return new
    

    def insert_no_event_tokens(self, rate=5, domain="padding"):
        """
        Insert synthetic 'no-event' tokens up to max_age_per_subject.
        Requires compute_max_ages() to have been called.
        Returns new EventSet.
        """
    
        if "max_age_per_subject" not in self.meta:
            raise ValueError("Call compute_max_ages() before insert_no_event_tokens().")
    
        max_age = self.meta["max_age_per_subject"]
    
        if domain not in self.domains:
            raise ValueError(f"Domain '{domain}' not found.")
    
        arr = self.domains[domain].cpu().numpy()
        out = []
    
        for sid in self.subject_ids:
            m = max_age.get(int(sid), None)
            if m is None or m <= 0:
                continue
    
            # generate evenly spaced ages
            pad_ages = np.arange(0, m, rate * 365.25)
            pad_tokens = np.zeros_like(pad_ages, dtype=int)  # token_id = 0
    
            # existing
            mask = arr[:, 0] == sid
            tok = arr[mask][:, 2]
            age = arr[mask][:, 1]
    
            # merge
            new_tok = np.concatenate([tok, pad_tokens])
            new_age = np.concatenate([age, pad_ages])
            order = np.argsort(new_age)
    
            out.append(
                np.stack([np.full_like(new_tok, sid), new_age[order], new_tok[order]], axis=1)
            )
    
        new = deepcopy(self)
        new.domains[domain] = torch.tensor(np.concatenate(out), dtype=self.domains[domain].dtype)
        return new
    
    def mask_tokens_after_max_age(self):
        """
        Remove events that occur AFTER the max age of each subject.
        Uses meta['max_age_per_subject'] computed earlier.
        """
    
        if "max_age_per_subject" not in self.meta:
            raise ValueError("Call compute_max_ages() before mask_tokens_after_age().")
    
        max_age = self.meta["max_age_per_subject"]
    
        new = deepcopy(self)
    
        for dname, arr in self.domains.items():
            arr_np = arr.cpu().numpy()
            sids = arr_np[:, 0]
            ages = arr_np[:, 1]
    
            keep = np.array([
                ages[i] <= max_age.get(int(sids[i]), -1)
                for i in range(len(arr_np))
            ])
    
            new.domains[dname] = torch.tensor(arr_np[keep], dtype=arr.dtype)
    
        return new


        # ─────────────────────────── Ajuste a seqlen ───────────────────────────

    def adjust_to_seqlen(
        self,
        seqlen: int,
        *,
        pad_domain: str = "padding",
        trim_domains: set[str] = frozenset({"diseases"}),
        PADDING_TOKEN: int = 0,
        PAD_AGE: float = -10000.0,
        mode: str = "fast",   # "fast" o "slow"
    ) -> "EventSet":
        """
        Garantiza que cada sujeto tenga EXACTAMENTE `seqlen` eventos en total,
        contando todos los dominios excepto `pad_domain`.

        - Si tiene más: se recortan los eventos más recientes (edad mayor) solo
          en los dominios listados en `trim_domains`.
        - Si tiene menos: se agregan eventos de padding en `pad_domain`.

        `mode="fast"`: versión vectorizada.
        `mode="slow"`: versión de referencia (más simple de leer / debugear).
        """
        if mode not in {"fast", "slow"}:
            raise ValueError("mode must be 'fast' or 'slow'")

        if mode == "fast":
            new_domains = self._adjust_to_seqlen_fast(
                seqlen,
                pad_domain=pad_domain,
                trim_domains=trim_domains,
                PADDING_TOKEN=PADDING_TOKEN,
                PAD_AGE=PAD_AGE,
            )
        else:
            new_domains = self._adjust_to_seqlen_slow(
                seqlen,
                pad_domain=pad_domain,
                trim_domains=trim_domains,
                PADDING_TOKEN=PADDING_TOKEN,
                PAD_AGE=PAD_AGE,
            )

        new = deepcopy(self)
        new.domains = new_domains
        new._update_subject_ids()
        return new

    # ------------------------------------------------------------------ #
    #   Versión lenta / referencia (loops explícitos por sujeto)
    # ------------------------------------------------------------------ #

    def _adjust_to_seqlen_slow(
        self,
        seqlen: int,
        *,
        pad_domain: str,
        trim_domains: set[str],
        PADDING_TOKEN: int,
        PAD_AGE: float,
    ) -> dict[str, torch.Tensor]:
        """
        Versión 'naive' / de referencia:
        - recorre sujetos en Python,
        - decide cuántos eventos borrar y cuáles (los más recientes),
        - luego padcea igual que la rápida.

        Útil para comparar resultados con la versión rápida.
        """
        from collections import defaultdict

        device = next(iter(self.domains.values())).device
        # dominios que cuentan para el total, excluyendo el de padding
        count_domains = [d for d in self.domains.keys() if d != pad_domain]
        trim_domains = [d for d in trim_domains if d in self.domains]

        # total de eventos por sujeto (antes de truncar)
        all_sids = torch.cat([self.domains[d][:, 0].long() for d in count_domains])
        unique_sids, counts = torch.unique(all_sids, return_counts=True)
        total_per_subject = {int(s.item()): int(c.item()) for s, c in zip(unique_sids, counts)}

        # ── truncar: elegir qué borrar en trim_domains ──
        to_drop_by_domain = defaultdict(list)
        for sid, tot in total_per_subject.items():
            diff = seqlen - tot
            if diff >= 0:
                continue
            R = -diff  # cuántos eventos hay que bajar

            candidates = []
            for d in trim_domains:
                arr = self.domains[d]
                sids_d = arr[:, 0].long()
                ages_d = arr[:, 1].float()
                idx = (sids_d == sid).nonzero(as_tuple=True)[0]
                for j in idx.tolist():
                    candidates.append((float(ages_d[j].item()), d, j))

            if not candidates:
                continue

            # ordenar por edad y sacar los más recientes
            candidates.sort(key=lambda t: t[0])
            drop = candidates[-min(R, len(candidates)):]
            for _, d, j in drop:
                to_drop_by_domain[d].append(j)

        new_domains = dict(self.domains)
        for d, lst in to_drop_by_domain.items():
            if not lst:
                continue
            arr = self.domains[d]
            mask = torch.ones(arr.shape[0], dtype=torch.bool, device=device)
            mask[torch.tensor(sorted(set(lst)), device=device)] = False
            new_domains[d] = arr[mask]

        # ── recomputar totales tras truncar ──
        all_sids2 = torch.cat([new_domains[d][:, 0].long() for d in count_domains])
        unique_sids2, counts2 = torch.unique(all_sids2, return_counts=True)
        total_per_subject2 = {int(s.item()): int(c.item()) for s, c in zip(unique_sids2, counts2)}

        # ── pad ──
        pad_rows = []
        # usamos el primer dominio que cuenta para inferir dtypes
        ref_dom = count_domains[0]
        ref = new_domains[ref_dom]
        sid_dtype = ref[:, 0].dtype
        age_dtype = ref[:, 1].dtype
        tok_dtype = ref[:, 2].dtype

        for sid, tot in total_per_subject2.items():
            diff = seqlen - tot
            if diff <= 0:
                continue
            pad_sid = torch.full((diff,), sid, device=device, dtype=sid_dtype)
            pad_age_col = torch.full((diff,), PAD_AGE, device=device, dtype=age_dtype)
            pad_token_col = torch.full((diff,), PADDING_TOKEN, device=device, dtype=tok_dtype)
            pad_rows.append(torch.stack([pad_sid, pad_age_col, pad_token_col], dim=1))

        if pad_rows:
            pad_rows = torch.cat(pad_rows, dim=0)
            if pad_domain in new_domains:
                new_domains[pad_domain] = torch.cat([new_domains[pad_domain], pad_rows], dim=0)
            else:
                new_domains[pad_domain] = pad_rows

        return new_domains

    # ------------------------------------------------------------------ #
    #   Versión rápida / vectorizada
    # ------------------------------------------------------------------ #

    def _adjust_to_seqlen_fast(
        self,
        seqlen: int,
        *,
        pad_domain: str,
        trim_domains: set[str],
        PADDING_TOKEN: int,
        PAD_AGE: float,
    ) -> dict[str, torch.Tensor]:
        """
        Versión vectorizada:
        - cuenta tokens por sujeto con torch.unique,
        - construye un gran array solo de dominios recortables (trim_domains),
        - ordena por (subject_id, age) y decide en bloque qué eventos borrar,
        - luego padcea en un solo shot usando repeat_interleave.
        """
        device = next(iter(self.domains.values())).device

        # dominios que cuentan para el total, excluyendo padding
        count_domains = [d for d in self.domains.keys() if d != pad_domain]
        trim_domains = [d for d in trim_domains if d in self.domains]

        # ── totales por sujeto (antes de truncar) ──
        all_sids_count = torch.cat([self.domains[d][:, 0].long() for d in count_domains])
        total_sids, total_counts = torch.unique(all_sids_count, return_counts=True)
        total_counts = total_counts.to(torch.int64)

        # ordenamos sujetos para poder usar searchsorted
        total_sids_sorted, perm = torch.sort(total_sids)
        total_counts_sorted = total_counts[perm]

        # ── construir arrays globales para dominios recortables ──
        if trim_domains:
            trim_sids_list = []
            trim_ages_list = []
            trim_dom_ids_list = []
            trim_idx_list = []

            for dom_idx, d in enumerate(trim_domains):
                t = self.domains[d]
                n = t.shape[0]
                trim_sids_list.append(t[:, 0].long())
                trim_ages_list.append(t[:, 1].float())
                trim_dom_ids_list.append(torch.full((n,), dom_idx, device=device, dtype=torch.long))
                trim_idx_list.append(torch.arange(n, device=device, dtype=torch.long))

            trim_sids = torch.cat(trim_sids_list)
            trim_ages = torch.cat(trim_ages_list)
            trim_dom_ids = torch.cat(trim_dom_ids_list)
            trim_idx = torch.cat(trim_idx_list)

            # sort por (subject_id, age) usando numpy.lexsort (más simple)
            key = np.lexsort(
                np.stack([trim_ages.cpu().numpy(), trim_sids.cpu().numpy()])
            )
            key = torch.from_numpy(key).to(device)

            trim_sids_sorted = trim_sids[key]
            trim_ages_sorted = trim_ages[key]          # noqa: F841  # (por si luego lo querés inspeccionar)
            trim_dom_ids_sorted = trim_dom_ids[key]
            trim_idx_sorted = trim_idx[key]

            # sujetos presentes en dominios recortables + cuántos eventos tiene cada uno
            trim_sids_unique, trim_counts = torch.unique_consecutive(
                trim_sids_sorted, return_counts=True
            )
            num_trim_subj = trim_sids_unique.shape[0]

            # alinear con total_sids_sorted para saber total de eventos por sujeto
            pos = torch.searchsorted(total_sids_sorted, trim_sids_unique)
            tot_for_trim = total_counts_sorted[pos]           # total (todos los dominios) de ese sujeto
            diff = tot_for_trim.to(torch.int64) - seqlen      # >0 => hay que borrar
            drop_needed = torch.clamp(diff, min=0)

            # offsets por sujeto dentro del array global ordenado
            offsets = torch.cumsum(trim_counts, dim=0) - trim_counts
            subj_idx_per_event = torch.repeat_interleave(
                torch.arange(num_trim_subj, device=device), trim_counts
            )
            offsets_per_event = offsets[subj_idx_per_event]
            within_group = torch.arange(trim_sids_sorted.shape[0], device=device) - offsets_per_event

            drop_per_subj = torch.minimum(drop_needed, trim_counts)
            thresholds = (trim_counts - drop_per_subj)[subj_idx_per_event]

            # True => este evento se borra
            drop_mask_sorted = within_group >= thresholds

            # mapear de vuelta a cada dominio
            new_domains = dict(self.domains)
            for di, d in enumerate(trim_domains):
                t = self.domains[d]
                mask_keep = torch.ones(t.shape[0], dtype=torch.bool, device=device)

                sel = trim_dom_ids_sorted == di          # eventos en este dominio
                idx_global = trim_idx_sorted[sel]        # índice local dentro de t
                drop_local = drop_mask_sorted[sel]

                mask_keep[idx_global[drop_local]] = False
                new_domains[d] = t[mask_keep]
        else:
            # nada que recortar, solo paddear
            new_domains = dict(self.domains)

        # ── totales tras truncar ──
        all_sids_after = torch.cat([new_domains[d][:, 0].long() for d in count_domains])
        final_sids, final_counts = torch.unique(all_sids_after, return_counts=True)
        final_sids_sorted, perm2 = torch.sort(final_sids)
        final_counts_sorted = final_counts[perm2].to(torch.int64)

        pad_needed = seqlen - final_counts_sorted
        pad_needed = torch.clamp(pad_needed, min=0)

        # ── construir filas de padding ──
        if pad_needed.sum() > 0:
            subj_ids_to_pad = final_sids_sorted[pad_needed > 0]
            counts_to_pad = pad_needed[pad_needed > 0]

            pad_sid = torch.repeat_interleave(subj_ids_to_pad, counts_to_pad).to(device)

            # dtypes de referencia
            ref_dom = count_domains[0]
            ref = new_domains[ref_dom]
            sid_dtype = ref[:, 0].dtype
            age_dtype = ref[:, 1].dtype
            tok_dtype = ref[:, 2].dtype

            pad_sid_col = pad_sid.to(dtype=sid_dtype)
            pad_age_col = torch.full(
                (pad_sid.shape[0],),
                PAD_AGE,
                device=device,
                dtype=age_dtype,
            )
            pad_token_col = torch.full(
                (pad_sid.shape[0],),
                PADDING_TOKEN,
                device=device,
                dtype=tok_dtype,
            )

            pad_rows = torch.stack([pad_sid_col, pad_age_col, pad_token_col], dim=1)

            if pad_domain in new_domains:
                new_domains[pad_domain] = torch.cat([new_domains[pad_domain], pad_rows], dim=0)
            else:
                new_domains[pad_domain] = pad_rows

        return new_domains
    

    def to_model_inputs(self, device=None):

        """
        Return (tokens, ages, subject_ids) dictionaries compatible with Delphi.forward().
        Shapes: each is a dict[domain] -> 1D tensor of length (#events in that domain).
        """
    
        tokens = {}
        ages = {}
        subject_ids = {}
    
        for dname, arr in self.domains.items():
            # arr has shape (N_d, 3): [subject_id, age, token_id]
            if arr.numel() == 0:
                # empty domain → empty tensors matching expected dtype
                tokens[dname]      = torch.empty(0, dtype=torch.long)
                ages[dname]        = torch.empty(0, dtype=torch.float)
                subject_ids[dname] = torch.empty(0, dtype=torch.long)
                continue
    
            tokens[dname]      = arr[:, 2].long()
            ages[dname]        = arr[:, 1].float()
            subject_ids[dname] = arr[:, 0].long()
    
            if device is not None:
                tokens[dname]      = tokens[dname].to(device)
                ages[dname]        = ages[dname].to(device)
                subject_ids[dname] = subject_ids[dname].to(device)
    
        return tokens, ages, subject_ids
    
    def get_case_control_masks(
        self,
        disease_domain: str,
        disease_label: str,     # por ejemplo "I10"
        min_age: float,
        max_age: float,
        sex_domain: str = "sex",
        sex_label: str = None,
    ):
        """
        Retorna máscaras de CASOS y CONTROLES para un dominio de enfermedades.
        
        CASO  = tokens cuyo siguiente token es la enfermedad de interés,
                en el rango etario, y del sexo solicitado.
        
        CONTROL = sujetos que JAMÁS tuvieron la enfermedad,
                  con tokens en el rango etario y del sexo solicitado.
    
        Retorna:
          mask_cases      : boolean mask (N_events,)
          mask_controls   : boolean mask (N_events,)
          df              : dataframe mergeado usado para las máscaras
        """
    
        # ——————————————————————————————————————
        # 1. Convertir dataset a DF (solo 1 vez)
        # ——————————————————————————————————————
        df = self.merge_data()  # columnas: subject_id, age_days, token_id, domain_id, domain
    
        # ——————————————————————————————————————
        # 2. Obtener tokenizer del dominio de enfermedad
        # ——————————————————————————————————————
        if not hasattr(self, "tokenizer"):
            raise ValueError("EventSetV2 necesita self.tokenizer[domain] para mapear token labels.")
    
        tokmap = self.tokenizer[disease_domain]     # ej: {0:'I10', 1:'E11', ...}
        inv_tokmap = {v: k for k, v in tokmap.items()}
    
        if disease_label not in inv_tokmap:
            raise ValueError(f"La enfermedad '{disease_label}' no existe en tokenizer[{disease_domain}]")
    
        disease_token_id = inv_tokmap[disease_label]
    
        # ——————————————————————————————————————
        # 3. Mascara base por dominio y edad
        # ——————————————————————————————————————
        mask_domain = (df["domain"] == disease_domain)
        mask_age    = df["age_days"].between(min_age * 365.25, max_age * 365.25)
    
        # ——————————————————————————————————————
        # 4. Filtrar sexo si corresponde
        # ——————————————————————————————————————
        if sex_label is not None:
            sex_tokmap = self.tokenizer[sex_domain]
            inv_sex = {v: k for k, v in sex_tokmap.items()}
    
            if sex_label not in inv_sex:
                raise ValueError(f"Sexo '{sex_label}' no válido en tokenizer[{sex_domain}]")
    
            sex_token_id = inv_sex[sex_label]
    
            # sujetos cuyo token de sexo coincide
            df_sex = df[df["domain"] == sex_domain]
            valid_subjects = df_sex[df_sex["token_id"] == sex_token_id]["subject_id"].unique()
    
            mask_sex = df["subject_id"].isin(valid_subjects)
        else:
            mask_sex = True
    
        # ——————————————————————————————————————
        # 5. CASOS: evento cuyo siguiente token == enfermedad
        # ——————————————————————————————————————
        df_sorted = df.sort_values(["subject_id", "age_days"]).reset_index(drop=True)
    
        # token siguiente dentro del mismo sujeto
        df_sorted["next_token_id"] = df_sorted.groupby("subject_id")["token_id"].shift(-1)
        df_sorted["next_domain"]   = df_sorted.groupby("subject_id")["domain"].shift(-1)
    
        mask_case_core = (
            (df_sorted["next_token_id"] == disease_token_id) &
            (df_sorted["next_domain"]   == disease_domain)
        )
    
        mask_cases = (
            mask_case_core &
            df_sorted["age_days"].between(min_age * 365.25, max_age * 365.25) &
            (df_sorted["subject_id"].isin(valid_subjects) if sex_label else True)
        )
    
        # ——————————————————————————————————————
        # 6. CONTROLES: sujetos sin la enfermedad + tokens válidos de ese rango
        # ——————————————————————————————————————
        # sujetos enfermos
        sick_subjects = df_sorted[
            (df_sorted["domain"] == disease_domain) &
            (df_sorted["token_id"] == disease_token_id)
        ]["subject_id"].unique()
    
        # sujetos válidos por sexo
        if sex_label:
            cand_subjects = valid_subjects
        else:
            cand_subjects = df_sorted["subject_id"].unique()
    
        # sujetos que jamás tuvieron esa enfermedad
        control_subjects = np.setdiff1d(cand_subjects, sick_subjects)
    
        mask_controls = (
            df_sorted["subject_id"].isin(control_subjects) &
            df_sorted["age_days"].between(min_age * 365.25, max_age * 365.25)
        )
    
        return mask_cases.to_numpy(), mask_controls.to_numpy(), df_sorted


    def case_control_masks(
        es, 
        disease_id: int,
        sex_token_id: int,
        age_min: float,
        age_max: float,
        disease_domain="diseases",
    ):
        """
        Returns:
          case_mask:     boolean mask over merged dataframe rows  
          control_mask:  boolean mask over merged dataframe rows
          df: merged dataframe (cached)
        """
    
        df = es.merge_data(as_dataframe=True)
    
        # ─────────────────────────────────────
        # SEX FILTER
        # ─────────────────────────────────────
        # sujetos cuyo sexo == sex_token_id
        # asumo que el dominio "sex" tiene UN solo token por sujeto en df
        subject_sex = (
            df[df.domain == "sex"]
            .groupby("subject_id")["token_id"]
            .first()
        )
    
        allowed_subjects = subject_sex[subject_sex == sex_token_id].index
    
        df = df[df.subject_id.isin(allowed_subjects)]
    
        # ─────────────────────────────────────
        # AGE FILTER
        # ─────────────────────────────────────
        df_in_age = df[
            (df.age_days >= age_min * 365.25) &
            (df.age_days <  age_max * 365.25)
        ]
    
        # ─────────────────────────────────────
        # FIND FIRST DIAGNOSIS PER SUBJECT
        # ─────────────────────────────────────
        disease_df = df[df.domain == disease_domain]
    
        # primer diagnóstico por sujeto
        first_diag = (
            disease_df[disease_df.token_id == disease_id]
            .groupby("subject_id")["age_days"]
            .min()
        )
    
        subjects_with_disease = set(first_diag.index)
        subjects_without_disease = set(df.subject_id.unique()) - subjects_with_disease
    
        # ─────────────────────────────────────
        # CASE MASK
        # ─────────────────────────────────────
        # evento inmediatamente anterior al diagnóstico
        def is_case_row(row):
            sid = row.subject_id
            if sid not in subjects_with_disease:
                return False
            diag_age = first_diag[sid]
            return (
                row.domain == disease_domain and
                row.age_days < diag_age and
                (diag_age - row.age_days) == (row.age_next_gap if "age_next_gap" in df.columns else min(df.age_days[df.subject_id==sid] - row.age_days[df.domain==disease_domain and df.age_days > row.age_days]))
            )
    
        # Mejor versión: calculado vectorizado
        df_d = df[df.domain == disease_domain].copy()
    
        # para cada sujeto: ordeno por edad y encuentro el anterior al diagnóstico
        df_d["is_case"] = False
        for sid, g in df_d.groupby("subject_id"):
            if sid not in subjects_with_disease:
                continue
            diag_age = first_diag[sid]
            # tokens previos al diagnóstico
            prev = g[g.age_days < diag_age]
            if len(prev) == 0:
                continue
            # el último antes del diag
            idx = prev.age_days.idxmax()
            df_d.loc[idx, "is_case"] = True
    
        case_mask = df_in_age.index.isin(df_d[df_d.is_case].index)
    
        # ─────────────────────────────────────
        # CONTROL MASK  — one token per subject
        # ─────────────────────────────────────
        df_ctrl = df_in_age[df_in_age.subject_id.isin(subjects_without_disease)]
        df_ctrl = df_ctrl[df_ctrl.domain == disease_domain]
    
        # sample one per subject
        ctrl_idx = df_ctrl.groupby("subject_id").sample(n=1, random_state=1337).index
        control_mask = df_in_age.index.isin(ctrl_idx)
    
        return case_mask, control_mask, df_in_age