from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable
import numpy as np
import torch
from dataclasses import dataclass
import numpy as np
import weakref


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