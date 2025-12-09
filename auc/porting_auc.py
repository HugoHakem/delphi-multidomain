# %%
import os, sys
DELPHI_DIR = f"{os.getenv('HOME')}/repos/delphi"
os.chdir(DELPHI_DIR)
if DELPHI_DIR not in sys.path:
    sys.path.insert(0, DELPHI_DIR)

import scipy.stats
import scipy
import torch

from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

from old_model import (
    Delphi as OldDelphi,
    DelphiConfig as OldDelphiConfig,
)

from utils.utils import get_batch, get_p2i
from utils.cv_utils import ( get_best_ckpt_from_mlflow )
from itertools import pairwise
import data.event_set
import importlib

import ast
from data.dataset import DelphiDataset, DelphiDataloader
from utils.cv_utils import get_data_partitions
from delphi.model.transformer import (
    Delphi,
    EmbedConfig,
    DelphiConfig,
)
import importlib 
import delphi
importlib.reload(delphi)
Delphi = delphi.model.transformer.Delphi
from easydict import EasyDict

# Nothing in this script requires gradients
torch.set_grad_enabled(False)

device = 'cpu'

def get_common_diseases(delphi_labels, filter_min_total=100):
    chapters_of_interest = [
        "I. Infectious Diseases",
        "II. Neoplasms",
        "III. Blood & Immune Disorders",
        "IV. Metabolic Diseases",
        "V. Mental Disorders",
        "VI. Nervous System Diseases",
        "VII. Eye Diseases",
        "VIII. Ear Diseases",
        "IX. Circulatory Diseases",
        "X. Respiratory Diseases",
        "XI. Digestive Diseases",
        "XII. Skin Diseases",
        "XIII. Musculoskeletal Diseases",
        "XIV. Genitourinary Diseases",
        "XV. Pregnancy & Childbirth",
        "XVI. Perinatal Conditions",
        "XVII. Congenital Abnormalities",
        "Death",
    ]
    labels_df = delphi_labels[
        delphi_labels["ICD-10 Chapter (short)"].isin(chapters_of_interest) * (delphi_labels["count"] > filter_min_total)
    ]
    return labels_df["index"].tolist()


def optimized_bootstrapped_auc_gpu(case, control, n_bootstrap=1):
    """
    Computes bootstrapped AUC estimates using PyTorch on CUDA.

    Parameters:
        case: 1D tensor of scores for positive cases
        control: 1D tensor of scores for controls
        n_bootstrap: Number of bootstrap replicates

    Returns:
        Tensor of shape (n_bootstrap,) containing AUC for each bootstrap replicate
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a GPU.")

    # Convert inputs to CUDA tensors
    if not torch.is_tensor(case):
        case = torch.tensor(case, device="cuda", dtype=torch.float32)
    else:
        case = case.to("cuda", dtype=torch.float32)

    if not torch.is_tensor(control):
        control = torch.tensor(control, device="cuda", dtype=torch.float32)
    else:
        control = control.to("cuda", dtype=torch.float32)

    n_case = case.size(0)
    n_control = control.size(0)
    total = n_case + n_control

    # Generate bootstrap samples
    boot_idx_case = torch.randint(0, n_case, (n_bootstrap, n_case), device="cuda")
    boot_idx_control = torch.randint(0, n_control, (n_bootstrap, n_control), device="cuda")

    boot_case = case[boot_idx_case]
    boot_control = control[boot_idx_control]

    combined = torch.cat([boot_case, boot_control], dim=1)

    # Mask to identify case entries
    mask = torch.zeros((n_bootstrap, total), dtype=torch.bool, device="cuda")
    mask[:, :n_case] = True

    # Compute ranks and AUC
    ranks = combined.argsort(dim=1).argsort(dim=1)
    case_ranks_sum = torch.sum(ranks.float() * mask.float(), dim=1)
    min_case_rank_sum = n_case * (n_case - 1) / 2.0
    U = case_ranks_sum - min_case_rank_sum
    aucs = U / (n_case * n_control)
    return aucs.cpu().tolist()


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float32)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float32)
    ty = np.empty([k, n], dtype=np.float32)
    tz = np.empty([k, m + n], dtype=np.float32)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def get_auc_delong_var(healthy_scores, diseased_scores):
    """
    Computes ROC AUC value and variance using DeLong's method

    Args:
        healthy_scores: Values for class 0 (healthy/controls)
        diseased_scores: Values for class 1 (diseased/cases)
    Returns:
        AUC value and variance
    """
    # Create ground truth labels (1 for diseased, 0 for healthy)
    ground_truth = np.array([1] * len(diseased_scores) + [0] * len(healthy_scores))
    predictions = np.concatenate([diseased_scores, healthy_scores])

    # Compute statistics needed for DeLong method
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]

    # Calculate AUC and covariance
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"

    return aucs[0], delongcov


def get_calibration_auc(j, k, d, p, offset=365.25, age_groups=range(10, 80, 5), precomputed_idx=None, n_bootstrap=1, use_delong=False):
    
    # Nomenclature: w->where, k->cases, c->controls

    X_tokens, X_ages, Y_tokens, Y_ages = d 
    
    # Indexes of cases with disease k
    wk = np.where(Y_tokens == k)

    n_cases = len(wk[0])
    if n_cases < 2:
        return None

    # For control tokens, we need to exclude subjects disease k
    # TODO: check if the two conditions are actually needed or the second is enough
    wc = np.where((Y_tokens != k) * (~(Y_tokens == k).any(-1))[..., None])

    wall = (np.concatenate([wk[0], wc[0]]), np.concatenate([wk[1], wc[1]]))  # All cases and controls

    # Use the tokens for prediction that are at least "offset" days before the event
    if precomputed_idx is None:
        pred_idx = (X_ages[wall[0]] <= Y_ages[wall].reshape(-1, 1) - offset).sum(1) - 1 
    else:
        # pred_idx: [n_subjects x block_size]
        pred_idx = precomputed_idx[wall]  # It's actually much faster to precompute this

    allowed = pred_idx != (NOT_ALLOWED := 1)
    z  = X_ages[(wall[0], pred_idx)][allowed]  # Times of the tokens for prediction
    zk = Y_ages[wall][allowed]                 # Target times

    # x = np.exp(p[..., j][(wall[0], pred_idx)]) * 365.25
    # x = 1 - np.exp(-x * age_step)  # the function is monotinic, so we don't need to do this for the AUC
    x = p[..., j][(wall[0], pred_idx)][allowed]

    wk = (wk[0][pred_idx[:n_cases] != NOT_ALLOWED], wk[1][pred_idx[:n_cases] != NOT_ALLOWED])
    p_idx = wall[0][allowed] 

    # z, z_k, p_idx

    out = []
    from itertools import pairwise

    for i, (start, end) in enumerate(pairwise(age_groups)):
    
        z_years = z / 365.25
        a = np.logical_and(z_years >= start, z_years < end)
    
        # Optionally, add extra filtering on the time difference, for example:
        # a *= (zk - z < 365.25)
   
        # We sample one token per subject (in the right age interval) 
        selected_groups = p_idx[a]
        perm = np.random.permutation(len(selected_groups))
        _, indices = np.unique(selected_groups[perm], return_index=True)
        indices = perm[indices]

        selected = np.zeros(np.sum(a), dtype=bool)
        selected[indices] = True

        # a mask that filters for the right age range, while sampling only one token per subject
        a[a] = selected 

        case    = x[:n_cases][a[:n_cases]]
        control = x[n_cases:][a[n_cases:]] 

        if len(control) == 0 or len(case) == 0:
            continue

        # ———————— DeLong ————————————————————————————————————————————————————————————————————————————————
        if use_delong:
            auc_value_delong, auc_variance_delong = get_auc_delong_var(control, case)
            auc_delong_dict = {"auc_delong": auc_value_delong, "auc_variance_delong": auc_variance_delong}
        else:
            auc_delong_dict = {}

        # ———————— Bootstrapping —————————————————————————————————————————————————————————————————————————
        if n_bootstrap > 1:
            aucs_bootstrapped = optimized_bootstrapped_auc_gpu(case, control, n_bootstrap)

        for bootstrap_idx in range(n_bootstrap):
            y = auc_value_delong if n_bootstrap == 1 else aucs_bootstrapped[bootstrap_idx]
            out_item = { "token": k, "auc": y, "age": f"{start}-{end}", "n_healthy": len(control), "n_diseased": len(case) }
            out.append(out_item | auc_delong_dict)
            if n_bootstrap > 1:
                out_item["bootstrap_idx"] = bootstrap_idx
        # ————————————————————————————————————————————————————————————————————————————————————————————————

    return out 


def evaluate_auc_pipeline(
    model,
    d100k,
    output_path,
    delphi_labels,
    diseases_of_interest=None,
    filter_min_total=10,
    disease_chunk_size=200,
    age_groups=np.arange(10, 80, 5),
    offset=0.1,
    batch_size=2048,
    device="cpu",
    seed=1337,
    n_bootstrap=1,
    meta_info={},
):
    """
    Runs the AUC evaluation pipeline.

    Args:
        model (torch.nn.Module): The loaded model set to eval().
        d100k (tuple): Data batch from get_batch, i.e. (input_tokens, input_ages, target_tokens, target_ages,)
        delphi_labels (pd.DataFrame): DataFrame with label info (token names, etc. "delphi_labels_chapters_colours_icd.csv").
        output_path (str | None): Directory where CSV files will be written. If None, files will not be saved.
        diseases_of_interest (np.ndarray or list, optional): If provided, these disease indices are used.
        filter_min_total (int): Minimum total token count to include a token.
        disease_chunk_size (int): Maximum chunk size for processing diseases.
        age_groups (np.ndarray): Age groups to use in calibration.
        offset (float): Offset used in get_calibration_auc.
        batch_size (int): Batch size for model forwarding.
        device (str): Device identifier.
        seed (int): Random seed for reproducibility.
        n_bootstrap (int): Number of bootstrap samples. (1 for no bootstrap)
    Returns:
        tuple: (df_auc_unpooled, df_auc, df_both) DataFrames.
    """

    assert n_bootstrap > 0, "n_bootstrap must be greater than 0"

    X_tokens, X_ages, Y_tokens, Y_ages = d100k

    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model.to(device)

    from functools import partial
    _get_calibration_auc = partial(get_calibration_auc, use_delong=True, n_bootstrap=n_bootstrap, age_groups=age_groups)

    def get_disease_chunks(diseases, chunk_size):
        # Split diseases into chunks for processing
        num_chunks = (len(diseases_of_interest) + disease_chunk_size - 1) // disease_chunk_size
        diseases_chunks = np.array_split(diseases_of_interest, num_chunks)
        return diseases_chunks

    # Get common diseases
    diseases_of_interest = diseases_of_interest or get_common_diseases(delphi_labels, filter_min_total)
    diseases_chunks = get_disease_chunks(diseases_of_interest, disease_chunk_size)

    # Precompute prediction indices for calibration
    pred_idx_precompute = (X_ages[:, :, np.newaxis] < Y_ages[:, np.newaxis, :] - offset).sum(1) - 1

    all_aucs = []
    disease_pbar_cfg = {"desc": "Processing disease chunks", "total": len(diseases_chunks)}

    print(f"{len(diseases_chunks)=}")
    for disease_chunk_idx, diseases_chunk in tqdm(enumerate(diseases_chunks), **disease_pbar_cfg):

        # Process the evaluation data in batches
        disease_chunk_pbar_cfg = { "desc": f"Model inference, chunk {disease_chunk_idx}", "total": d100k[0].shape[0] // batch_size + 1 }

        p100k = []
        for dd in tqdm(zip(*[torch.split(x, batch_size) for x in d100k]), **disease_chunk_pbar_cfg):
            dd = [x.to(device) for x in dd]
            outputs = model(*dd)[0].cpu().detach().numpy()
            # Keep only the columns corresponding to the current disease chunk
            p100k.append(outputs[:, :, diseases_chunk].astype("float16"))  # enough to store logits, but not rates
        p100k = np.vstack(p100k)

        # Loop over each disease (token) in the current chunk, sexes separately
        for sex, sex_idx in [("female", 2), ("male", 3)]:
            sex_mask = ((d100k[0] == sex_idx).sum(1) > 0).cpu().detach().numpy()
            p_sex = p100k[sex_mask]
            d100k_sex = [d_[sex_mask].cpu().detach().numpy() for d_ in d100k]
            precomputed_idx_subset = pred_idx_precompute[sex_mask].cpu().detach().numpy()
            
            for j, k in tqdm( list(enumerate(diseases_chunk)), desc=f"Processing diseases in chunk {disease_chunk_idx}, {sex}"):
                # Get calibration AUC for the current disease token.
                out = _get_calibration_auc(j, k, d100k_sex, p_sex, precomputed_idx=precomputed_idx_subset)
                if out is None:
                    print(f"No data for disease {k} and sex {sex}")
                    continue
                for out_item in out:
                    out_item["sex"] = sex
                    all_aucs.append(out_item)

    df_auc_unpooled = pd.DataFrame(all_aucs)

    for key, value in meta_info.items():
        df_auc_unpooled[key] = value

    delphi_labels_subset = delphi_labels[['index', 'ICD-10 Chapter (short)', 'name', 'color', 'count']]
    df_auc_unpooled_merged = df_auc_unpooled.merge(delphi_labels_subset, left_on="token", right_on="index", how="inner")

    def aggregate_age_brackets_delong(group):
        # For normal distributions, when averaging n of them:
        # The variance of the sum is the sum of variances
        # The variance of the average is the sum of variances divided by n^2
        n = len(group)
        mean = group['auc_delong'].mean()
        # Since we're taking the average, divide combined variance by n^2
        var = group['auc_variance_delong'].sum() / (n**2)
        return pd.Series({
            'auc': mean,
            'auc_variance_delong': var,
            'n_samples': n, 
            'n_diseased': group['n_diseased'].sum(),
            'n_healthy': group['n_healthy'].sum(),
        })

    print('Using DeLong method to calculate AUC confidence intervals..')
    
    df_auc = df_auc_unpooled.groupby(["token"]).apply(aggregate_age_brackets_delong).reset_index()
    df_auc_merged = df_auc.merge(delphi_labels, left_on="token", right_on="index", how="inner")
    
    if output_path is not None:
        Path(output_path).mkdir(exist_ok=True, parents=True)
        print(f"Created this folder to store the parquet file: {output_path}")
        df_auc_merged.to_parquet(f"{output_path}/df_both_onlywhite.parquet", index=False)
        df_auc_unpooled_merged.to_parquet(f"{output_path}/df_auc_unpooled_onlywhite.parquet", index=False)

    return df_auc_unpooled_merged, df_auc_merged


@dataclass
class EventSet:
    
    """
    Represents a collection of temporal token sequences, one per subject.
    Each subject has a stream of (token, age) pairs, optionally with other metadata.
    This class provides high-level, semantic operations on those sequences.
    """

    X_tokens: np.ndarray           # shape: [n_subjects, seq_len]
    X_ages:   np.ndarray           # shape: [n_subjects, seq_len]
    Y_tokens: np.ndarray           # shape: [n_subjects, seq_len]
    Y_ages:   np.ndarray           # shape: [n_subjects, seq_len]

    meta: Dict[str, Any] = field(default_factory=dict)

    subject_ids:  Optional[np.ndarray] = None  # [n_subjects]

    # Optional metadata
    # domain_info:  Optional[Dict[str, Any]] = None            # e.g. model.transformer.embed.domain_embed
    # token_maps:   Optional[Dict[str, Dict[int, str]]] = None  # {domain: {id: name}}
    # model_config: Optional[Any] = None                      # could be DelphiConfig

    cache: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_batch(cls, batch_tuple, subject_ids=None):
        """
        Build an EventSet from a 4-tuple of arrays/tensors:
        (X_tokens, X_ages, Y_tokens, Y_ages).
        """
        X_tokens, X_ages, Y_tokens, Y_ages = batch_tuple
        if isinstance(X_tokens, torch.Tensor):
            X_tokens, X_ages, Y_tokens, Y_ages = [
                x.detach().cpu().numpy() for x in batch_tuple
            ]
        return cls(X_tokens, X_ages, Y_tokens, Y_ages, subject_ids)

    
    # --------------------------------------------------------------------------------
    # Core utilities
    # --------------------------------------------------------------------------------
    def for_disease(self, disease_id: int):
      """
      Return a new EventSet view focused on a specific disease ID.
      Keeps only subjects that have at least one token equal to disease_id
      in Y_tokens (i.e. they experienced this disease).
      """
      # Boolean mask of shape [n_subjects]
      has_disease = (self.Y_tokens == disease_id).any(axis=1)

      # Filter all arrays by subject
      X_tokens_f = self.X_tokens[has_disease]
      X_ages_f   = self.X_ages[has_disease]
      Y_tokens_f = self.Y_tokens[has_disease]
      Y_ages_f   = self.Y_ages[has_disease]
      
      if self.subject_ids is not None:
          subj_ids_f = self.subject_ids[has_disease]
      else:
          subj_ids_f = None
 
      # Shallow copy of metadata
      meta = dict(self.meta)
      meta["focus_disease"] = disease_id
  
      # Return a new view
      return EventSet(
          X_tokens_f,
          X_ages_f,
          Y_tokens_f,
          Y_ages_f,
          subject_ids=subj_ids_f,
          meta=meta
      )

    
    def filter_by_age(self, min_age: float, max_age: float):
        """
        Return a new EventSet view including only tokens within [min_age, max_age) years.
        Tokens outside this range are masked out (set to 0).
        """
        # Convert from days to years if necessary
        z_years = self.X_ages / 365.25
    
        # Boolean mask of valid ages
        age_mask = (z_years >= min_age) & (z_years < max_age)
    
        # Apply the same mask to all arrays
        X_tokens_f = np.where(age_mask, self.X_tokens, 0)
        X_ages_f   = np.where(age_mask, self.X_ages, 0)
        Y_tokens_f = np.where(age_mask, self.Y_tokens, 0)
        Y_ages_f   = np.where(age_mask, self.Y_ages, 0)
    
        meta = dict(self.meta)
        meta["age_range"] = (min_age, max_age)
    
        return EventSet(
            X_tokens_f,
            X_ages_f,
            Y_tokens_f,
            Y_ages_f,
            subject_ids=self.subject_ids,
            meta=self.meta | {"age_range": (min_age, max_age)},
        )

    def exclude_disease(self, disease_id: int):
        """
        Return a new EventSet view excluding all subjects that ever had the given disease.
        This is typically used to select control subjects for a case-control comparison.
        """
        # Find subjects that never had this disease
        never_had = ~(self.Y_tokens == disease_id).any(axis=1)
    
        # Filter arrays by subject
        X_tokens_f = self.X_tokens[never_had]
        X_ages_f   = self.X_ages[never_had]
        Y_tokens_f = self.Y_tokens[never_had]
        Y_ages_f   = self.Y_ages[never_had]
        # subj_ids_f = self.subject_ids[never_had]
    
        meta = dict(self.meta)
        meta["excluded_disease"] = disease_id
    
        return EventSet(
            X_tokens_f,
            X_ages_f,
            Y_tokens_f,
            Y_ages_f,
            # subj_ids_f,
            meta=meta
        )

    def get_prediction_context(self, offset: float):
        """
        For each target token, find the last input token occurring at least
        `offset` days (or years) in the past.
        Returns an array of indices with shape [n_subjects, seq_len].
        """
        raise NotImplementedError

    
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


    
    def case_mask(self, disease_id: int):
        """
        Return a boolean mask selecting the disease event token (case) for each subject.
    
        Args:
            disease_id (int): Disease ID of interest.
    
        Returns:
            np.ndarray: Boolean mask [n_subjects, seq_len] where True marks the disease event.
        """
        # One-hot where disease appears in the output tokens
        mask = (self.Y_tokens == disease_id)
    
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
  

# %%
MLFLOW_URI = f"{os.getenv('HOME')}/repos/delphi/output/mlruns"
mlflow.set_tracking_uri(MLFLOW_URI)



# Choosed run
runid = "f945b0c1f1b84294bf29a6d39e6ef831"

# Load model
ckpt_path = get_best_ckpt_from_mlflow(runid)
checkpoint = torch.load(ckpt_path, map_location=device)
# checkpoint['model_args'].pop("vocab_size")
checkpoint['model_args'].pop("t_min")
checkpoint['model_args'].pop("mask_ties")
checkpoint['model_args'].pop("ignore_tokens")

conf = OldDelphiConfig(**checkpoint["model_args"])

model = OldDelphi(conf)
state_dict = checkpoint["model"]
state_dict = { k.replace("_orig_mod.", ""): v for k, v in state_dict.items() }
model.load_state_dict(state_dict)
model.eval()
model = model.to(device)

# %%

input_path = f"{os.environ['HOME']}/repos/delphi/data/transforms/deprecated/ukb_real_data/"
val = np.fromfile(f"{input_path}/ukb_real_hla4d_val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
val_p2i = get_p2i(val)

dataset_subset_size = len(val_p2i)

d100k = get_batch(
    range(dataset_subset_size),
    val,
    val_p2i,
    select="left",
    block_size=128,
    device=device,
    padding="random",
    no_event_token_rate=5,
    # health_token_replacement_prob=health_token_replacement_prob,
)

delphi_labels = "./data/transforms/deprecated/ukb_real_data_4digit/labels.csv"
delphi_labels = "./data/delphi_labels_chapters_colours_icd_with_hla4d.csv"
delphi_labels = pd.read_csv(delphi_labels)

# %% 

event_set = EventSet(*d100k, meta={"block_size": 128, "device": device})

TOKEN_ID = 1628
age_range = (60, 85)

cases_mask    = event_set.\
    filter_by_age(*age_range).\
    for_disease(TOKEN_ID).\
    case_mask(TOKEN_ID)

controls_mask = event_set.\
    exclude_disease(TOKEN_ID).\
    filter_by_age(*age_range).\
    sample_one_token_per_subject(return_type="mask")

event_set.\
    filter_by_age(*age_range).\
        sample_one_token_per_subject(return_type="mask", allowed_token_ids=range(1628-1256, 1628))

event_set.sample_one_token_per_subject(return_type="indices")
event_set.sample_one_token_per_subject(return_type="values")

'''
df_auc_unpooled, df_auc_merged = evaluate_auc_pipeline(
    model,
    d100k,
    "auc_test",
    delphi_labels,
    diseases_of_interest=get_common_diseases(delphi_labels, 10000),
    # filter_min_total=1000,
    disease_chunk_size=10000,
    device=device,
    seed=42,
    n_bootstrap=1,
)


df_auc_unpooled.to_csv("auc_unpooled.csv")
df_auc_merged.to_csv("auc_merged.csv")
'''

root_path = Path("./data/transforms")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

tokens_path = root_path / 'tokens'

from data.dataset import DelphiDataset, DelphiDataloader
from utils.cv_utils import get_data_partitions
from delphi.model.transformer import (
    # Delphi,
    # EmbedConfig,
    DelphiConfig,
)

import re
from pathlib import Path

train_ids, val_ids, test_ids = get_data_partitions("./data/transforms/subject_lists", fold=0)

default_cfg_per_domain = {
    # 'genetic_pcs': EmbedConfig(projector="linear", path=tokens_path / 'genetic_pcs', type='continuous', at_birth=True),
    'diseases':    EmbedConfig(projector="embed", path=tokens_path / 'diseases',    predict=True),
    'death':       EmbedConfig(projector="embed", path=tokens_path / 'death',       predict=True),
    'lifestyle':   EmbedConfig(projector="embed", path=tokens_path / 'lifestyle',   age_jitter=True),  
    "hla_alleles": EmbedConfig(projector="embed", path=tokens_path / 'hla_alleles', at_birth=True),
    "sex":         EmbedConfig(projector="embed", path=tokens_path / 'sex',         at_birth=True),
    "padding":     EmbedConfig(projector="embed")
}

domain_cfg = default_cfg_per_domain


def get_last_epoch_checkpoint(run_dir: str):
    """
    Given the run directory (the one containing artifacts/checkpoints),
    return the checkpoint file corresponding to the highest epoch.
    """

    ckpt_dir = Path(run_dir) / "artifacts" / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Find all .pt files
    ckpts = list(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoint files found.")

    # Regex to extract epoch number: looks for 'epoch<digits>'
    epoch_re = re.compile(r"epoch(\d+)", re.IGNORECASE)

    best = None
    best_epoch = -1

    for ck in ckpts:
        m = epoch_re.search(ck.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best = ck

    if best is None:
        raise RuntimeError("No checkpoint contained an epoch number.")

    return best, best_epoch


def infer_delphi_config_from_state_dict(sd):
    cfg = EasyDict()

    # --- 1. Infer number of transformer layers ---
    layer_indices = []
    for key in sd:
        m = re.match(r"transformer\.h\.(\d+)\.", key)
        if m:
            layer_indices.append(int(m.group(1)))
    cfg.n_layer = max(layer_indices) + 1

    # --- 2. Infer embedding dimension ---
    # Use the first c_attn.weight
    for key, val in sd.items():
        if "attn.c_attn.weight" in key:
            W = val
            cfg.n_embd = W.shape[1]
            break

    # --- 3. Infer MLP hidden size ---
    for key, val in sd.items():
        if "mlp.c_fc.weight" in key:
            cfg.mlp_hidden_dim = val.shape[0]
            break

    # --- 4. Infer domains ---
    cfg.domains = []
    for key in sd:
        m = re.match(r"transformer\.embed\.domain_embed\.(\w+)\.projector\.weight", key)
        if m:
            cfg.domains.append(m.group(1))

    # --- 5. Presence of age embedding ---
    cfg.use_age_embedding = any("age_embedding" in k for k in sd)

    # --- 6. Final ln_f layer ---
    cfg.use_final_layernorm = "transformer.ln_f.weight" in sd

    # --- 7. Infer vocabulary sizes per domain ---
    cfg.vocab_sizes = EasyDict()
    for d in cfg.domains:
        key = f"embedding_to_logits.embedding_layer_dict.domain_embed.{d}.projector.weight"
        if key in sd:
            W = sd[key]   # shape = [n_tokens_in_domain, n_embd]
            cfg.vocab_sizes[d] = W.shape[0]  # num tokens

    return cfg

# %%
# ————————————————————————————————————————————————————————————————————————————————————————————————

MLFLOW_URI = Path("/home/bonazzola/repos/delphi/train_scripts/mlruns")
mlflow.set_tracking_uri(MLFLOW_URI)
experiment_id = "276216673358193607"

ckpt_path = get_last_epoch_checkpoint(MLFLOW_URI / experiment_id / runid)
print(ckpt_path[0])
weights = torch.load(ckpt_path[0])['state_dict']


runid = "10a09e55a89d440298155eb74a1cc6b1"
run = mlflow.get_run(runid)
params = run.data.params

# 1. Extract attention scheme
attention_scheme = params.get("attention_scheme")

config = infer_delphi_config_from_state_dict(weights)
n_layer = config['n_layer']
n_embd = config['n_embd']

root_path = Path("./data/transforms")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
tokens_path = root_path / 'tokens'

train_ids, val_ids, test_ids = get_data_partitions("./data/transforms/subject_lists", fold=0)

default_cfg_per_domain = {
    # 'genetic_pcs': EmbedConfig(projector="linear", path=tokens_path / 'genetic_pcs', type='continuous', at_birth=True),
    'diseases':    EmbedConfig(projector="embed", path=tokens_path / 'diseases',    predict=True),
    'death':       EmbedConfig(projector="embed", path=tokens_path / 'death',       predict=True),
    'lifestyle':   EmbedConfig(projector="embed", path=tokens_path / 'lifestyle',   age_jitter=True),  
    "hla_alleles": EmbedConfig(projector="embed", path=tokens_path / 'hla_alleles', at_birth=True),
    "sex":         EmbedConfig(projector="embed", path=tokens_path / 'sex',         at_birth=True),
    "padding":     EmbedConfig(projector="embed")
}

domain_cfg = default_cfg_per_domain

attention_scheme = ast.literal_eval(attention_scheme)

config = DelphiConfig(
    n_embd=n_embd,
    n_layer=n_layer, 
    token_dropout=0.1, 
    domains=domain_cfg, 
    attention_scheme=attention_scheme
)    
# %%

new_model = Delphi(config)
new_model.to("cuda")

# %% 
importlib.reload(data.event_set)
EventSetV2 = data.event_set.EventSetV2

dataset    = DelphiDataset(domains=domain_cfg, root="./data/transforms", subjects=train_ids).to('cuda')
dataloader =  DelphiDataloader(dataset, batch_size=128)

for i, batch in enumerate(dataloader):
    print(i)
    if i == 10:
        break
    
    # pipe(lambda es: es.insert_no_event_tokens(...)).\
    es = EventSetV2(batch).adjust_to_seqlen(
            seqlen=128,
            pad_domain="padding",
            trim_domains={"diseases"},
            PADDING_TOKEN=0,
            PAD_AGE=-10000.0,
            mode="fast"
        )

    tokens, ages, subject_ids = es.to_model_inputs()

    logits, _ = new_model(tokens, ages, subject_ids)
    disease_logits = logits['diseases']

es_df = es.merge_data(as_dataframe=True)

# %%
DISEASE_ID = 486
es_df.set_index("subject_id").loc[[subject_id for subject_id, gg in es_df.groupby("subject_id") if DISEASE_ID in set(gg.token_id)]]

# %%
# case_subject_idx, case_token_idx, ctrl_subject_idx, ctrl_token_idx, d = 

a, b, c = es.get_case_control_indices(disease_token_id=486, min_age=0, max_age=90, sex_token_id=0)
[ [x, y % 128] for x, y in a ]
[ [x, y % 128] for x, y in b ]

# %%
df = es.merge_data(as_dataframe=True)

# %%
print("domain_to_int:", es.domain_to_int)
print("df domain_id unique:", df.domain_id.unique())
print("df domain unique:", df.domain.unique())
print("min age:", df.age_days.min(), "max age:", df.age_days.max())
print(df.head())

# %%
print(es.domain_to_int)

df = es.merge_data(as_dataframe=True)
print(df["domain"].value_counts())
print(df["domain_id"].unique())

d_dom = df[df["domain_id"] == es.domain_to_int["diseases"]]
print("Total disease events:", len(d_dom))
print("Token present?:", (d_dom["token_id"] == 486).any())
print(d_dom[d_dom["token_id"] == 486].head())

sex_dom = es.domain_to_int["sex"]
df_sex = df[df["domain_id"] == sex_dom]
print(df_sex["token_id"].unique()[:20])
print("Subjects with sex=1:", 
      df_sex[df_sex["token_id"] == 1]["subject_id"].nunique())

age_min = 30 * 365.25
age_max = 60 * 365.25
print("Min age in diseases:", d_dom["age_days"].min())
print("Max age in diseases:", d_dom["age_days"].max())

print("Any inside age range?", d_dom["age_days"].between(age_min, age_max).any())


d_dom_sorted = d_dom.sort_values(["subject_id", "age_days"]).reset_index(drop=True)
d_dom_sorted["prev_age"] = d_dom_sorted.groupby("subject_id")["age_days"].shift(+1)

mask = (
    (d_dom_sorted["token_id"] == 486) &
    (d_dom_sorted["prev_age"].between(age_min, age_max))
)

print("Number of case events:", mask.sum())

# %%
