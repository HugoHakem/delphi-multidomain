"""
SHAP-like HLA allele effect estimation, optionally stratified by sex and age bracket.

Computes the delta-logit: the difference in disease log-odds between a subject's
original HLA genotype and counterfactually injected donor HLA blocks. Subjects who
carry the target allele AND have the disease in their record are used as cases;
non-carriers serve as HLA donors.

Supersedes custom_hla_shap2.py (no sex stratification) and custom_hla_shap3.py
(with sex stratification). Omitting --sex reproduces the custom_hla_shap2.py
behaviour exactly.

Output .pkl keys:
  - "delta":        np.ndarray (N,) — Δlogit (original − counterfactual)
  - "ages":         np.ndarray (N,) — age in days at the disease event
  - "sexes":        np.ndarray (N,) — int (0=female, 1=male, -1=unknown)
  - "age_brackets": np.ndarray (N,) — bracket label string or None

Usage:
    python shap/custom_hla_shap.py \\
        --allele_id <int> \\
        --disease "<ICD description>" \\
        --n_counterfactuals 10 \\
        [--sex female|male] \\
        --subjects data/transforms/subject_lists/genetic_white_ids.txt \\
        --output shap/delta_logits/{disease_id}__{allele_id}__{sex}.pkl

Arguments:
    --disease           Disease name (fuzzy-matched against tokenizer)
    --disease_id        Disease token ID (alternative to --disease)
    --hla_allele        HLA allele prefix (e.g. "HLA-A*02"); all matching alleles grouped
    --allele_id         Single allele token ID (alternative to --hla_allele)
    --n_counterfactuals Number of donor HLA injections to average over (default: 5)
    --sex               Restrict cases and donors to this sex: "male" or "female" (optional)
    --subjects          Path to file with subject IDs to restrict to (optional)
    --output            Output .pkl path; supports {disease_id}, {allele_id}, {sex} placeholders
"""

import os
import sys
import random
import argparse
import pickle as pkl
import warnings
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from loguru import logger
from scipy import stats
from tqdm import tqdm
import mlflow

warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)

DELPHI_DIR = Path("/nfs/research/birney/users/bonazzola/repos/delphis/delphi-refactor")
if str(DELPHI_DIR) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

os.chdir(DELPHI_DIR)

from data.dataset import DelphiDataset, DelphiCollateFn, AgeSampler
from torch.utils.data import DataLoader
from utils import reconstruct_model
from utils.utils import read_ids

device = "cpu"
DAYS_PER_YEAR = 365.25

SEX_TOKENS = {"female": 0, "male": 1}
INT_TO_SEX = {v: k for k, v in SEX_TOKENS.items()}
AGE_RANGES = [(a, a + 20) for a in range(0, 70, 10)]

BATCH_SIZE = 512
CACHE_DIR = "/hps/nobackup/birney/users/bonazzola/delphi/output/cache"

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

hla_tokenizer = yaml.safe_load(
    open(DELPHI_DIR / "data/transforms/tokens/hla_alleles/tokenizer.yaml", "rt")
)
disease_tokenizer = np.array(
    yaml.safe_load(open(DELPHI_DIR / "data/transforms/tokens/diseases/tokenizer.yaml", "rt"))
)


def assign_age_bracket(age_days):
    age_years = age_days / DAYS_PER_YEAR
    for lo, hi in AGE_RANGES:
        if lo <= age_years < hi:
            return f"{lo}-{hi}"
    return None


def _make_continuous_domains(model):
    return {
        dname: cfg.n_latent_tokens or 1
        for dname, cfg in model.domain_cfg.items()
        if cfg.type == "continuous"
    }


def _make_collate(model):
    continuous_domains = _make_continuous_domains(model)
    age_sampler = AgeSampler(
        insertion_mode=model.config.no_event_token_insertion_mode,
        token_rate=model.config.no_event_token_rate,
        seed=model.config.seed,
    )
    return DelphiCollateFn(
        age_sampler=age_sampler,
        block_size=model.block_size,
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_token_id=1,
        continuous_domains=continuous_domains,
    )


def get_dataloader(model, all_test_ids, subject_ids=None, block_size=None):
    if block_size is not None:
        model.set_block_size(block_size)

    if subject_ids is not None:
        if isinstance(subject_ids, str):
            assert os.path.exists(subject_ids), f"File {subject_ids} does not exist."
            subject_ids = read_ids(subject_ids)
        subject_ids = [sid for sid in all_test_ids if sid in set(subject_ids)]
    else:
        subject_ids = all_test_ids

    continuous_domains = _make_continuous_domains(model)
    dataset = DelphiDataset(
        root=DELPHI_DIR / "data" / "transforms",
        domains_cfg=model.domain_cfg,
        domain_to_int=model.domain_to_int,
        block_size=model.block_size,
        subjects=subject_ids,
        required_domains=["diseases"],
        no_event_token_rate=model.config.no_event_token_rate,
        no_event_insertion_mode=model.config.no_event_token_insertion_mode,
        continuous_domains=continuous_domains,
    )
    collate = _make_collate(model)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    return dataset, dataloader


def get_tokens_df_from_dataset(model, dataset):
    """Build a flat tokens DataFrame from a DelphiDataset. Vectorized: no Python loop over subjects."""
    N, T = dataset._domain_ids.shape
    subj_expanded = dataset._subject_ids.unsqueeze(1).expand(N, T)
    seq_idx = torch.arange(T).unsqueeze(0).expand(N, T)
    real_mask = seq_idx < dataset._real_counts.unsqueeze(1)

    df = pd.DataFrame({
        "subject_id": subj_expanded[real_mask].numpy(),
        "domain_id":  dataset._domain_ids[real_mask].numpy(),
        "token_id":   dataset._local_token_ids[real_mask].numpy(),
        "age":        dataset._ages[real_mask].numpy(),
    })
    df["domain"] = df["domain_id"].map(model.int_to_domain)
    return df


def get_tokens_df_cached(model, test_ids, run_id, cache_dir=None):
    if cache_dir is not None:
        dataset_hash = hash(tuple(sorted(test_ids)))
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True, parents=True)
        cache_file = cache_path / f"tokens_df_{run_id}_{dataset_hash}.parquet"

        if cache_file.exists():
            print(f"Loading cached tokens_df from {cache_file}")
            return pd.read_parquet(cache_file)

    print("Computing tokens_df...")
    dataset, _ = get_dataloader(model, all_test_ids=test_ids)
    tokens_df = get_tokens_df_from_dataset(model, dataset)

    if cache_dir is not None:
        print(f"Caching tokens_df at {cache_file}")
        tokens_df.to_parquet(cache_file)

    return tokens_df


def find_most_similar_token(query, token_names, n=3, cutoff=0.6):
    match = get_close_matches(query, token_names, n=n, cutoff=cutoff)
    if not match:
        raise ValueError("No similar disease found.")
    best_name = match[0]
    tid = next(i for i, name in enumerate(token_names) if name == best_name)
    return tid, best_name


@torch.no_grad()
def disease_prev_logits_from_embeddings(model, h, batch, disease_domain, disease_token_id):
    """
    Compute per-token logits for disease_token_id and find positions where
    the *next* token is that disease.
    """
    dom_id = model.domain_to_int[disease_domain]
    global_disease_id = disease_token_id + model.domain_offsets[dom_id]

    W = model.embed._get_domain_weight(disease_domain)[disease_token_id]  # [n_embd]
    logits = (h @ W).float()  # [B, T]

    next_is_disease = (
        (batch.global_token_ids[:, 1:] == global_disease_id) &
        (batch.domain_ids[:, 1:] == dom_id)
    )
    b_idx, t_prev = torch.where(next_is_disease)
    return logits[b_idx, t_prev], torch.stack([b_idx, t_prev], dim=1)


def inject_hla_item(item_rec, item_don, hla_domain_int, padding_domain_id, padding_age=-10000.0):
    """Return a copy of item_rec with its HLA tokens replaced by those of item_don."""
    rec_real = int(item_rec["real_count"])
    don_real = int(item_don["real_count"])

    rec_doms = item_rec["domain_ids"][:rec_real]
    don_doms = item_don["domain_ids"][:don_real]

    rec_hla = rec_doms == hla_domain_int
    don_hla = don_doms == hla_domain_int

    non_hla_dom = rec_doms[~rec_hla]
    non_hla_tok = item_rec["local_token_ids"][:rec_real][~rec_hla]
    non_hla_age = item_rec["ages"][:rec_real][~rec_hla]

    don_hla_dom = don_doms[don_hla]
    don_hla_tok = item_don["local_token_ids"][:don_real][don_hla]
    don_hla_age = item_don["ages"][:don_real][don_hla]

    all_dom = torch.cat([don_hla_dom, non_hla_dom])
    all_tok = torch.cat([don_hla_tok, non_hla_tok])
    all_age = torch.cat([don_hla_age, non_hla_age])

    # Sort by age, breaking ties by domain to keep deterministic ordering
    sort_key = all_age + all_dom.float() * 0.001
    _, idx = sort_key.sort()
    all_dom = all_dom[idx]
    all_tok = all_tok[idx]
    all_age = all_age[idx]

    block_size = item_rec["domain_ids"].shape[0]
    n_real = min(len(all_dom), block_size)

    new_item = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in item_rec.items()}
    new_item["domain_ids"] = torch.full(
        (block_size,), padding_domain_id, dtype=item_rec["domain_ids"].dtype
    )
    new_item["local_token_ids"] = torch.zeros(block_size, dtype=item_rec["local_token_ids"].dtype)
    new_item["ages"] = torch.full((block_size,), padding_age, dtype=item_rec["ages"].dtype)
    new_item["domain_ids"][:n_real]      = all_dom[:n_real]
    new_item["local_token_ids"][:n_real] = all_tok[:n_real]
    new_item["ages"][:n_real]            = all_age[:n_real]
    new_item["real_count"] = torch.tensor(n_real, dtype=item_rec["real_count"].dtype)

    return new_item


def extract_sex_map(tokens_df):
    return (
        tokens_df[tokens_df["domain"] == "sex"]
        .drop_duplicates("subject_id")
        .set_index("subject_id")["token_id"]
        .to_dict()
    )


def compute_delta_for_run(
    model, test_ids, tokens_df, disease_id, allele_ids,
    sex_map, sex_filter=None, n_counterfactuals=1,
    disease_name="disease", allele_name="allele",
):
    """
    Returns (delta, ages, sexes) as 1-D tensors, one entry per disease event.
    sex_filter=None means no sex filtering (equivalent to custom_hla_shap2.py behaviour).
    """
    hla = tokens_df.query('domain == "hla_alleles"')[["subject_id", "token_id"]].drop_duplicates()
    all_subjects = pd.Index(hla["subject_id"].unique())

    subjects_by_allele = (
        hla.groupby("token_id")["subject_id"]
        .apply(lambda s: pd.Index(s.unique()))
    )

    # Only compute non-carrier sets for the alleles actually needed
    allele_ids_set = set(allele_ids)
    non_carriers_by_allele = {
        allele: all_subjects.difference(carriers)
        for allele, carriers in subjects_by_allele.items()
        if allele in allele_ids_set
    }

    subjects_with_disease = set(
        tokens_df.loc[
            (tokens_df["token_id"] == disease_id) & (tokens_df["domain"] == "diseases"),
            "subject_id",
        ].astype(int)
    )
    subjects_with_allele = set(
        tokens_df.loc[
            (tokens_df["token_id"].isin(allele_ids)) & (tokens_df["domain"] == "hla_alleles"),
            "subject_id",
        ].astype(int)
    )
    subjects_with_allele_disease = subjects_with_allele & subjects_with_disease

    donor_subjects = set.intersection(*[
        set(non_carriers_by_allele[a].astype(int)) for a in allele_ids
    ])

    if sex_filter is not None:
        sex_int = SEX_TOKENS[sex_filter]
        subjects_with_allele_disease = {s for s in subjects_with_allele_disease if sex_map.get(s) == sex_int}
        donor_subjects = {s for s in donor_subjects if sex_map.get(s) == sex_int}

    if not subjects_with_allele_disease:
        logger.warning("No cases found after filtering.")
        return torch.tensor([]), torch.tensor([]), torch.tensor([], dtype=torch.int)
    if not donor_subjects:
        logger.warning("No donors found after filtering.")
        return torch.tensor([]), torch.tensor([]), torch.tensor([], dtype=torch.int)

    filtered_dataset, filtered_dataloader = get_dataloader(
        model=model,
        all_test_ids=test_ids,
        subject_ids=subjects_with_allele_disease,
        block_size=96,
    )

    n_donors = min(len(subjects_with_allele_disease) * 2, len(donor_subjects))
    donor_dataset, _ = get_dataloader(
        model=model,
        all_test_ids=test_ids,
        subject_ids=random.sample(list(donor_subjects), n_donors),
        block_size=96,
    )

    hla_domain_int = model.domain_to_int["hla_alleles"]
    padding_domain_id = model.domain_to_int["padding"]
    collate = _make_collate(model)

    donor_subjects_list = donor_dataset.subject_list
    donor_idx = 0

    delta_all, ages_all, sexes_all = [], [], []

    for batch in tqdm(
        filtered_dataloader,
        total=len(filtered_dataloader),
        desc=f"Δlogit: {disease_name} vs {allele_name}",
    ):
        batch = batch.to(device)

        with torch.no_grad():
            _, _, h = model(batch, return_embeddings=True)

        logits_orig, idx_prev = disease_prev_logits_from_embeddings(
            model, h, batch, "diseases", disease_id
        )
        n_orig = len(logits_orig)
        if n_orig == 0:
            continue

        b_idx, t_prev = idx_prev[:, 0], idx_prev[:, 1]
        ages_at_event  = batch.ages[b_idx, t_prev].cpu()
        sids_at_event  = batch.subject_ids[b_idx].cpu()
        sexes_at_event = torch.tensor([sex_map.get(int(s), -1) for s in sids_at_event])

        rec_sids  = batch.subject_ids.tolist()
        rec_items = [filtered_dataset[filtered_dataset._sid_to_idx[sid]] for sid in rec_sids]

        logits_sw_samples = []
        for _ in tqdm(range(n_counterfactuals), desc="Counterfactuals", leave=False):
            don_items = []
            for _ in rec_sids:
                don_sid = donor_subjects_list[donor_idx % len(donor_subjects_list)]
                donor_idx += 1
                don_items.append(donor_dataset[donor_dataset._sid_to_idx[don_sid]])

            modified_items = [
                inject_hla_item(r, d, hla_domain_int, padding_domain_id)
                for r, d in zip(rec_items, don_items)
            ]
            batch_sw = collate(modified_items).to(device)

            with torch.no_grad():
                _, _, h_sw = model(batch_sw, return_embeddings=True)

            logits_sw, _ = disease_prev_logits_from_embeddings(
                model, h_sw, batch_sw, "diseases", disease_id
            )
            if len(logits_sw) == 0:
                continue
            n = min(n_orig, len(logits_sw))
            logits_sw_samples.append(logits_sw[:n].cpu())

        if not logits_sw_samples:
            continue

        n = min(n_orig, min(len(s) for s in logits_sw_samples))
        logits_sw_avg = torch.stack([s[:n] for s in logits_sw_samples]).mean(dim=0)
        delta_all.append(logits_orig[:n].cpu() - logits_sw_avg)
        ages_all.append(ages_at_event[:n])
        sexes_all.append(sexes_at_event[:n])

    if not delta_all:
        return torch.tensor([]), torch.tensor([]), torch.tensor([], dtype=torch.int)

    return torch.cat(delta_all), torch.cat(ages_all), torch.cat(sexes_all)


def process_fold(
    fold, runs_df, disease_id, allele_ids, sex_filter, n_counterfactuals,
    cache_dir, subjects_include=None, disease_name="disease", allele_name="allele",
):
    run_id = runs_df.query("test_fold == @fold").run_id.iloc[0]

    model, test_ids, _, _ = reconstruct_model(run_id)
    model = model.to("cpu")
    model.eval()

    if subjects_include is not None:
        test_ids = [int(sid) for sid in test_ids if int(sid) in subjects_include]
        logger.info(f"fold {fold}: {len(test_ids)} subjects after filtering")

    tokens_df = get_tokens_df_cached(model, test_ids, run_id=run_id, cache_dir=cache_dir)
    sex_map = extract_sex_map(tokens_df)

    delta_fold, ages_fold, sexes_fold = compute_delta_for_run(
        model, test_ids, tokens_df, disease_id, allele_ids,
        sex_map=sex_map, sex_filter=sex_filter,
        n_counterfactuals=n_counterfactuals,
        disease_name=disease_name, allele_name=allele_name,
    )

    print(f"fold {fold} n={len(delta_fold)}")
    return delta_fold, ages_fold, sexes_fold


def _resolve_experiment_id(prefix: str) -> str:
    """
    Return the unique experiment ID whose string starts with `prefix`.
    Aborts if zero or more than one match is found.
    """
    all_experiments = mlflow.search_experiments()
    matches = [e for e in all_experiments if e.experiment_id.startswith(prefix)]
    if not matches:
        raise SystemExit(f"No experiment found with ID prefix '{prefix}'.")
    if len(matches) > 1:
        candidates = "\n  ".join(
            f"{e.experiment_id}  ({e.name})" for e in matches
        )
        raise SystemExit(
            f"Ambiguous prefix '{prefix}' matches {len(matches)} experiments:\n  {candidates}"
        )
    return matches[0].experiment_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_id", type=str, required=True,
        help="MLflow experiment ID or unique prefix thereof.",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, metavar="REGEX",
        help="Regex applied to run name; only matching runs are kept.",
    )
    parser.add_argument(
        "--param", type=str, action="append", default=[], metavar="NAME=VALUE",
        help="Filter by parameter value: NAME=VALUE. Can be repeated.",
    )
    parser.add_argument("--disease", type=str)
    parser.add_argument("--disease_id", type=int)
    parser.add_argument("--hla_allele", type=str)
    parser.add_argument("--allele_id", type=int)
    parser.add_argument("--n_counterfactuals", type=int, default=5)
    parser.add_argument(
        "--sex", type=str, default=None, choices=["male", "female", "both"],
        help="Restrict cases and donors to this sex. 'both' or omitted = no filtering.",
    )
    parser.add_argument(
        "--subjects", type=str, default=None,
        help="Path to file with subject IDs to intersect with test set.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .pkl path. Supports {disease_id}, {allele_id}, {sex} placeholders.",
    )
    parser.add_argument(
        "--dry-run", "--dryrun", "--dry_run", dest="dry_run", action="store_true",
        help="Print the disease, alleles and runs that would be processed, then exit.",
    )
    args = parser.parse_args()

    experiment_id = _resolve_experiment_id(args.experiment_id)

    runs_df = mlflow.search_runs(experiment_ids=[experiment_id])
    runs_df = runs_df.loc[:, runs_df.nunique() > 1]
    runs_df = runs_df.rename(columns=lambda c: (
        c.replace("metrics.", "").replace("params.", "").replace("tags.", "")
    ))
    runs_df = runs_df.loc[:, ~runs_df.columns.duplicated()]

    if args.run_name is not None:
        mask = runs_df["mlflow.runName"].str.contains(args.run_name, regex=True, na=False)
        runs_df = runs_df[mask]
        if runs_df.empty:
            raise SystemExit(f"No runs matched --run_name pattern '{args.run_name}'.")

    for spec in args.param:
        if "=" not in spec:
            raise SystemExit(f"Invalid --param format: '{spec}'. Expected NAME=REGEX.")
        col, pattern = spec.split("=", 1)
        col = col.strip()
        if col not in runs_df.columns:
            available = [c for c in runs_df.columns if not c.startswith("mlflow.")]
            raise SystemExit(f"Parameter '{col}' not found. Available params: {available}")
        mask = runs_df[col].astype(str) == pattern
        runs_df = runs_df[mask]
        if runs_df.empty:
            raise SystemExit(f"No runs matched --param '{spec}'.")

    runs_df = runs_df.sort_values("test_fold").reset_index(drop=True)
    runs_df["test_fold"] = runs_df["test_fold"].astype(int)
    print(f"Selected {len(runs_df)} run(s) across fold(s): {sorted(runs_df.test_fold.unique())}")
    if args.sex == "both":
        args.sex = None

    # --- disease ---
    if args.disease_id is not None:
        disease_id = args.disease_id
        disease_name = disease_tokenizer[disease_id]
    elif args.disease is not None:
        disease_id, disease_name = find_most_similar_token(args.disease, disease_tokenizer)
    else:
        raise ValueError("You must provide either --disease or --disease_id")

    # --- allele ---
    if args.allele_id is not None:
        allele_ids = [args.allele_id]
        allele_name = hla_tokenizer[args.allele_id]
    elif args.hla_allele is not None:
        allele_ids = [i for i, hla in enumerate(hla_tokenizer) if hla.startswith(args.hla_allele)]
        allele_name = hla_tokenizer[allele_ids[0]]
    else:
        raise ValueError("You must provide either --hla_allele or --allele_id")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(f"Disease : {disease_name} (id={disease_id})")
        print(f"Allele  : {allele_name} (ids={allele_ids})")
        print(f"Runs    ({len(runs_df)}):")
        for _, row in runs_df.iterrows():
            print(f"  fold={row['test_fold']}  run_id={row['run_id']}  name={row['mlflow.runName']}")
        raise SystemExit(0)

    print(f"Computing Δlogit for {disease_name} and {allele_name}"
          + (f" (sex={args.sex})" if args.sex else ""))

    # --- subjects ---
    subjects_include = None
    if args.subjects is not None:
        assert os.path.exists(args.subjects), f"File {args.subjects} does not exist."
        subjects_include = set(read_ids(args.subjects))
        logger.info(f"Loaded {len(subjects_include)} subject IDs from {args.subjects}")

    # --- output ---
    sex_label = args.sex or "both_sexes"
    output_file = args.output or str(
        DELPHI_DIR / "shap/output_delta_logit/{disease_id}__{allele_id}__{sex}.pkl"
    )
    output_file = output_file.format(disease_id=disease_id, allele_id=allele_ids[0], sex=sex_label)

    # --- run folds ---
    n_folds = runs_df.test_fold.nunique()
    fold_results = list(tqdm(
        Parallel(n_jobs=-1, backend="loky", return_as="generator")(
            delayed(process_fold)(
                fold, runs_df, disease_id, allele_ids, args.sex,
                args.n_counterfactuals, CACHE_DIR, subjects_include,
                disease_name, allele_name,
            )
            for fold in sorted(runs_df.test_fold.unique())
        ),
        total=n_folds,
        desc="Folds",
    ))

    all_delta, all_ages, all_sexes = zip(*fold_results)
    delta = torch.cat(all_delta).numpy()
    ages  = torch.cat(all_ages).numpy()
    sexes = torch.cat(all_sexes).numpy()
    age_brackets = np.array([assign_age_bracket(a) for a in ages])

    # --- summary ---
    MIN_N_WILCOXON = 10
    stat, p_two_sided = stats.wilcoxon(delta)
    print(f"mean Δlogit = {delta.mean():.4f}")
    print(f"Wilcoxon two-sided p = {p_two_sided:.2e}")

    groupby_cols = ["age_bracket", "sex"] if args.sex is None else ["age_bracket"]
    summary_df = pd.DataFrame({
        "delta":       delta,
        "age_bracket": age_brackets,
        "sex":         np.vectorize(INT_TO_SEX.get)(sexes, "unknown"),
    })

    print(f"\nΔlogit summary: {disease_name}  |  {allele_name}"
          + (f"  |  sex={args.sex}" if args.sex else ""))
    header = f"{'Age bracket':<12}  {'Sex':<8}  {'n':>6}  {'mean Δlogit':>12}  {'p (Wilcoxon)':>14}"
    print(header)
    print("-" * len(header))

    for keys, grp in summary_df.groupby(groupby_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        bracket = keys[0]
        sex_col = keys[1] if len(keys) > 1 else (args.sex or "all")
        n       = len(grp)
        mean_d  = grp["delta"].mean()
        if n >= MIN_N_WILCOXON:
            _, p = stats.wilcoxon(grp["delta"])
            p_str = f"{p:.2e}"
        else:
            p_str = f"n<{MIN_N_WILCOXON}"
        print(f"{str(bracket):<12}  {str(sex_col):<8}  {n:>6}  {mean_d:>12.4f}  {p_str:>14}")

    # --- save ---
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    pkl.dump({"delta": delta, "ages": ages, "sexes": sexes, "age_brackets": age_brackets},
             open(output_file, "wb"))
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
