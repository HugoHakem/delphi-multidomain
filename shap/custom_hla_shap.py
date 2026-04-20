# %%
import os, sys
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from loguru import logger
import argparse
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from difflib import get_close_matches
import ast, re
import mlflow
import random

torch.set_grad_enabled(False)

DELPHI_DIR = Path("/homes/bonazzola/repos/delphi")
if str(DELPHI_DIR) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

os.chdir(DELPHI_DIR)

from data.dataset import DelphiDataset, DelphiDataloader
from utils import reconstruct_model, read_ids

device = 'cpu'
DAYS_PER_YEAR = 365.25

SEX_TOKENS = {"female": 0, "male": 1}
AGE_RANGES = [(a, a+5) for a in range(0, 85, 5)]

BATCH_SIZE = 512
NPROC = -1

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

hla_tokenizer = yaml.safe_load(open(DELPHI_DIR / "data/transforms/tokens/hla_alleles/tokenizer.yaml", "rt"))
disease_tokenizer = yaml.safe_load(open(DELPHI_DIR / "data/transforms/tokens/diseases/tokenizer.yaml", "rt"))
disease_tokenizer = np.array(disease_tokenizer)


def tokens_to_df(model, x, ages, subject_ids, domains):

    B, T = x.shape
    subject_idx = torch.arange(B).unsqueeze(1).expand(B, T)
    seq_idx     = torch.arange(T).unsqueeze(0).expand(B, T)
    subject_id  = subject_ids.unsqueeze(1).expand(B, T)

    rows = {
        "subject_idx": subject_idx.reshape(-1).cpu().numpy(),
        "seq_idx":     seq_idx.reshape(-1).cpu().numpy(),
        "subject_id":  subject_id.reshape(-1).cpu().numpy(),
        "age":         ages.reshape(-1).cpu().numpy(),
        "token_id":    x.reshape(-1).cpu().numpy(),
        "domain_id":   domains.reshape(-1).cpu().numpy(),
    }

    rows["domain"] = [
        model.int_to_domain_name[d]
        for d in rows["domain_id"]
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values(["subject_idx", "seq_idx"]).reset_index(drop=True)
    return df


def get_dataloader(model, all_test_ids, subject_ids=None, block_size=None, prediction_domains=None):

    if block_size is not None:
        model.set_block_size(block_size)

    if subject_ids is not None:
        if isinstance(subject_ids, str):
            assert os.path.exists(subject_ids), f"File {subject_ids} does not exist."
            logger.info(f"Before filtering: ", len(all_test_ids))
            subject_ids = read_ids(subject_ids)
            logger.info(f"After filtering for {subject_ids}: ", len(subject_ids))
        subject_ids = [test_id for test_id in all_test_ids if test_id in subject_ids]
    else:
        subject_ids = all_test_ids

    dataset = DelphiDataset(
        domains_cfg=model.domain_cfg,
        root=DELPHI_DIR / "data" / "transforms",
        subjects=subject_ids
    ).to(device)

    dataloader = DelphiDataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_tokens_df = []

    torch.set_grad_enabled(False)

    logger.info("Iterating through dataloader to extract logits and token information...")
    for bi, batch in tqdm(enumerate(dataloader)):

        x, ages, subject_ids = model.prepare_input(batch)

        x, \
        ages, \
        subject_ids, \
        domains, \
        input_embeddings = model.from_dicts_to_tensors(
            x,\
            ages, \
            input_embeddings := model.embed(x),\
            subject_ids
        )

        tokens_df = tokens_to_df(model, x, ages, subject_ids, domains)

        all_tokens_df.append(tokens_df)

        gc.collect()

    return \
        dataset, \
        dataloader, \
        pd.concat(all_tokens_df, axis=0)


def find_most_similar_token(query, token_names, n=3, cutoff=0.6):

    match = get_close_matches(query, token_names, n=n, cutoff=cutoff)

    if not match:
        raise ValueError("No similar disease found.")

    best_name = match[0]

    for tid, name in enumerate(token_names):
        if name == best_name:
            return tid, best_name


@torch.no_grad()
def forward_from_dicts(model, x_dict, ages_dict, sids_dict):
    """
    Forward pass from dict-formatted inputs.
    Assumes insert_no_event_tokens and adjust_to_seqlen have already been applied.
    """
    _, _, output_embeddings = model(
        x_dict, ages_dict, sids_dict,
        return_embeddings=True
    )

    x, ages, subject_ids, domains, _ = model.from_dicts_to_tensors(
        x_dict,
        ages_dict,
        model.embed(x_dict),
        sids_dict
    )

    return output_embeddings, x, ages, domains


@torch.no_grad()
def disease_prev_logits_from_embeddings(
    model,
    output_embeddings,
    x,
    domains,
    disease_domain: str,
    disease_token_id: int
):
    dom_id = model.domain_to_int[disease_domain]

    W = model.embed.domain_embed[disease_domain].projector.weight[disease_token_id]
    logits = (output_embeddings @ W).float()

    next_is_disease = (
        (x[:, 1:] == disease_token_id) &
        (domains[:, 1:] == dom_id)
    )

    b_idx, t_prev = torch.where(next_is_disease)

    logits_prev = logits[b_idx, t_prev]

    return logits_prev, torch.stack([b_idx, t_prev], dim=1)


@torch.no_grad()
def inject_hla_dicts(
    x_rec, ages_rec, sids_rec,
    x_don, ages_don, sids_don,
    hla_domain="hla_alleles",
    generator=None
):
    """
    Replaces the whole HLA domain of the receptor with the donors' HLA block.

    x_rec, ages_rec, sids_rec: dict of receptor batch
    x_don, ages_don, sids_don: dict of donor batch

    Returns modified dictionaries.
    """

    if generator is None:
        generator = random

    x_new = dict(x_rec)
    ages_new = dict(ages_rec)
    sids_new = dict(sids_rec)

    tok_rec = x_rec[hla_domain]
    age_rec = ages_rec[hla_domain]
    sid_rec = sids_rec[hla_domain]

    tok_don = x_don[hla_domain]
    age_don = ages_don[hla_domain]
    sid_don = sids_don[hla_domain]

    device = tok_rec.device

    rec_unique = torch.unique(sid_rec)
    don_unique = torch.unique(sid_don).tolist()

    new_tok = []
    new_age = []
    new_sid = []

    for sid in rec_unique:

        donor_sid = generator.choice(don_unique)

        donor_mask = (sid_don == donor_sid)

        donor_tokens = tok_don[donor_mask].to(device)
        donor_ages = age_don[donor_mask].to(device)

        sid_rep = torch.full(
            (len(donor_tokens),),
            sid,
            dtype=sid_rec.dtype,
            device=device
        )

        new_tok.append(donor_tokens)
        new_age.append(donor_ages)
        new_sid.append(sid_rep)

    # replace domain
    x_new[hla_domain] = torch.cat(new_tok)
    ages_new[hla_domain] = torch.cat(new_age)
    sids_new[hla_domain] = torch.cat(new_sid)

    return x_new, ages_new, sids_new


def prepare_input_manual(model, x_dict, ages_dict, sids_dict):

    max_ages = model.get_max_ages_per_subject(ages_dict, sids_dict)

    x_dict, ages_dict, sids_dict = model.insert_no_event_tokens(
        x_dict, ages_dict, sids_dict, max_ages
    )

    x_dict, ages_dict, sids_dict = model.adjust_to_seqlen(
        x_dict, ages_dict, sids_dict, model.block_size
    )

    return x_dict, ages_dict, sids_dict


def permute_hla_within_batch(
    x_dict, ages_dict, sids_dict,
    hla_domain="hla_alleles",
    generator=None
):
    """
    Permutes HLA blocks across subjects within the same batch.
    Guarantees same batch structure and avoids donor dataloaders.
    """

    if generator is None:
        generator = random

    x_new = dict(x_dict)
    ages_new = dict(ages_dict)
    sids_new = dict(sids_dict)

    tok = x_dict[hla_domain]
    age = ages_dict[hla_domain]
    sid = sids_dict[hla_domain]

    device = tok.device

    subjects = torch.unique(sid).tolist()
    permuted = subjects.copy()
    generator.shuffle(permuted)

    # mapping: receptor subject → donor subject
    donor_map = dict(zip(subjects, permuted))

    new_tok = []
    new_age = []
    new_sid = []

    for sid_rec in subjects:

        sid_don = donor_map[sid_rec]

        donor_mask = sid == sid_don
        donor_tokens = tok[donor_mask]
        donor_ages = age[donor_mask]

        sid_rep = torch.full(
            (len(donor_tokens),),
            sid_rec,
            dtype=sid.dtype,
            device=device
        )

        new_tok.append(donor_tokens)
        new_age.append(donor_ages)
        new_sid.append(sid_rep)

    x_new[hla_domain] = torch.cat(new_tok)
    ages_new[hla_domain] = torch.cat(new_age)
    sids_new[hla_domain] = torch.cat(new_sid)

    return x_new, ages_new, sids_new


def parse_mlflow_dict(s):
    s = re.sub(r"\w*Path\('([^']*)'\)", r"'\1'", s)
    return ast.literal_eval(s)


def get_tokens_df_cached(model, test_ids, cache_dir=None):

    if cache_dir is not None:
        dataset_hash = hash(tuple(sorted(test_ids)))
        CACHE_DIR = Path(cache_dir)
        CACHE_DIR.mkdir(exist_ok=True, parents=True)
        cache_file = CACHE_DIR / f"tokens_df_{dataset_hash}.parquet"

        if cache_file.exists():
            print(f"Loading cached tokens_df from {cache_file}")
            return pd.read_parquet(cache_file)

    print("Computing tokens_df...")
    _, _, tokens_df = get_dataloader(model, all_test_ids=test_ids)

    if cache_dir is not None:
        print(f"Caching tokens_df at {cache_file}")
        tokens_df.to_parquet(cache_file)

    return tokens_df


def compute_delta_for_run(model, test_ids, tokens_df, disease_id, allele_ids, n_counterfactuals=1):

    hla = tokens_df.query('domain == "hla_alleles"')[['subject_id', 'token_id']].drop_duplicates()
    all_subjects = pd.Index(hla['subject_id'].unique())

    subjects_by_allele = (
        hla.groupby('token_id')['subject_id']
        .apply(lambda s: pd.Index(s.unique()))
    )

    non_carriers_by_allele = {
        allele: all_subjects.difference(carriers)
        for allele, carriers in subjects_by_allele.items()
    }

    subjects_with_disease = tokens_df.loc[
        (tokens_df["token_id"] == disease_id) &
        (tokens_df["domain"] == "diseases"),
        "subject_id"
    ].astype(int)

    subjects_with_allele = tokens_df.loc[
        (tokens_df["token_id"].isin(allele_ids)) &
        (tokens_df["domain"] == "hla_alleles"),
        "subject_id"
    ].astype(int)

    subjects_with_allele_disease = set(subjects_with_allele) & set(subjects_with_disease)

    filtered_dataset, filtered_dataloader, _ = get_dataloader(
        model=model,
        all_test_ids=test_ids,
        subject_ids=subjects_with_allele_disease,
        block_size=96
    )

    donor_subjects = set.intersection(*[
        set(non_carriers_by_allele[a].astype(int))
        for a in allele_ids
    ])

    donor_dataset, donor_dataloader, _ = get_dataloader(
        model=model,
        all_test_ids=test_ids,
        subject_ids=random.sample(list(donor_subjects), len(subjects_with_allele_disease) * 2),
        block_size=96
    )

    donor_iter = iter(donor_dataloader)
    delta_all = []

    for batch_rec in filtered_dataloader:

        x_rec, ages_rec, sids_rec = model.get_tensors_from_batch(batch_rec)
        x_rec_p, ages_rec_p, sids_rec_p = prepare_input_manual(model, x_rec, ages_rec, sids_rec)
        out_orig, x_orig, ages_orig, dom_orig = forward_from_dicts(model, x_rec_p, ages_rec_p, sids_rec_p)

        logits_orig, idx_prev = disease_prev_logits_from_embeddings(
            model, out_orig, x_orig, dom_orig, "diseases", disease_id
        )

        n_orig = len(logits_orig)
        if n_orig == 0:
            continue

        b_idx, t_prev = idx_prev[:, 0], idx_prev[:, 1]
        ages_at_event = ages_orig[b_idx, t_prev].cpu()

        # Average swapped logits across n_counterfactuals donor draws.
        # Note: HLA injection may shift disease token positions, so we find
        # positions independently in each swapped sequence.
        logits_sw_samples = []
        for _ in range(n_counterfactuals):
            try:
                batch_don = next(donor_iter)
            except StopIteration:
                donor_iter = iter(donor_dataloader)
                batch_don = next(donor_iter)

            x_don, ages_don, sids_don = model.get_tensors_from_batch(batch_don)
            x_sw, ages_sw, sids_sw = inject_hla_dicts(
                x_rec, ages_rec, sids_rec,
                x_don, ages_don, sids_don
            )
            x_sw_p, ages_sw_p, sids_sw_p = prepare_input_manual(model, x_sw, ages_sw, sids_sw)
            out_sw, x_sw_t, ages_sw_t, dom_sw = forward_from_dicts(model, x_sw_p, ages_sw_p, sids_sw_p)

            logits_sw_prev, _ = disease_prev_logits_from_embeddings(
                model, out_sw, x_sw_t, dom_sw, "diseases", disease_id
            )

            if len(logits_sw_prev) == 0:
                continue

            n = min(n_orig, len(logits_sw_prev))
            logits_sw_samples.append(logits_sw_prev[:n].cpu())

        if not logits_sw_samples:
            continue

        n = min(n_orig, min(len(s) for s in logits_sw_samples))
        logits_sw_avg = torch.stack([s[:n] for s in logits_sw_samples]).mean(dim=0)
        delta = logits_orig[:n].cpu() - logits_sw_avg
        delta_all.append((delta, ages_at_event[:n]))

    deltas, ages = zip(*delta_all)
    return torch.cat(deltas), torch.cat(ages)


# %%
mlflow.search_experiments()

runs_df = mlflow.search_runs(experiment_ids=["263078128312970150"])
runs_df = runs_df.loc[:, runs_df.nunique() > 1]
runs_df = runs_df.rename(columns=lambda c: (
    c.replace("metrics.", "")
     .replace("params.", "")
     .replace("tags.", "")
))
runs_df = runs_df.query("n_head == '12'").reset_index(drop=True)
runs_df = runs_df[
    runs_df['mlflow.runName'].apply(lambda run_name: "mix" in run_name)
]
runs_df = runs_df.sort_values("test_fold")
runs_df['test_fold'] = runs_df['test_fold'].astype(int)

parser = argparse.ArgumentParser()
parser.add_argument("--disease", type=str)
parser.add_argument("--disease_id", type=int)
parser.add_argument("--hla_allele", type=str)
parser.add_argument("--allele_id", type=int)
parser.add_argument("--n_counterfactuals", type=int, default=1)
parser.add_argument("--subjects", type=str, default=None, help="Path to file with subject IDs to intersect with test set")
parser.add_argument("--output", type=str, default=None, help="Output file path. Supports {disease_id} and {allele_id} placeholders.")

args = parser.parse_args()

disease, allele = args.disease, args.hla_allele

# --- disease ---
if args.disease_id is not None:
    disease_id = args.disease_id
    disease_name = disease_tokenizer[disease_id]

elif args.disease is not None:
    disease_id, disease_name = find_most_similar_token(
        args.disease,
        disease_tokenizer
    )

else:
    raise ValueError("You must provide either --disease or --disease_id")


# --- allele ---
if args.allele_id is not None:
    allele_ids = [args.allele_id]
    allele_name = hla_tokenizer[args.allele_id]

elif args.hla_allele is not None:
    allele_pattern = args.hla_allele
    allele_ids = [
        i for i, hla in enumerate(hla_tokenizer)
        if hla.startswith(allele_pattern)
    ]
    allele_name = hla_tokenizer[allele_ids[0]]

else:
    raise ValueError("You must provide either --hla_allele or --allele_id")

print(f"Computing Δlogit for {disease_name} and {allele_name}")

# --- subjects ---
subjects_include = None
if args.subjects is not None:
    assert os.path.exists(args.subjects), f"File {args.subjects} does not exist."
    subjects_include = set(read_ids(args.subjects))
    logger.info(f"Loaded {len(subjects_include)} subject IDs from {args.subjects}")

# --- output ---
output_file = args.output or str(DELPHI_DIR / "shap/output_delta_logit/{disease_id}__{allele_id}.pkl")
output_file = output_file.format(disease_id=disease_id, allele_id=allele_ids[0])

# %%
CACHE_DIR = "/hps/nobackup/birney/users/bonazzola/delphi/output/cache"


def process_fold(fold, runs_df, disease_id, allele_ids, n_counterfactuals, cache_dir, subjects_include=None):

    run_id = runs_df.query("test_fold == @fold").run_id.iloc[0]

    model, test_ids, _, _ = reconstruct_model(run_id)
    model = model.to("cpu")
    model.eval()

    if subjects_include is not None:
        test_ids = [sid for sid in test_ids if sid in subjects_include]
        logger.info(f"fold {fold}: {len(test_ids)} subjects after filtering")

    tokens_df = get_tokens_df_cached(model, test_ids, cache_dir=cache_dir)

    delta_fold, ages_fold = compute_delta_for_run(
        model, test_ids, tokens_df, disease_id, allele_ids,
        n_counterfactuals=n_counterfactuals
    )

    print(f"fold {fold} n={len(delta_fold)}")

    return delta_fold, ages_fold


fold_results = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_fold)(fold, runs_df, disease_id, allele_ids, args.n_counterfactuals, CACHE_DIR, subjects_include)
    for fold in sorted(runs_df.test_fold.unique())
)

all_delta, all_ages = zip(*fold_results)
delta = torch.cat(all_delta).numpy()
ages  = torch.cat(all_ages).numpy()

stat, p_two_sided = stats.wilcoxon(delta)
mean_delta = delta.mean()
print(f"mean Δlogit = {mean_delta:.4f}")
print(f"Wilcoxon two-sided p = {p_two_sided:.2e}")

import pickle as pkl

Path(output_file).parent.mkdir(parents=True, exist_ok=True)
pkl.dump({"delta": delta, "ages": ages}, open(output_file, "wb"))
print(f"Saved to {output_file}")

exit(0)
