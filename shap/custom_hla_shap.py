# %%
import pandas as pd 

import os, sys, shutil
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import argparse
from easydict import EasyDict

from joblib import Parallel, delayed

import gc

import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import yaml

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from difflib import get_close_matches

import ast, re
import mlflow
import random

torch.set_grad_enabled(False)

if ( DELPHI_DIR := Path("/homes/bonazzola/repos/delphi") ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

os.chdir(DELPHI_DIR)

from data.dataset import DelphiDataset, DelphiDataloader
from utils.utils import reconstruct_model, read_ids

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
        subject_ids    = [ test_id for test_id in all_test_ids if test_id in subject_ids]        
    else:
        subject_ids = all_test_ids        

    dataset = DelphiDataset(
        domains_cfg=model.domain_cfg, 
        root=DELPHI_DIR / "data" / "transforms", 
        subjects=subject_ids
    ).to(device)

    dataloader = DelphiDataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)
 
    # ————————————————————————————————————————————————————————————————————————————————————————————————

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
    
    # recuperar token_id
    for tid, name in enumerate(token_names):
        if name == best_name:
            return tid, best_name
        

@torch.no_grad()
def forward_from_dicts(model, x_dict, ages_dict, sids_dict):
    """
    Asume que ya pasaste por:
      - insert_no_event_tokens
      - adjust_to_seqlen
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

    return output_embeddings, x, domains


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
    Reemplaza el dominio HLA del receptor usando sujetos del donor.

    x_rec, ages_rec, sids_rec: dict del batch receptor
    x_don, ages_don, sids_don: dict del batch donor

    Devuelve nuevos diccionarios modificados.
    """

    if generator is None:
        generator = random

    # clonar diccionarios
    x_new = dict(x_rec)
    ages_new = dict(ages_rec)
    sids_new = dict(sids_rec)

    # tensores HLA
    tok_rec = x_rec[hla_domain]
    age_rec = ages_rec[hla_domain]
    sid_rec = sids_rec[hla_domain]

    tok_don = x_don[hla_domain]
    age_don = ages_don[hla_domain]
    sid_don = sids_don[hla_domain]

    device = tok_rec.device

    # sujetos únicos
    rec_unique = torch.unique(sid_rec)
    don_unique = torch.unique(sid_don).tolist()

    new_tok = []
    new_age = []
    new_sid = []

    for sid in rec_unique:

        # elegir donante
        donor_sid = generator.choice(don_unique)

        donor_mask = (sid_don == donor_sid)

        donor_tokens = tok_don[donor_mask].to(device)
        donor_ages = age_don[donor_mask].to(device)

        # repetir subject_id del receptor
        sid_rep = torch.full(
            (len(donor_tokens),),
            sid,
            dtype=sid_rec.dtype,
            device=device
        )

        new_tok.append(donor_tokens)
        new_age.append(donor_ages)
        new_sid.append(sid_rep)

    # reemplazar dominio
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


def parse_mlflow_dict(s):
    s = re.sub(r"\w*Path\('([^']*)'\)", r"'\1'", s)
    return ast.literal_eval(s)

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

# %%
# ———————————————— SELECT MLFLOW RUN ————————————————

# %%

TEST_FOLD = 1
RUNID = runs_df.query("test_fold == @TEST_FOLD").run_id.iloc[0]

model, test_ids, _, _ = reconstruct_model(RUNID)
model.to(device)
model.eval()

# %%

def get_tokens_df_cached(model, test_ids):

    dataset_hash = hash(tuple(sorted(test_ids)))
    CACHE_DIR = Path("/hps/nobackup/birney/users/bonazzola/delphi/output/cache")
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    cache_file = CACHE_DIR / f"tokens_df_{dataset_hash}.parquet"

    if cache_file.exists():
        print(f"Loading cached tokens_df for {run_id}")
        tokens_df = pd.read_parquet(cache_file)

        return tokens_df

    print(f"Computing tokens_df for {run_id}")

    _, _, tokens_df = get_dataloader(
        model,
        all_test_ids=test_ids
    )

    tokens_df.to_parquet(cache_file)

    return tokens_df

# %%

# dataset, dataloader, tokens_df = get_dataloader(model, all_test_ids=test_ids)

_, _, tokens_df = get_dataloader(model, all_test_ids=test_ids)
# tokens_df_backup = tokens_df.copy()

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


# %%
def compute_delta_for_run(run_id, pair, index):
    
    model, test_ids, _, _ = reconstruct_model(run_id)
    model.to(device)
    model.eval()

    tokens_df = get_tokens_df_cached(model, test_ids)      

    hla = tokens_df.query('domain == "hla_alleles"')[['subject_id','token_id']].drop_duplicates()
    all_subjects = pd.Index(hla['subject_id'].unique())

    subjects_by_allele = (
        hla.groupby('token_id')['subject_id']
        .apply(lambda s: pd.Index(s.unique()))
    )

    non_carriers_by_allele = {
        allele: all_subjects.difference(carriers)
        for allele, carriers in subjects_by_allele.items()
    }

    allele_pattern = pair[1]

    allele_ids = [
        i for i, hla in enumerate(hla_tokenizer)
        if hla.startswith(allele_pattern)
    ]

    disease_id, disease_name = find_most_similar_token(pair[0], disease_tokenizer)

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
        subject_ids=random.sample(list(donor_subjects),
        len(subjects_with_allele_disease)*2),
        block_size=96
    )

    donor_iter = iter(donor_dataloader)
    delta_all = []

    for batch_rec in filtered_dataloader:

        try:
            batch_don = next(donor_iter)
        except StopIteration:
            donor_iter = iter(donor_dataloader)
            batch_don = next(donor_iter)

        x_rec, ages_rec, sids_rec = model.get_tensors_from_batch(batch_rec)

        x_rec_p, ages_rec_p, sids_rec_p = prepare_input_manual(
            model, x_rec, ages_rec, sids_rec
        )

        out_orig, x_orig, dom_orig = forward_from_dicts(
            model, x_rec_p, ages_rec_p, sids_rec_p
        )

        logits_orig, idx_prev = disease_prev_logits_from_embeddings(
            model,
            out_orig,
            x_orig,
            dom_orig,
            "diseases",
            disease_id
        )

        x_don, ages_don, sids_don = model.get_tensors_from_batch(batch_don)

        x_sw, ages_sw, sids_sw = inject_hla_dicts(
            x_rec, ages_rec, sids_rec,
            x_don, ages_don, sids_don
        )

        x_sw_p, ages_sw_p, sids_sw_p = prepare_input_manual(
            model, x_sw, ages_sw, sids_sw
        )

        out_sw, _, _ = forward_from_dicts(
            model, x_sw_p, ages_sw_p, sids_sw_p
        )

        logits_sw = (
            out_sw @ model.embed.domain_embed["diseases"]
            .projector.weight[disease_id]
        ).float()

        b_idx = idx_prev[:,0]
        t_prev = idx_prev[:,1]

        try:
            logits_sw_prev = logits_sw[b_idx, t_prev]
        except:
            continue        

        delta = logits_orig - logits_sw_prev
        delta_all.append(delta.cpu())

    return torch.cat(delta_all)

# %%
pairs = [
    ("psoriasis", "HLA-C*06"),
    ("ankylosing spondylitis", "HLA-B*27"),
    ("rheumatoid arthritis", "HLA-DRB1*04"),
    ("intestinal malabsorption", "HLA-DQB1*02:01"),
    ("e14 unspecified diabetes mellitus", "HLA-DRB1*03"),
    ("multiple sclerosis", "HLA-DRB1*15:01"),
    ("narcolepsy", "HLA-DQB1*06:02"),
]

# %%
INDEX = 3
all_delta = []

for fold in sorted(runs_df.test_fold.unique()):

    run_id = runs_df.query("test_fold == @fold").run_id.iloc[0]

    delta_fold = compute_delta_for_run(
        run_id,
        pair=pairs[INDEX],
        index=INDEX
    )

    print(f"fold {fold} n={len(delta_fold)}")

    all_delta.append(delta_fold)

delta_all = torch.cat(all_delta)
delta = delta_all.numpy()

# %%
disease_name = pairs[INDEX][0]
allele_pattern = pairs[INDEX][1]

# %%

# some positive controls
INDEX = 0
allele_pattern = pairs[INDEX][1]

allele_ids = [ i for i, hla in enumerate(hla_tokenizer) if hla.startswith(allele_pattern)]
allele_names = [ hla_tokenizer[allele_id] for allele_id in allele_ids ]
allele_ids, allele_names

disease_id, disease_name = find_most_similar_token(pairs[INDEX][0], disease_tokenizer)

subjects_with_token = set(
    tokens_df.loc[tokens_df["token_id"] == disease_id, "subject_id"]
      .unique()
)

print(allele_names, disease_name)

subjects_with_disease = tokens_df.loc[(tokens_df["token_id"] == disease_id) & (tokens_df["domain"] == "diseases"), "subject_id"].astype(int)

subjects_with_allele = tokens_df.loc[(tokens_df["token_id"].isin(allele_ids)) & (tokens_df["domain"] == "hla_alleles"), "subject_id"].astype(int)
subjects_with_allele_disease = set(subjects_with_allele) & set(subjects_with_disease)

filtered_dataset, \
filtered_dataloader, \
filtered_tokens_df = get_dataloader(
    model=model, 
    all_test_ids=test_ids,
    subject_ids=subjects_with_allele_disease, 
    block_size=96
)

donor_subjects = set.intersection(*[ 
    set(non_carriers_by_allele[allele_id].to_series().astype(int).tolist()) 
    for allele_id in allele_ids 
])
# non_carriers_by_allele[allele_id].to_series().reset_index(drop=True)

donor_dataset, \
donor_dataloader, \
donor_tokens_df = get_dataloader(
    model=model, 
    all_test_ids=test_ids,
    subject_ids=random.sample(list(donor_subjects), len(subjects_with_allele_disease)*10), 
    block_size=96 
)

delta_all = []
donor_iter = iter(donor_dataloader)

for batch_rec in tqdm(filtered_dataloader):

    try:
        batch_don = next(donor_iter)
    except StopIteration:
        donor_iter = iter(donor_dataloader)
        batch_don = next(donor_iter)

    # -------- ORIGINAL --------
    x_rec, ages_rec, sids_rec = model.get_tensors_from_batch(batch_rec)

    x_rec_p, ages_rec_p, sids_rec_p = prepare_input_manual(
        model, x_rec, ages_rec, sids_rec
    )

    out_orig, x_orig, dom_orig = forward_from_dicts(
        model, x_rec_p, ages_rec_p, sids_rec_p
    )

    logits_orig, idx_prev = disease_prev_logits_from_embeddings(
        model,
        out_orig,
        x_orig,
        dom_orig,
        "diseases",
        disease_id
    )

    # -------- SWAP --------
    x_don, ages_don, sids_don = model.get_tensors_from_batch(batch_don)

    x_sw, ages_sw, sids_sw = inject_hla_dicts(
        x_rec, ages_rec, sids_rec,
        x_don, ages_don, sids_don,
        hla_domain="hla_alleles"
    )

    x_sw_p, ages_sw_p, sids_sw_p = prepare_input_manual(
        model, x_sw, ages_sw, sids_sw
    )

    out_sw, x_sw_t, dom_sw = forward_from_dicts(
        model, x_sw_p, ages_sw_p, sids_sw_p
    )

    logits_sw = (out_sw @ model.embed.domain_embed["diseases"]
                 .projector.weight[disease_id]).float()

    # usar MISMAS posiciones
    b_idx = idx_prev[:, 0]
    t_prev = idx_prev[:, 1]

    try:
        logits_sw_prev = logits_sw[b_idx, t_prev]
    except:
        print("Error: ")
        continue

    delta = logits_orig - logits_sw_prev
    delta_all.append(delta.cpu())

delta_all = torch.concat(delta_all)


delta = delta_all.cpu().numpy()

# %%
# --- Wilcoxon 1-sided (H1: Δ > 0)
stat, p_two_sided = stats.wilcoxon(delta)

if np.median(delta) > 0:
    p_one_sided = p_two_sided / 2
else:
    p_one_sided = 1 - (p_two_sided / 2)

mean_delta = delta.mean()

# --- Plot
plt.figure(figsize=(8, 5))

sns.histplot(
    delta,
    bins=60,
    stat="density",
    color="#4C72B0",
    edgecolor="white",
    alpha=0.85
)

plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.axvline(mean_delta, color="red", linewidth=2)

plt.title(f"Effect of {allele_pattern} on risk of {disease_name}\nΔ logit (original − HLA swapped)")
plt.xlabel("Δ logit")
plt.ylabel("Density")

# stats box
textstr = (
    f"mean Δ = {mean_delta:.4f}\n"
    f"Wilcoxon (one-sided)\n"
    f"p = {p_one_sided:.2e}"
)

plt.text(
    0.98, 0.95, textstr,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

plt.tight_layout()
plt.show()