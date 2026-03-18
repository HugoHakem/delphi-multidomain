# %%
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
torch.set_grad_enabled(False)

if ( DELPHI_DIR := Path(__file__).resolve().parent.parent ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

os.chdir(DELPHI_DIR)

from data.dataset import DelphiDataset, DelphiDataloader
from utils.utils import reconstruct_model, read_ids
from scripts.auc_utils import compute_all_stats

device = 'cuda'
DAYS_PER_YEAR = 365.25

SEX_TOKENS = {"female": 0, "male": 1}
AGE_RANGES = [(a, a+5) for a in range(0, 85, 5)]

BATCH_SIZE = 512
NPROC = -1

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True


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


def get_subject_sex(df):
    return df.\
      query('domain == "sex"').\
      groupby("subject_idx")["token_id"].\
      first().\
      to_dict()



def add_sex_column(df):
    return df.assign(sex=lambda df: df['subject_idx'].map(get_subject_sex(df)))


def add_age_bin(tokens_df):
    
    # TODO:
    # Add option to select "previous token" to be the last token before the current cluster of diagnoses

    tokens_df = add_sex_column(tokens_df)
    tokens_df['previous_idx'] = tokens_df.global_idx.apply(lambda x: x-1)
    tokens_df['age_previous'] = tokens_df.age.shift(1)
    DAYS_PER_YEAR = 365.25
    age_bins = [ DAYS_PER_YEAR*i-1  for i in range(0, 90, 5) ]
    tokens_df['age_bin'] = pd.cut(tokens_df['age_previous'], age_bins)
    tokens_df['age_bin'] = (tokens_df.age_bin.cat.codes+1) * 5
    tokens_df = tokens_df[tokens_df.age_bin.notna()]

    return tokens_df


def get_case_ctrl_prevtoken_indices(df, disease_id, sex_token_id, age_min, age_max, dom_of_interest):

    age_years   = df["age"].to_numpy() / 365.25
    subj_arr    = df["subject_idx"].to_numpy()
    token_arr   = df["token_id"].to_numpy()
    dom_arr     = df["domain_id"].to_numpy()
    global_arr  = df["global_idx"].to_numpy()
    df["next_domain"] = df["domain_id"].shift(-1)
    df["next_token_id"] = df["token_id"].shift(-1)

    next_domain = df["next_domain"].to_numpy()
    next_token_id = df["next_token_id"].to_numpy()

    subj_sex_arr = df.sex.to_numpy()

    # FIND CASE INDICES
    mask_sex = (subj_sex_arr == sex_token_id)
    mask_age = (age_years > age_min) & (age_years <= age_max)
    mask_dom = (next_domain == dom_of_interest)
    mask_token = mask_dom & (next_token_id == disease_id)

    case_mask     = (mask_token & mask_sex & mask_age )
    case_subjects = subj_arr[case_mask]
    case_seq      = global_arr[case_mask]

    subj, prev_seq = [], []
    for s, t in zip(case_subjects, case_seq):
        if t > 0:
            subj.append(s)
            prev_seq.append(t)# - 1)

    if len(subj) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    subj = np.array(subj)
    prev_seq  = np.array(prev_seq)

    prev_mask = (np.isin(subj_arr, subj) & np.isin(global_arr, prev_seq))
    case_prev_global_idx = global_arr[prev_mask]

    # FIND CONTROL INDICES
    subjects_with_dis = np.unique(subj_arr[mask_token])
    all_subjects = np.unique(subj_arr)
    subjects_without_dis = np.setdiff1d(all_subjects, subjects_with_dis)
    mask_never_had_disease = np.isin(subj_arr, subjects_without_dis)    
    ctrl_mask = ( mask_sex & mask_age & mask_never_had_disease )

    rng = np.random.default_rng(seed=42)
    ctrl_candidates = np.where(ctrl_mask)[0]
    ctrl_subjects   = subj_arr[ctrl_candidates]
    
    ctrl_global_idx = []
    
    for s in np.unique(ctrl_subjects):
        idx = ctrl_candidates[ctrl_subjects == s]
        chosen = rng.choice(idx)
        ctrl_global_idx.append(global_arr[chosen])
    
    ctrl_global_idx = np.array(ctrl_global_idx, dtype=int)

    return case_prev_global_idx, ctrl_global_idx


def get_case_ctrl_indices(token_id=486, sex="female", age_range=(40, 45)):
    ind_token_of_interest = case_ctrl_indices.query("level_0 == @sex & level_1 == @age_range & level_3 == @token_id")
    if len(ind_token_of_interest) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    cases = ind_token_of_interest.iloc[0].cases
    controls = ind_token_of_interest.iloc[0].controls.astype(int)
    return cases, controls


def compute_auc_from_indices(logits_for_token, cases, controls, block_size): 
    auc_stats = compute_all_stats(
      logits_for_token[cases // block_size, cases % block_size-1],
      logits_for_token[controls // block_size, controls % block_size],
    )
    return auc_stats


def process_disease(logits_for_token, token_id, lookup_disease, block_size):

    import warnings
    warnings.filterwarnings("ignore")

    rows = []
    for age_range in AGE_RANGES:
        for sex in ['female', 'male']:
            case, ctrl = lookup_disease.get((sex, age_range), EMPTY)

            stats = compute_auc_from_indices(logits_for_token, case, ctrl, block_size)
            stats.update({
                "domain": token_id[0],
                "token_id": token_id[1],
                "age_start": age_range[0],
                "age_end": age_range[1],
                "sex": sex,
                "n_case": len(case),
                "n_ctrl": len(ctrl),
            })
            rows.append(stats)

    return rows


def extract_case_ctrl_for_sex_age(sex, sex_id, a0, a1):
    
    key = (sex, (a0, a1))
    DAYS_PER_YEAR = 365.25

    all_sids = ALL_SIDS[sex_id]
    
    mask = age_masks[(a0, a1)] & (TOK_SEX == sex_id)

    subj = TOK_SUBJ[mask]
    gidx = TOK_GIDX[mask]

    df = pd.DataFrame({"subject_id": subj, "global_idx": gidx})
    subset = (
        df.groupby("subject_id", group_keys=False)
          .sample(n=1, random_state=42)
          .set_index("subject_id")
          .reindex(all_sids)
          .sort_index()
    )
    
    local_case, local_ctrl = {}, {}

    DAYS_PER_YEAR = 365.25    

    for dd in ctrl_subjects[sex].keys():

        local_ctrl[(sex, (a0, a1), *dd)] = (
            subset.loc[ctrl_subjects[sex][dd]]
            .dropna()
            .global_idx
        )
        
        gidx_arr, age_arr = DISEASE_IDX[sex_id][dd]
        mask = (age_arr > a0 * DAYS_PER_YEAR) & (age_arr <= a1 * DAYS_PER_YEAR)
        local_case[(sex, (a0, a1), *dd)] = gidx_arr[mask]

    return local_ctrl, local_case


def log_aucs_once(auc_df, run_id, artifact_dir="aucs", final_name="aucs.csv"):
    """
    Each job writes a unique temp file, uploads it, and only the first job
    publishes the stable artifact name. Others exit without overwriting.
    """

    import os, tempfile
    import mlflow
    from mlflow.tracking import MlflowClient

    # 1) write unique local temp file
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        tmp_path = tmp.name
        
    auc_df.to_csv(tmp_path, index=False, na_rep="NA")

    client = MlflowClient()
    final_path = f"{artifact_dir}/{final_name}"

    # 2) check if final artifact already exists
    existing = {
        a.path for a in client.list_artifacts(run_id, artifact_dir)
    }

    if final_path not in existing:
        # 3) publish final artifact (only one job will succeed logically)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(tmp_path, artifact_path=artifact_dir)
        print(f"Published {final_name}")
    else:
        print(f"{final_name} already exists — skipping")

    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri.replace("file://", "")

    src = os.path.join(artifact_uri, artifact_dir, tmp_path)
    dst = os.path.join(artifact_uri, artifact_dir, final_name)

    # if final doesn't exist, promote
    if not os.path.exists(dst):
        shutil.move(src, dst)  # atomic on same filesystem
        print(f"Renamed artifact to {final_name}")
    else:
        # another job already published final
        os.remove(src)
        print(f"{final_name} already exists — discarded temp")

    # 4) cleanup
    try:
        os.remove(tmp_path)
    except OSError:
        pass


def main(args):

    # ————————————————————————————————————————————————————————————————————————————————————————————————

    model, test_ids, _, _ = reconstruct_model(args.runid)
    model.to(device)
    model.eval()

    if args.block_size is not None:
        model.set_block_size(args.block_size)

    if args.subject_ids is not None:
        assert os.path.exists(args.subject_ids), f"File {args.subject_ids} does not exist."        
        logger.info("Before: ", len(test_ids))
        subject_ids = read_ids(args.subject_ids)
        logger.info(f"After filtering for {args.subject_ids}: ", len(test_ids))
        test_ids    = [ test_id for test_id in test_ids if test_id in subject_ids]
        
    domain_to_id = { k: i for i, k in enumerate(model.domains) }
    
    if args.prediction_domains is None:
        prediction_domains = model.predicted_domains
        dom_targets = { dom: domain_to_id[dom] for dom in prediction_domains }

    dataset = DelphiDataset(
        domains_cfg=model.domain_cfg, 
        root=DELPHI_DIR / "data" / "transforms", 
        subjects=test_ids
    ).to(device)

    dataloader = DelphiDataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)
 
    # ————————————————————————————————————————————————————————————————————————————————————————————————

    all_tokens_df, _all_output_embeddings = [], []

    torch.set_grad_enabled(False)

    logger.info("Iterating through dataloader to extract logits and token information...")
    for bi, batch in tqdm(enumerate(dataloader)):
    
        x, ages, subject_ids = model.prepare_input(batch)
        
        _, _, output_embeddings = model(x, ages, subject_ids, return_embeddings=True)
        
        _all_output_embeddings.append(output_embeddings)
        
        del output_embeddings
        
        # here, everything gets transformed to tensors
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
    
    all_output_embeddings = torch.cat(_all_output_embeddings, dim=0).half()

    tokens_df = pd.concat(all_tokens_df, ignore_index=True)
    tokens_df = tokens_df.reset_index().rename(columns={"index": "global_idx"})
    tokens_df['subject_idx'] = tokens_df.global_idx // model.block_size
    tokens_df = add_sex_column(tokens_df)
    tokens_df = add_age_bin(tokens_df)

    del _all_output_embeddings, all_tokens_df
    del x, ages, input_embeddings, subject_ids, domains
    
    # ————————————————————————————————————————————————————————————————————————————————————————————————

    logger.info("Obtaining case and control subject indices for each disease and sex")
    global age_masks, age_previous_masks, case_subjects, ctrl_subjects

    age_masks, age_previous_masks = {} , {}      
    for (a0, a1) in AGE_RANGES:
        age_masks[(a0, a1)] = tokens_df.age.between(a0*DAYS_PER_YEAR, a1*DAYS_PER_YEAR, inclusive="right")
        age_previous_masks[(a0, a1)] = tokens_df.age_previous.between(a0*DAYS_PER_YEAR, a1*DAYS_PER_YEAR, inclusive="right")

    by_disease_dfs = dict(female=None, male=None) 
    case_subjects  = dict(female={}, male={})
    ctrl_subjects  = dict(female={}, male={})    

    # Maybe have model expose this as a property.
    predicted_tokens = [ 
        (domain_name, token_id) 
        for domain_name in model.predicted_domains 
        for token_id in range(model.vocab_lens[model.domain_to_int[domain_name]])
    ]

    for sex, sex_id in SEX_TOKENS.items():

        _by_disease_dfs = tokens_df.\
            query("sex == @sex_id").\
            groupby(["domain", "token_id"])
        
        _by_disease_dfs = dict(list(_by_disease_dfs))
        by_disease_dfs[sex_id] = _by_disease_dfs
    
        tokens_df_sex = tokens_df.query("sex== @sex_id")
        unique_subjects = tokens_df_sex.subject_id.unique()
        
        for (domain_name, token_id) in predicted_tokens:
          
            if (domain_name, token_id) not in _by_disease_dfs:
                continue
          
            case_subjects[sex][domain_name, token_id] = _by_disease_dfs[(domain_name, token_id)].subject_id
            cases_ids = set(case_subjects[sex][(domain_name, token_id)])
            ctrl_subjects[sex][(domain_name, token_id)] = unique_subjects[
                ~pd.Series(unique_subjects).isin(cases_ids)
            ]

    logger.info("Parallelizing indices extraction for each sex and age group...")
    
    # ——————————————————————————————————————————————————————————————————————————————————————————
    global ALL_SIDS, TOK_SUBJ, TOK_SEX, TOK_GIDX, DISEASE_IDX

    ALL_SIDS = {
        sex_id: tokens_df.loc[tokens_df.sex == sex_id, "subject_id"].unique()
        for sex_id in SEX_TOKENS.values()
    }

    TOK_SUBJ = tokens_df["subject_id"].to_numpy()
    TOK_SEX  = tokens_df["sex"].to_numpy()
    TOK_GIDX = tokens_df["global_idx"].to_numpy()

    DISEASE_IDX = {
        sex_id: {
            dd: (
                df["global_idx"].to_numpy(),
                df["age_previous"].to_numpy()
            )
            for dd, df in by_disease_dfs[sex_id].items()
        }
        for sex_id in SEX_TOKENS.values()
    }

    del by_disease_dfs, tokens_df

    # ——————————————————————————————————————————————————————————————————————————————————————————
 
    tasks = [
        (sex, sex_id, a0, a1)
        for sex, sex_id in SEX_TOKENS.items()
        for (a0, a1) in AGE_RANGES
    ]

    results = Parallel(n_jobs=NPROC, backend="threading")(
        delayed(extract_case_ctrl_for_sex_age)(sex, sex_id, a0, a1)
        for sex, sex_id, a0, a1 in tqdm(tasks)
    )

    ctrl_indices, case_indices = {}, {}
    
    for local_ctrl, local_case in results:
        ctrl_indices.update(local_ctrl)
        case_indices.update(local_case)
    
    del results
    logger.info("Indices extracted")
    
    case_indices_df = pd.Series(case_indices).to_frame()
    case_indices_df = case_indices_df.loc[case_indices_df.apply(lambda x: len(x[0]) > 0, axis=1)]

    ctrl_indices_df = pd.Series(ctrl_indices).to_frame()
    
    case_ctrl_indices_df = case_indices_df.\
        merge(ctrl_indices_df, left_index=True, right_index=True).\
        reset_index().\
        rename({"0_x": "cases", "0_y": "controls"}, axis=1)           

    logger.info("Obtaining case and control subject indices for each disease and sex")    
    
    global EMPTY
    EMPTY = (np.array([], dtype=int), np.array([], dtype=int))
    
    # case_ctrl_indices_df.rename({
    #     "level_0": "sex",
    #     "level_1": "age_range",
    #     "level_2": "domain",
    #     "level_3": "token_id"
    # }, axis=1)

    case_ctrl_lookup_by_disease = defaultdict(dict)
    for row in case_ctrl_indices_df.itertuples(index=False):
        # row.level_3 = disease
        case_ctrl_lookup_by_disease[(row.level_2, row.level_3)][
            (row.level_0, row.level_1)
        ] = (
            row.cases,
            row.controls.astype(int)
        )

    del case_ctrl_indices_df
    gc.collect()

    # ——————————————————————————————————————————————————————————————————————————————————————————

    logger.info("Parallelizing AUC computation for each disease")  

    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_disease)(
            logits_for_d := (
                all_output_embeddings @ \
                model.embed.domain_embed[domain_name].projector.weight[token_id].half()
            ).detach().cpu().numpy(), 
            (domain_name, token_id), 
            case_ctrl_lookup_by_disease.get((domain_name, token_id), {}),
            block_size=model.block_size
        ) 
        for domain_name, token_id in tqdm(predicted_tokens)
    )
    
    # ——————————————————————————————————————————————————————————————————————————————————————————

    logger.info(f"Saving AUC to {args.output_file}")

    output_file = args.output_file.replace(".parquet", ".csv")

    auc_data = [row for sublist in results for row in sublist]
    auc_df = pd.DataFrame(auc_data)
    auc_df = auc_df.drop(['auc_bootstrap_mean', 'auc_bootstrap_std'], axis=1)
    auc_df = auc_df.assign(runid=args.runid, block_size=args.block_size)

    COLUMNS_IN_ORDER = [
        "runid", "domain", "token_id", 
        "sex" , "age_start", "age_end" , 
        "n_case", "n_ctrl" , 
        "auc_delong", "auc_delong_var", "mann_u", "mann_p"
    ]

    auc_df = auc_df[COLUMNS_IN_ORDER]	

    log_aucs_once(auc_df, args.runid, final_name=output_file)


# ————————————————————————————————————————————————————————————————————————————————————————————————

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--prediction_domains", type=str, default=None, nargs='+')
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--subject_ids", type=str, default=None)    

    args = parser.parse_args()    

    main(args)
