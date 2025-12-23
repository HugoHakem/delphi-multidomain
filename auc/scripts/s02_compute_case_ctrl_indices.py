#!/usr/bin/env python3

import argparse
from pathlib import Path
from tqdm import tqdm
import re
import ast
from easydict import EasyDict
import os, sys

import numpy as np
import pandas as pd
import torch
import mlflow

if ( DELPHI_DIR := Path(__file__).resolve().parent.parent.parent ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))
    
from data.event_set import EventSet

from utils.utils import reconstruct_model

SEX_TOKENS = {"female": 0, "male": 1}
AGE_RANGES = [(a, a+5) for a in range(0, 90, 5)]
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
INDICES_FILE_PATTERN = "indices__chunk_{ci}_of_{n_chunks}__dchunk_{dchunk}_of_{n_dchunks}.parquet"
LOGITS_FILE_PATTERN  = "indices__chunk_{ci}_of_{n_chunks}__dchunk_{dchunk}_of_{n_dchunks}.pt"


def get_subject_sex(df, dom_sex):
    rows = df[df["domain_id"] == dom_sex]
    return (
        rows.groupby("subject_idx")["token_id"]
        .first()
        .to_dict()
    )


def get_case_ctrl_prevtoken_indices(df, disease_id, sex_token_id, age_min, age_max, dom_diseases, dom_sex):

    age_years   = df["age"].to_numpy() / 365.25
    subj_arr    = df["subject_idx"].to_numpy()
    token_arr   = df["token_id"].to_numpy()
    dom_arr     = df["domain_id"].to_numpy()
    # seq_arr     = df["seq_idx"].to_numpy()
    global_arr  = df["global_idx"].to_numpy()

    subj2sex = get_subject_sex(df, dom_sex)
    subj_sex_arr = np.array([subj2sex.get(s, -1) for s in subj_arr])

    # FIND CASE INDICES
    mask_sex = (subj_sex_arr == sex_token_id)
    mask_age = (age_years >= age_min) & (age_years < age_max)
    mask_dom = (dom_arr == dom_diseases)

    case_mask     = ( (token_arr == disease_id) & mask_dom & mask_sex & mask_age )
    case_subjects = subj_arr[case_mask]
    case_seq      = global_arr[case_mask]

    subj, prev_seq = [], []
    for s, t in zip(case_subjects, case_seq):
        if t > 0:
            subj.append(s)
            prev_seq.append(t - 1)

    if len(subj) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    subj = np.array(subj)
    prev_seq  = np.array(prev_seq)

    prev_mask = (np.isin(subj_arr, subj) & np.isin(global_arr, prev_seq))
    case_prev_global_idx = global_arr[prev_mask]

    # FIND CONTROL INDICES
    subjects_with_dis = np.unique(subj_arr[case_mask])
    all_subjects = np.unique(subj_arr)
    subjects_without_dis = np.setdiff1d(all_subjects, subjects_with_dis)
    mask_never_had_disease = np.isin(subj_arr, subjects_without_dis)
    # ctrl_mask = ( mask_sex & mask_age & mask_dom & mask_never_had_disease )
    ctrl_mask = ( mask_sex & mask_age & mask_never_had_disease )
    ctrl_global_idx = global_arr[ctrl_mask]

    return case_prev_global_idx, ctrl_global_idx


def get_predicted_domains(model):
    return [ dom for dom, cfg in model.config.domains.items() if getattr(cfg, "predict", True) ]



def main(args):

    runid, ci, dchunk, n_dchunks = args.runid, args.chunk_index, args.dchunk, args.n_dchunks
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    ( outdir := Path(args.output_dir) / runid).mkdir(exist_ok=True, parents=True )
    logits_dir = Path(args.logits_dir) # / runid
    
    pattern = f"{runid}_chunk_{ci}_of_*_df.parquet"
    print(logits_dir / pattern)
    matches = list(logits_dir.glob(pattern))
    
    assert len(matches) > 0, f"No DF file found for chunk_index={ci} in {logits_dir} for pattern {pattern}"
    
    if len(matches) > 1:
        print("[WARN] Multiple matches found, using first:", matches)
    
    df_path = matches[0]    
    fname_parts = df_path.stem.split("_")  
    n_chunks = int(fname_parts[4])
    
    print(f"[INFO] Found DF: {df_path}  → n_chunks = {n_chunks}")    
    
    df = pd.read_parquet(df_path).assign(age=lambda df: df.age_days)

    model, _, _, _, _ = reconstruct_model(runid)
    domain_to_id = df[['domain_id', 'domain']].drop_duplicates().set_index("domain").domain_id.to_dict()   
    print("[INFO] domain_to_id =", domain_to_id)

    assert "sex" in domain_to_id, "Model does not contain 'sex' domain."
    dom_sex = domain_to_id["sex"]
    
    if args.prediction_domains is None or len(args.prediction_domains) == 0:
        print("[INFO] Using model-predicted domains:", prediction_domains := get_predicted_domains(model))
    else:
        prediction_domains = args.prediction_domains
        print("[INFO] Using user-specified domains:", prediction_domains)

    for d in prediction_domains:
        if d not in domain_to_id:
            raise RuntimeError(f"Domain '{d}' not present in model.")

    dom_targets = { dom: domain_to_id[dom] for dom in prediction_domains }
    
    rows = []

    for dom in prediction_domains:
        domid = dom_targets[dom]
    
        max_token = df.loc[df["domain_id"] == domid, "token_id"].max()
        if pd.isna(max_token):
            print(f"[WARN] Domain '{dom}' has no tokens in chunk → skipping")
            continue
    
        all_tokens = np.arange(max_token + 1)
        my_tokens = np.array_split(all_tokens, n_dchunks)[dchunk]
    
        print(f"[INFO] chunk={ci}, dchunk={dchunk}, domain={dom}, tokens={len(my_tokens)}")
    
        for tok in tqdm(my_tokens, desc=f"{dom}-dchunk-{dchunk}"):
            for sex, sex_token in SEX_TOKENS.items():
                for (a0, a1) in AGE_RANGES:
                    ci_arr, co_arr = get_case_ctrl_prevtoken_indices(df, tok, sex_token, a0, a1, domid, dom_sex)    
                    rows.append({ "domain": dom, "token_id": tok, "sex": sex, "age_start": a0, "age_end": a1, 
                                 "case_indices": ci_arr, "ctrl_indices": co_arr })
    
    df_out = pd.DataFrame(rows)
    df_out["chunk_index"] = ci
    df_out["dchunk"] = dchunk
        
    indices_file = indices_file_pattern.format(ci=ci, n_chunks=n_chunks, dchunk=dchunk, n_dchunks=n_dchunks)
    df_out.to_parquet(outfile := outdir / indices_file)
    print("[OK] saved:", outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", required=True)
    parser.add_argument("--chunk_index", type=int, required=True)     # DF/logits shard
    parser.add_argument("--dchunk", type=int, required=True)          # domain token shard index
    parser.add_argument("--n_dchunks", type=int, required=True)       # total domain shards
    parser.add_argument("--logits_dir", type=str, default="/hps/nobackup/birney/users/bonazzola/auc/full_logits")
    parser.add_argument("--output_dir", type=str, default="/hps/nobackup/birney/users/bonazzola/auc/indices")
    parser.add_argument("--prediction_domains", type=str, default=None, nargs='+')
    args = parser.parse_args()

    main(args)
    
# %%
