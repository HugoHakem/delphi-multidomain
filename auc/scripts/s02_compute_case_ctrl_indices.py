#!/usr/bin/env python3

import argparse
from pathlib import Path
from itertools import product
import os, sys

import numpy as np
import pandas as pd

if ( DELPHI_DIR := Path(__file__).resolve().parent.parent.parent ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))
    
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
    ctrl_mask = ( mask_sex & mask_age & mask_never_had_disease )
    
    # choose only one control token per subject
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


def get_predicted_domains(model):
    return [ dom for dom, cfg in model.config.domains.items() if getattr(cfg, "predict", True) ]


def compute_indices(model, tokens_df, chunk_index, disease_chunk, n_disease_chunks):
        
    domain_to_id = tokens_df[['domain_id', 'domain']].drop_duplicates().set_index("domain").domain_id.to_dict()   
    # print("[INFO] domain_to_id =", domain_to_id)

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

    for dom in model.predicted_domains:
    
        # max_token = tokens_df.query("domain == @dom").token_id.max()        
        max_token = model.vocab_lens[model.domain_to_int[dom]]
        all_tokens = np.arange(max_token + 1)
        
        my_tokens  = np.array_split(all_tokens, n_disease_chunks)[disease_chunk-1]
        print(f"[DEBUG] Domain '{dom}': max_token={max_token}, total_tokens={len(all_tokens)}, assigned_tokens={len(my_tokens)}")
        
        # if pd.isna(max_token):
            # print(f"[WARN] Domain '{dom}' has no tokens in chunk → skipping")
            # continue
            
        print(f"[INFO] chunk={chunk_index}, disease_chunk={disease_chunk}, domain={dom}, tokens={len(my_tokens)}")            

        brackets = product( my_tokens, SEX_TOKENS.items(), AGE_RANGES )     

        for token_id, (sex, sex_token), (a0, a1) in brackets:
        
            ci_arr, co_arr = get_case_ctrl_prevtoken_indices(
                tokens_df,
                token_id,
                sex_token,
                a0,
                a1,
                (domid := dom_targets[dom]),
                dom_sex,
            )
        
            row = [ dom, token_id, sex, a0, a1, ci_arr, co_arr, chunk_index, disease_chunk ]           
            rows.append(row)
    
    columns = [ "domain","token_id","sex","age_start","age_end","case_indices","ctrl_indices","chunk_index","dchunk" ]
    indices_df = pd.DataFrame(rows, columns=columns)    

    return indices_df
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--runid", required=True)
    parser.add_argument("--tokens_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--chunk_index", type=int, required=True)     # DF/logits shard
    parser.add_argument("--dchunk",      type=int, required=True)     # domain token shard index
    parser.add_argument("--n_dchunks",   type=int, required=True)     # total domain shards
    parser.add_argument("--subject_ids", type=str, required=False, default=None)    # subset of subjects to compute AUCs for 
    parser.add_argument("--prediction_domains", type=str, default=None, nargs='+')
    
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(exist_ok=True, parents=True)
    
    model, _, _, _ = reconstruct_model(args.runid)
    tokens_df = pd.read_parquet(args.tokens_file)

    if args.subject_ids:
        subject_ids = pd.read_csv(args.subject_ids)
        tokens_df = tokens_df[ tokens_df["subject_idx"].isin(subject_ids["subject_idx"]) ]

    indices_df = compute_indices(model, tokens_df, args.chunk_index, args.dchunk, args.n_dchunks)        
    
    indices_df.to_parquet(args.output_file)
    
