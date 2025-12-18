#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import mlflow
import re
import ast
from easydict import EasyDict
import os, sys

DELPHI_DIR = f"{os.getenv('HOME')}/repos/delphi"

sys.path.insert(0, DELPHI_DIR)
from delphi.model.transformer import Delphi, DelphiConfig, EmbedConfig
from data.event_set import EventSet

SEX_TOKENS = {"female": 0, "male": 1}
AGE_RANGES = [(a, a+5) for a in range(0, 90, 5)]
MLFLOW_URI = f"{os.environ['HOME']}/repos/delphi/train_scripts/mlruns"

def infer_delphi_config_from_state_dict(sd):

    cfg = EasyDict()

    # capas
    layer_indices = []
    for k in sd:
        m = re.match(r"transformer\.h\.(\d+)\.", k)
        if m:
            layer_indices.append(int(m.group(1)))
    cfg.n_layer = max(layer_indices) + 1

    for k, v in sd.items():
        if "attn.c_attn.weight" in k:
            cfg.n_embd = v.shape[1]
            break

    # dominios detectados
    cfg.domains = []
    for k in sd:
        m = re.match(r"transformer\.embed\.domain_embed\.(\w+)\.projector\.weight", k)
        if m:
            cfg.domains.append(m.group(1))

    return cfg


# 2) Cargar run, checkpoint y reconstruir modelo
def setup_mlflow():    
    mlflow.set_tracking_uri(uri:=MLFLOW_URI)
    return uri


def load_run_info(run_id):
    mlflow.set_tracking_uri(MLFLOW_URI)
    run = mlflow.get_run(run_id)
    params = run.data.params
    attn_scheme = ast.literal_eval(params["attention_scheme"])
    return params, attn_scheme


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



def load_checkpoint(mlruns_uri, experiment_id, run_id):
    ckpt_path = get_last_epoch_checkpoint(mlruns_uri / experiment_id / run_id)[0]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt, ckpt_path


def build_domain_config(config, tokens_path: Path):

    default_cfg = {
    # 'genetic_pcs': EmbedConfig(projector="linear", path=tokens_path / 'genetic_pcs', type='continuous', at_birth=True),
      'diseases':    EmbedConfig(projector="embed", path=tokens_path / 'diseases',    predict=True),
      'death':       EmbedConfig(projector="embed", path=tokens_path / 'death',       predict=True),
      'cv_drugs':    EmbedConfig(projector="embed", path=tokens_path / 'cv_drugs',    predict=True),
      'ns_drugs':    EmbedConfig(projector="embed", path=tokens_path / 'ns_drugs',    predict=True),
      'lifestyle':   EmbedConfig(projector="embed", path=tokens_path / 'lifestyle',   age_jitter=True),  
      "hla_alleles": EmbedConfig(projector="embed", path=tokens_path / 'hla_alleles', at_birth=True),
      "sex":         EmbedConfig(projector="embed", path=tokens_path / 'sex',         at_birth=True),
      "padding":     EmbedConfig(projector="embed")    
    }    

    return {name: default_cfg[name] for name in config.domains}

def reconstruct_model(run_id):

    mlflow.set_tracking_uri(MLFLOW_URI)
    runinfo = mlflow.get_run(run_id)

    params = runinfo.data.params
    attn_scheme = ast.literal_eval(params["attention_scheme"])

    experiment_id = runinfo.info.experiment_id

    ckpt_path = get_last_epoch_checkpoint(
        Path(MLFLOW_URI) / experiment_id / run_id
    )[0]

    ckpt = torch.load(ckpt_path, map_location="cpu")

    weights = ckpt["state_dict"]
    test_ids = ckpt["metadata"]["test_ids"]

    # 5) inferir config del modelo
    cfg_inf = infer_delphi_config_from_state_dict(weights)

    tokens_path = Path(f"{DELPHI_DIR}/data/transforms") / "tokens"
    domain_cfg_full = build_domain_config(cfg_inf, tokens_path)

    delphi_cfg = DelphiConfig(
        n_embd=cfg_inf.n_embd,
        n_layer=cfg_inf.n_layer,
        token_dropout=0.1,
        domains=domain_cfg_full,
        attention_scheme=attn_scheme,
    )

    model = Delphi(delphi_cfg)
    model.load_state_dict(weights)
    model.eval()

    return model, test_ids, params, experiment_id, ckpt_path


def get_subject_sex(df, dom_sex):
    rows = df[df["domain_id"] == dom_sex]
    return (
        rows.groupby("subject_idx")["token_id"]
        .first()
        .to_dict()
    )


def get_case_ctrl_prevtoken_indices(
    df,
    disease_id,
    sex_token_id,
    age_min,
    age_max,
    dom_diseases,
    dom_sex
):

    age_years   = df["age"].to_numpy() / 365.25
    subj_arr    = df["subject_idx"].to_numpy()
    token_arr   = df["token_id"].to_numpy()
    dom_arr     = df["domain_id"].to_numpy()
    seq_arr     = df["seq_idx"].to_numpy()
    global_arr  = df["global_idx"].to_numpy()

    subj2sex = get_subject_sex(df, dom_sex)
    subj_sex_arr = np.array([subj2sex.get(s, -1) for s in subj_arr])

    mask_sex = (subj_sex_arr == sex_token_id)
    mask_age = (age_years >= age_min) & (age_years < age_max)
    mask_dom = (dom_arr == dom_diseases)

    case_mask = (
        (token_arr == disease_id)
        & mask_dom
        & mask_sex
        & mask_age
    )

    case_subjects = subj_arr[case_mask]
    case_seq      = seq_arr[case_mask]

    prev_subj = []
    prev_seq  = []

    for s, t in zip(case_subjects, case_seq):
        if t > 0:
            prev_subj.append(s)
            prev_seq.append(t - 1)

    if len(prev_subj) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    prev_subj = np.array(prev_subj)
    prev_seq  = np.array(prev_seq)

    prev_mask = (np.isin(subj_arr, prev_subj) & np.isin(seq_arr, prev_seq))
    case_prev_global_idx = global_arr[prev_mask]

    subjects_with_dis = np.unique(subj_arr[case_mask])

    all_subjects = np.unique(subj_arr)
    subjects_without_dis = np.setdiff1d(all_subjects, subjects_with_dis)

    ctrl_mask = (
        mask_sex
        & mask_age
        & mask_dom
        & np.isin(subj_arr, subjects_without_dis)
    )

    ctrl_global_idx = global_arr[ctrl_mask]

    return case_prev_global_idx, ctrl_global_idx


def nested3_to_df(d):
    rows = []
    for disease, v1 in d.items():
        for sex, v2 in v1.items():
            for (a0, a1), (case_idx, ctrl_idx) in v2.items():
                rows.append({
                    "disease": disease,
                    "sex": sex,
                    "age_start": a0,
                    "age_end": a1,
                    "case_indices": case_idx,
                    "ctrl_indices": ctrl_idx
                })
    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", required=True)
    parser.add_argument("--chunk_index", type=int, required=True)     # DF/logits shard
    parser.add_argument("--dchunk", type=int, required=True)          # domain token shard index
    parser.add_argument("--n_dchunks", type=int, required=True)       # total domain shards
    parser.add_argument("--logits_dir", type=str, default="/hps/nobackup/birney/users/bonazzola/auc/full_logits")
    parser.add_argument("--output_dir", type=str, default="/hps/nobackup/birney/users/bonazzola/auc/indices")
    parser.add_argument("--prediction_domains", type=str, default=None, nargs='+')
    args = parser.parse_args()

    runid, ci, dchunk, n_dchunks = args.runid, args.chunk_index, args.dchunk, args.n_dchunks
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    runinfo = mlflow.get_run(runid)
    experiment_id = runinfo.info.experiment_id

    outdir = Path(args.output_dir) / runid
    outdir.mkdir(exist_ok=True, parents=True)

    logits_dir = Path(args.logits_dir) / runid
    
    pattern = f"chunk_{ci}_of_*_df.parquet"
    matches = list(logits_dir.glob(pattern))
    
    if len(matches) == 0:
        raise RuntimeError(f"No DF file found for chunk_index={ci} in {logits_dir}")
    
    if len(matches) > 1:
        print("[WARN] Multiple matches found, using first:", matches)
    
    df_path = matches[0]
    
    fname_parts = df_path.stem.split("_")  
    n_chunks = int(fname_parts[3])
    
    print(f"[INFO] Found DF: {df_path}  → n_chunks = {n_chunks}")
    
    df = pd.read_parquet(df_path)
    df = df.assign(age=df.age_days)

    model, test_ids, params, experiment_id, ckpt_path = reconstruct_model(runid)
    domain_cfg = model.config.domains  # EXACT order and predict flags

    domain_to_id = df[['domain_id', 'domain']].drop_duplicates().set_index("domain").domain_id.to_dict()
    
    print("[INFO] domain_to_id =", domain_to_id)

    assert "sex" in domain_to_id, "Model does not contain 'sex' domain."
    dom_sex = domain_to_id["sex"]
    
    if args.prediction_domains is None or len(args.prediction_domains) == 0:
        prediction_domains = [
            dom for dom, cfg in domain_cfg.items()
            if getattr(cfg, "predict", True)
        ]
        print("[INFO] Using model-predicted domains:", prediction_domains)
    else:
        prediction_domains = args.prediction_domains
        print("[INFO] Using user-specified domains:", prediction_domains)

    for d in prediction_domains:
        if d not in domain_to_id:
            raise RuntimeError(f"Domain '{d}' not present in model.")

    dom_targets = { dom: domain_to_id[dom] for dom in prediction_domains }
    
    case_ctrl = {}
    
    for dom in prediction_domains:
    
        domid = dom_targets[dom]
    
        # Cantidad de tokens reales en este dominio dentro del chunk (0..max)
        max_token = df.loc[df["domain_id"] == domid, "token_id"].max()
        if pd.isna(max_token):
            print(f"[WARN] Domain '{dom}' has no tokens in chunk → skipping")
            continue
    
        all_tokens = np.arange(max_token + 1)
    
        # SLURM split
        token_blocks = np.array_split(all_tokens, n_dchunks)
        my_tokens = token_blocks[dchunk]
    
        print(f"[INFO] chunk={ci}, dchunk={dchunk}, domain={dom}, tokens={len(my_tokens)}")
    
        case_ctrl[dom] = {}
    
        for tok in tqdm(my_tokens, desc=f"{dom}-dchunk-{dchunk}"):
            case_ctrl[dom][tok] = {}
    
            for sex, sex_token in SEX_TOKENS.items():
                case_ctrl[dom][tok][sex] = {}
    
                for (a0, a1) in AGE_RANGES:
    
                    ci_arr, co_arr = get_case_ctrl_prevtoken_indices(
                        df=df,
                        disease_id=tok,         # not only diseases
                        sex_token_id=sex_token,
                        age_min=a0,
                        age_max=a1,
                        dom_diseases=domid,
                        dom_sex=dom_sex,
                    )
    
                    case_ctrl[dom][tok][sex][(a0, a1)] = (ci_arr, co_arr)
    
    # Convertimos el dict anidado en un DataFrame
    rows = []
    for dom, v_dom in case_ctrl.items():
        for tok, v_tok in v_dom.items():
            for sex, v_sex in v_tok.items():
                for (a0, a1), (ci_arr, co_arr) in v_sex.items():
                    rows.append({
                        "domain": dom,
                        "token_id": tok,
                        "sex": sex,
                        "age_start": a0,
                        "age_end": a1,
                        "case_indices": ci_arr,
                        "ctrl_indices": co_arr
                    })
    
    df_out = pd.DataFrame(rows)
    df_out["chunk_index"] = ci
    df_out["dchunk"] = dchunk
    
    outfile = outdir / f"indices_runid_{runid}__chunk_{ci}__dchunk_{dchunk}.parquet"
    df_out.to_parquet(outfile)
    
    print("[OK] saved:", outfile)


if __name__ == "__main__":
    main()
