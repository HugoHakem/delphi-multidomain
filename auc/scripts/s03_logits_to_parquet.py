#!/usr/bin/env python3

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from auc_utils import compute_all_stats
from s01_compute_logits_per_chunk          import LOGITS_FILE_PATTERN
from scripts.s02_compute_case_ctrl_indices import INDICES_FILE_PATTERN

INDICES_FILE_PATTERN = INDICES_FILE_PATTERN.format(ci="*", n_chunks="*", dchunk="*", n_dchunks="*")
LOGITS_FILE_PATTERN  = LOGITS_FILE_PATTERN.format(ci="*", n_chunks="*", dchunk="*", n_dchunks="*")

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
def load_logits(runid, logits_root):
    """
    Carga logits por chunk_index → {chunk_idx: tensor[T, D]}
    """
    base = Path(logits_root) / runid
    out = {}

    files = sorted(base.glob(LOGITS_FILE_PATTERN))
    print(f"[INFO] Logits found: {len(files)} files")

    for f in tqdm(files, desc="Loading logits"):
        ci = int(f.stem.split("_")[1])
        out[ci] = torch.load(f, map_location="cpu")

    return out


def load_indices(runid, indices_root):
    """
    Concatena TODOS los parquets de índices del run.
    """
    files = sorted(Path(indices_root).glob(f"{runid}/{INDICES_FILE_PATTERN}"))
    if not files:
        raise RuntimeError(f"No index files found for run {runid}")

    print(f"[INFO] Index files: {len(files)}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


<<<<<<< Updated upstream
=======
# def get_global_token_id(domain, local_token_id):


>>>>>>> Stashed changes
def collect_logits_merged(runid, indices_root, logits_root):
    """
    Junta logits por (disease, sex, age_start, age_end).
    Cada key acumula datos de múltiples chunks.
    Devuelve un dict con vectores concatenados.
    """

    indices_root = Path(indices_root) / runid
    idx_df = load_indices(runid, indices_root)
    logits_by_chunk = load_logits(runid, logits_root)

    merged = {}   # key → {"case": list, "ctrl": list}

    # Group indices by chunk_index para minimizar cargas
    for chunk_idx, g in tqdm(idx_df.groupby("chunk_index"), desc="Processing chunks"):
    
        if chunk_idx not in logits_by_chunk:
            raise RuntimeError(f"Missing logits for chunk {chunk_idx}")
    
        logits = logits_by_chunk[chunk_idx]   # [T, D]
    
        if logits.dim() != 2:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
        for _, row in g.iterrows():
    
            domain   = row["domain"]                # string: "diseases", "death", ...
            token_id = int(row["token_id"])         # columna dentro del dominio
            global_token_id = get_global_token_id(domain, token_id)

            sex      = row["sex"]
            a0       = int(row["age_start"])
            a1       = int(row["age_end"])
    
            key = (domain, token_id, sex, a0, a1)
    
            case_idx = np.array(row["case_indices"], dtype=int)
            ctrl_idx = np.array(row["ctrl_indices"], dtype=int)
    
            # Chequear que el token_id entra en rango
            if token_id >= logits.shape[1]:
                raise ValueError(
                    f"token_id={token_id} fuera de rango en logits (shape={logits.shape})"
                )
    
            # Extraer logits
            case_vals = logits[case_idx, token_id].tolist() if len(case_idx) else []
            ctrl_vals = logits[ctrl_idx, token_id].tolist() if len(ctrl_idx) else []
    
            # Inicializar el entry si no existe
            if key not in merged:
                merged[key] = { "case": [], "ctrl": [] }
    
            merged[key]["case"].extend(case_vals)
            merged[key]["ctrl"].extend(ctrl_vals)
    
    return merged


def save_merged_as_parquet(runid, merged, outdir):
    
    rows = []
    for (domain, token_id, sex, a0, a1), vals in merged.items():
        rows.append({ 
          "runid": runid, "domain": domain, "token_id": token_id, "sex": sex, "age_start": a0, "age_end": a1,
          "n_case": len(vals["case"]), "n_ctrl": len(vals["ctrl"]), "case_logits": vals["case"], "ctrl_logits": vals["ctrl"],
        })

    df = pd.DataFrame(rows)

    ( outdir := Path(outdir) / runid ).mkdir(exist_ok=True, parents=True)        
    outpath = outdir / "logits.parquet"
    df.to_parquet(outpath, compression="zstd", engine="pyarrow")
    print(f"[OK] Saved merged logits → {outpath}")
    return df, outpath


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", required=True)
    parser.add_argument("--indices_root", default="/hps/nobackup/birney/users/bonazzola/auc/indices")
    parser.add_argument("--logits_root", default="/hps/nobackup/birney/users/bonazzola/auc/full_logits")
    parser.add_argument("--logits_merged_outdir", default="/hps/nobackup/birney/users/bonazzola/auc/logits_merged")
    parser.add_argument("--auc_output_dir", default="auc_results")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n_bootstrap", type=int, default=200)
    args = parser.parse_args()

    # Merge logits
    merged = collect_logits_merged( args.runid, indices_root=args.indices_root, logits_root=args.logits_root )    
<<<<<<< Updated upstream

    logits_df, output_path = save_merged_as_parquet(args.runid, merged, outdir=args.logits_merged_outdir)

    # Compute AUCs
    ( auc_output_dir := Path(args.auc_output_dir) / args.runid ).mkdir(exist_ok=True, parents=True)

    results = []

    for _, row in tqdm(logits_df.iterrows(), total=logits_df.shape[0]):
        
        token = row["domain"], row["token_id"]
        sex = row["sex"]
        age_bin = row["age_start"], row["age_end"]
        
        print("[INFO] Computing AUC / Mann–Whitney for ...")

        case_logits, ctrl_logits = row["case_logits"], row["ctrl_logits"]
        
        stats = compute_all_stats( case_logits, ctrl_logits, do_bootstrap=args.bootstrap, n_bootstrap=args.n_bootstrap )

        results.append({ 
            "runid": args.runid, 
            "domain": token[0], "token_id": token[1], 
            "sex": sex, "age_start": age_bin[0], "age_end": age_bin[1], 
            "n_case": row["n_case"], "n_ctrl": row["n_ctrl"],
            **stats
        })

    out_df = pd.DataFrame(results)
    out_path = auc_output_dir / f"aucs.csv"
    out_df.to_csv(out_path, index=False)
=======
    logits_df, output_path = save_merged_as_parquet(args.runid, merged, outdir=args.logits_merged_outdir)

    # Compute AUCs
    ( auc_output_dir := Path(args.auc_output_dir) / args.runid ).mkdir(exist_ok=True, parents=True)
>>>>>>> Stashed changes

    results = []
    for _, row in tqdm(logits_df.iterrows(), total=logits_df.shape[0]):
        
        token = row["domain"], row["token_id"]
        sex = row["sex"]
        age_bin = row["age_start"], row["age_end"]
        
        print("[INFO] Computing AUC / Mann–Whitney for ...")

        case_logits, ctrl_logits = row["case_logits"], row["ctrl_logits"]
        
        stats = compute_all_stats( case_logits, ctrl_logits, do_bootstrap=args.bootstrap, n_bootstrap=args.n_bootstrap )

        results.append({ 
            "runid": runid, 
            "domain": token[0], "token_id": token[1], 
            "sex": sex, "age_start": age_bin[0], "age_end": age_bin[1], 
            "n_case": row["n_case"], "n_ctrl": row["n_ctrl"],
            **stats
        })

    out_df = pd.DataFrame(results)
    out_path = auc_output_dir / f"aucs.csv"
    out_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
