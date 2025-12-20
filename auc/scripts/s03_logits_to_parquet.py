#!/usr/bin/env python3

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------
# LOAD HELPERS
# ---------------------------------------------------------------

def load_logits(runid, logits_root="outputs"):
    """
    Carga logits por chunk_index → {chunk_idx: tensor[T, D]}
    """
    base = Path(logits_root) / runid
    out = {}

    files = sorted(base.glob("chunk_*_logits.pt"))
    print(f"[INFO] Logits found: {len(files)} files")

    for f in tqdm(files, desc="Loading logits"):
        ci = int(f.stem.split("_")[1])
        out[ci] = torch.load(f, map_location="cpu")
    return out


def load_indices(runid, indices_root="indices"):
    """
    Concatena TODOS los parquets de índices del run.
    """
    files = sorted(Path(indices_root).glob(f"indices_runid_{runid}__*.parquet"))
    if not files:
        raise RuntimeError(f"No index files found for run {runid}")

    print(f"[INFO] Index files: {len(files)}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------
# MAIN MERGE LOGIC
# ---------------------------------------------------------------

def collect_logits_merged(runid, indices_root="indices_2", logits_root="outputs"):
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
                merged[key] = {
                    "case": [],
                    "ctrl": [],
                }
    
            merged[key]["case"].extend(case_vals)
            merged[key]["ctrl"].extend(ctrl_vals)
    
    return merged



# ---------------------------------------------------------------
# SAVE TO PARQUET
# ---------------------------------------------------------------

def save_merged_as_parquet(runid, merged, outdir="logits_merged"):
    Path(outdir).mkdir(exist_ok=True, parents=True)

    rows = []
    for (domain, token_id, sex, a0, a1), vals in merged.items():
        rows.append({
            "runid": runid,
            "domain": domain,
            "token_id": token_id,
            "sex": sex,
            "age_start": a0,
            "age_end": a1,
            "n_case": len(vals["case"]),
            "n_ctrl": len(vals["ctrl"]),
            "case_logits": vals["case"],
            "ctrl_logits": vals["ctrl"],
        })

    df = pd.DataFrame(rows)

    outpath = Path(outdir) / f"logits_{runid}.parquet"
    df.to_parquet(outpath, compression="zstd", engine="pyarrow")

    print(f"[OK] Saved merged logits → {outpath}")
    return outpath


# ---------------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", required=True)
    parser.add_argument("--indices_root", default="/hps/nobackup/birney/users/bonazzola/auc/indices")
    parser.add_argument("--logits_root", default="/hps/nobackup/birney/users/bonazzola/auc/full_logits")
    parser.add_argument("--outdir", default="/hps/nobackup/birney/users/bonazzola/auc/logits_merged")
    args = parser.parse_args()

    merged = collect_logits_merged(
        args.runid,
        indices_root=args.indices_root,
        logits_root=args.logits_root
    )

    save_merged_as_parquet(args.runid, merged, outdir=args.outdir)


if __name__ == "__main__":
    main()

