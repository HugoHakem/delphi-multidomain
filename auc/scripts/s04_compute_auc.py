#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from auc_utils import compute_all_stats

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", required=True)
    parser.add_argument("--logits_parquet_dir", default="/hps/nobackup/birney/users/bonazzola/auc/logits_merged")
    parser.add_argument("--output_dir", default="auc_results")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n_bootstrap", type=int, default=200)
    args = parser.parse_args()

    runid = args.runid

    logits_path = Path(args.logits_parquet_dir) / f"logits_{runid}.parquet"
    print(f"[INFO] Loading: {logits_path}")
<<<<<<< Updated upstream
    
=======
>>>>>>> Stashed changes
    logits_df = pd.read_parquet(logits_path)
    ( out_dir := Path(args.output_dir) ).mkdir(exist_ok=True, parents=True)

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
    out_path = out_dir / f"auc_{runid}.csv"
    out_df.to_csv(out_path, index=False)

    print(f"[OK] Saved AUC results → {out_path}")


if __name__ == "__main__":
    main()
