import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Merge SHAP chunk files by run id.")
parser.add_argument('--dry-run', action='store_true', help='Only show grouping and order, do not load or save anything')
parser.add_argument('--run-idx', type=int, default=None, help='Index of run id to process (alphabetical order). If not set, process all.')
args = parser.parse_args()

folder = os.getcwd()

# 1. Group by run id
files = [f for f in os.listdir(folder)
         if f.startswith("shap_values_chunk") and f.endswith(".pkl") and "of" in f and "chunk" in f.split("_")[2]]
grouped = defaultdict(list)
for fname in files:
    run_id = fname.rsplit("_", 1)[-1].replace(".pkl", "")
    grouped[run_id].append(fname)

# Sort run ids for consistent indexing
run_ids_sorted = sorted(grouped.keys())

# Select which run ids to process
if args.run_idx is not None:
    if args.run_idx < 0 or args.run_idx >= len(run_ids_sorted):
        raise ValueError(f"run-idx {args.run_idx} out of range (0-{len(run_ids_sorted)-1})")
    run_ids_to_process = [run_ids_sorted[args.run_idx]]
else:
    run_ids_to_process = run_ids_sorted

# 2. Process each run id
for run_id in tqdm(run_ids_to_process, desc="Run IDs"):
    flist = grouped[run_id]
    print(flist)
    # Sort by chunk number
    def chunk_number(fname):
        base = os.path.basename(fname)
        parts = base.split("_")
        if len(parts) < 3 or not parts[2].startswith("chunk"):
            return float('inf')
        try:
            chunk_part = parts[2]  # e.g. 'chunk9of1000'
            return int(chunk_part.replace("chunk", "").split("of")[0])
        except Exception:
            return float('inf')
    flist_sorted = sorted(flist, key=chunk_number)

    if args.dry_run:
        print(f"Run ID: {run_id}")
        for fname in flist_sorted:
            base = os.path.basename(fname)
            try:
                chunk_part = base.split("_")[2]
                chunk_num = int(chunk_part.replace("chunk", "").split("of")[0])
            except Exception:
                chunk_num = "?"
            print(f"  {fname}   (chunk {chunk_num})")
        continue

    # 3. Concatenate the data
    merged = {}
    for i, fname in enumerate(tqdm(flist_sorted, desc=f"Chunks for {run_id}", leave=False)):
        with open(os.path.join(folder, fname), "rb") as f:
            data = pickle.load(f)
        if i == 0:
            merged = {k: v.copy() for k, v in data.items()}
        else:
            for k in merged:
                merged[k] = np.concatenate([merged[k], data[k]], axis=0)
    # 4. Save the result
    outname = f"shap_values_merged_{run_id}.pkl"
    with open(os.path.join(folder, outname), "wb") as f:
        pickle.dump(merged, f)
    print(f"Saved: {outname} ({len(flist_sorted)} chunks)") 

