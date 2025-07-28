import os
import glob
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sys

def process_group(group, file_list, output_folder):
    rows = []
    for f, embedding in file_list:
        try:
            print(f)
            df = pd.read_csv(f, sep='\s+')
            df = df[["CHR", "SNP", "BP", "A1", "BETA", "P"]].copy()
            df["embedding"] = embedding
            df["group"] = group
            rows.append(df)
        except Exception as e:
            print(f"❌ Failed on {f}: {e}")

    if rows:
        full = pd.concat(rows, ignore_index=True)
        outname = os.path.join(output_folder, f"gwas_summary_{group}.parquet")
        full.to_parquet(outname, engine="pyarrow", compression="zstd", index=False)
        return f"✅ Saved {outname} ({len(full):,} rows)"
    return f"⚠️ No data for {group}"

def process_gwas_folder_parallel(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_folder, "embedding_*.assoc.linear")))
    grouped = {}

    for f in tqdm(files):
        base = os.path.basename(f)
        parts = base.replace(".assoc.linear", "").split("_")
        if len(parts) < 3:
            continue
        embedding = int(parts[1])
        group = "_".join(parts[2:])
        grouped.setdefault(group, []).append((f, embedding))

    # Ejecutar en paralelo
    with ProcessPoolExecutor() as executor:
        futures = []
        for group, file_list in grouped.items():
            futures.append(executor.submit(process_group, group, file_list, output_folder))
        for f in tqdm(futures, desc="Saving groups"):
            print(f.result())

# Uso:
if __name__ == "__main__":
    age = sys.argv[1]
    process_gwas_folder_parallel(f"gwas_outputs_{age}", f"parquets_{age}")

