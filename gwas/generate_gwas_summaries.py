import pandas as pd
import numpy as np
import os
import glob
import re

INPUT_FOLDER = "gwas_outputs_20"
OUTPUT_FOLDER = "summaries"
BIN_SIZE = 1_000_000  # 1 Mb

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def bin_gwas_by_megabase(df, component, group, bin_size, output_folder, chrom):
    chr_df = df[df["CHR"] == chrom]
    max_bp = chr_df["BP"].max()
    bins = range(0, int(max_bp) + bin_size, bin_size)
    rows = []

    for start in bins:
        end = start + bin_size
        region = chr_df[(chr_df["BP"] >= start) & (chr_df["BP"] < end)]
        if not region.empty:
            best = region.loc[region["P"].idxmin()]
            rows.append({
                "CHR": chrom,
                "BIN_START": start,
                "BIN_END": end,
                "POS": best["BP"],
                "SNP": best["SNP"],
                "P": best["P"]
            })

    if rows:
        outname = f"{output_folder}/embedding_{component}_{group}_chr{chrom}_{bin_size // 1_000_000}Mb.csv"
        pd.DataFrame(rows).to_csv(outname, index=False)
        print(f"✅ Saved: {outname}")

def process_all_files(input_folder, output_folder, bin_size):
    files = glob.glob(os.path.join(input_folder, "*.assoc.linear"))
    pattern = re.compile(r"embedding_(\d+)_([^.]+)\.assoc\.linear")

    for path in files:
        base = os.path.basename(path)
        match = pattern.match(base)
        if not match:
            print(f"⚠️  Skipping: {base}")
            continue

        component = int(match.group(1))
        group = match.group(2)

        try:
            df = pd.read_csv(path, sep=r"\s+")
            df = df[df["P"] > 0].dropna(subset=["CHR", "BP", "P", "SNP"])
            df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce").astype(int)

            for chrom in sorted(df["CHR"].unique()):
                bin_gwas_by_megabase(df, component, group, bin_size, output_folder, chrom)

        except Exception as e:
            print(f"❌ Error processing {base}: {e}")

# Ejecutar
if __name__ == "__main__":
    process_all_files(INPUT_FOLDER, OUTPUT_FOLDER, BIN_SIZE)

