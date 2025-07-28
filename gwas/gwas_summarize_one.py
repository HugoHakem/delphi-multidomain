# summarize_one.py
import pandas as pd
import numpy as np
import os
import re
import sys

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
        outname = os.path.join(output_folder, f"embedding_{component}_{group}_chr{chrom}_{bin_size // 1_000_000}Mb.csv")
        pd.DataFrame(rows).to_csv(outname, index=False)
        print(f"✅ Saved: {outname}")

def summarize_file(path, bin_size=1_000_000):
    pattern = re.compile(r"embedding_(\d+)_([^.]+)\.assoc\.linear")
    base = os.path.basename(path)
    match = pattern.match(base)
    if not match:
        print(f"⚠️ Skipping: {base}")
        return

    component = int(match.group(1))
    group = match.group(2)
    input_folder = os.path.dirname(path)
    output_folder = os.path.join(input_folder, "summaries")
    os.makedirs(output_folder, exist_ok=True)

    try:
        df = pd.read_csv(path, sep=r"\s+")
        df = df[df["P"] > 0].dropna(subset=["CHR", "BP", "P", "SNP"])
        df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce").astype(int)

        for chrom in sorted(df["CHR"].unique()):
            bin_gwas_by_megabase(df, component, group, bin_size, output_folder, chrom)

    except Exception as e:
        print(f"❌ Error processing {base}: {e}")

if __name__ == "__main__":
    path = sys.argv[1]
    summarize_file(path)

