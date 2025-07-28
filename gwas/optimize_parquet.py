import sys
import pandas as pd
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = input_path.with_name(input_path.stem + "_optimized.parquet")

print(f"🔄 Reading {input_path}")
df = pd.read_parquet(input_path)
print(f"✅ Read {len(df)} rows")

df_sorted = df.sort_values("SNP")

print(f"💾 Writing sorted to {output_path}")
df_sorted.to_parquet(
    output_path,
    engine="pyarrow",
    compression="zstd",
    row_group_size=100_000,
    index=False
)
print(f"✅ Done: {output_path}")

