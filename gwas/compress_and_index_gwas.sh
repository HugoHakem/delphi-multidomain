#!/bin/bash
#SBATCH --job-name=bgzip_index                     # SLURM job name
#SBATCH --output=logs/tabix/bgzip_%A_%a.out        # STDOUT log
#SBATCH --error=logs/tabix/bgzip_%A_%a.err         # STDERR log
#SBATCH --time=00:05:00                            # Max walltime
#SBATCH --mem=2G                                   # Memory per job

# === Input preparation ===
# Each SLURM array task processes one file listed in files_to_compress_test.txt
FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" files_to_compress.txt)

echo "Processing $FILE"

BASENAME=$(basename "$FILE" .assoc.linear)
DIRNAME=$(dirname "$FILE")
SORTED="$DIRNAME/${BASENAME}.sorted.assoc.linear"
GZ="$SORTED.gz"
TMP_TSV="$DIRNAME/${BASENAME}.tsv"

# === Sorting ===
# Clean header and data lines: trim whitespace, replace with single tab
# Then sort numerically by CHR (col 1) and BP (col 4)
awk '
NR==1 {
  gsub(/^[ \t]+|[ \t]+$/, "", $0);
  gsub(/[ \t]+/, "\t", $0);
  print "#" $0;
  next
}
{
  gsub(/^[ \t]+|[ \t]+$/, "", $0);
  gsub(/[ \t]+/, "\t", $0);
  print
}' "$FILE" > "$TMP_TSV"

(head -n 1 "$TMP_TSV" && tail -n +2 "$TMP_TSV" | sort -k1,1n -k3,3n) > "$SORTED"
rm "$TMP_TSV"

# === Compression ===
# Compress with bgzip (compatible with tabix indexing)
bgzip -f "$SORTED"  # -f: force overwrite if exists

# === Indexing ===
# Create tabix index:
# -s 1 → CHR in column 1
# -b 4 → BP (start) in column 4
# -e 4 → BP (end) in column 4 (same as start)
tabix -f -s 1 -b 3 -e 3 "$GZ"

echo "✅ Finished: $GZ"
