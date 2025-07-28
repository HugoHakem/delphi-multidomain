#!/bin/bash

BFILE="/nfs/research/birney/projects/association/snp_gwas/regenie/resources/ukb22828_allChr_b0_v3_maf01_04_merge"
PHENO_CSV="phenos.plink"
PHENOS=$(head -n 1 "$PHENO_CSV" | cut -d' ' -f3-)

for pheno in $PHENOS; do
    echo "Running GWAS for $pheno"

    for group in all males females; do
        case $group in
            all)
                FILTER=""
                ;;
            males)
                FILTER="--filter-males"
                ;;
            females)
                FILTER="--filter-females"
                ;;
        esac

        OUTFILE="gwas_outputs/${pheno}_${group}"
        plink --bfile "$BFILE" \
              $FILTER \
              --keep keep_white.txt \
              --pheno phenos.plink \
              --pheno-name "$pheno" \
              --linear \
              --out "$OUTFILE"
    done
done
