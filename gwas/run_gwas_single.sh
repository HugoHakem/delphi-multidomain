#!/bin/bash
#SBATCH --job-name=gwas
#SBATCH --output=logs/gwas/gwas_%A_%a.out
#SBATCH --error=logs/gwas/gwas_%A_%a.err
#SBATCH --array=0-239
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

# Paths
AGE=20
PHENOS=phenos_${AGE}.plink
BFILE=/nfs/research/birney/projects/association/snp_gwas/regenie/resources/ukb22828_allChr_b0_v3_maf01_04_merge
KEEP=keep_white.txt

# Obtener fenotipo según ID del array
PHENO=embedding_$(printf "%03d" ${SLURM_ARRAY_TASK_ID})

echo "Running GWAS for phenotype: $PHENO"

# Todos
plink --bfile $BFILE \
      --keep $KEEP \
      --pheno $PHENOS \
      --pheno-name $PHENO \
      --linear \
      --out gwas_outputs_${AGE}/${PHENO}_white_all

# Hombres
plink --bfile $BFILE \
      --keep $KEEP \
      --filter-males \
      --pheno $PHENOS \
      --pheno-name $PHENO \
      --linear \
      --out gwas_outputs_${AGE}/${PHENO}_white_males

# Mujeres
plink --bfile $BFILE \
      --keep $KEEP \
      --filter-females \
      --pheno $PHENOS \
      --pheno-name $PHENO \
      --linear \
      --out gwas_outputs_${AGE}/${PHENO}_white_females
