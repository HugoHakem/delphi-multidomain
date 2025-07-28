import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from biomart import BiomartServer
from gprofiler.gprofiler import GProfiler

### ----------------------------
### A. Gene TSS and Nearby SNPs
### ----------------------------

BFILE = "/nfs/research/birney/projects/association/snp_gwas/regenie/resources/ukb22828_allChr_b0_v3_maf01_04_merge"
bim_path = f"{BFILE}.bim"
bim = pd.read_csv(bim_path, sep='\s+', header=None)

parquet_files = glob("parquets_*/gwas_summary*_optimized.parquet")

from biomart import BiomartServer
import gzip

def get_gene_tss(gene_name):
    server = BiomartServer("http://grch37.ensembl.org/biomart")
    dataset = server.datasets['hsapiens_gene_ensembl']

    response = dataset.search({
        'filters': {'external_gene_name': gene_name},
        'attributes': [
            'chromosome_name',
            'transcription_start_site',
            'strand',
            'external_gene_name'
        ]
    })

    lines = response.content.decode().strip().split("\n")
    rows = [line.split("\t") for line in lines]
    rows = [r for r in rows if r[0].isdigit() or r[0] in ['X', 'Y']]
    rows.sort(key=lambda x: int(x[1]) * x[2])

    if not rows:
        return None

    chrom, tss, strand, _ = rows[0]
    return chrom, int(tss), int(strand)


def get_snps_near_tss(gene_name, window=50_000):
    """
    Filter SNPs in a BIM file that fall within a window around the gene TSS.
    """

    chrom, tss, strand = get_gene_tss(gene_name)    
    bim.columns = ['chrom', 'snp', 'cm', 'pos', 'a1', 'a2']
    bim_chr = bim[bim['chrom'].astype(str) == str(chrom)]
    mask = (bim_chr['pos'] >= tss - window) & (bim_chr['pos'] <= tss + window)
    return bim_chr[mask]


def query_snps(snp_list):

    dfs = []
    for path in parquet_files:
        dataset = ds.dataset(path, format="parquet")
        table = dataset.to_table(filter=ds.field("SNP").isin(snp_list))
        df = table.to_pandas()
        if not df.empty:
            df["source_file"] = path
            dfs.append(df)
    
    df_final = pd.concat(dfs, ignore_index=True)
    return(df_final)

### -------------------------------
### B. Plot SNP association by age
### -------------------------------

def plot_snp_associations(snp_id, assoc_df, outdir="figures/"):
    """
    Plot the -log10(p-values) of a given SNP across age strata.
    Colors reflect the strength of association (blue = significant).
    """
    df = assoc_df[assoc_df['snp'] == snp_id]
    if df.empty:
        print(f"SNP {snp_id} not found.")
        return

    df = df.sort_values('age')
    pvals = df['pval'].values
    ages = df['age'].values
    neglogp = -np.log10(pvals)

    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=0, vmax=-np.log10(5e-8))

    fig, ax = plt.subplots(figsize=(6, 2))
    sc = ax.scatter(ages, neglogp, c=neglogp, cmap=cmap, norm=norm, edgecolor='k')
    ax.axhline(-np.log10(5e-8), color='gray', linestyle='--', label='Genome-wide sig.')
    ax.set_title(snp_id)
    ax.set_xlabel("Age")
    ax.set_ylabel("-log10(p-value)")
    plt.colorbar(sc, ax=ax, label='-log10(p)')
    plt.tight_layout()
    fig.savefig(f"{outdir}/{snp_id}.png", dpi=200)
    plt.close()


### ----------------------------------------------
### C. Visualize top ICD-10 tokens for a variable
### ----------------------------------------------

def plot_top_icd10_terms(embedding_matrix, var_idx, icd10_map, top_n=20, title=None):
    """
    Plot the top contributing ICD-10 terms for a given variable (embedding).
    """
    weights = embedding_matrix[var_idx]
    top_idx = np.argsort(np.abs(weights))[::-1][:top_n]
    top_weights = weights[top_idx]
    top_terms = icd10_map.iloc[top_idx]

    plt.figure(figsize=(8, 6))
    plt.barh(top_terms['code'], top_weights, color='blue')
    plt.gca().invert_yaxis()
    plt.title(title or f"Top ICD-10 terms for variable {var_idx}")
    plt.xlabel("Weight")
    plt.tight_layout()
    plt.show()


### ------------------------------------------------------
### D. Nearby gene mapping and enrichment using g:Profiler
### ------------------------------------------------------

def get_significant_snps(df_assoc, pval_thresh=5e-8):
    """
    Return SNPs with p-values below the given threshold.
    """
    return df_assoc[df_assoc['P'] < pval_thresh]


def map_snps_to_nearby_genes(signif_snps_df, bim_df, biomart_genes, window=100_000):
    """
    Map significant SNPs to nearby genes using genomic distance.
    """
    merged = pd.merge(signif_snps_df[['SNP']], bim_df[['snp', 'chrom', 'pos']],
                      left_on='SNP', right_on='snp')
    results = []
    for _, row in merged.iterrows():
        snp_chr = str(row['chrom'])
        snp_pos = int(row['pos'])
        nearby = biomart_genes[
            (biomart_genes['chromosome'] == snp_chr) &
            (np.abs(biomart_genes['start'] - snp_pos) <= window)
        ]
        for _, gene in nearby.iterrows():
            results.append(gene['gene_name'])
    return list(set(results))


def enrich_genes(gene_list, organism="hsapiens"):
    """
    Run gene set enrichment analysis using g:Profiler.
    """
    gp = GProfiler(return_dataframe=True)
    results = gp.profile(organism=organism, query=gene_list)
    return results

