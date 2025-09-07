# %%
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import re
import glob

client = MlflowClient()
HLA_SCORE_CSV = "hla_score_per_icd10_with_justification_COMPLETE.csv"


# %%
def fix_artifact_uri(artifact_dir, on_codon=False):
    if not on_codon:
        artifact_dir = artifact_dir.replace("/homes", "/home")
        artifact_dir = artifact_dir.replace('/nfs/research/birney/users', "/home")
    return artifact_dir


def get_auc_dfs(run, suffix=""):
    artifact_dir = fix_artifact_uri(run.info.artifact_uri, on_codon="codon" in os.environ.get('HOSTNAME', ''))
    AUCDIR = f"{artifact_dir}/auc/"
    try:
        unpooled_auc = pd.read_parquet(f"{AUCDIR}/df_auc_unpooled{suffix}.parquet").query("n_diseased > 20")
    except FileNotFoundError:
        unpooled_auc = pd.read_parquet(f"{AUCDIR}/df_auc_unpooled.parquet").query("n_diseased > 20")
    try:
        both_auc = pd.read_parquet(f"{AUCDIR}/df_both{suffix}.parquet")
    except FileNotFoundError:
        both_auc = pd.read_parquet(f"{AUCDIR}/df_both.parquet")
    unpooled_auc = unpooled_auc.drop(["auc"], axis=1)
    unpooled_auc = unpooled_auc[~unpooled_auc.duplicated()]
    unpooled_auc = unpooled_auc.drop(['ICD-10 Chapter (short)', 'color'], axis=1)
    # unpooled_auc = unpooled_auc.drop(['auc_variance_delong', 'count'], axis=1)
    return unpooled_auc, both_auc


def load_5fold_cv_auc():

    EXP_NOHLA = "437945341875567335"
    EXP_HLA   = "278131880607437980"

    runs_hla_df = mlflow.search_runs(experiment_ids=[EXP_HLA])
    runs_nohla_df = mlflow.search_runs(experiment_ids=[EXP_NOHLA])
    
    unpooled_auc_mergeds, both_auc_mergeds = [], []

    for fold_i in range(1, 6):
        fold_i = str(fold_i)
        run_nohla_id = runs_nohla_df.query("`params.fold` == @fold_i").run_id.iloc[0]
        run_hla_id = runs_hla_df.query("`params.fold` == @fold_i").run_id.iloc[0]
        run_nohla  = client.get_run(run_nohla_id)
        run_hla  = client.get_run(run_hla_id)
        unpooled_auc_nohla, both_auc_nohla = get_auc_dfs(run_nohla, suffix="_onlywhite")
        unpooled_auc_hla, both_auc_hla = get_auc_dfs(run_hla, suffix="_onlywhite")

        unpooled_auc_merged = pd.merge(unpooled_auc_hla, unpooled_auc_nohla, on=['age', 'name', 'sex'], suffixes=['_hla', '_nohla']).\
            drop(["n_healthy_hla", "n_healthy_nohla"], axis=1).\
            assign(diff=lambda x: x.auc_delong_hla - x.auc_delong_nohla).\
            sort_values("diff", ascending=False).\
            merge(pd.read_csv(HLA_SCORE_CSV), left_on="index_nohla", right_on="index")# .\
            # drop(["index_hla", "index_nohla", "token_hla", "token_nohla", "n_diseased_hla", "n_diseased_nohla"], axis=1).\
            # loc[:, ["age", "sex", "name", "auc_delong_hla", "auc_delong_nohla", "diff", 'score', 'genes', 'justification']]
        
        cols_to_discard = ['color', 'ICD-10 Chapter', 'index',
            #'auc_variance_delong_nohla', 'auc_variance_delong_hla', 
            # 'n_samples_hla', 'n_diseased_hla', 'n_healthy_hla', 'index_hla', 'count_hla', 'token_hla', 
            # 'n_samples_nohla', 'n_diseased_nohla', 'n_healthy_nohla', 'index_nohla', 'count_nohla', 'token_nohla'
        ]

        both_auc_merged = pd.merge(both_auc_hla, both_auc_nohla, on=['name', 'ICD-10 Chapter', 'ICD-10 Chapter (short)', 'color'], suffixes=['_hla', '_nohla']).\
            assign(diff=lambda x: x.auc_hla - x.auc_nohla).\
            sort_values("diff", ascending=False).\
            merge(pd.read_csv(HLA_SCORE_CSV), left_on="index_nohla", right_on="index").\
            drop(cols_to_discard, axis=1)

        unpooled_auc_mergeds.append(unpooled_auc_merged.assign(fold=fold_i))
        both_auc_mergeds.append(both_auc_merged.assign(fold=fold_i)) 
    
    unpooled_auc_mergeds = pd.concat(unpooled_auc_mergeds)
    both_auc_mergeds = pd.concat(both_auc_mergeds)

    return unpooled_auc_mergeds, both_auc_mergeds


# %%
unpooled_auc_both_df, auc_both_df = load_5fold_cv_auc()

# %%
auc_fold1 = unpooled_auc_both_df.query("fold == '1'")
auc_fold2 = unpooled_auc_both_df.query("fold == '5'")

mean_auc_fold1 = auc_fold1.groupby(["name"]).auc_delong_hla.mean()
mean_auc_fold2 = auc_fold2.groupby(["name"]).auc_delong_hla.mean()

sns.scatterplot(
    x=mean_auc_fold1,
    y=mean_auc_fold2,
    alpha=0.5  # Agrega transparencia a los puntos
)

min_val = min(mean_auc_fold1.min(), mean_auc_fold2.min())
max_val = max(mean_auc_fold1.max(), mean_auc_fold2.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='x = y')

plt.xlabel("Using pseudosequences")
plt.ylabel("Using 2-field nomenclature")
plt.legend()

# %%
unpooled_auc_both_df.sort_values("diff", ascending=False).\
    drop(["auc_variance_delong_hla", "auc_variance_delong_nohla", "index_hla", "index_nohla", "token_hla", "token_nohla", "n_diseased_hla", "n_diseased_nohla"], axis=1).\
    groupby(["name", "sex", "age"]).agg({'diff': 'mean'}).sort_values("diff", ascending=False).reset_index().to_csv("DeltaAUC.csv", index=False)
