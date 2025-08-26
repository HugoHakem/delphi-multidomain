# %%
import os, sys
os.getcwd()
sys.path.append("gwas")
import pandas as pd

from importlib import reload 
import gwas_covariates_helpers

reload(gwas_covariates_helpers)
gcov = gwas_covariates_helpers

# %%
import yaml

yaml_str = '''
../data/datasets/genetic_pcs_22009.txt:
  - id: "f.eid"
  - f.22009.0.1: pc0
  - f.22009.0.2: pc1
  - f.22009.0.3: pc2
  - f.22009.0.4: pc3
  - f.22009.0.5: pc4
  - f.22009.0.6: pc5
  - f.22009.0.7: pc6
  - f.22009.0.8: pc7
  - f.22009.0.9: pc8
  - f.22009.0.10: pc9
  - f.22009.0.11: pc10
  - f.22009.0.12: pc11
  - f.22009.0.13: pc12
  - f.22009.0.14: pc13
  - f.22009.0.15: pc14
  - f.22009.0.16: pc15
  - f.22009.0.17: pc16
  - f.22009.0.18: pc17
  - f.22009.0.19: pc18
  - f.22009.0.20: pc19
  - f.22009.0.21: pc20
  - f.22009.0.22: pc21
  - f.22009.0.23: pc22
  - f.22009.0.24: pc23
  - f.22009.0.25: pc24
  - f.22009.0.26: pc25
  - f.22009.0.27: pc26
  - f.22009.0.28: pc27
  - f.22009.0.29: pc28
  - f.22009.0.30: pc29
  - f.22009.0.31: pc30
  - f.22009.0.32: pc31
  - f.22009.0.33: pc32
  - f.22009.0.34: pc33
  - f.22009.0.35: pc34
  - f.22009.0.36: pc35
  - f.22009.0.37: pc36
  - f.22009.0.38: pc37
  - f.22009.0.39: pc38
  - f.22009.0.40: pc39

../data/datasets/sex_31.txt:
  - id: "f.eid"
  - f.31.0.0: sex
'''

covariates = yaml.safe_load(yaml_str)
print(covariates)

cov_df = gcov.generate_covariates_df(covariates_config=covariates)

# %%
ethn_df = pd.read_csv("~/Delphi/data/datasets/22006.csv")
ethn_df = ethn_df.rename({"eid": "ID", "22006-0.0": "white"}, axis=1)
ethn_df.white = ethn_df.white.notnull().astype(int)

emb120_df = pd.read_csv("/home/bonazzola/Delphi/embeddings/data/embedding_120/embeddings_20.csv")
emb120_df

# %%
all_df = pd.concat([
    emb120_df.rename({"subject_id": "ID"}, axis=1).set_index("ID").rename_axis(None).rename(index=lambda x: int(x)),
    cov_df.set_index("ID").rename_axis(None).rename(index=lambda x: int(x)),
    ethn_df.set_index("ID").rename_axis(None).rename(index=lambda x: int(x))
], axis=1)

all_df = all_df.query("white == 1").drop("white", axis=1)
# %%
all_df
# %%
"1".zfill(4)
# %%
from tqdm import tqdm
fit = {}
for j in tqdm(range(120)):
    fit[j] = gcov.fit_linear_model(all_df, f'embedding_{str(j).zfill(3)}', [f'pc{i}' for i in range(40)] + ['sex'])

# %%
fit.summary()

# %%
