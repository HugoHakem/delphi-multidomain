# %%
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")

import time
import math
import pickle as pkl
from contextlib import nullcontext
# import ipdb

import numpy as np
import pandas as pd
import torch

from ast import literal_eval

from torch.utils.data import Dataset

from pprint import pprint
from collections import defaultdict

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional

from omegaconf import OmegaConf

import yaml
import warnings
from typing import List, Dict, Set
from delphi.data.ukb import UKBDataConfig, UKBDataset

# %%
tokens_lst = []
for fold in range(1, 6):
    file = f"/home/bonazzola/repos/delphi/data/transforms/ukb_real_5_folds_4digit/fold{fold}.bin"
    tokens = np.memmap(file, dtype=np.uint32, mode='r').reshape(-1, 3)
    tokens_lst.append(tokens)
    # open(f"fold{fold}.csv", "wt").write("\n".join([str(x) for x in np.unique(tokens[:,0])]))
# %%
# file = f"/home/bonazzola/repos/delphi/data/transforms/ukb_real_5_folds_4digit/fold1.bin"
# tokens = np.memmap(file, dtype=np.uint32, mode='r').reshape(-1, 3)
tokens = np.concatenate(tokens_lst)

# %%
tokens
# pd.read_csv("/home/bonazzola/repos/delphi/data/transforms/ukb_real_data_4digit/labels.csv")

# %%
tokens[:,2].max()

# %%
labels = pd.read_csv("/home/bonazzola/repos/delphi/data/transforms/ukb_real_data_4digit/labels.csv", header=None)
labels = list(labels.iloc[:,0].to_dict().values())

domain_ranges = { 
  "sex": [2, 3], 
  "hla_alleles": range(4, 363), 
  "lifestyle": range(363, 372), 
  "diseases": range(372, 1627), 
  "death": [1627] 
}

for domain, indices in domain_ranges.items():

    print(f"{domain}: {len(indices)}")
    # print([labels[i] for i in indices])
    print()
    domain_tokens = tokens[pd.Series(tokens[:,2]).isin(indices)]
    domain_tokens[:,2] = domain_tokens[:,2] - indices[0]
    
    print(domain_tokens)
    pd.DataFrame(domain_tokens, columns=["subject_id", "age", "token_id"]).\
        to_csv(f"../data/transforms/{domain}/tokens.csv", index=False)

# %%
hla_tokens = tokens[(tokens[:,2] >= 3) & (tokens[:,2] < 362)]
hla_tokens[:,2] = hla_tokens[:,2] - 4
pd.DataFrame(hla_tokens).to_csv("../data/transforms/hla_alleles/tokens.csv", index=False, header=["subject_id", "age", "token_id"])

lifestyle_tokens = tokens[(tokens[:,2] >= 363) & (tokens[:,2] < 372)]
lifestyle_tokens[:,2] = lifestyle_tokens[:,2] - 363
pd.DataFrame(lifestyle_tokens).to_csv("../data/transforms/lifestyle/tokens.csv", index=False, header=["subject_id", "age", "token_id"])

sex_tokens = tokens[(tokens[:,2] >= 1) & (tokens[:,2] < 3)]
sex_tokens[:,2] = sex_tokens[:, 2] - 1
pd.DataFrame(sex_tokens).to_csv("../data/transforms/sex/tokens.csv", index=False, header=["subject_id", "age", "token_id"])

disease_tokens = tokens[(tokens[:,2] >= 372) & (tokens[:,2] < 1628)]
disease_tokens[:,2] = disease_tokens[:,2] - 372
labels_diseases = labels[372:1628]
pd.Series(labels_diseases).to_csv("../data/transforms/diseases/tokenizer.yaml", index=False, header=False, sep="|")

# %%
tokens[tokens[:,2] == 4]
