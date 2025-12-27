# %%
import os, sys
from pathlib import Path
from tqdm import tqdm

from loguru import logger

import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
torch.set_grad_enabled(False)

import argparse

import numpy as np
import pandas as pd

import torch
torch.set_grad_enabled(False)

if ( DELPHI_DIR := Path(__file__).resolve().parent.parent.parent ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from data.dataset import DelphiDataset, DelphiDataloader
from data.event_set import EventSet
from utils.utils import reconstruct_model

device = 'cpu'

BLOCK_SIZE = 128
BATCH_SIZE = 16

TOKENS_FILE_PATTERN = "chunk_{shard_id}_of_{n_chunks}_df.parquet"
LOGITS_FILE_PATTERN = "chunk_{shard_id}_of_{n_chunks}_logits.pt"


def split_subjects(subjects, n_chunks, chunk_id):
    indices = np.array_split(np.arange(len(subjects)), n_chunks)
    idx = indices[chunk_id]
    return [subjects[i] for i in idx]


def token_df_from_dataloder(dataset, batch_size):

    dataloader = DelphiDataloader(dataset, batch_size=batch_size, shuffle=False)

    offset, all_rows = 0, []
    for bi, batch in tqdm(enumerate(dataloader)):
        es = EventSet(batch)
        es = es.insert_no_event_tokens(rate=5)
        es = es.adjust_to_seqlen(seqlen=BLOCK_SIZE, pad_domain="padding", trim_domains={"diseases"}, PADDING_TOKEN=0, PAD_AGE=-10000.0, mode="fast")
        df = es.merge_domains(as_dataframe=True)
        df = df.sort_values(["subject_idx", "seq_idx"]).reset_index(drop=True)
        df["global_idx"] = range(offset, offset + len(df))
        offset = offset + len(df)
        all_rows.append(df)

    df = pd.concat(all_rows, ignore_index=True)
    return df
  
 
def running_in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


if not running_in_notebook():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", "--run-id", "--run_id", dest="runid", type=str, default=None)
    parser.add_argument("--chunk_index", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--logits_file", type=str, default=None)
    parser.add_argument("--tokens_file", type=str, default=None)
    
    args = parser.parse_args()
    
    shard_id = args.chunk_index
    n_chunks = args.n_chunks
    runid    = args.runid

    if args.logits_file is None:
        logits_file = LOGITS_FILE_PATTERN.format(shard_id=shard_id, n_chunks=n_chunks)
    else:
        logits_file = args.logits_file

    if args.tokens_file is None:
        tokens_file = TOKENS_FILE_PATTERN.format(shard_id=shard_id, n_chunks=n_chunks)
    else:
        tokens_file = args.tokens_file

else:

    shard_id = 0
    n_chunks = 10
    runid    = "10a09e55a89d440298155eb74a1cc6b1"
    output_dir = "test_logits"

# —————————————————————————————————————————————————————————————————————————————————————

print(f"Processing chunk {shard_id}", flush=True)

model, test_ids, _, _ = reconstruct_model(runid)
model.to(device)
shard_subjects = split_subjects(test_ids, n_chunks, shard_id-1)

dataset = DelphiDataset(domains=model.domain_cfg, root=DELPHI_DIR / "data/transforms", subjects=shard_subjects).to(device)

Path(tokens_file).parent.mkdir(exist_ok=True, parents=True)
Path(logits_file).parent.mkdir(exist_ok=True, parents=True)

logits    = model.run_inference(dataset, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
tokens_df = token_df_from_dataloder(dataset, batch_size=BATCH_SIZE)

tokens_df.to_parquet(tokens_file, index=False)

vocab_len = logits.shape[-1]
torch.save(logits.reshape(-1, vocab_len), logits_file)

print(f"Chunk {shard_id} ready ({len(tokens_df)} rows)", flush=True)  
print(f"Files created {logits_file} and {tokens_file}", flush=True)  
