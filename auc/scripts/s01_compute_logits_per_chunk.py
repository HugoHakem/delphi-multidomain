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

BATCH_SIZE = 128

def split_subjects(subjects, n_chunks=None, chunk_id=None):
    if n_chunks is None or n_chunks <= 1:
        return subjects
    indices = np.array_split(np.arange(len(subjects)), n_chunks)
    idx = indices[chunk_id]
    return [subjects[i] for i in idx]

# %%

def tokens_to_df(model, x, ages, subject_ids, domains):
    
    B, T = x.shape
    subject_idx = torch.arange(B).unsqueeze(1).expand(B, T)
    seq_idx     = torch.arange(T).unsqueeze(0).expand(B, T)
    subject_id  = subject_ids.unsqueeze(1).expand(B, T)
    
    rows = {
        "subject_idx": subject_idx.reshape(-1).cpu().numpy(),
        "seq_idx":     seq_idx.reshape(-1).cpu().numpy(),
        "subject_id":  subject_id.reshape(-1).cpu().numpy(),
        "age":         ages.reshape(-1).cpu().numpy(),
        "token_id":    x.reshape(-1).cpu().numpy(),
        "domain_id":   domains.reshape(-1).cpu().numpy(),
    }
    
    rows["domain"] = [model.int_to_domain_name[d] for d in rows["domain_id"]]
    df = pd.DataFrame(rows)
    df = df.sort_values(["subject_idx", "seq_idx"]).reset_index(drop=True)
    return df



def token_df_from_dataloder(model, dataset, batch_size):

    dataloader = DelphiDataloader(dataset, batch_size=batch_size, shuffle=False)

    offset, all_rows = 0, []
    for bi, batch in tqdm(enumerate(dataloader)):
        
        x, ages, subject_ids = model.prepare_input(batch)
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
    parser.add_argument("--logits_file", type=str)
    parser.add_argument("--tokens_file", type=str)
    
    args = parser.parse_args()
    
    shard_id = args.chunk_index
    n_chunks = args.n_chunks
    runid    = args.runid
    logits_file = args.logits_file
    tokens_file = args.tokens_file

# —————————————————————————————————————————————————————————————————————————————————————

if __name__ == "__main__":
    
    print(f"Processing chunk {shard_id}", flush=True)
    
    model, test_ids, _, _ = reconstruct_model(runid)
    model.to(device)
    shard_subjects = split_subjects(test_ids, n_chunks, shard_id-1)
    
    dataset = DelphiDataset(domains=model.domain_cfg, root=DELPHI_DIR / "data/transforms", subjects=shard_subjects).to(device)
    
    Path(tokens_file).parent.mkdir(exist_ok=True, parents=True)
    Path(logits_file).parent.mkdir(exist_ok=True, parents=True)
    
    # ——————————————————————————————————————————————————————————————————————————————————————————
    
    dataloader = DelphiDataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    offset, all_rows = 0, []
    all_logits = []
    all_tokens_df = []
    inputs = []
   
    model.eval() 
    torch.set_grad_enabled(False)

    for bi, batch in tqdm(enumerate(dataloader)):
    
        x, ages, subject_ids = model.prepare_input(batch)    
    
        with torch.no_grad():
            logits_dict, _ = model(x, ages, subject_ids)                
        
        logits_all_domains = []
        for dom in model.predicted_domains:
            assert dom  in logits_dict, f"Domain '{dom}' not found in model output ({model.predicted_domains})."        
            logits_all_domains.append(logits_dict[dom])
        logits_all_domains = torch.cat(logits_all_domains, dim=-1)  # [B * L, sum(Ds)]
        
        inputs.append( (x, ages, subject_ids) )
            
        all_logits.append(logits_all_domains)
        
        # here, everything get transformed to tensors
        x, ages, emb, subject_ids, domains = model.to_tensor(x, ages, emb := model.transformer.embed(x), subject_ids)
        tokens_df = tokens_to_df(model, x, ages, subject_ids, domains)
    
        all_tokens_df.append(tokens_df)
    
    logits = torch.cat(all_logits, dim=0)
    
    all_tokens_df = pd.concat(all_tokens_df, ignore_index=True)
    all_tokens_df = all_tokens_df.reset_index().rename(columns={"index": "global_idx"})
    all_tokens_df['subject_idx'] = all_tokens_df.global_idx // model.block_size
    
    # ——————————————————————————————————————————————————————————————————————————————————————————

    all_tokens_df.to_parquet(tokens_file, index=False)
    
    vocab_len = logits.shape[-1]
    torch.save(logits.reshape(-1, vocab_len), logits_file)
    
    print(f"Chunk {shard_id} ready ({len(tokens_df)} rows)", flush=True)  
    print(f"Files created {logits_file} and {tokens_file}", flush=True)  
