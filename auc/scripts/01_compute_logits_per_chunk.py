# %%
import os, sys
import re
from loguru import logger
from easydict import EasyDict

if ( DELPHI_DIR := Path(__file__).resolve().parent.parent ) not in sys.path:
    sys.path.insert(0, DELPHI_DIR)

import ast
import argparse
import torch
torch.set_grad_enabled(False)

from tqdm import tqdm
import pandas as pd
import numpy as np

from pathlib import Path

from easydict import EasyDict
from dataclasses import dataclass, field
from tqdm import tqdm

from data.dataset import DelphiDataset, DelphiDataloader
from utils.cv_utils import get_data_partitions
from delphi.model.transformer import (
    Delphi,
    EmbedConfig,
    DelphiConfig,
)
from data.event_set import EventSet

from utils.utils import (
    load_run_info,
    load_checkpoint,
    get_last_epoch_checkpoint,
    reconstruct_model,
    setup_mlflow
)

device = 'cpu'

BLOCK_SIZE = 128
BATCH_SIZE = 512


def split_subjects(subjects, n_chunks, chunk_id):
    indices = np.array_split(np.arange(len(subjects)), n_chunks)
    idx = indices[chunk_id]
    return [subjects[i] for i in idx]


def process_chunk(shard_subjects, shard_id, n_chunks, model, domain_cfg, root, output_dir):

    print(f"Procesando chunk {shard_id}", flush=True)
    
    dataset    = DelphiDataset(domains=domain_cfg, root=root, subjects=shard_subjects).to("cpu")
    dataloader = DelphiDataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_rows, all_logits, offset = [], [], 0

    selected_domains = [ k for k, v in domain_cfg.items() if v.predict]

    with torch.no_grad():

        for bi, batch in tqdm(enumerate(dataloader)):
            es = EventSet(batch)
            es = es.insert_no_event_tokens(rate=5)
            es = es.adjust_to_seqlen(seqlen=BLOCK_SIZE, pad_domain="padding", trim_domains={"diseases"}, PADDING_TOKEN=0, PAD_AGE=-10000.0, mode="fast")

            tokens, ages, subject_ids_tensor = es.to_model_inputs()

            logits_dict, _ = model(tokens, ages, subject_ids_tensor)

            flat_parts = []
            for dom in selected_domains:
                assert dom  in logits_dict, f"Domain '{dom}' not found in model output ({selected_domains})."
                x = logits_dict[dom]   # [B, L, D]
                B, L, D_dom = x.shape
                flat_parts.append(x.reshape(B * L, D_dom))

            flat_logits = torch.cat(flat_parts, dim=1)  # [B * L, sum(Ds)]

            df = es.merge_domains(as_dataframe=True)
            df = df.sort_values(["subject_idx", "seq_idx"]).reset_index(drop=True)

            N = B * L
            df["global_idx"] = range(offset, offset + N)

            all_rows.append(df)
            all_logits.append(flat_logits)

            offset += N

    shard_df = pd.concat(all_rows, ignore_index=True)
    shard_logits = torch.cat(all_logits, dim=0)

    if output_dir is not None:
        (output_dir := Path(output_dir)).mkdir(exist_ok=True)
        df_path = output_dir / f"chunk_{shard_id}_of_{n_chunks}_df.parquet"
        logits_path = output_dir / f"chunk_{shard_id}_of_{n_chunks}_logits.pt"
        shard_df.to_parquet(df_path, index=False)
        torch.save(shard_logits, logits_path)

    return shard_df, shard_logits

    print(f"Chunk {shard_id} listo ({len(shard_df)} filas)", flush=True)


def running_in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


if False: # not running_in_notebook():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_index", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--runid", type=str, default=None)
    parser.add_argument("--experiment_id", type=str, default=None)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.runid
    os.makedirs(output_dir, exist_ok=True)
        
else:

    args = EasyDict({
        "chunk_index": 0,
        "n_chunks": 10,
        "output_dir": None,
        "runid": "10a09e55a89d440298155eb74a1cc6b1",
        "experiment_id": None,
    })
# —————————————————————————————————————————————————————————————————————————————————————
# %%

if args.experiment_id is None:
    import mlflow        
    setup_mlflow()
    experiment_id = mlflow.get_run(args.runid).info.experiment_id
    logger.info(f"Inferring MLflow experiment ID from run ID: {experiment_id}")
else: 
    experiment_id = args.experiment_id


model, test_ids, ckpt_path, params, domain_cfg = reconstruct_model(args.runid, experiment_id)

chunk_subjects = split_subjects(test_ids, args.n_chunks, args.chunk_index)
processed_chunk = process_chunk(chunk_subjects, args.chunk_index, args.n_chunks, model, domain_cfg, "./data/transforms", args.output_dir)

import ipdb; ipdb.set_trace()
# %%
