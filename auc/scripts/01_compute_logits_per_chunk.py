# %%
import os, sys
import re
DELPHI_DIR = f"{os.getenv('HOME')}/repos/delphi"
os.chdir(DELPHI_DIR)
if DELPHI_DIR not in sys.path:
    sys.path.insert(0, DELPHI_DIR)

import argparse

import scipy.stats
import scipy
import torch

from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow

from dataclasses import dataclass, field

from utils.cv_utils import ( get_best_ckpt_from_mlflow )
import data.event_set
import importlib

import ast
from data.dataset import DelphiDataset, DelphiDataloader
from utils.cv_utils import get_data_partitions
from delphi.model.transformer import (
    Delphi,
    EmbedConfig,
    DelphiConfig,
)

import importlib 
import delphi
importlib.reload(delphi)
Delphi = delphi.model.transformer.Delphi

from easydict import EasyDict

torch.set_grad_enabled(False)
device = 'cpu'

from data.event_set import EventSetV2

def infer_delphi_config_from_state_dict(sd):
    cfg = EasyDict()

    # --- 1. Infer number of transformer layers ---
    layer_indices = []
    for key in sd:
        m = re.match(r"transformer\.h\.(\d+)\.", key)
        if m:
            layer_indices.append(int(m.group(1)))
    cfg.n_layer = max(layer_indices) + 1

    # --- 2. Infer embedding dimension ---
    # Use the first c_attn.weight
    for key, val in sd.items():
        if "attn.c_attn.weight" in key:
            W = val
            cfg.n_embd = W.shape[1]
            break

    # --- 3. Infer MLP hidden size ---
    for key, val in sd.items():
        if "mlp.c_fc.weight" in key:
            cfg.mlp_hidden_dim = val.shape[0]
            break

    # --- 4. Infer domains ---
    cfg.domains = []
    for key in sd:
        m = re.match(r"transformer\.embed\.domain_embed\.(\w+)\.projector\.weight", key)
        if m:
            cfg.domains.append(m.group(1))

    # --- 5. Presence of age embedding ---
    cfg.use_age_embedding = any("age_embedding" in k for k in sd)

    # --- 6. Final ln_f layer ---
    cfg.use_final_layernorm = "transformer.ln_f.weight" in sd

    # --- 7. Infer vocabulary sizes per domain ---
    cfg.vocab_sizes = EasyDict()
    for d in cfg.domains:
        key = f"embedding_to_logits.embedding_layer_dict.domain_embed.{d}.projector.weight"
        if key in sd:
            W = sd[key]   # shape = [n_tokens_in_domain, n_embd]
            cfg.vocab_sizes[d] = W.shape[0]  # num tokens

    return cfg

# ———————————————————————————————————————————————————————————————————————————————————————————————————

def setup_mlflow():
    uri = Path(f"{os.environ['HOME']}/repos/delphi/train_scripts/mlruns")
    mlflow.set_tracking_uri(uri)
    return uri


def load_run_info(run_id):
    run = mlflow.get_run(run_id)
    params = run.data.params
    attn_scheme = ast.literal_eval(params["attention_scheme"])
    return params, attn_scheme


def load_checkpoint(mlruns_uri, experiment_id, run_id):
    ckpt_path = get_last_epoch_checkpoint(mlruns_uri / experiment_id / run_id)[0]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt, ckpt_path


def get_last_epoch_checkpoint(run_dir: str):
    """
    Given the run directory (the one containing artifacts/checkpoints),
    return the checkpoint file corresponding to the highest epoch.
    """

    ckpt_dir = Path(run_dir) / "artifacts" / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Find all .pt files
    ckpts = list(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoint files found.")

    # Regex to extract epoch number: looks for 'epoch<digits>'
    epoch_re = re.compile(r"epoch(\d+)", re.IGNORECASE)

    best = None
    best_epoch = -1

    for ck in ckpts:
        m = epoch_re.search(ck.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best = ck

    if best is None:
        raise RuntimeError("No checkpoint contained an epoch number.")

    return best, best_epoch


def build_domain_config(config, tokens_path: Path):

    default_cfg = {
    # 'genetic_pcs': EmbedConfig(projector="linear", path=tokens_path / 'genetic_pcs', type='continuous', at_birth=True),
      'diseases':    EmbedConfig(projector="embed", path=tokens_path / 'diseases',    predict=True),
      'death':       EmbedConfig(projector="embed", path=tokens_path / 'death',       predict=True),
      'cv_drugs':    EmbedConfig(projector="embed", path=tokens_path / 'cv_drugs',    predict=True),
      'ns_drugs':    EmbedConfig(projector="embed", path=tokens_path / 'ns_drugs',    predict=True),
      'lifestyle':   EmbedConfig(projector="embed", path=tokens_path / 'lifestyle',   age_jitter=True),  
      "hla_alleles": EmbedConfig(projector="embed", path=tokens_path / 'hla_alleles', at_birth=True),
      "sex":         EmbedConfig(projector="embed", path=tokens_path / 'sex',         at_birth=True),
      "padding":     EmbedConfig(projector="embed")    
    }    

    return {name: default_cfg[name] for name in config.domains}


def reconstruct_model(run_id, experiment_id):

    mlruns_uri = setup_mlflow()
    params, attn_scheme = load_run_info(run_id)

    ckpt, ckpt_path = load_checkpoint(mlruns_uri, experiment_id, run_id)
    weights = ckpt["state_dict"]
    test_ids = ckpt["metadata"]["test_ids"]

    cfg = infer_delphi_config_from_state_dict(weights)
    n_layer = cfg["n_layer"]
    n_embd = cfg["n_embd"]

    root_path = Path(f"{DELPHI_DIR}/data/transforms")
    tokens_path = root_path / "tokens"
    print(tokens_path)
    domain_cfg = build_domain_config(cfg, tokens_path)

    delphi_cfg = DelphiConfig(
        n_embd=n_embd,
        n_layer=n_layer,
        token_dropout=0.1,
        domains=domain_cfg,
        attention_scheme=attn_scheme,
    )

    # model
    model = Delphi(delphi_cfg)
    model.load_state_dict(weights)
    model.to("cpu")
    model.eval()

    return model, test_ids, ckpt_path, params, domain_cfg


def split_subjects(subjects, n_chunks, chunk_id):
    indices = np.array_split(np.arange(len(subjects)), n_chunks)
    idx = indices[chunk_id]
    return [subjects[i] for i in idx]



def extract_flat_logits(logits_dict, selected_domains):
    """
    logits_dict: dict con entradas como logits['diseases'] = [B,L,D]
    selected_domains: ordered list, e.g. ['diseases', 'death']

    Returns:
        flat_logits  → tensor [B*L, sum(D_domain)]
        domain_dims → dict {domain: (start_idx, end_idx)}
    """
    flat_parts = []
    domain_dims = {}
    offset = 0

    for dom in selected_domains:
        if dom not in logits_dict:
            raise ValueError(f"Domain '{dom}' not present in logits!")

        x = logits_dict[dom]        # [B, L, D_dom]
        B, L, D_dom = x.shape
        
        x_flat = x.reshape(B*L, D_dom)
        flat_parts.append(x_flat)

        domain_dims[dom] = (offset, offset + D_dom)
        offset += D_dom

    return torch.cat(flat_parts, dim=1), domain_dims


def process_chunk(shard_subjects, shard_id, n_chunks, model, domain_cfg, root, output_dir):

    print(f"Procesando chunk {shard_id}", flush=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    BLOCK_SIZE = 128
    BATCH_SIZE = 512

    dataset = DelphiDataset(domains=domain_cfg, root=root, subjects=shard_subjects).to("cpu")
    dataloader = DelphiDataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_rows = []
    all_logits = []
    offset = 0

    selected_domains = [ k for k, v in domain_cfg.items() if v.predict]

    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            es = EventSetV2(batch)
            es = es.insert_no_event_tokens(rate=5)
            es = es.adjust_to_seqlen(
                seqlen=BLOCK_SIZE,
                pad_domain="padding",
                trim_domains={"diseases"},
                PADDING_TOKEN=0,
                PAD_AGE=-10000.0,
                mode="fast"
            )

            tokens, ages, subject_ids_tensor = es.to_model_inputs()

            logits_dict, _ = model(tokens, ages, subject_ids_tensor)

            flat_parts = []
            for dom in selected_domains:
                if dom not in logits_dict:
                    raise ValueError(f"Domain '{dom}' not found in model output.")
                x = logits_dict[dom]   # [B,L,D]
                B, L, D_dom = x.shape
                flat_parts.append(x.reshape(B * L, D_dom))

            flat_logits = torch.cat(flat_parts, dim=1)  # [B*L, sum(Ds)]

            df = es.merge_data(as_dataframe=True)
            df = df.sort_values(["subject_idx", "seq_idx"]).reset_index(drop=True)

            N = B * L
            df["global_idx"] = range(offset, offset + N)

            all_rows.append(df)
            all_logits.append(flat_logits)

            offset += N

    shard_df = pd.concat(all_rows, ignore_index=True)
    shard_logits = torch.cat(all_logits, dim=0)

    df_path = output_dir / f"chunk_{shard_id}_of_{n_chunks}_df.parquet"
    logits_path = output_dir / f"chunk_{shard_id}_of_{n_chunks}_logits.pt"

    shard_df.to_parquet(df_path, index=False)
    torch.save(shard_logits, logits_path)

    print(f"Chunk {shard_id} listo ({len(shard_df)} filas)", flush=True)

# —————————————————————————————————————————————————————————————————————————————————————


parser = argparse.ArgumentParser()
parser.add_argument("--chunk_index", type=int, default=0)
parser.add_argument("--n_chunks", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="outputs")
parser.add_argument("--runid", type=str, default="10a09e55a89d440298155eb74a1cc6b1", help="MLflow run ID")
parser.add_argument("--experiment_id", type=str)
args = parser.parse_args()

output_dir = Path(args.output_dir) / args.runid
os.makedirs(output_dir, exist_ok=True)

experiment_id = args.experiment_id
model, test_ids, ckpt_path, params, domain_cfg = reconstruct_model(args.runid, experiment_id)

chunk_subjects = split_subjects(test_ids, args.n_chunks, args.chunk_index)

process_chunk(chunk_subjects, args.chunk_index, args.n_chunks, model, domain_cfg, "./data/transforms", output_dir)

# —————————————————————————————————————————————————————————————————————————————————————
# %%
