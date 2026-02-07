import numpy as np
import pandas as pd
import torch
import re
import os, sys
import ast 
from pathlib import Path
import mlflow
from easydict import EasyDict
import yaml

DELPHI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, DELPHI_DIR)

MLFLOW_TRACKING_URI = Path( os.getenv("MLFLOW_TRACKING_URI", DELPHI_DIR / "mlruns" ) )

from delphi.model import (
    Delphi,
    DomainConfig,
    DelphiConfig
)


def load_embed_config(cfg_path, tokens_path):

    raw = yaml.safe_load(Path(cfg_path).read_text())

    cfg = {}
    for domain, params in raw.items():
        p = dict(params)
        if "path" in p:
            p["path"] = tokens_path / p["path"]
        cfg[domain] = DomainConfig(**p)

    return cfg




def fix_artifact_uri(artifact_uri):
    artifact_uri = re.sub(pattern="^file://", repl="", string=artifact_uri)
    artifact_uri = re.sub(pattern=".*/mlruns", repl="mlruns", string=artifact_uri)
    import pathlib
    artifact_uri = pathlib.Path(artifact_uri)
    return artifact_uri


def get_epoch_from_ckpt(ckpt_path):
    return int(ckpt_path.split("_")[-1].split(".")[0])


def get_ignored_tokens(runinfo, validation_loss_mode = True):
    """
    Get the list of ignored tokens from the runinfo.
    """
    ignored_tokens = ast.literal_eval(runinfo['ignore_tokens'])
    if validation_loss_mode:
        ignored_tokens += [NO_EVENT_TOKEN_ID]    
    
    if isinstance(ignored_tokens, int):
        ignored_tokens = [ignored_tokens]
    return ignored_tokens


def get_top_counts(data, labels, top_n=200, ignored_tokens=[]):

    id_to_token = dict(zip(labels.index-1, labels.name))

    counts = pd.DataFrame(data, columns=["subject_id", "age", "token_id"]).\
        query("token_id not in @ignored_tokens").\
        assign(token=lambda df: df.token_id.apply(lambda x: id_to_token[x])).\
        token.value_counts(ascending=False).\
        head(top_n).\
        sort_values()
    
    return counts


def get_wte(model):
    wte = model.transformer.wte.weight.detach().numpy()
    return pd.DataFrame(wte, index=[ id_to_token[i] for i in range(-1, len(id_to_token)-1) ])


def get_best_ckpt(runinfo):
    """
    Get the path to the best checkpoint from the runinfo.
    """
    ckpt_dir = fix_artifact_uri(runinfo.artifact_uri) / "checkpoints"
    best_ckpt_path = ckpt_dir / sorted(os.listdir(ckpt_dir), key=get_epoch_from_ckpt)[-1]
    return best_ckpt_path



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
    
    global MLFLOW_TRACKING_URI
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MLFLOW_TRACKING_URI


def load_run_params(run_id):

    runinfo = mlflow.get_run(run_id)
    params = runinfo.data.params    
    params['attention_scheme'] = ast.literal_eval(params["attention_scheme"])
    return params


def get_experiment_id_from_runid(run_id):
    return mlflow.get_run(run_id).info.experiment_id


def load_checkpoint(run_id):
    
    mlflow_uri = Path(mlflow.get_tracking_uri().replace("file:", ""))
    experiment_id = get_experiment_id_from_runid(run_id)
    ckpt_path = get_last_epoch_checkpoint( mlflow_uri / experiment_id / run_id )[0]
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



# ----------------------------------------------------------------------------------------------------

def config_from_runid(runid):

    VAL_BATCH_SIZE = 256

    # Retrieve run info (contains experiment ID and tags)
    runinfo = mlflow.get_run(runid)
    
    artifact_uri = re.sub(".*mlruns", "mlruns", runinfo.info.artifact_uri)
    
    if "batch_size" in runinfo.data.params:
        batch_size = int(runinfo.data.params.pop("batch_size"))
    else:
        batch_size = 16

    if "test_fold" in runinfo.data.params:
        test_fold = runinfo.data.params.pop("test_fold")
    else:
        test_fold = 0
    
    if "learning_rate" in runinfo.data.params:
        learning_rate = runinfo.data.params.pop("learning_rate")

    # ------------------------------------------------------------------------------------------------
    runinfo.data.params.pop("ema_alpha")
    runinfo.data.params['attention_scheme'] = ast.literal_eval(runinfo.data.params['attention_scheme'])            
    s = runinfo.data.params['domains']        
    s_clean = re.sub(r"PosixPath\(([^)]+)\)", r"\1", s)
    runinfo.data.params['domains'] = s_clean
    runinfo.data.params['domains'] = ast.literal_eval(runinfo.data.params['domains'])
    runinfo.data.params['domains'] = { k: DomainConfig(**v) for k, v in runinfo.data.params['domains'].items() }
    for param, value in runinfo.data.params.items():
        if "drop"in param:
            runinfo.data.params[param] = float(value)
        if param in {"n_embd", "n_head", "n_layer", "block_size", "n_layer"}:
            runinfo.data.params[param] = int(value)
        if param in {"zero_inflate", "bias"}:
            runinfo.data.params[param] = True if runinfo.data.params[param] == "True" else False                    
    # ------------------------------------------------------------------------------------------------
    
    tracking_uri = Path(os.path.dirname(mlflow.get_tracking_uri()))

    ckpt_dir = tracking_uri / (artifact_uri + "/checkpoints")
    ckpt_files = sorted(Path(ckpt_dir).glob("*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found for run {runid}")
    latest_ckpt = ckpt_files[-1]

    print(f"Loading latest checkpoint: {latest_ckpt}")
    delphi_cfg = DelphiConfig(**runinfo.data.params)
    model = Delphi(delphi_cfg).to(DEVICE)
    ckpt = torch.load(latest_ckpt)

    start_epoch = ckpt.get("metadata", {}).get("epoch", 0) + 1
    weights = ckpt['state_dict']
    model.load_state_dict(weights, strict=False)
    torch.compile(model)
    
    optimizer, scheduler = configure_optimizers(model=model, cfg=OptimConfig(), device_type=DEVICE)                      
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    DDS = DelphiDataset
    datasets = [
        train_dataset := DDS(root="../data/transforms", domains=delphi_cfg.domains, subjects=ckpt['metadata']['train_ids']).to("cuda"),
        valid_dataset := DDS(root="../data/transforms", domains=delphi_cfg.domains, subjects=ckpt['metadata']['valid_ids']).to("cuda"),
        test_dataset  := DDS(root="../data/transforms", domains=delphi_cfg.domains, subjects=ckpt['metadata']['test_ids']).to("cuda")
    ]

    dataloaders = [
        train_loader := DelphiDataloader(train_dataset, batch_size=batch_size),
        valid_loader := DelphiDataloader(valid_dataset, batch_size=VAL_BATCH_SIZE), 
        test_loader  := DelphiDataloader(test_dataset,  batch_size=VAL_BATCH_SIZE)
    ]    

    previous_run_name = runinfo.data.tags.get("mlflow.runName", None)
    logged_params = { "test_fold": test_fold, "batch_size": batch_size }    

    return model, dataloaders, optimizer, scheduler, logged_params, previous_run_name


def reconstruct_model(run_id):
    
    params = load_run_params(run_id)

    attn_scheme = params['attention_scheme']
    
    ckpt, ckpt_path = load_checkpoint(run_id)

    weights = ckpt["state_dict"]
    test_ids = ckpt["metadata"]["test_ids"]

    cfg = infer_delphi_config_from_state_dict(weights)
    n_layer = cfg["n_layer"]
    n_embd = cfg["n_embd"]

    domain_cfg = get_domain_configs_from_string(params['domains'])

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

    return model, test_ids, ckpt_path, params

def get_domain_configs_from_string(s):
    return EasyDict(eval(s, {"PosixPath": Path, "__builtins__": {}}, {}))
