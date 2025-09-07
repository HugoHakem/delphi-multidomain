# %%
import ipdb
import os, sys
os.chdir("/home/bonazzola/Delphi")
sys.path.append(os.getcwd())

import mlflow

display = print

import re
import torch

import importlib
import model.model as model
model = importlib.reload(model)
Delphi = model.Delphi

# from model import Delphi
from utils import get_p2i, get_batch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from easydict import EasyDict

import ast
import torch.nn.functional as F

# %%
device = 'cpu'
dtype = getattr(torch, 'float32')

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# %%
PADDING_TOKEN = 0
NO_EVENT_TOKEN_ID = 1
labels = pd.read_csv("delphi_labels_chapters_colours_icd.csv")

# %%
def get_val_data(val_filename):
    val_data = np.memmap(val_filename, dtype=np.int32).reshape(-1, 3)
    val_p2i = get_p2i(val_data)
    return val_data, val_p2i

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


def get_top_counts(top_n=200):
    counts = pd.DataFrame(val_data, columns=["subject_id", "age", "token_id"]).\
        query("token_id not in @ignored_tokens").\
        assign(token=lambda df: df.token_id.apply(lambda x: id_to_token[x])).\
        token.value_counts(ascending=False).\
        head(top_n).\
        sort_values()
    return counts


def get_wte(model):
   wte = model.transformer.wte.weight.detach().numpy()
   return pd.DataFrame(wte, index=[ id_to_token[i] for i in range(-1, len(id_to_token)-1) ])


def show_embedding_autocorr_from_wte():
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    import matplotlib.pyplot as plt
    # linkage requires flattened distance matrix
    from scipy.spatial.distance import squareform
    wte = get_wte(model)
    corr = pd.DataFrame.corr(wte)
    plt.figure(figsize=(10, 6))
    flatten_dist = squareform(1 - corr.values**2, checks=False)
    Z = linkage(flatten_dist, method="average")
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=corr.columns, leaf_rotation=90)
    plt.show()
    clusters = fcluster(Z, t=15, criterion="maxclust")  # 5 clusters
    order = np.argsort(clusters)
    sns.heatmap(corr.iloc[order, order], cmap="vlag", center=0)


def get_best_ckpt(runinfo):
    """
    Get the path to the best checkpoint from the runinfo.
    """
    ckpt_dir = fix_artifact_uri(runinfo.artifact_uri) / "checkpoints"
    best_ckpt_path = ckpt_dir / sorted(os.listdir(ckpt_dir), key=get_epoch_from_ckpt)[-1]
    return best_ckpt_path

# %%
runs_df = mlflow.search_runs(experiment_ids = [exp.experiment_id for exp in mlflow.search_experiments()])
runs_df = runs_df[runs_df['fold'].notnull()]
grouped = runs_df.groupby('experiment_id')

exp_ids = []

for exp_id, group in grouped:
    
    if group['fold'].nunique() == 5:
        # print(group[['run_id', 'params.val_filename']])
        print(f"Experiment ID: {exp_id}")
        exp_ids.append(exp_id)

runs_df = runs_df[runs_df['experiment_id'].isin(exp_ids)]
display(runs_df)

runinfo = runs_df.iloc[0]

# %%
run = mlflow.get_run("6b47df60154d4f67a70a7f0c94f712b6")
runinfo = pd.Series({"run_id": run.info.run_id, "status": run.info.status, "artifact_uri": run.info.artifact_uri, **run.data.params, **run.data.metrics, **run.data.tags})

ignored_tokens = get_ignored_tokens(runinfo)
t_min = float(runinfo['t_min'])
mask_ties = ast.literal_eval(runinfo['mask_ties'])

# %%
val_data, val_p2i = get_val_data(val_filename=runinfo['val_filename'])
best_ckpt_path    = get_best_ckpt(runinfo)
model = Delphi.from_checkpoint(best_ckpt_path).eval()

# %%
id_to_token = dict(zip(labels.index-1, labels.name))
get_top_counts(100).plot.barh(figsize=(10, 20), title="Top 200 tokens in validation set", xlabel="Count", ylabel="Token")

# %%
import concurrent.futures

mini_batch_size = 32
n_chunks = 512
block_size = 128

def get_batch_wrapper(token_stream):
    return get_batch(ix=token_stream, data=val_data, p2i=val_p2i, select='left', block_size=block_size, device=device, padding='random')

ix = torch.randint(len(val_p2i), (batch_size := n_chunks * mini_batch_size,))

with concurrent.futures.ThreadPoolExecutor() as executor:
    batches = list(executor.map(get_batch_wrapper, ix.chunk(n_chunks)))

d = EasyDict({
  "X":     torch.stack([b[0] for b in batches]),
  "age_X": torch.stack([b[1] for b in batches]),
  "Y":     torch.stack([b[2] for b in batches]),
  "age_Y": torch.stack([b[3] for b in batches])
})

assert block_size == d.X.size(-1)
n_total_tokens = n_chunks * mini_batch_size

token_stream =    d.X.view(n_total_tokens, block_size)
age =         d.age_X.view(n_total_tokens, block_size)
targets =         d.Y.view(n_total_tokens, block_size)
targets_age = d.age_Y.view(n_total_tokens, block_size)

def model_forward(X, age_X):
    with torch.no_grad():
        return model(X, age_X)

with concurrent.futures.ThreadPoolExecutor() as executor:
    outputs = list(executor.map(lambda b: model_forward(b[0], b[1]), batches))

logits = torch.concat([ outputs[b][0] for b in range(len(outputs)) ])
del outputs

# %%
tokens_of_interest = [ x for x in range(len(labels)) if x not in ignored_tokens ]
logits[..., tokens_of_interest].shape


# %%
# token_stream (N, B)
from debug_utils import instrument
attn_mask_hooked = instrument(model.build_attention_mask)
attn_mask, history = attn_mask_hooked(token_stream, age, targets, targets_age, mask_ties)

# ipdb.set_trace()

'''
N, B = token_stream.size()
d = dict(device=device)
attn_mask = (token_stream!=PADDING_TOKEN).view(N, 1, 1, B) * (token_stream!=PADDING_TOKEN).view(N, 1, B, 1)  # Do not attend to padded positions
attn_mask_1 = attn_mask.clone()[0,0].int().numpy()

attn_mask *= torch.tril(torch.ones(B, B, **d))[None, None, :, :] > 0 
attn_mask_2 = attn_mask.clone()[0,0].int().numpy()

#self.transformer.h[0].attn.bias[:,:,:token_stream.size(1),:token_stream.size(1)] > 0
if targets is not None and mask_ties:
    # Mask co-occurring tokens
    attn_mask *= ((age.view(N, 1, 1, B) != targets_age.view(N, 1, B, 1))) 
    attn_mask_3 = attn_mask.clone()[0,0].int().numpy()
    attn_mask += (attn_mask.sum(-1, keepdim=True)==0) * torch.diag(torch.ones(B, **d)) > 0

attn_mask_4 = attn_mask.clone()[0,0].int().numpy()
attn_mask += (token_stream==PADDING_TOKEN).view(N, 1, 1, B) * torch.diag(torch.ones(B, **d)) > 0 # Except for padding
attn_mask_5 = attn_mask.clone()[0,0].int().numpy()
attn_mask *= torch.tril(torch.ones(B, B, **d))[None, None,: , :] > 0 
attn_mask_6 = attn_mask.clone()[0,0].int().numpy()

fig, ax = plt.subplots(2, 3)
ax[0,0].imshow(attn_mask_1, cmap='gray', vmin=0, vmax=1)
ax[0,1].imshow(attn_mask_2, cmap='gray', vmin=0, vmax=1)
ax[0,2].imshow(attn_mask_3, cmap='gray', vmin=0, vmax=1)
ax[1,0].imshow(attn_mask_4, cmap='gray', vmin=0, vmax=1)
ax[1,1].imshow(attn_mask_5, cmap='gray', vmin=0, vmax=1)
ax[1,2].imshow(attn_mask_6, cmap='gray', vmin=0, vmax=1);
fig.tight_layout()
'''

# %%
import ipywidgets
@ipywidgets.interact
def show_attention_mask(token_stream=ipywidgets.IntSlider(min=0, max=token_stream.size(0)-1, step=1, value=0)):
    plt.imshow(attn_mask[token_stream][0].int());

# %%
list( map(lambda x: id_to_token[x-1], ignored_tokens) )

# %%
with torch.no_grad():

    # if we are given some desired targets also calculate the loss
    # ignored_tokens = self.config.ignore_tokens.copy()    
    
    # "filter" columns
    if (validation_loss_mode := True):
        logits = model.blackout_ignored(logits, ignored_tokens)

    # filter rows
    pass_tokens = model.get_allowed_tokens_mask(targets, ignored_tokens)
    
    time_to_next_event = targets_age - age
    loss_ce = model.cross_entropy_loss(logits, targets, pass_tokens, agg='per_disease')
    ipdb.set_trace()
    loss_dt = model.time_to_event_loss(logits, time_to_next_event, pass_tokens, attn_mask, mask_ties, t_min, agg=None)

    loss = dict(loss_ce=loss_ce, loss_dt=loss_dt)

# %%
(tokens_of_interest := labels.iloc[tokens_of_interest].sort_values("count", ascending=False).index.tolist())

# %%
loss, loss_ce, loss_dt = EasyDict(), EasyDict(), EasyDict()

with torch.no_grad():

    targets = targets.reshape(-1)
    flattened_logits = logits.reshape(-1, logits.size(-1))
    flattened_targets_age = targets_age.reshape(-1)
    flattened_age = age.reshape(-1)
    flattened_lse = lse.reshape(-1)

    total_loss = 0
    for allowed_token in tqdm(tokens_of_interest):
        pass_tokens = targets != -1
        pass_tokens *= targets == allowed_token
        if pass_tokens.sum() > 0:

            flattened_targets_age_pass = flattened_targets_age[pass_tokens]
            flattened_age_pass = flattened_age[pass_tokens]
            flattened_lse_pass = flattened_lse[pass_tokens]
            ce_for_token = -torch.log(F.softmax(flattened_logits[pass_tokens])[:, allowed_token]).sum().item() / len(logits)
            dt_for_token = -torch.log(torch.clamp((flattened_targets_age - flattened_age)[pass_tokens], min=1.0) + t_min)
            dt_for_token = - (flattened_lse[pass_tokens] - torch.exp(flattened_lse[pass_tokens] - dt_for_token)).sum() / len(logits) ## Exponential log-likelihood (real statistics, TM)

            loss[f'loss_ce_{allowed_token}'] = ce_for_token
            loss[f'loss_dt_{allowed_token}'] = dt_for_token
            loss_ce[str(allowed_token)] = ce_for_token
            loss_dt[str(allowed_token)] = dt_for_token
            
            # print(f"{ce_for_token:.4f}")
            # print(f"{dt_for_token:.4f}")
            total_loss += ce_for_token
            total_loss += dt_for_token
            print(total_loss)

# %%
sum(loss_ce.values())

# %%
labels['ce'] = labels['index'].apply(lambda x: loss_ce.get(str(x), 0))
labels.sort_values('ce', ascending=False).head(600)

# %%
print(allowed_token, pass_tokens.sum(), ce_for_token)      

# %%
loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1))[pass_tokens], targets[pass_tokens], ignore_index=-1)
loss_ce

# %%
flattened_logits[pass_tokens].shape

# %%
F.cross_entropy(flattened_logits[pass_tokens], targets[pass_tokens], weight=1/sum(pass_tokens))
sum(pass_tokens)

# %%
