# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: delphi
#     language: python
#     name: python3
# ---

# %%
import os
import torch

from ipywidgets import interact
import ipywidgets as widgets

from model import DelphiConfig, Delphi
from tqdm import tqdm
import pandas as pd
import numpy as np
import textwrap
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format='retina'

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'axes.grid': True,
                     'grid.linestyle': ':',
                     'axes.spines.bottom': False,
          'axes.spines.left': False,
          'axes.spines.right': False,
          'axes.spines.top': False})
plt.rcParams['figure.dpi'] = 72
plt.rcParams['pdf.fonttype'] = 42

#Green
light_male = '#BAEBE3'
normal_male = '#0FB8A1'
dark_male = '#00574A'


#Purple
light_female = '#DEC7FF'
normal_female = '#8520F1'
dark_female = '#7A00BF'
 
DELPHI_LABELS_FILE = 'delphi_labels_chapters_colours_icd_with_hla4d.csv'
delphi_labels = pd.read_csv(DELPHI_LABELS_FILE)

# %%
out_dir = './Delphi-hla'
ckpt_file = 'ckpt__f945b0c1f1b84294bf29a6d39e6ef831__100000.pt'
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'mps', etc.
dtype ='float32' #'bfloat16' # 'float32' or 'bfloat16' or 'float16'
dtype = {'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

ckpt_path = os.path.join(out_dir, ckpt_file)
checkpoint = torch.load(ckpt_path, map_location=device)
conf = DelphiConfig(**checkpoint['model_args'])
model = Delphi(conf)
state_dict = checkpoint['model']
state_dict = { k.replace("_orig_mod.", ""): v for k, v in state_dict.items() }

model.load_state_dict(state_dict)

model.eval()
model = model.to(device)

# %%
from utils import get_batch, get_p2i

# train = np.fromfile('data/ukb_simulated_data/train.bin', dtype=np.uint32).reshape(-1,3)
# val = np.fromfile('data/ukb_simulated_data/val.bin', dtype=np.uint32).reshape(-1,3)

train = np.fromfile('./data/ukb_real_data/ukb_real_hla4d_train.bin', dtype=np.uint32).reshape(-1,3)
val = np.fromfile('./data/ukb_real_data/ukb_real_hla4d_val.bin', dtype=np.uint32).reshape(-1,3)

train_p2i = get_p2i(train) # mapping trajectory id to its position in the dataset
val_p2i = get_p2i(val)

dataset_subset_size = 100000 # len(val_p2i) # can be set to smaller number (e.g. 2048) for a quick run

# %%
# lets split the large data chanks to smaller batches and calculate the logits for the whole dataset
# p = []
# batch_size = 256
# subset_size = min(dataset_subset_size, 10_000)
# with torch.no_grad():
#     for d_batch in tqdm(zip(*map(lambda x: torch.split(x, batch_size), d)), total=d[0].shape[0]//batch_size+1):
#         p.append(model(*d_batch)[0].cpu().detach())
# p = torch.vstack(p)
# 
# d = [d_.cpu() for d_ in d]

# %%
PATTERNS_OF_INTEREST = ["erythema multiforme"]
hla_diseases_of_interest = delphi_labels[delphi_labels.name.apply(lambda x: any([pattern in x.lower() for pattern in PATTERNS_OF_INTEREST]))].index.to_list()

# %%
hla_scores = pd.read_csv("hla_score_per_icd10_with_justification_COMPLETE.csv").sort_values("score", ascending=False)

# %%
d = get_batch(range(dataset_subset_size), val, val_p2i,  
              select='left', block_size=96, 
              device=device, padding='random')


which_subjects = np.where(torch.isin(d[2].cpu(), torch.tensor(hla_diseases_of_interest)).sum(axis=1))
which_subjects = (which_subjects[0][:10000],)

att = model(*list(map(lambda x: x[which_subjects[0],:], d)))[2].cpu().detach().numpy().squeeze()
att.shape

# %%
which_subjects

# %%
w = which_subjects
BLOCK_SIZE = 96

@interact
def show_attention_maps(subject=widgets.SelectionSlider(options=which_subjects[0])):
    
    i = subject
    sub_index = which_subjects[0].tolist().index(i)
    j = (d[0][i]==0).sum()
    
    plt.figure(figsize=(6 * (d[3][i,-1]-d[1][i,0])/365.25/70, 20 * (BLOCK_SIZE-j)/BLOCK_SIZE))
    x = torch.concatenate([d[1][i], d[3][i,[-1]]])/365.25
    plt.pcolormesh( x[j:], np.arange(j, BLOCK_SIZE+1, 1), att[0,sub_index,:,j:,j:].max((0)).T, cmap='Blues')

    y_ticks = np.arange(j, 96) + .5
    
    y_ticklabels = []    
    for k, t in zip(d[0][i, j:].detach().numpy().squeeze(), d[1][i, j:].detach().numpy().squeeze()/365.25):
        label = textwrap.shorten(delphi_labels.loc[k,'name'], 50).replace(' ', r'\ ') if k in (hla_diseases_of_interest) else \
           (f"{textwrap.shorten(delphi_labels.loc[k,'name'], 50)}" if j > 1 else "")
        y_ticklabels.append(f"$\\bf{{{label}}}$" if k in (hla_diseases_of_interest) else label)

    _ = plt.yticks(y_ticks, y_ticklabels)
    
    plt.gca().invert_yaxis()
    plt.xlabel('Age')
    plt.tick_params(axis='y', labelsize=10)
    plt.show()        

# %%
