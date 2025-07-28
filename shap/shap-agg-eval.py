import os
import pickle
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from IPython import embed
import shap

from model import DelphiConfig, Delphi
from utils import get_batch, get_p2i, shap_custom_tokenizer, shap_model_creator

parser = argparse.ArgumentParser()
parser.add_argument('--delphi_labels', type=str, required=True, help='Path to delphi_labels_chapters_colours_icd.csv')
parser.add_argument('--labels', type=str, required=True, help='Path to labels.csv')
parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--data_root', type=str, required=True, help='Directory containing train.bin and val.bin')
parser.add_argument('--output_pickle', type=str, required=True, help='Path to save shap output pickle')

parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., "cuda", "cpu")')
parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64', 'bfloat16', 'float16'], help='Torch dtype')
parser.add_argument('--seed', type=int, default=1337, help='Random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = args.device
dtype = getattr(torch, args.dtype)
device_type = 'cuda' if 'cuda' in device else 'cpu'

delphi_labels = pd.read_csv(args.delphi_labels)
labels = pd.read_csv(args.labels, header=None, sep="\t")
model = Delphi.from_checkpoint(args.ckpt_path).eval()

train = np.fromfile(os.path.join(args.data_root, 'train.bin'), dtype=np.uint32).reshape(-1, 3)
val = np.fromfile(os.path.join(args.data_root, 'val.bin'), dtype=np.uint32).reshape(-1, 3)
train_p2i = get_p2i(train)
val_p2i = get_p2i(val)

id_to_token = labels.to_dict()[0]
token_to_id = {v: k for k, v in id_to_token.items()}

def tokens_to_ids(tokens):
    return [token_to_id[t] for t in tokens]

def ids_to_tokens(ids):
    return [id_to_token[int(id_)] for id_ in ids]

def split_person(p):
    tokens = [i[0] for i in p]
    ages = [i[1] for i in p]
    return tokens, ages

def get_person(idx):
    x, y, _, time = get_batch([idx], val, val_p2i, select='left', block_size=64, device=device, padding='random', cut_batch=True)
    x, y = x[y > -1], y[y > -1]
    person = [(id_to_token[xi.item()], yi.item()) for xi, yi in zip(x, y)]
    return person, y, time[0][-1]


DISEASES_OF_INTEREST = ['M06', 'M45', 'E14', 'L40', 'K90']
DISEASES_OF_INTEREST = 'all'

shaply_val = []

for person_idx in tqdm(range(len(val_p2i))):
    
    try:
        person_to_process, time, time_target = get_person(person_idx)
        time_passed = (time_target - time).cpu().detach().numpy()        

        person_tokens, person_ages = split_person(person_to_process)
        person_tokens_ids = tokens_to_ids(person_tokens)

        of_interest = True if DISEASES_OF_INTEREST == 'all' else any([ y in x for x in person_tokens for y in DISEASES_OF_INTEREST ])
        
        if not of_interest:
            continue

        masker = shap.maskers.Text(shap_custom_tokenizer, output_type='str', mask_token='10000', collapse_mask_token=False)
        
        # embed()

        model_shap = shap_model_creator(model, labels.index.values, person_tokens_ids, person_ages, device)
        explainer = shap.Explainer(model_shap, masker, output_names=labels[0].values)

        shap_values = explainer([' '.join(map(lambda x: str(token_to_id[x]), person_tokens))])
        shap_values.data = np.array([[f"{x[0]}({x[1]/365:.1f})" for x in person_to_process]])
        shaply_val.append((person_tokens_ids, shap_values.values.astype(np.float16), time_passed, [person_idx] * len(person_tokens_ids)))

    except Exception as e:
        print(repr(e))
        continue

all_tokens = np.concatenate([i[0] for i in shaply_val])
all_values = np.concatenate([i[1] for i in shaply_val], axis=1)[0]
all_times_passed = np.concatenate([i[2] for i in shaply_val], axis=0)
all_people = np.concatenate([i[3] for i in shaply_val])

with open(args.output_pickle, 'wb') as f:
    pickle.dump({
        'tokens': all_tokens,
        'values': all_values,
        'times': all_times_passed,
        'model': args.ckpt_path,
        'people': all_people
    }, f)
