import pickle
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from IPython import embed

from joblib import Parallel, delayed

from model import DelphiConfig, Delphi
from utils import get_batch, get_p2i, DelphiData

import shap
from shap_utils import shap_custom_tokenizer, shap_model_creator

'''
Usage:
python shap-agg-eval.py --delphi_labels /path/to/delphi_labels_chapters_colours_icd.csv --labels /path/to/labels.csv --ckpt_path /path/to/model.pt --data_root /path/to/data --output_pickle /path/to/output.pkl
'''

def process_person(person_idx):
    try:
        person_to_process, time, time_target = delphi_data.get_person(person_idx, "validation")
        time_passed = (time_target - time).cpu().detach().numpy()        

        person_tokens, person_ages = delphi_data.split_person(person_to_process)
        person_tokens_ids = delphi_data.tokens_to_ids(person_tokens)

        masker = shap.maskers.Text(shap_custom_tokenizer, output_type='str', mask_token='10000', collapse_mask_token=False)
        model_shap = shap_model_creator(model, labels.index.values, person_tokens_ids, person_ages, device)
        explainer = shap.Explainer(model_shap, masker, output_names=labels[0].values)

        shap_values = explainer([' '.join(map(lambda x: str(token_to_id[x]), person_tokens))])
        shap_values.data = np.array([[f"{x[0]}({x[1]/365:.1f})" for x in person_to_process]])
        return (person_tokens_ids, shap_values.values.astype(np.float16), time_passed, [person_idx] * len(person_tokens_ids))
    except Exception as e:
        print(repr(e))
        return None

from utils import DelphiData

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--delphi_labels', type=str, required=True, help='Path to delphi_labels_chapters_colours_icd.csv')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels.csv')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Directory containing train.bin and val.bin')
    parser.add_argument('--output_pickle', type=str, required=True, help='Path to save shap output pickle')
    
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., "cuda", "cpu")')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64', 'bfloat16', 'float16'], help='Torch dtype')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    
    
    parser.add_argument('--num_chunks', type=int, default=1, help='Total number of chunks to split the validation set')
    parser.add_argument('--chunk_idx', type=int, default=0, help='Index of the current chunk (starting from 0)')

    # Re-parse the arguments to include the new ones (necessary if parser was already used)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    device = args.device
    dtype = getattr(torch, args.dtype)
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    delphi_data = DelphiData(args.data_root, 5, args.delphi_labels, args.labels, args.ckpt_path, args.device, args.dtype, args.seed)
    delphi_data.get_p2i()
    delphi_data.get_id_to_token()    
    
    train_data,  val_data    = delphi_data.train_data,  delphi_data.val_data
    train_p2i,   val_p2i     = delphi_data.train_p2i,   delphi_data.val_p2i
    id_to_token, token_to_id = delphi_data.id_to_token, delphi_data.token_to_id

    delphi_labels = delphi_data.delphi_labels
    labels = delphi_data.labels

    model = Delphi.from_checkpoint(args.ckpt_path, device=device).eval()
    
    shaply_val = []
    
    print(device)
    # Divide the validation set into chunks according to the provided arguments

    total_people = len(val_p2i)
    people_per_chunk = total_people // args.num_chunks
    remainder = total_people % args.num_chunks

    # Calculate start and end indices for this chunk
    if args.chunk_idx < remainder:
        start = args.chunk_idx * (people_per_chunk + 1)
        end = start + people_per_chunk + 1
    else:
        start = args.chunk_idx * people_per_chunk + remainder
        end = start + people_per_chunk

    print(f"Processing chunk {args.chunk_idx+1}/{args.num_chunks}: people {start} to {end-1} of {total_people}")

    # for person_idx in tqdm(range(start, end)):
    results = Parallel(n_jobs=-1)(
        delayed(process_person)(person_idx) for person_idx in tqdm(range(start, end))  
    )

    shaply_val = [r for r in results if r is not None]
                
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