# %%
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")
from pathlib import Path
root_path = Path("../data/transforms")

import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader

from ast import literal_eval
import importlib

from easydict import EasyDict

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

from tqdm import tqdm
from dataclasses import asdict, dataclass, field

from omegaconf import OmegaConf
import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import delphi
from delphi.optim import OptimConfig, configure_optimizers
from delphi.model.transformer import (
    Delphi,
    EmbedConfig,
    DelphiConfig,
)

# from sklearn.model_selection import train_test_split
# from utils.utils import get_p2i, get_batch

import data.dataset

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ————————————————————————————————————————————————————————————————————————————————————————————

@dataclass
class TrainBaseConfig:

    ckpt_dir: str = "."
    eval_interval: int = 2000
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    init_from: str = "scratch"

    seed: int = 42
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes

    # if gradient_accumulation_steps > 1, this is the micro-batch size
    batch_size: int = 128

    # system
    device: str = DEVICE
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float32"
    # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

    train_data: dict = field(default_factory=dict)
    val_data: dict = field(default_factory=dict)

    model: dict = field(default_factory=dict)
    optim: OptimConfig = field(default_factory=OptimConfig)
    # log: TrainLogConfig = field(default_factory=TrainLogConfig)

# ————————————————————————————————————————————————————————————————————————————————————————————

# def batch_to_tensors(df, block_size):
#     
#     df = df.sort_values(["subject_id", "age"])
#     grouped = df.groupby("subject_id")
# 
#     tokens, ages, mask = [], [], []
# 
#     for _, g in grouped:
#         g = g.head(block_size).copy()  # or pad if there are less
#         if len(g) < block_size:
#             pad_len = block_size - len(g)
#             g = pd.concat([g, pd.DataFrame({
#                 "token_id": [0]*pad_len,
#                 "age": [0]*pad_len,
#                 "predict": [False]*pad_len
#             })], ignore_index=True)
# 
#         tokens.append(torch.tensor(g["token_id"].values, dtype=torch.long))
#         ages.append(torch.tensor(g["age"].values, dtype=torch.float32))
#         masks.append(torch.tensor(g["predict"].astype(int).values, dtype=torch.bool))
#     
#     ts = torch.stack
#     tokens, ages, masks = ts(tokens), ts(ages), ts(masks)
#     
#     return tokens, ages, masks

# ————————————————————————————————————————————————————————————————————————————————————————————

class Trainer():

    def __init__(self, model, training_loader, valid_loader, test_loader, optimizer):

        self.model           = model
        self.training_loader = training_loader
        self.valid_loader    = valid_loader
        self.test_loader     = test_loader
        self.optimizer       = optimizer

    # ——————————————————————————————————————————————————————————————————————————————
   
    def get_vocab_len(self, domain_name):
        self.transformer.embed.domain_embed[domain_name].weight.shape[0]


    def train(self):
        

        for batch in tqdm(self.training_loader):
            
            self.optimizer.zero_grad()
            
            domain_to_int = { k: i for i, k in enumerate(self.model.transformer.embed.domain_embed.keys()) }
            predicted_domains = [ dname for dname, config in domain_config.items() if config.predict ]
            predicted_domains_int = torch.tensor([ domain_to_int[dname] for dname in predicted_domains])
            
            x, ages, subject_ids = trainer.get_tensors_from_batch(batch)            
            max_ages = model.get_max_ages_per_subject(ages, subject_ids)
            x, ages, subject_ids = model.insert_no_event_tokens(x, ages, subject_ids)
            x, ages, subject_ids = model.mask_tokens_after_age (x, ages, subject_ids, max_ages)
            x, ages, subject_ids = trainer.pad_to_seqlen(x, ages, subject_ids, seqlen:=128)
            
            logits, att = model(x, ages, subject_ids)
            
            domains     = model._trace['domains']
            targets     = model._trace['tokens'][:,1:]
            target_ages = model._trace['ages'][:,1:]
            input_ages  = model._trace['ages'][:,:-1]
            target_domains = domains[:,1:];
            
            predict_mask = torch.isin(target_domains, predicted_domains_int.to(DEVICE))
            
            logits    = torch.cat([logits[dname] for dname in predicted_domains], axis=-1)
            f_logits  = logits[predict_mask]
            f_domains = target_domains[predict_mask]
            local_ids = targets[predict_mask]
            
            offsets_per_domain = get_offset_per_domain(model, domains_of_interest=predicted_domains)
            
            offsets = offsets_per_domain[f_domains]
            global_ids = offsets + local_ids
                        
            loss_ce = model.cross_entropy_loss(f_logits, global_ids)
            loss_ce_per_disease = model.cross_entropy_loss(f_logits, global_ids, agg="per_disease")
            print(loss_ce_per_disease)

            # loss = torch.nn.functional.cross_entropy(f_logits, global_ids)

            before = model.transformer.embed.domain_embed['diseases'].weight.clone()
            loss_ce.backward(retain_graph=True)
            print(f"{loss_ce=}")
            
            embed_param = model.transformer.embed.domain_embed['diseases'].weight

            if os.environ.get("DEBUG", False):
                import ipdb; ipdb.set_trace()

            self.optimizer.step()
            print(torch.mean((model.transformer.embed.domain_embed['diseases'].weight - before).abs()))


        # from torch.profiler import profile, record_function, ProfilerActivity    
        # for batch in tqdm(self.training_loader):
        #     tokens, ages, subject_ids = self.get_tensors_from_batch(batch)
        #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        #         with record_function("forward_delphi"):
        #             # output = model(batch)
        #             logits, _ = model(tokens, ages, subject_ids, validation_loss_mode=True)
        #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # ——————————————————————————————————————————————————————————————————————————————
    def get_tensors_from_batch(self, batch):

        SUBJECT_ID_COLUMN, AGE_COLUMN, TOKEN_COLUMN = 0, 1, 2
        tokens, ages, subject_ids = EasyDict(), EasyDict(), EasyDict()
        
        for dname in batch:   
            domain_data = batch.get(dname, [])  
            if len(domain_data) == 0:
                domain_data = domain_data.view(0, 3)
            tokens[dname] = domain_data[:, TOKEN_COLUMN].int()
            ages[dname] = domain_data[:, AGE_COLUMN]
            subject_ids[dname] = domain_data[:, SUBJECT_ID_COLUMN]

        return tokens, ages, subject_ids


    def pad_to_seqlen(self, x, ages, subject_ids, seqlen, PADDING_TOKEN=0, PAD_AGE=-10000):

        device = subject_ids['diseases'].device
    
        subject_ids_uniq, token_count = np.unique(
            torch.concat(list(subject_ids.values())).cpu().int().numpy(), 
            return_counts=True
        )
        
        necessary_padding_tokens = { 
            k: seqlen - v for k, v in dict(zip(subject_ids_uniq, token_count)).items() 
        }    
        
        x_pad  = torch.cat([ torch.full((v,), PADDING_TOKEN) for _, v in necessary_padding_tokens.items() ]).to(device)
        a_pad  = torch.cat([ torch.full((v,), PAD_AGE) for _, v in necessary_padding_tokens.items() ]).to(device)
        t_subj = torch.cat([ torch.full((v,), k) for k, v in necessary_padding_tokens.items() ]).to(device)
    
        x['padding'] = torch.cat([x['padding'], x_pad])
        ages['padding'] = torch.cat([ages['padding'], a_pad])
        subject_ids['padding'] = torch.cat([subject_ids['padding'], t_subj])
    
        return x, ages, subject_ids

    
    def get_token_domains_from_dict(self, x, ages, subjects_flat):

        all_domains = []
        for domain_idx, dname in enumerate(x.keys()):
            t = x[dname]
            n = t.shape[0]
            # domain ids (move it from dict keys to a separate tensor)
            d = torch.full((n,), domain_idx, dtype=torch.long, device=t.device)
            all_domains.append(d)
        
        domains_flat  = torch.cat(all_domains)
        unique_subjects = subjects_flat.unique(sorted=True)
        batch_domains = []
        for subj in unique_subjects:
            mask = subjects_flat == subj
            d_subj = domains_flat[mask]
            # sort by age
            order = torch.argsort(a_subj)
            t_subj = t_subj[order]
            a_subj = a_subj[order]
            d_subj = d_subj[order]
    
            batch_tokens.append(t_subj.unsqueeze(0))
            batch_ages.append(a_subj.unsqueeze(0))
            batch_domains.append(d_subj.unsqueeze(0))
    
        batch_tokens = torch.stack(batch_tokens)
        batch_ages   = torch.stack(batch_ages)
        batch_domains = torch.stack(batch_domains)
    
        return batch_tokens, batch_ages, unique_subjects, batch_domains
        return domains_flat

    # ——————————————————————————————————————————————————————————————————————————————        
    def evaluate(self):

        out = {}
        model.eval()
        
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters, 2)
            data = self.train_dataloader if split == 'train' else self.val_dataloader
            p2i = train_p2i if split == 'train' else val_p2i

            for k in range(eval_iters):
                ix = torch.randint(len(p2i), (batch_size,))
                X, A, Y, B = get_batch(ix, data, p2i, block_size=block_size,
                                       device=device, select='left', lifestyle_augmentations=True,
                                       no_event_token_rate=no_event_token_rate, 
                                       cut_batch=True)
                with ctx:
                    logits, loss, _, _ = model(X, A, Y, B, validation_loss_mode=True)

                losses[k] = torch.stack([loss['loss_ce'], loss['loss_dt']])

            out[split] = losses.mean(0)

        model.train()   
        return out


def get_offset_per_domain(model, domains_of_interest):
    
    domain_to_int = { k: i for i, k in enumerate(model.transformer.embed.domain_embed.keys()) } 
    vocab_lens = { domain_to_int[k]: v.vocab_len for k, v in model.transformer.embed.domain_embed.items() if k in domains_of_interest }
    offsets_per_domain = np.array([0] + list(vocab_lens.values())).cumsum()[:-1]
    offsets_per_domain = torch.tensor(offsets_per_domain).to(DEVICE)
    return offsets_per_domain


# ——————————————— CONFIG ———————————————————————————————————————————————————————————————

domain_config = {
  'diseases':    EmbedConfig(projector="embed", path=root_path / 'diseases', predict=True),
  'death':       EmbedConfig(projector="embed", path=root_path / 'death', predict=True),
  'lifestyle':   EmbedConfig(projector="embed", path=root_path / 'lifestyle', age_jitter=True),
  "hla_alleles": EmbedConfig(projector="embed", path=root_path / 'hla_alleles'),
  "sex":         EmbedConfig(projector="embed", path=root_path / 'sex'),
  "padding":     EmbedConfig(projector="embed")    
}

ATTENTION_SCHEME = 12 * [ "[hla_alleles, sex]:bidirectional,[sex,disease,lifestyle,death]:causal(mask_ties=True)" ]

cfg = DelphiConfig(    
    token_dropout=0.1,
    domains=domain_config,
    attention_scheme=ATTENTION_SCHEME
)

# ——————————————————————————————————————————————————————————————————————————————————————

import data.dataset
data.dataset = importlib.reload(data.dataset)
DelphiDataset = data.dataset.DelphiDataset
DelphiDataloader = data.dataset.DelphiDataloader

folds = [ f"subject_lists/subset{i}of5.csv" for i in range(1, 6) ]
test_fold, dev_folds = [folds.pop(0)], folds

dataset_config = dict(root="../data/transforms", domains=domain_config, exclusions=[]) # "subject_lists/genetic_white_ids.txt"])
dev_dataset  = DelphiDataset(subjects=dev_folds, **dataset_config).to(DEVICE)
test_dataset = DelphiDataset(subjects=test_fold, **dataset_config)

n_valid      = len(dev_dataset) - (n_train := int(0.8*len(dev_dataset)))
train_dataset, valid_dataset = random_split(
    dev_dataset, [ n_train, n_valid ],
    generator=torch.Generator().manual_seed(42)
)
dataloaders = [ 
    DelphiDataloader(d, batch_size=4) 
    for d in [train_dataset, valid_dataset, test_dataset] 
]

delphi = importlib.reload(delphi)
Delphi = delphi.model.transformer.Delphi

config = DelphiConfig(n_embd=120, domains=domain_config)
model  = Delphi(config).to(DEVICE)
torch.compile(model)

optimizer, scheduler = configure_optimizers(model=model, cfg=OptimConfig(), device_type=DEVICE)

# x = EasyDict()
# ages = EasyDict()
# pp = EasyDict()
# 
# for batch in tqdm(dataloaders[0]):  
#     for dname in model.transformer.embed.domain_embed:   
# 
#         domain_data = batch.get(dname, [])
#     
#         if len(domain_data) == 0:
#             continue
#             
#         x[dname] = domain_data[:,2].int() # torch.tensor(domain_data.token_id.cat.codes.values).type(torch.int32).to(DEVICE)
#         ages[dname] = domain_data[:,1] # torch.tensor(domain_data.age.values).type(torch.int32).to(DEVICE)
#         
#         # print(domain_data[:,0])
#         # print(f"{x=}")
#         # print(f"{dname}: {ages[dname]}")
#     
#         x_embed   = model.transformer.embed.domain_embed[dname].projector(x[dname])
#         age_embed = model.transformer.embed.age_encoding(ages[dname].unsqueeze(1))
#         pp[dname] = x_embed + age_embed





trainer = Trainer(model, *dataloaders, optimizer)
trainer.train()  


# %%
predicted_domains = [ dname for dname, config in domain_config.items() if config.predict ]
predicted_domains_int = torch.tensor([ domain_to_int[dname] for dname in predicted_domains])

batch = next(iter(dataloaders[0]))
x, ages, subject_ids = trainer.get_tensors_from_batch(batch)            
max_ages = model.get_max_ages_per_subject(ages, subject_ids)
x, ages, subject_ids = model.insert_no_event_tokens(x, ages, subject_ids)
x, ages, subject_ids = model.mask_tokens_after_age (x, ages, subject_ids, max_ages)
x, ages, subject_ids = trainer.pad_to_seqlen(x, ages, subject_ids, seqlen:=128)

logits, att = model(x, ages, subject_ids)

domains     = model._trace['domains']
targets     = model._trace['tokens'][:,1:];
target_ages = model._trace['ages'][:,1:];
input_ages  = model._trace['ages'][:,:-1];
target_domains = domains[:,1:];

predict_mask = torch.isin(target_domains, predicted_domains_int.to(DEVICE))

logits    = torch.cat([logits[dname] for dname in predicted_domains], axis=-1)
f_logits  = logits[predict_mask]
f_domains = target_domains[predict_mask]
local_ids = targets[predict_mask]

offsets_per_domain = get_offset_per_domain(model, domains_of_interest=predicted_domains)

offsets = offsets_per_domain[f_domains]
global_ids = offsets + local_ids
global_ids

f_logits.shape

torch.nn.functional.cross_entropy(f_logits, global_ids)

# %%
def local_to_global_ids(local_ids):

    vocab_lens = { domain_to_int[k]: v.vocab_len for k, v in model.transformer.embed.domain_embed.items() if k in domains_of_interest }
    offsets_per_domain = np.array([0] + list(vocab_lens.values())).cumsum()[:-1]
    offsets_per_domain = torch.tensor(offsets_per_domain).to(DEVICE)
    local_ids = targets[torch.isin(target_domains, domains_of_interest_int.to(DEVICE))]
    f_domains = target_domains[mask]
    offsets = offsets_per_domain[f_domains]
    global_ids = offsets + local_ids
    return global_ids

# for dname in ['diseases', 'death']:
    # loss = model.compute_loss(logits, tokens[dname][:,1:], ages[dname][:,1:])

# %%
