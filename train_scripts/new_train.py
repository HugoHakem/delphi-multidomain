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
from data.dataset import DelphiDataset, DelphiDataloader

from delphi.optim import OptimConfig, configure_optimizers
from delphi.model.transformer import (
    Delphi,
    EmbedConfig,
    DelphiConfig,
)

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
RUN_NAME = os.getenv("RUN_NAME", "default")
odir = f"output/{RUN_NAME}"
os.makedirs(odir, exist_ok=True)

# ————————————————————————————————————————————————————————————————————————————————————————————
'''
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
'''
# ————————————————————————————————————————————————————————————————————————————————————————————

EACH_VAL = 3236

def local_to_global_ids(local_ids):

    vocab_lens = { domain_to_int[k]: v.vocab_len for k, v in model.transformer.embed.domain_embed.items() if k in domains_of_interest }
    offsets_per_domain = np.array([0] + list(vocab_lens.values())).cumsum()[:-1]
    offsets_per_domain = torch.tensor(offsets_per_domain).to(DEVICE)
    local_ids = targets[torch.isin(target_domains, domains_of_interest_int.to(DEVICE))]
    f_domains = target_domains[mask]
    offsets = offsets_per_domain[f_domains]
    global_ids = offsets + local_ids
    return global_ids


class EarlyStopping:

    def __init__(self, patience=10, min_delta=0.0, mode='min', ckpt_dir="checkpoints"):
        """
        mode: 'min' for losses, 'max' for metrics like AUROC
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.ckpt_dir = ckpt_dir
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score, model=None):
        # Adjust sign depending on mode
        score = -current_score if self.mode == 'min' else current_score
        
        if self.best_score is None:
            self.best_score = score
            if model and self.ckpt_path:
                self._save_model(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if model and self.ckpt_path:
                self._save_model(model)
        
        return self.should_stop

    def _save_model(self, model, metadata=None):
        
        if metadata is None:
            metadata = {}

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.ckpt_dir) / f"best_model_{timestamp}.pt"
        torch.save(model.state_dict() | metadata, path)
        print(f"Saved checkpoint at {path}")
        return path


class Trainer():

    def __init__(self, model, training_loader, valid_loader, test_loader, optimizer, scheduler, n_train_batches=None, n_val_batches=None):

        self.model           = model
        self.training_loader = training_loader
        self.valid_loader    = valid_loader
        self.test_loader     = test_loader
        self.optimizer       = optimizer
        self.scheduler       = scheduler
        self.early_stopper   = EarlyStopping(patience=args.patience, min_delta=0.001, mode='min')                        

        self.domain_to_int = { k: i for i, k in enumerate(self.model.transformer.embed.domain_embed.keys()) }
        self.predicted_domains = [ dname for dname, config in domain_config.items() if config.predict ]
        self.predicted_domains_int = torch.tensor([ self.domain_to_int[dname] for dname in self.predicted_domains]).to(DEVICE)

        self.n_train_batches = n_train_batches
        self.n_val_batches = n_val_batches

        self.train_outputs = []
        self.valid_outputs = []
        self.test_outputs  = []

        self.current_epoch   = 0
        self._valid_counter = 0
        

    # ——————————————————————————————————————————————————————————————————————————————
   
    def get_vocab_len(self, domain_name):
        self.transformer.embed.domain_embed[domain_name].weight.shape[0]


    def train(self, max_epochs=1000):

        self.valid_epoch(n_batches=10)

        for epoch in range(self.current_epoch, max_epochs):
            
            self.current_epoch = epoch            
            self.train_epoch(n_batches=10, eval_every=EACH_VAL)
            # self.valid_epoch(n_batches=10)
            
            if self.early_stopper.step(self.mean_val_loss, model=self.model):
                print(f"Early stopping triggered at epoch {self.current_epoch}")
                break


    def shared_step(self, batch, batch_idx, epoch, log_per_disease=False, return_logits=False, return_att=False, stage="training"):

        model = self.model
        x, ages, subject_ids = trainer.get_tensors_from_batch(batch)            
        max_ages = model.get_max_ages_per_subject(ages, subject_ids)
        x, ages, subject_ids = model.insert_no_event_tokens(x, ages, subject_ids)
        x, ages, subject_ids = model.mask_tokens_after_age (x, ages, subject_ids, max_ages)
        x, ages, subject_ids = trainer.pad_to_seqlen(x, ages, subject_ids, seqlen:=160)
        
        logits, att = model(x, ages, subject_ids)
        
        domains     = model._trace['domains']
        targets     = model._trace['tokens'][:,1:]
        input_ages  = model._trace['ages'][:,:-1]
        target_ages = model._trace['ages'][:,1:]
        target_domains = domains[:,1:];
        
        predict_mask = torch.isin(target_domains, self.predicted_domains_int)
        
        logits    = torch.cat([logits[dname] for dname in self.predicted_domains], axis=-1)
        f_logits  = logits[predict_mask]
        f_domains = target_domains[predict_mask]
        local_ids = targets[predict_mask]
        age_diff = (target_ages - input_ages)[predict_mask]
        
        offsets_per_domain = self.get_offset_per_domain(model, domains_of_interest=self.predicted_domains)
        
        offsets = offsets_per_domain[f_domains]
        global_ids = offsets + local_ids

        if os.environ.get("DEBUG", False):
            import ipdb; ipdb.set_trace()

        loss_ce = model.cross_entropy_loss(f_logits, global_ids)            
        time_loss = model.time_to_event_loss(f_logits, age_diff, t_min=1e-1, agg='mean')

        outputs = getattr(self, stage[:5] + "_outputs")

        loss = EasyDict({                
            'ce_loss': loss_ce,            
            'time_loss': time_loss,
            'ce_ema_loss': torch.lerp(outputs[-1]['ce_ema_loss'], loss_ce, weight=0.002) if outputs else loss_ce,
            'time_ema_loss': torch.lerp(outputs[-1]['time_ema_loss'], time_loss, weight=0.002) if outputs else time_loss,
            'total': loss_ce + time_loss,            
        })

        if log_per_disease:
            loss_ce_per_disease = model.cross_entropy_loss(
                f_logits, global_ids, agg="per_disease"
            ).to_frame().assign(batch_idx=batch_idx)

            loss['ce_loss_per_disease'] = loss_ce_per_disease

        if return_att and return_logits:
            return loss, logits, att 
        elif return_logits:
            return loss, logits
        elif return_att:
            return loss, att
        else:
            return loss


    def valid_epoch_end(self, loss_outputs):
        import pandas as pd
        
        self._valid_counter += 1        

        return pd.concat([x['ce_loss_per_disease'] for x in loss_outputs]).\
            reset_index().\
            pivot(index="token_id", columns="batch_idx", values="log_p").\
            fillna(0).\
            sum(axis=1).\
            sort_values()
        

    def train_epoch(self, n_batches=None, eval_every=None):

        pbar = tqdm(self.training_loader)

        for i, batch in enumerate(pbar):
            
            self.optimizer.zero_grad()                     
            loss = self.shared_step(batch, batch_idx=i, epoch=self.current_epoch, stage="training")
            self.train_outputs.append(loss)
            
            loss['total'].backward()

            if eval_every is not None and (i % eval_every) == 0:
                self.mean_val_loss = self.valid_epoch()                        

            pbar.set_postfix({
                "ce_loss":      f"{loss['ce_loss'].item():.4f}", 
                "time_loss":    f"{loss['time_loss'].item():.4f}",
                "loss_sm":      f"{loss['ce_ema_loss'].item():.4f}", 
                "time_loss_sm": f"{loss['time_ema_loss'].item():.4f}",
            })

            self.optimizer.step() 
            self.scheduler.step()

            if n_batches is not None and i == n_batches:
                break


    def epoch_end(self):

        self.train_outputs = []
        self.valid_outputs = []


    def valid_epoch(self, n_batches=None):
        
        pbar = tqdm(self.valid_loader)

        with torch.no_grad():
            loss_outputs = []
            for i, batch in enumerate(pbar):
                loss = self.shared_step(batch, batch_idx=i, epoch=self.current_epoch, log_per_disease=True, stage="validation")
                loss_outputs.append(loss)
            
                pbar.set_postfix({
                    "ce_loss":      f"{loss['ce_loss'].item():.4f}", 
                    "time_loss":    f"{loss['time_loss'].item():.4f}",
                    "loss_sm":      f"{loss['ce_ema_loss'].item():.4f}", 
                    "time_loss_sm": f"{loss['time_ema_loss'].item():.4f}",
                })
            
                if n_batches is not None and i == n_batches:
                    break
        
        loss_per_disease_df = self.valid_epoch_end(loss_outputs).reset_index()
        loss_per_disease_df.to_csv(f"{odir}/loss_outputs_{self._valid_counter}.csv", index=False)

        loss = torch.stack([loss['ce_loss'] for loss in loss_outputs]).mean()

        return loss


    def mlflow_logging(self):

        pass


    # ——————————————————————————————————————————————————————————————————————————————
    def get_tensors_from_batch(self, batch):
        
        tokens, ages, subject_ids = EasyDict(), EasyDict(), EasyDict()
        
        for dname in batch:               
            SUBJECT_ID_COLUMN, AGE_COLUMN, TOKEN_COLUMN = 0, 1, 2
            domain_data = batch.get(dname, [])  
            if len(domain_data) == 0:
                domain_data = domain_data.view(0, 3)
            if dname == "genetic_pcs":
                tokens[dname] = domain_data[:, 1:-1].float()
                ages[dname] = domain_data[:, -1]
                subject_ids[dname] = domain_data[:, 0]
            else:    
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


    def get_offset_per_domain(self, model, domains_of_interest):
        
        domain_to_int = { k: i for i, k in enumerate(model.transformer.embed.domain_embed.keys()) } 
        vocab_lens = { 
            domain_to_int[k]: v.vocab_len 
            for k, v in model.transformer.embed.domain_embed.items() if k in domains_of_interest 
        }
        offsets_per_domain = np.array([0] + list(vocab_lens.values())).cumsum()[:-1]
        offsets_per_domain = torch.tensor(offsets_per_domain).to(DEVICE)
        return offsets_per_domain


# ——————————————— CONFIG ———————————————————————————————————————————————————————————————


# import argparse
# 
# parser = argparse.ArgumentParser()
# parser.add_argument("--attention_scheme", default="[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death]:causal(mask_ties=True)", nargs="+")
# parser.add_argument("--n_layers",         default=12)
# parser.add_argument("--test_fold")
# parser.add_argument("--domains", default=)
# 
# args = parser.parse_args()

args = EasyDict({
    "attention_scheme": "[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death]:causal(mask_ties=True)",
    "n_layers": 12,
    "test_fold": 3
})

tokens_path = root_path / 'tokens'

domain_config = {
  # 'genetic_pcs': EmbedConfig(projector="linear", path=tokens_path / 'genetic_pcs', type='continuous', at_birth=True),
  'diseases':    EmbedConfig(projector="embed", path=tokens_path / 'diseases',  predict=True),
  'death':       EmbedConfig(projector="embed", path=tokens_path / 'death',     predict=True),
  'lifestyle':   EmbedConfig(projector="embed", path=tokens_path / 'lifestyle', age_jitter=True),  
  "hla_alleles": EmbedConfig(projector="embed", path=tokens_path / 'hla_alleles', at_birth=True),
  "sex":         EmbedConfig(projector="embed", path=tokens_path / 'sex', at_birth=True),
  "padding":     EmbedConfig(projector="embed")    
}

# —————————————————————————————————————————————————————————————————————————————————————————————————————————

from utils.cv_utils import get_data_partitions, generate_splits, load_fold_ids
# args.test_fold = 

train_ids, val_ids, test_ids = get_data_partitions("../data/transforms/subject_lists", fold=args.test_fold)

# %%
len(train_ids), len(val_ids), len(test_ids)
# %%

# generate_splits()

folds = [ f"subject_lists/subset{i}of5.csv" for i in range(1, 6) ]
test_fold, dev_folds = [folds.pop(0)], folds

dataset_config = dict(root=root_path, domains=domain_config, exclusions=[]) # "subject_lists/genetic_white_ids.txt"])
dev_dataset  = DelphiDataset(subjects=dev_folds, **dataset_config).to(DEVICE)
test_dataset = DelphiDataset(subjects=test_fold, **dataset_config)

n_valid      = len(dev_dataset) - (n_train := int(0.8*len(dev_dataset)))
train_dataset, valid_dataset = random_split(
    dev_dataset, [ n_train, n_valid ],
    generator=torch.Generator().manual_seed(42)
)
dataloaders = [ DelphiDataloader(d, batch_size=32) for d in [train_dataset, valid_dataset, test_dataset] ]

# %%

# —————————————————————————————————————————————————————————————————————————————————————————————————————————

# ATTENTION_SCHEME = "[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death]:causal(mask_ties=True)"
# ATTENTION_SCHEME = "[hla_alleles,sex,diseases,lifestyle,death]:causal(mask_ties=True)"
# ATTENTION_SCHEME = "[sex,diseases,lifestyle,death]:causal(mask_ties=True)"
# ATTENTION_SCHEME = "[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,genetic_pcs,death]:causal(mask_ties=True)"
# ATTENTION_SCHEME = "[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death]:causal(mask_ties=True)"
# ATTENTION_SCHEME = "[h,s]:bidirectional,[s,dis,l,de]:causal(mask_ties=True)"

assert len(args.attention_scheme) in {1, args.n_layers}, f"--attention_scheme should be either 1 or args.n_layers (={args.n_layers})"

if len(args.attention_scheme) == 1:
    attention_scheme = args.n_layers * args.attention_scheme
elif args.n_layers == len(args.attention_scheme):
    attention_scheme = args.attention_scheme

config = DelphiConfig(n_layers=args.n_layers, token_dropout=0.1, domains=domain_config, attention_scheme=attention_scheme)

model  = Delphi(config).to(DEVICE)
torch.compile(model)

optimizer, scheduler = configure_optimizers(model=model, cfg=OptimConfig(), device_type=DEVICE)

# —————————————————————————————————————————————————————————————————————————————————————————————————————————

trainer = Trainer(model, *dataloaders, optimizer, scheduler, n_train_batches=1000, n_val_batches=1000)
trainer.train(max_epochs=1000)

# %%