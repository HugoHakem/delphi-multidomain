# %%
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")
from pathlib import Path
root_path = Path("../data/transforms")

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, random_split, DataLoader

from easydict import EasyDict

import mlflow

from tqdm import tqdm
import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from collections import defaultdict

from data.dataset import DelphiDataset, DelphiDataloader
from utils.cv_utils import get_data_partitions

from delphi.optim import OptimConfig, configure_optimizers
from delphi.model.transformer import (
    Delphi,
    EmbedConfig,
    DelphiConfig,
)

from copy import deepcopy

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

EVERY_VAL = 3236

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
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        mode: 'min' for losses, 'max' for metrics like AUROC.
        This class only decides when to stop; it doesn't handle checkpointing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.is_improvement = False

    def step(self, current_score):
        """
        Update the early stopping state given a new validation score.

        Returns
        -------
        should_stop : bool
            Whether training should stop.
        is_improvement : bool
            Whether the current score improved over the best one.
        """
        score = -current_score if self.mode == 'min' else current_score

        if self.best_score is None:
            self.best_score = score
            self.is_improvement = True
            return False, True

        if score < self.best_score + self.min_delta:
            # No improvement
            self.counter += 1
            self.is_improvement = False
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            # Improvement found
            self.best_score = score
            self.counter = 0
            self.is_improvement = True

        return self.should_stop, self.is_improvement



class NullLogger:

    def log_params(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_artifact(self, path, artifact_path=None):
        pass

    def log_model(self, model, artifact_path="model"):
        pass
    

class MLFlowLogger:

    def __init__(self, run_name=None, autostart=True, nested=False):
        self.run_name = run_name
        self.active_run = None
        if autostart:
            self.start(nested=nested)

    def start(self, nested=False):
        if self.active_run is None:
            self.active_run = mlflow.start_run(run_name=self.run_name, nested=nested)
        return self.active_run

    def end(self):
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path, artifact_path=None):
        mlflow.log_artifact(path, artifact_path=artifact_path)



class Trainer():

    def __init__(self, model, training_loader, valid_loader, test_loader, optimizer, scheduler, patience=3, n_train_batches=None, n_val_batches=None, logger=NullLogger()):

        self.model           = model        
        
        self.optimizer       = optimizer
        self.scheduler       = scheduler        
        self.early_stopper   = EarlyStopping(patience=patience, min_delta=0.001, mode='min')                        

        self.training_loader, self.valid_loader, self.test_loader = training_loader, valid_loader, test_loader

        self.n_train_batches, self.n_val_batches = n_train_batches, n_val_batches
        self.train_outputs, self.valid_outputs, self.test_outputs = [], [], []            

        self.current_epoch  = 0
        self._valid_counter = 0
        
        self.logger = logger
        

    # ——————————————————————————————————————————————————————————————————————————————
   
    @property
    def predicted_domains(self):
        return [ dname for dname, config in self.model.config.domains.items() if config.predict ]

    @property
    def predicted_domains_as_int(self):
        return torch.tensor([ self.domain_to_int[dname] for dname in self.predicted_domains]).to(DEVICE)

    @property
    def domain_to_int(self):
        return { dname: i for i, dname in enumerate(self.model.transformer.embed.domain_embed.keys()) }


    @property
    def vocab_lens(self):
        return { 
            dname: self.model.transformer.embed.domain_embed[dname].weight.shape[0]
            for dname in self.model.transformer.embed.domain_embed
        }


    def get_model_metadata(self):

        metadata = { 
            "train_ids": sorted(self.training_loader.dataset.subjects),
            "valid_ids": sorted(self.valid_loader.dataset.subjects),
            "test_ids":  sorted(self.test_loader.dataset.subjects),
        }
        return metadata


    def get_vocab_len(self, domain_name):
        self.transformer.embed.domain_embed[domain_name].weight.shape[0]


    def train(self, max_epochs=1000):

        self.logger.log_params(self.model.config)
        # self.logger.log_params(self.optimizer.config)
        # self.logger.log_params(self.scheduler.config)

        for epoch in range(self.current_epoch, max_epochs):
            
            self.current_epoch = epoch            
            train_loss = self.train_epoch(eval_every=EVERY_VAL)

            metrics = {"train_loss": train_loss}
             
            if val_loss is not None:
                metrics["val_loss"] = val_loss

            self.logger.log_metrics(metrics, step=epoch)            
            # self.logger.log_model(self.model)

            should_stop, improved = self.early_stopper.step(self.mean_val_loss)

            if improved:                
                ckpt_dir = Path("checkpoints")
                ckpt_dir.mkdir(exist_ok=True)
                self.best_epoch = epoch
                ckpt_path = ckpt_dir / f"best_model_epoch{epoch}_valloss{self.mean_val_loss:.4f}.ckpt"
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"New best model saved → {ckpt_path}")
 
            if should_stop:
                print(f"Early stopping triggered at epoch {self.current_epoch}")
            


    def shared_step(self, batch, batch_idx, epoch, log_per_disease=False, return_logits=False, return_att=False, stage="training", add_prefix=None):

        model = self.model

        x, ages, subject_ids = trainer.get_tensors_from_batch(batch)

        max_ages = model.get_max_ages_per_subject(ages, subject_ids)
        x, ages, subject_ids = model.insert_no_event_tokens(x, ages, subject_ids)
        x, ages, subject_ids = model.mask_tokens_after_age (x, ages, subject_ids, max_ages)
        x, ages, subject_ids = trainer.adjust_to_seqlen(x, ages, subject_ids, seqlen:=96)
        
        logits, att = model(x, ages, subject_ids)
        
        domains     = model._trace['domains']
        targets     = model._trace['tokens'][:,1:]
        input_ages  = model._trace['ages'][:,:-1]
        target_ages = model._trace['ages'][:,1:]
        target_domains = domains[:,1:];
        
        predict_mask = torch.isin(target_domains, self.predicted_domains_as_int)
        
        logits    = torch.cat([logits[dname] for dname in self.predicted_domains], axis=-1)
        f_logits  = logits[predict_mask]
        f_domains = target_domains[predict_mask]
        local_ids = targets[predict_mask]
        
        offsets_per_domain = self.get_offset_per_domain(model, domains_of_interest=self.predicted_domains)
        
        offsets = offsets_per_domain[f_domains]
        global_ids = offsets + local_ids

        loss_ce = model.cross_entropy_loss(f_logits, global_ids)

        age_diff = (target_ages - input_ages)[predict_mask]
        time_loss = model.time_to_event_loss(f_logits, age_diff, t_min=1e-1, agg='mean')

        outputs = getattr(self, stage[:5] + "_outputs")
        
        prefix = "" if add_prefix is None else add_prefix + "_"

        loss = EasyDict({                
            f'{prefix}ce_loss': loss_ce,            
            f'{prefix}time_loss': time_loss,
            f'{prefix}ce_ema_loss': torch.lerp(outputs[-1][f'{prefix}ce_ema_loss'], loss_ce, weight=0.002) if outputs else loss_ce,
            f'{prefix}time_ema_loss': torch.lerp(outputs[-1][f'{prefix}time_ema_loss'], time_loss, weight=0.002) if outputs else time_loss,
            f'{prefix}total': loss_ce + time_loss,            
        })

        if log_per_disease:
            loss_ce_per_disease = model.cross_entropy_loss(
                f_logits, global_ids, agg="per_disease"
            ).to_frame().assign(batch_idx=batch_idx)

            loss[f'{prefix}ce_loss_per_disease'] = loss_ce_per_disease


        if return_att and return_logits: return loss, logits, att 
        elif return_logits:              return loss, logits
        elif return_att:                 return loss, att
        else:                            return loss


    def valid_epoch_end(self, loss_outputs):        
        
        self._valid_counter += 1        

        return pd.concat([x['val_ce_loss_per_disease'] for x in loss_outputs]).\
            reset_index().\
            pivot(index="token_id", columns="batch_idx", values="log_p").\
            fillna(0).\
            sum(axis=1).\
            sort_values()
        

    def train_epoch(self, n_batches=None, eval_every=None):

        pbar = tqdm(self.training_loader)

        for i, batch in enumerate(pbar):
            
            self.optimizer.zero_grad()
            loss = self.shared_step(batch, batch_idx=i, epoch=self.current_epoch, stage="training", add_prefix="train")
            self.train_outputs.append(loss)
            
            loss['train_total'].backward()

            if eval_every is not None and (i % eval_every) == 0:
                self.mean_val_loss = self.valid_epoch(n_batches=1000)                        

            pbar.set_postfix({
                "ce_loss":      f"{loss['train_ce_loss'].item():.4f}", 
                "time_loss":    f"{loss['train_time_loss'].item():.4f}",
                "loss_sm":      f"{loss['train_ce_ema_loss'].item():.4f}", 
                "time_loss_sm": f"{loss['train_time_ema_loss'].item():.4f}",
            })

            self.optimizer.step() 
            self.scheduler.step()

            if (n_batches is not None) and (i == n_batches):
                break


    def epoch_end(self):

        self.train_outputs = []
        self.valid_outputs = []


    def valid_epoch(self, n_batches=None):
        
        pbar = tqdm(self.valid_loader)

        with torch.no_grad():
            loss_outputs = []
            for i, batch in enumerate(pbar):
                loss = self.shared_step(batch, batch_idx=i, epoch=self.current_epoch, log_per_disease=True, stage="validation", add_prefix="val")
                loss_outputs.append(loss)
            
                pbar.set_postfix({
                    "ce_loss":      f"{loss['val_ce_loss'].item():.4f}", 
                    "time_loss":    f"{loss['val_time_loss'].item():.4f}",
                    "loss_sm":      f"{loss['val_ce_ema_loss'].item():.4f}", 
                    "time_loss_sm": f"{loss['val_time_ema_loss'].item():.4f}",
                })
            
                if n_batches is not None and i == n_batches:
                    break
        
        loss_per_disease_df = self.valid_epoch_end(loss_outputs).reset_index()
        loss_per_disease_df.to_csv(f"{odir}/loss_outputs_{self._valid_counter}.csv", index=False)

        loss = torch.stack([loss['val_ce_loss'] for loss in loss_outputs]).mean()

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


    def lengths_by_subject(self, subject_ids, ignore=None):
        out = {}
        for d, sids in subject_ids.items():
            if ignore is not None and d in ignore: 
                continue
            for sid in torch.unique(sids):
                out.setdefault(int(sid.item()), 0)
                out[int(sid.item())] += int((sids == sid).sum().item())
        return out
    

    def truncate_subjects_by_age(self,
        x: dict[str, torch.Tensor],
        ages: dict[str, torch.Tensor],
        subject_ids: dict[str, torch.Tensor],
        total_per_subject: dict[int, int],
        seqlen: int,
        trim_domains: set[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Truncate subjects with more than `seqlen` tokens, globally across trim_domains.
    
        Drops the most recent (highest age) events until each subject has exactly `seqlen` tokens total.
        Returns updated dicts with tokens removed.
        """
        device = next(iter(subject_ids.values())).device
    
        # Accumulate indices to drop per domain
        to_drop_by_domain: dict[str, list[int]] = defaultdict(list)
    
        for subj_id, tot in total_per_subject.items():
            diff = seqlen - tot
            if diff >= 0:
                continue  # nothing to drop
    
            R = -diff  # number of tokens to remove
    
            # Collect all candidate (age, domain, idx) from trim_domains
            candidates = []
            for d in trim_domains:
                if d not in subject_ids:
                    continue
                sids_d, ages_d = subject_ids[d], ages[d]
                idx = (sids_d == subj_id).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                for j in idx.tolist():
                    candidates.append((float(ages_d[j].item()), d, j))
    
            if not candidates:
                # No eligible domains for trimming
                continue
    
            # Sort by age (ascending) and remove the R most recent
            candidates.sort(key=lambda t: t[0])
            drop = candidates[-min(R, len(candidates)):]
            for _, d, j in drop:
                to_drop_by_domain[d].append(j)
    
        # Apply drops per domain
        for d, drop_list in to_drop_by_domain.items():
            if not drop_list:
                continue
            mask = torch.ones(len(subject_ids[d]), dtype=torch.bool, device=device)
            mask[torch.tensor(sorted(set(drop_list)), device=device)] = False
            x[d] = x[d][mask]
            ages[d] = ages[d][mask]
            subject_ids[d] = subject_ids[d][mask]
    
        return x, ages, subject_ids


    def pad_subjects(self,
        x: dict[str, torch.Tensor],
        ages: dict[str, torch.Tensor],
        subject_ids: dict[str, torch.Tensor],
        total_per_subject: dict[int, int],
        seqlen: int,
        pad_domain: str,
        PADDING_TOKEN: int,
        PAD_AGE: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Pad subjects with fewer than `seqlen` tokens by appending padding tokens
        in the specified `pad_domain`.
        """
        device = next(iter(subject_ids.values())).device
        dtype_x = next(iter(x.values())).dtype
        dtype_age = next(iter(ages.values())).dtype
        dtype_sid = next(iter(subject_ids.values())).dtype
    
        pad_x, pad_a, pad_sid = [], [], []
    
        for subj_id, tot in total_per_subject.items():
            diff = seqlen - tot
            if diff <= 0:
                continue
            pad_x.append(torch.full((diff,), PADDING_TOKEN, device=device, dtype=dtype_x))
            pad_a.append(torch.full((diff,), PAD_AGE, device=device, dtype=dtype_age))
            pad_sid.append(torch.full((diff,), subj_id, device=device, dtype=dtype_sid))
    
        if pad_x:
            x[pad_domain] = torch.cat([x[pad_domain], torch.cat(pad_x)])
            ages[pad_domain] = torch.cat([ages[pad_domain], torch.cat(pad_a)])
            subject_ids[pad_domain] = torch.cat([subject_ids[pad_domain], torch.cat(pad_sid)])
    
        return x, ages, subject_ids


    def adjust_to_seqlen(self,
        x: dict[str, torch.Tensor],
        ages: dict[str, torch.Tensor],
        subject_ids: dict[str, torch.Tensor],
        seqlen: int,
        *,
        pad_domain: str = "padding",
        trim_domains: set[str] = frozenset({"diseases"}),
        PADDING_TOKEN: int = 0,
        PAD_AGE: float = -10000.0,
    ):
        """
        Main orchestrator: ensures all subjects have exactly `seqlen` tokens in total,
        truncating by age when too long, padding otherwise.
        """
                
        x  = deepcopy(x)
        ages  = deepcopy(ages)
        subject_ids  = deepcopy(subject_ids)
     
        # Domains to consider for counting (exclude pad domain)
        domains = [d for d in subject_ids.keys() if d != pad_domain]
    
        # Count total tokens per subject (excluding padding)
        if domains:
            all_sids = torch.cat([subject_ids[d] for d in domains])
            subj_uniq, counts = np.unique(all_sids.cpu().int().numpy(), return_counts=True)
            total_per_subject = {int(k): int(v) for k, v in zip(subj_uniq, counts)}
        else:
            total_per_subject = {}
    
        # Step 1: truncate subjects with too many tokens
        x, ages, subject_ids = self.truncate_subjects_by_age(
            x, ages, subject_ids, total_per_subject, seqlen, trim_domains
        )
    
        # Step 2: recompute totals (after truncation)
        if domains:
            all_sids = torch.cat([subject_ids[d] for d in domains])
            subj_uniq, counts = np.unique(all_sids.cpu().int().numpy(), return_counts=True)
            total_per_subject = {int(k): int(v) for k, v in zip(subj_uniq, counts)}
    
        # Step 3: pad subjects that are still short
        x, ages, subject_ids = self.pad_subjects(
            x, ages, subject_ids, total_per_subject, seqlen, pad_domain, PADDING_TOKEN, PAD_AGE
        )
    
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

def get_cli_args():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_scheme", default="[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death]:causal(mask_ties=True)", nargs="+")
    parser.add_argument("--n_layer",          default=12)
    parser.add_argument("--test_fold",        default=1)
    parser.add_argument("--domains",          default=["diseases", "death", "lifestyle", "hla_alleles", "sex", "padding"])
    parser.add_argument("--run_name",         default="default")
    parser.add_argument("--batch_size",       default=4)
    args = parser.parse_args()

    return args

if __name__ == "__main__":    
    args =  get_cli_args()
else:
    args = EasyDict({
        "attention_scheme": ["[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death]:causal(mask_ties=True)"],
        "n_layer": 12,
        "test_fold": 3,
        "patience": 2,
        "batch_size": 4,
        "run_name": os.getenv("RUN_NAME", "default2"),
    })

if isinstance(args.attention_scheme, str):
    args.attention_scheme = [args.attention_scheme]

tokens_path = root_path / 'tokens'

default_cfg_per_domain = {
# 'genetic_pcs': EmbedConfig(projector="linear", path=tokens_path / 'genetic_pcs', type='continuous', at_birth=True),
  'diseases':    EmbedConfig(projector="embed", path=tokens_path / 'diseases',    predict=True),
  'death':       EmbedConfig(projector="embed", path=tokens_path / 'death',       predict=True),
  'lifestyle':   EmbedConfig(projector="embed", path=tokens_path / 'lifestyle',   age_jitter=True),  
  "hla_alleles": EmbedConfig(projector="embed", path=tokens_path / 'hla_alleles', at_birth=True),
  "sex":         EmbedConfig(projector="embed", path=tokens_path / 'sex',         at_birth=True),
  "padding":     EmbedConfig(projector="embed")    
}

# k, v doesn't work for some reason!
domain_cfg = { k: default_cfg_per_domain[k] for k in default_cfg_per_domain for k in args.domains }

assert all([k in default_cfg_per_domain for k in args.domains])

# —————————————————————————————————————————————————————————————————————————————————————————————————————————

train_ids, val_ids, test_ids = get_data_partitions("../data/transforms/subject_lists", fold=args.test_fold)

dataset_config = dict(root=root_path, domains=domain_cfg, exclusions=[])

train_dataset = DelphiDataset(subjects=train_ids, **dataset_config).to(DEVICE)
valid_dataset = DelphiDataset(subjects=val_ids,   **dataset_config).to(DEVICE)
test_dataset  = DelphiDataset(subjects=test_ids,  **dataset_config).to(DEVICE)

dataloaders = [ DelphiDataloader(d, batch_size=args.batch_size) for d in [train_dataset, valid_dataset, test_dataset] ]

# —————————————————————————————————————————————————————————————————————————————————————————————————————————

# ATTENTION_SCHEME = "[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death]:causal(mask_ties=True)"
# ATTENTION_SCHEME = "[hla_alleles,sex,diseases,lifestyle,death]:causal(mask_ties=True)"
# ATTENTION_SCHEME = "[sex,diseases,lifestyle,death]:causal(mask_ties=True)"
# ATTENTION_SCHEME = "[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,genetic_pcs,death]:causal(mask_ties=True)"
# ATTENTION_SCHEME = "[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death]:causal(mask_ties=True)"
# ATTENTION_SCHEME = "[h,s]:bidirectional,[s,dis,l,de]:causal(mask_ties=True)"

assert len(args.attention_scheme) in {1, args.n_layer}, f"len of the --attention_scheme argument should be either 1 or args.n_layer (={args.n_layer})"

if len(args.attention_scheme) == 1:
    attention_scheme = args.n_layer * args.attention_scheme
elif args.n_layer == len(args.attention_scheme):
    attention_scheme = args.attention_scheme

print(domain_cfg)

config = DelphiConfig(
    n_layer=args.n_layer, 
    token_dropout=0.1, 
    domains=domain_cfg, 
    attention_scheme=attention_scheme
)

model  = Delphi(config).to(DEVICE)
torch.compile(model)

optim_config = OptimConfig()
optimizer, scheduler = configure_optimizers(model=model, cfg=optim_config, device_type=DEVICE)
# %%


# %%
# —————————————————————————————————————————————————————————————————————————————————————————————————————————

assert args.run_name != 'default', f"You are using the 'default' value for run_name."
logger = MLFlowLogger(run_name=args.run_name)

trainer = Trainer(model, *dataloaders, optimizer, scheduler, logger=logger)
trainer.train(max_epochs=1000)

# %%
