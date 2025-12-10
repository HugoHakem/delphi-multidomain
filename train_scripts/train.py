# %%
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["DELPHI_DATA_DIR"] = os.getenv("DELPHI_DATA_DIR", "../data")
os.environ["DELPHI_CKPT_DIR"] = os.getenv("DELPHI_CKPT_DIR", "../output/checkpoints")
from pathlib import Path
root_path = Path("../data/transforms")

import re
import ast
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, random_split, DataLoader

import tempfile
import shutil
from urllib.parse import urlparse
from typing import List, Dict
from copy import deepcopy

from easydict import EasyDict
import mlflow

from tqdm import tqdm
import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from collections import defaultdict

from data.dataset import DelphiDataset, DelphiDataloader
from utils.cv_utils import get_data_partitions
from utils.profiling import profile_and_print

from delphi.optim import OptimConfig, configure_optimizers
from delphi.model.transformer import (
    Delphi,
    EmbedConfig,
    DelphiConfig,
)

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

RUN_NAME = os.getenv("RUN_NAME", "default")
odir = f"output/{RUN_NAME}"

# os.makedirs(odir, exist_ok=True)

print(f"{sys.stdout.isatty()=}")

USE_TQDM = sys.stdout.isatty()
print(f"{USE_TQDM=}")

def clone_run_to_new_experiment(
    old_run_id: str,
    new_experiment_name: str,
    new_run_name: str = None
) -> str:
    """
    Clone an existing MLflow run into a NEW experiment (fresh run + copied artifacts).

    Parameters
    ----------
    old_run_id : str
        The MLflow run ID to clone.
    new_experiment_name : str
        Name for the new experiment (created if it doesn't exist).
    new_run_name : str, optional
        Name for the new run. Defaults to "<old_run_name>_resumed".

    Returns
    -------
    new_run_id : str
        ID of the newly created run in the new experiment.
    """
    # Fetch old run data
    old_run = mlflow.get_run(old_run_id)
    old_run_name = old_run.data.tags.get("mlflow.runName", "unnamed_run")

    # Create or get target experiment
    exp = mlflow.get_experiment_by_name(new_experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(new_experiment_name)
    else:
        exp_id = exp.experiment_id

    # Start the new run
    new_run = mlflow.start_run(
        experiment_id=exp_id,
        run_name=new_run_name or f"{old_run_name}_resumed"
    )
    new_run_id = new_run.info.run_id

    # Copy params and tags
    # mlflow.log_params(old_run.data.params)
    for k, v in old_run.data.tags.items():
        mlflow.set_tag(k, v)
    mlflow.set_tag("resumed_from", old_run_id)

    # Copy all artifacts
    src_dir = mlflow.artifacts.download_artifacts(run_id=old_run_id)
    dst_dir = Path(mlflow.get_artifact_uri()).as_posix().replace("file://", "")
    dst_dir = Path(dst_dir)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    print(f"✅ Cloned run {old_run_id} → {new_run_id} (experiment: {new_experiment_name})")
    print(f"Artifacts copied from {src_dir} to {dst_dir}")

    mlflow.end_run()
    return new_run_id


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

VAL_EVERY_NSAMPLES = 100000

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
    """
    Simple MLflow wrapper that manages an active run and
    provides convenient methods for logging parameters, metrics, and artifacts.
    """

    def __init__(self, experiment_name, run_name=None, autostart=True, nested=False):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.active_run = None
        self.tracking_base = None
        if autostart:
            self.start(nested=nested)

    def start(self, nested=False, resume_run_id=None):
      """
      Start a new MLflow run, or resume an existing one if resume_run_id is provided.
      """
      mlflow.set_experiment(self.experiment_name)
      if self.active_run is None:
          if resume_run_id:
            # Resume an existing run
            self.active_run = mlflow.start_run(run_id=resume_run_id)
          else:
            # Start a fresh run
            self.active_run = mlflow.start_run(run_name=self.run_name, nested=nested)

      # Store tracking base for relative URI resolution
      self.tracking_base = self._strip_file_prefix(mlflow.get_tracking_uri())
      return self.active_run

    def end(self):
        """
        End the active MLflow run.
        """
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params):
        """
        Log a dictionary of parameters.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        """
        Log a dictionary of metrics.
        Optionally specify a global step.
        """
        mlflow.log_metrics(metrics, step=step)


    def log_artifact(self, path, artifact_path=None, relative_uri=True):
        """
        Log a single artifact file and optionally return a relative artifact URI
        instead of the absolute one (useful for portable runs).
        """
        mlflow.log_artifact(path, artifact_path=artifact_path)
        uri = mlflow.get_artifact_uri(artifact_path)

        if relative_uri and self.tracking_base:
            uri = self._strip_file_prefix(uri)
            if uri.startswith(self.tracking_base):
                uri = os.path.relpath(uri, self.tracking_base)
        return uri


    def log_df_as_artifact(self, df: pd.DataFrame, filename="data.csv", artifact_path=None, relative_uri=True):
        """
        Save a DataFrame as a temporary CSV, log it as an artifact,
        and optionally return a relative artifact URI.
        """
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, filename)
        df.to_csv(tmp_path, index=False)

        try:
            uri = self.log_artifact(tmp_path, artifact_path=artifact_path, relative_uri=relative_uri)
        finally:
            shutil.rmtree(tmp_dir)
        return uri


    @staticmethod
    def _strip_file_prefix(uri: str) -> str:
        """
        Remove file:// prefix if present.
        """
        return urlparse(uri).path if uri.startswith("file://") else uri


    def save_model(self, model, optimizer, scheduler=None, metadata=None, filename=None):
        """
        Save a model checkpoint inside the current MLflow run's artifact directory.
    
        Args:
            model: The PyTorch model (nn.Module).
            metadata: Optional dict with extra info (epoch, val_loss, etc.).
            filename: Optional filename for the checkpoint.
        """
        from datetime import datetime
        import torch
        from pathlib import Path
    
        if metadata is None:
            metadata = {}
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"best_model_{timestamp}.pt"
    
        # Create a temporary checkpoint
        tmp_dir = tempfile.mkdtemp()
        tmp_path = Path(tmp_dir) / filename
        torch.save({
            "state_dict": model.state_dict(), 
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metadata": metadata
            }, 
            tmp_path
        )
    
        # Log to MLflow artifacts
        artifact_path = "checkpoints"
        mlflow.log_artifact(str(tmp_path), artifact_path=artifact_path)
        shutil.rmtree(tmp_dir)
    
        uri = mlflow.get_artifact_uri(artifact_path)
        print(f"Saved checkpoint at {uri}/{filename}")
        return Path(uri) / filename


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

class Trainer():

    def __init__(self, model, dataloaders, 
          optimizer, scheduler, patience=3, 
          n_train_batches=None, n_val_batches=None, 
          logger=NullLogger(), mlflow_params=dict(), start_epoch=0
        ):

        '''

        '''

        self.model           = model        
        self.optimizer       = optimizer
        self.scheduler       = scheduler        
        self.early_stopper   = EarlyStopping(patience=patience, min_delta=0.001, mode='min')                        

        assert isinstance(dataloaders, list), "Argument 'dataloaders' should be a list of either 2 or 3 dataloaders (train/val[/test])"
        if len(dataloaders) == 2:
            self.train_loader, self.valid_loader = dataloaders
        elif len(dataloaders) == 3:
            self.train_loader, self.valid_loader, self.test_loader = dataloaders
        else:
            raise ValueError(f"{len(dataloaders)=} ")
        
        self.n_train_batches, self.n_val_batches = n_train_batches or 'all', n_val_batches or 'all'
        
        self.train_outputs,   self.valid_outputs, self.test_outputs = [], [], []            

        self.current_epoch  = start_epoch
        self._valid_counter = 0
        
        self.logger = logger
        self.val_loss = None
        
        self.ema_alpha = 0.005
        
        self.additional_mlflow_params = mlflow_params | { "ema_alpha": self.ema_alpha }

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


    def get_subject_ids_per_partition(self):

        subject_ids = { 
            "train_ids": sorted(self.train_loader.dataset.subjects),
            "valid_ids": sorted(self.valid_loader.dataset.subjects),
            "test_ids":  sorted(self.test_loader.dataset.subjects),
        }
        return subject_ids


    def get_vocab_len(self, domain_name):
        self.model.transformer.embed.domain_embed[domain_name].weight.shape[0]


    def train(self, max_epochs=1000):

        self.logger.log_params(self.model.config)
        self.logger.log_params(self.additional_mlflow_params)

        # self.logger.log_params(self.optimizer.config)
        # self.logger.log_params(self.scheduler.config)
        # profile_and_print(self.shared_step, self.training_loader, self.optimizer, n_steps=2, top_k=20) 

        for epoch in range(self.current_epoch, max_epochs):
            
            self.current_epoch = epoch            
            train_loss = self.train_epoch(eval_every=VAL_EVERY_NSAMPLES//args.batch_size)

            metrics = { "train_loss": train_loss }
             
            if self.val_loss is not None:
                metrics["val_loss"] = self.val_loss
                val_loss_per_disease = metrics["val_loss"].pop("val_ce_loss_per_disease")

            self.logger.log_metrics(metrics['val_loss'], step=epoch)            
            self.logger.log_metrics(metrics['train_loss'], step=epoch)            

            should_stop, improved = self.early_stopper.step(self.mean_val_loss)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if improved:

                ckpt_uri = self.logger.save_model(
                    self.model,
                    self.optimizer,
                    self.scheduler,                    
                    metadata={
                      "epoch": epoch, 
                      "val_loss": float(self.mean_val_loss.cpu()),
                      "train_loss": float(train_loss["train_total"].cpu()),
                      "n_params": sum(p.numel() for p in self.model.parameters()),
                      "timestamp": datetime.now().isoformat(timespec="seconds"),
                      "attention_scheme": getattr(self.model.config, "attention_scheme", None),
                      "date_cutoff": None,  # placeholder - to be set later when needed
                    } | self.get_subject_ids_per_partition(),
                    filename=f"best_model_epoch{self.current_epoch}_trainloss_{train_loss['train_total']}_{timestamp}.pt"
                )
                print(f"New best model logged at {ckpt_uri}")
 
            if should_stop:
                print(f"Early stopping triggered at epoch {self.current_epoch}")
                break

            self.epoch_end()
            


    def shared_step(self, batch, batch_idx, epoch, log_per_disease=False, return_logits=False, return_att=False, stage="training", add_prefix=None):

        model = self.model

        # events = EventSet.from_batch(batch)
        # events = events.\
        #  compute_max_ages().\
        #  insert_no_event_tokens(rate=5).\
        #  mask_tokens_after_age().\
        #  adjust_to_seqlen(96)

        x, ages, subject_ids = trainer.get_tensors_from_batch(batch)
        max_ages             = model.get_max_ages_per_subject(ages, subject_ids)
        x, ages, subject_ids = model.insert_no_event_tokens(x, ages, subject_ids)
        x, ages, subject_ids = model.mask_tokens_after_age (x, ages, subject_ids, max_ages)
        x, ages, subject_ids = trainer.adjust_to_seqlen(x, ages, subject_ids, seqlen:=96)
        
        logits, att = model(x, ages, subject_ids)
        
        # This is necessary
        domains        = model._trace['domains']
        targets        = model._trace['tokens'][:,1:]
        input_ages     = model._trace['ages'][:,:-1]
        target_ages    = model._trace['ages'][:,1:]
        target_domains = domains[:,1:]
        
        predict_mask = torch.isin(target_domains, self.predicted_domains_as_int)        
        logits       = torch.cat([logits[dname] for dname in self.predicted_domains], axis=-1)
        logits       = logits[:,:-1,:]
        f_logits     = logits[predict_mask]
        f_domains    = target_domains[predict_mask]
        local_ids    = targets[predict_mask]
        
        # import ipdb; ipdb.set_trace()
        offsets_per_domain = self.get_offset_per_domain(model, domains_of_interest=self.predicted_domains)
        
        offsets = offsets_per_domain[f_domains]
        global_ids = offsets + local_ids

        # import ipdb; ipdb.set_trace()
        loss_ce = model.cross_entropy_loss(f_logits, global_ids)

        age_diff = (target_ages - input_ages)[predict_mask]
        time_loss = model.time_to_event_loss(f_logits, age_diff, t_min=1e-1, agg='mean')

        outputs = getattr(self, stage[:5] + "_outputs")
        
        prefix = "" if add_prefix is None else add_prefix + "_"

        loss = EasyDict({                
            f'{prefix}ce_loss':       loss_ce,            
            f'{prefix}time_loss':     time_loss,
            f'{prefix}ce_ema_loss':   torch.lerp(outputs[-1][f'{prefix}ce_ema_loss'], loss_ce, weight=0.002) if outputs else loss_ce,
            f'{prefix}time_ema_loss': torch.lerp(outputs[-1][f'{prefix}time_ema_loss'], time_loss, weight=0.002) if outputs else time_loss,
            f'{prefix}total':         loss_ce + time_loss,            
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
        

    def compute_mean(self, outputs: List[Dict]):

        out = {}
        if not outputs:
            return out
    
        for k in outputs[0].keys():
            try:
                vals = [
                    (v[k].detach() if torch.is_tensor(v[k]) else torch.as_tensor(v[k]))
                    for v in outputs
                    if k in v
                ]
                out[k] = torch.stack(vals).mean()
            except Exception as e:
                print(f"[compute_mean] Skipping key '{k}': {e}")
                out[k] = torch.tensor(float('nan'))
        return out
        


    def train_epoch(self, n_batches=None, eval_every=None):

        self.val_step = 0
        ce_ema, time_ema = None, None

        pbar = tqdm(
            total=len(self.train_loader) if n_batches == "all" else n_batches, 
            disable=not USE_TQDM
        )

        for i, batch in enumerate(self.train_loader):
            
            self.optimizer.zero_grad()
            loss = self.shared_step(batch, batch_idx=i, epoch=self.current_epoch, stage="training", add_prefix="train")
            
            loss['train_total'].backward()
            self.optimizer.step()
            self.scheduler.step() 

            with torch.no_grad():
                ce_val = loss['train_ce_loss'].detach()
                time_val = loss['train_time_loss'].detach()
                ce_ema = ce_val if ce_ema is None else (1 - self.ema_alpha) * ce_ema + self.ema_alpha * ce_val
                time_ema = time_val if time_ema is None else (1 - self.ema_alpha) * time_ema + self.ema_alpha * time_val
        
            self.train_outputs.append({
                'train_ce_loss': ce_val,
                'train_time_loss': time_val,
                'train_ce_ema_loss': ce_ema,
                'train_time_ema_loss': time_ema,
                'train_total': ce_val + time_val,
            })
            
            if eval_every is not None and (i % eval_every) == 0:
                self.val_loss, self.mean_val_loss = self.valid_epoch(n_batches=self.n_val_batches)
                self.val_step += 1                        
    
            pbar.set_postfix({
                "train_cce_loss":  f"{loss['train_ce_loss'].item():.3f} ({loss['train_ce_ema_loss'].item():.3f})", 
                "train_time_loss": f"{loss['train_time_loss'].item():.3f} ({loss['train_time_ema_loss'].item():.3f})"
            })

            pbar.update(self.train_loader.batch_size)
    
            if (n_batches is not None) and (i == n_batches):
                break
            
        return self.compute_mean(self.train_outputs)
                

    def valid_epoch(self, n_batches='all'):
        
        pbar = tqdm(
            total=len(self.valid_loader) if n_batches == "all" else n_batches, 
            disable=not USE_TQDM
        )

        with torch.no_grad():
            
            loss_outputs = []
            for i, batch in enumerate(self.valid_loader):

                loss = self.shared_step(batch, batch_idx=i, epoch=self.current_epoch, log_per_disease=True, stage="validation", add_prefix="val")
                loss_outputs.append(loss)
            
                pbar.set_postfix({
                    "val_cce":      f"{loss['val_ce_loss'].item():.3f} ({loss['val_ce_ema_loss'].item():.3f})", 
                    "val_time_loss":    f"{loss['val_time_loss'].item():.3f} ({loss['val_time_ema_loss'].item():.3f})"
                })
            
                pbar.update(self.valid_loader.batch_size)

                if n_batches != "all" and i == n_batches:
                    break
        
        pbar.close()

        loss_per_disease_df = self.valid_epoch_end(loss_outputs).reset_index()
        self.logger.log_df_as_artifact(df=loss_per_disease_df, filename=f"losses_epoch{self.current_epoch}_{self.val_step}.csv", artifact_path="val_loss_per_disease")
        loss_per_disease_df.to_csv(f"{odir}/loss_outputs_{self._valid_counter}.csv", index=False)

        mean_loss = torch.stack([loss['val_ce_loss'] for loss in loss_outputs]).mean()

        return loss, mean_loss,

    
    def epoch_end(self):
        self.train_outputs = []
        self.valid_outputs = []


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

        # TODO: check if this gives the same results
        # vocab_lens = model.vocab_lens
        vocab_lens = { 
            domain_to_int[k]: v.vocab_len 
            for k, v in self.model.transformer.embed.domain_embed.items() if k in domains_of_interest 
        }
        
        offsets_per_domain = np.array([0] + list(vocab_lens.values())).cumsum()[:-1]
        offsets_per_domain = torch.tensor(offsets_per_domain).to(DEVICE)
        return offsets_per_domain

# ----------------------------------------------------------------------------------------------------

def config_from_runid(runid):

    VAL_BATCH_SIZE = 256

    # Retrieve run info (contains experiment ID and tags)
    runinfo = mlflow.get_run(runid)
    
    # experiment_id = runinfo.info.experiment_id
    # experiment_name = mlflow.get_experiment(experiment_id).name
    
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
    runinfo.data.params['domains'] = { k: EmbedConfig(**v) for k, v in runinfo.data.params['domains'].items() }
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
    print(ckpt_dir)
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
    train_dataset = DelphiDataset(root="../data/transforms", domains=delphi_cfg.domains, subjects=ckpt['metadata']['train_ids']).to("cuda")
    valid_dataset = DelphiDataset(root="../data/transforms", domains=delphi_cfg.domains, subjects=ckpt['metadata']['valid_ids']).to("cuda")
    test_dataset  = DelphiDataset(root="../data/transforms", domains=delphi_cfg.domains, subjects=ckpt['metadata']['test_ids']).to("cuda")
    
    dataloaders = [
        train_loader := DelphiDataloader(train_dataset, batch_size=batch_size),
        valid_loader := DelphiDataloader(valid_dataset, batch_size=VAL_BATCH_SIZE), 
        test_loader  := DelphiDataloader(test_dataset,  batch_size=VAL_BATCH_SIZE)
    ]    

    previous_run_name = runinfo.data.tags.get("mlflow.runName", None)
    logged_params = { "test_fold": test_fold, "batch_size": batch_size }    

    return model, dataloaders, optimizer, scheduler, logged_params, previous_run_name


# ——————————————— CONFIG ———————————————————————————————————————————————————————————————

def get_cli_args():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_scheme", default="[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death,padding]:causal(mask_ties=True)", nargs="+")
    parser.add_argument("--n_layer",          default=12,   type=int)
    parser.add_argument("--n_head",           default=10,   type=int)
    parser.add_argument("--n_embd",           default=120,  type=int)
    parser.add_argument("--test_fold",        default=1,    type=int)
    parser.add_argument("--subjects",         default=None, type=str)
    parser.add_argument("--domains",          default="diseases,death,cv_drugs,ns_drugs,lifestyle,hla_alleles,sex,padding")
    parser.add_argument("--experiment_name",  default="drugs-predicted")
    parser.add_argument("--run_name",         default=None)
    parser.add_argument("--batch_size",       default=16, type=int)
    parser.add_argument("--learning_rate", "--lr", dest="lr", default=1e-4, type=float)
    parser.add_argument("--resume_run_id", type=str, default=None,
                    help="Resume training from the latest checkpoint of this MLflow run")
    parser.add_argument("--dryrun", "--dry-run", "--dry_run", dest="dry_run", action="store_true", default=False)

    args = parser.parse_args()

    return args

if __name__ == "__main__":    
    args =  get_cli_args()
    
else:
    args = EasyDict({
        "attention_scheme": ["[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death, hla_alleles]:causal(mask_ties=True)"],
        "n_layer": 12,
        "test_fold": 3,
        "patience": 2,
        "batch_size": 4,
        "lr": 1e-4,
        "run_name": os.getenv("RUN_NAME", "default"),
    })

if isinstance(args.attention_scheme, str):
    args.attention_scheme = [args.attention_scheme]

# %%

if __name__ == "__main__":

  if not args.resume_run_id:

    ################################ FROM SCRATCH ################################

    tokens_path = root_path / 'tokens'

    default_cfg_per_domain = {
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
    
    domains = args.domains.split(",")
    # k, v with .items() doesn't work for some reason!
    domain_cfg = { k: default_cfg_per_domain[k] for k in default_cfg_per_domain for k in domains }
    
    assert all([k in default_cfg_per_domain for k in domains])
    
    # —————————————————————————————————————————————————————————————————————————————————————————————————————————
    
    train_ids, val_ids, test_ids = get_data_partitions("../data/transforms/subject_lists", fold=args.test_fold)
    
    if args.subjects is not None:
        subject_ids = pd.read_csv(args.subjects, header=None)[0].tolist()
        train_ids   = list(set(train_ids) & set(subject_ids))
        val_ids     = list(set(val_ids) & set(subject_ids))
        test_ids    = list(set(test_ids) & set(subject_ids))

    dataset_config = dict(root=root_path, domains=domain_cfg, exclusions=[], required_domains=["diseases"])
    
    train_dataset = DelphiDataset(subjects=train_ids, **dataset_config).to(DEVICE)
    valid_dataset = DelphiDataset(subjects=val_ids,   **dataset_config).to(DEVICE)
    test_dataset  = DelphiDataset(subjects=test_ids,  **dataset_config).to(DEVICE)
    
    dataloaders = [ DelphiDataloader(d, batch_size=[args.batch_size, 128, 128][i]) for i, d in enumerate([train_dataset, valid_dataset, test_dataset]) ]
    
    # —————————————————————————————————————————————————————————————————————————————————————————————————————————
      
    assert len(args.attention_scheme) in {1, args.n_layer}, f"len of the --attention_scheme argument should be either 1 or args.n_layer (={args.n_layer})"
    
    if len(args.attention_scheme) == 1:
        attention_scheme = args.n_layer * args.attention_scheme
    elif args.n_layer == len(args.attention_scheme):
        attention_scheme = args.attention_scheme
    
    config = DelphiConfig(
        n_embd=args.n_embd,
        n_layer=args.n_layer, 
        token_dropout=0.1, 
        domains=domain_cfg, 
        attention_scheme=attention_scheme
    )
    
    model  = Delphi(config).to(DEVICE)
    torch.compile(model)
    
    optim_config = OptimConfig(learning_rate=args.lr, min_lr=args.lr/10)
    optimizer, scheduler = configure_optimizers(model=model, cfg=optim_config, device_type=DEVICE)  
    
    assert args.run_name != 'default', f"You are using the 'default' value for run_name."
    logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=args.run_name)
    
    logged_params = { "test_fold": args.test_fold, "batch_size": args.batch_size, "learning_rate": args.lr }
   
  else:

    ################################ FROM PREVIOUS RUN ################################

    model, dataloaders, optimizer, scheduler, logged_params, previous_run_name = config_from_runid(args.resume_run_id)
        
    new_run_id = clone_run_to_new_experiment(args.resume_run_id, args.experiment_name)
    logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=previous_run_name, autostart=False)
    logger.start(resume_run_id=new_run_id)

    #TODO: Add possibility to change some parameters, e.g. attention scheme, or add domains (e.g. genetic PCs and HLA alleles)
    print(f"Resuming from MLflow run {args.resume_run_id} ...")

# —————————————————————————————————————————————————————————————————————————————————————————————————————————

  trainer = Trainer(model, dataloaders, optimizer, scheduler, logger=logger, mlflow_params=logged_params)
  trainer.train(max_epochs=1000)

# %%
