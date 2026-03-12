import pandas as pd
from typing import List, Dict, Union
import torch
import mlflow
from tqdm import tqdm
import os, sys
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from easydict import EasyDict
from urllib.parse import urlparse

if ( DELPHI_DIR := Path(__file__).resolve().parent.parent ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from data.event_set import EventSet

from dataclasses import asdict
from pathlib import Path
import json


def lod2dol(lod):
    """
    Convert list of dicts -> dict of lists.
    """
    from collections import defaultdict
    dol = defaultdict(list)
    for d in lod:
        for k, v in d.items():
            dol[k].append(v.item())
    return dict(dol)


def make_json_serializable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: make_json_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_json_serializable(v) for v in x]
    return x


def clone_run_to_new_experiment(old_run_id: str, new_experiment_name: str, new_run_name: str = None) -> str:

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

    print(f"Cloned run {old_run_id} → {new_run_id} (experiment: {new_experiment_name})")
    print(f"Artifacts copied from {src_dir} to {dst_dir}")

    mlflow.end_run()
    return new_run_id


# ————————————————————————————————————————————————————————————————————————————————————————————

VAL_EVERY_NSAMPLES = 100000
VAL_EVERY_NSAMPLES = 1e18

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


    def set_patience(self, patience):
        self.patience = patience


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

    def build_state_dict(self, model, optimizer, scheduler=None, metadata=None):

        return { 
            "state_dict": model.state_dict(), 
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metadata": metadata
        }


    def save_model(self, model, optimizer, scheduler=None, metadata=None, filename=None):
        """
        Save a model checkpoint inside the current MLflow run's artifact directory.
    
        Args:
            model: The PyTorch model (nn.Module).
            metadata: Optional dict with extra info (epoch, val_loss, etc.).
            filename: Optional filename for the checkpoint.
        """

        import torch        
    
        metadata = {} if metadata is None else metadata
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"best_model_{timestamp}.pt"
    
        # Create a temporary checkpoint
        tmp_dir = tempfile.mkdtemp()
        tmp_path = Path(tmp_dir) / filename
        torch.save( self.build_state_dict(model, optimizer, scheduler, metadata), tmp_path )
    
        # Log to MLflow artifacts
        artifact_path = "checkpoints"
        mlflow.log_artifact(str(tmp_path), artifact_path=artifact_path)
        shutil.rmtree(tmp_dir)
    
        uri = mlflow.get_artifact_uri(artifact_path)
        print(f"Saved checkpoint at {uri}/{filename}")
        return Path(uri) / filename


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# Just a base class that shows the expected interface for child classes
class BaseTrainer():

    def shared_step(self):     pass
    
    def train_epoch(self):     pass
    def train(self):           pass
    
    def val_epoch(self):       pass
    def valid_epoch_end(self): pass


class Trainer(BaseTrainer):

    LOSSES_PER_EPOCH_FILEPATTERN = "losses_epoch{current_epoch}_{val_step}.csv"

    def __init__(self, model, dataloaders, 
          optimizer, scheduler, patience=3, 
          n_train_batches=None, n_val_batches=None, n_validations_per_epoch=1,
          logger=NullLogger(), mlflow_params=dict(), start_epoch=0, log_loss_per_disease=False,
          use_tqdm=True
        ):

        '''
        Trainer class, mimicking Pytorch Lightning trainer
        '''

        self.model           = model        
        self.optimizer       = optimizer
        self.scheduler       = scheduler        
        self.early_stopper   = EarlyStopping(patience=patience, min_delta=0.001, mode='min')                        

        assert isinstance(dataloaders, list), \
               "Argument 'dataloaders' should be a list of either 2 or 3 dataloaders (train/val[/test])"

        if len(dataloaders) == 2:
            self.train_loader, self.valid_loader = dataloaders
        elif len(dataloaders) == 3:
            self.train_loader, self.valid_loader, self.test_loader = dataloaders
        else:
            raise ValueError(f"{len(dataloaders)=} ")
        
        self.n_train_batches, self.n_val_batches = n_train_batches or 'all', n_val_batches or 'all'
        
        self.train_outputs, \
        self.valid_outputs, \
        self.test_outputs = [], [], []            

        self.current_epoch = start_epoch
        self._validation_counter = 0
        self.n_validations_per_epoch = n_validations_per_epoch
        
        self.logger = logger
        self.val_loss = None
        self.ema_alpha = 0.02

        self.additional_mlflow_params = mlflow_params | { "ema_alpha": self.ema_alpha }

        self.use_tqdm = use_tqdm

        self.ce_ema = None
        self.time_ema = None
        self.log_loss_per_disease = log_loss_per_disease
        
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True

    # ——————————————————————————————————————————————————————————————————————————————
   
    @property
    def device(self):
        return self.model.device


    def get_subject_ids_per_partition(self):

        subject_ids = { 
            "train_ids": sorted(self.train_loader.dataset.subjects),
            "valid_ids": sorted(self.valid_loader.dataset.subjects),
            "test_ids":  sorted(self.test_loader.dataset.subjects),
        }
        return subject_ids


    def get_vocab_len(self, domain_name):
        self.model.transformer.embed.domain_embed[domain_name].weight.shape[0]


    def train(self, max_epochs=1000, patience=None):

        if patience is not None:
            self.early_stopper.set_patience(patience)
        
        n_batches_epoch = len(self.train_loader)
        if self.n_validations_per_epoch <= 0:
            eval_every = None
        else:
            eval_every = max(1, n_batches_epoch // self.n_validations_per_epoch)

        self.logger.log_params(self.model.config)
        self.logger.log_params(self.additional_mlflow_params)

        for epoch in range(self.current_epoch, max_epochs):
            
            self.current_epoch = epoch
            self.model.train()

            train_loss = self.train_epoch(eval_every=eval_every)

            metrics = { "train_loss": train_loss }
             
            if self.val_loss is not None:
                metrics["val_loss"] = self.val_loss
                if "val_ce_loss_per_disease" in metrics["val_loss"]:
                    val_loss_per_disease = metrics["val_loss"].pop("val_ce_loss_per_disease")

            self.logger.log_metrics(metrics['val_loss'], step=epoch)            
            self.logger.log_metrics(metrics['train_loss'], step=epoch)            

            should_stop, improved = self.early_stopper.step(self.mean_val_loss)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if improved:

                ckpt_filepath = f"best_model__epoch{self.current_epoch}__trainloss_{train_loss['train_total']:.4f}__{timestamp}.pt"

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
                    filename=ckpt_filepath
                )
                print(f"New best model logged at {ckpt_uri}")
 
            if should_stop:
                print(f"Early stopping triggered at epoch {self.current_epoch}")
                break

            self.epoch_end()
            

    def _prepare_input_simplified(self):
        
        '''
        Sample only one input tensor from the training loader and reuse it
        for profiling purposes
        '''

        if not hasattr(self, "_sample_input_tensor"):
            sample_batch = next(iter(self.train_loader))
            self._sample_input_tensor = self.model.prepare_input(sample_batch, seqlen=self.seqlen)
        return self._sample_input_tensor

    
    def shared_step(self, 
          batch, batch_idx, epoch,           
          return_logits=False, return_att=False, stage="training", add_prefix=None
        ):

        model = self.model

        x, ages, subject_ids = self.model.prepare_input(batch)

        logits, att = model(x, ages, subject_ids)

        # This is ugly but necessary at the moment
        domains        = model._trace['domains']
        targets        = model._trace['tokens'][:,1:]
        input_ages     = model._trace['ages'][:,:-1]
        target_ages    = model._trace['ages'][:,1:]
        target_domains = domains[:,1:]
        
        predict_mask = torch.isin(target_domains, self.model.predicted_domains_as_int)        
        logits       = torch.cat([logits[dname] for dname in self.model.predicted_domains], axis=-1)
        logits       = logits[:,:-1,:]
        f_logits     = logits[predict_mask]
        f_domains    = target_domains[predict_mask]
        local_token_ids    = targets[predict_mask]
        
        global_ids   = self.model.local_to_global_ids(f_domains, local_token_ids)

        loss_ce  = model.cross_entropy_loss(f_logits, global_ids)

        age_diff = (target_ages - input_ages)[predict_mask]
        time_loss = model.time_to_event_loss(f_logits, age_diff, t_min=1e-1, agg='mean')

        outputs = getattr(self, stage[:5] + "_outputs")
        
        prefix = "" if add_prefix is None else add_prefix + "_"

        loss = EasyDict({
            f'{prefix}ce_loss': loss_ce,
            f'{prefix}time_loss': time_loss,
            f'{prefix}total': loss_ce + time_loss,
        })

        if self.log_loss_per_disease:
            loss_ce_per_disease = model.cross_entropy_loss(
                f_logits, global_ids, agg="per_disease"
            ).to_frame().assign(batch_idx=batch_idx)

            loss[f'{prefix}ce_loss_per_disease'] = loss_ce_per_disease

        if return_att and return_logits: return loss, logits, att 
        elif return_logits:              return loss, logits
        elif return_att:                 return loss, att
        else:                            return loss
      

    def valid_epoch_end(self, loss_outputs):        
        
        self._validation_counter += 1        

        if len(loss_outputs) and 'val_ce_loss_per_disease' in loss_outputs[0]:
            self._val_ce_loss_per_disease_df = pd.concat([x['val_ce_loss_per_disease'] for x in loss_outputs]).\
                reset_index().\
                pivot(index="token_id", columns="batch_idx", values="log_p").\
                fillna(0).\
                sum(axis=1).\
                sort_values().\
                reset_index()

        return 1
        

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
          

    def train_epoch(self, n_batches='all', eval_every=None):

        self.val_step = 0
        ce_ema, time_ema = None, None                

        pbar = tqdm(
            total=len(self.train_loader.dataset) if n_batches == "all" else n_batches, 
            disable=not self.use_tqdm
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
            
                self.ce_ema = ce_val if self.ce_ema is None else \
                    (1 - self.ema_alpha) * self.ce_ema + self.ema_alpha * ce_val
            
                self.time_ema = time_val if self.time_ema is None else \
                    (1 - self.ema_alpha) * self.time_ema + self.ema_alpha * time_val
            
            self.train_outputs.append({
                'train_ce_loss': ce_val,
                'train_time_loss': time_val,
                'train_ce_ema_loss': self.ce_ema,
                'train_time_ema_loss': self.time_ema,
                'train_total': ce_val + time_val,
            })
            
            
            if eval_every is not None and (i % eval_every) == 0:
                self.val_loss, self.mean_val_loss = self.valid_epoch(n_batches=self.n_val_batches)
                self.val_step += 1                        
    
            current_loss = self.train_outputs[-1]
            pbar.set_postfix({
                "train_cce_loss":  f"{current_loss['train_ce_loss'].item():.3f}   ({current_loss['train_ce_ema_loss'].item():.3f})", 
                "train_time_loss": f"{current_loss['train_time_loss'].item():.3f} ({current_loss['train_time_ema_loss'].item():.3f})"
            })

            pbar.update(self.train_loader.batch_size)
    
            if (n_batches is not None) and (i == n_batches):
                break
            
        return self.compute_mean(self.train_outputs)
                

    def valid_epoch(self, n_batches='all'):
        
        self.model.eval()

        pbar = tqdm(
            total=len(self.valid_loader.dataset) if n_batches == "all" else n_batches, 
            disable=not self.use_tqdm,

        )

        with torch.no_grad():
            
            loss_outputs = []
            for i, batch in enumerate(self.valid_loader):

                loss = self.shared_step(batch, batch_idx=i, epoch=self.current_epoch, stage="validation", add_prefix="val")
                loss_outputs.append(loss)
            
                pbar.set_postfix({
                    "val_cce":      f"{loss['val_ce_loss'].item():.3f}", 
                    "val_time_loss":    f"{loss['val_time_loss'].item():.3f}"
                })
            
                pbar.update(self.valid_loader.batch_size)

                if n_batches != "all" and i == n_batches:
                    break
        
        pbar.close()

        self.valid_epoch_end(loss_outputs)

        if hasattr(self, "_val_ce_loss_per_disease_df"):            
            losses_per_epoch_file = self.LOSSES_PER_EPOCH_FILEPATTERN.format(current_epoch=self.current_epoch, val_step=self.val_step)
            self.logger.log_df_as_artifact(
                df=loss_per_disease_df, 
                filename=self._val_ce_loss_per_disease_df, 
                artifact_path="val_loss_per_disease"
            )
            loss_per_disease_df.to_csv(f"{odir}/loss_outputs_{self._validation_counter}.csv", index=False)
                    
        mean_loss = torch.stack([loss['val_ce_loss'] for loss in loss_outputs]).mean()

        self.model.train()

        return loss, mean_loss,

    
    def epoch_end(self):
        self.train_outputs = []
        self.valid_outputs = []


    def mlflow_logging(self):
        pass