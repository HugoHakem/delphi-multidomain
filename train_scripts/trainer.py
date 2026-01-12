import pandas as pd
from typing import List, Dict, Union
import torch
import numpy as np
import mlflow
from tqdm import tqdm
import os, sys
from pathlib import Path
import tempfile
import shutil
from datetime import datetime


if ( DELPHI_DIR := Path(__file__).resolve().parent.parent ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from data.event_set import EventSet
from easydict import EasyDict
from copy import deepcopy
from collections import defaultdict
from urllib.parse import urlparse

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
        torch.save( self.build_state_dict(model, optimizer, scheduler, metadata), tmp_path )
    
        # Log to MLflow artifacts
        artifact_path = "checkpoints"
        mlflow.log_artifact(str(tmp_path), artifact_path=artifact_path)
        shutil.rmtree(tmp_dir)
    
        uri = mlflow.get_artifact_uri(artifact_path)
        print(f"Saved checkpoint at {uri}/{filename}")
        return Path(uri) / filename


# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

class Trainer():

    LOSSES_PER_EPOCH_FILEPATTERN = "losses_epoch{current_epoch}_{val_step}.csv"

    def __init__(self, model, dataloaders, 
          optimizer, scheduler, patience=3, 
          n_train_batches=None, n_val_batches=None, 
          logger=NullLogger(), mlflow_params=dict(), start_epoch=0,
          use_tqdm=True
        ):

        '''
        Trainer class, mimicking PytorchLightning trainer
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
        self._validation_counter = 0
        
        self.logger = logger
        self.val_loss = None
        self.ema_alpha = 0.005

        self.additional_mlflow_params = mlflow_params | { "ema_alpha": self.ema_alpha }

        self.use_tqdm = use_tqdm

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


    def train(self, max_epochs=1000):

        self.logger.log_params(self.model.config)
        self.logger.log_params(self.additional_mlflow_params)

        # self.logger.log_params(self.optimizer.config)
        # self.logger.log_params(self.scheduler.config)
        # profile_and_print(self.shared_step, self.training_loader, self.optimizer, n_steps=2, top_k=20) 

        for epoch in range(self.current_epoch, max_epochs):
            
            self.current_epoch = epoch            
            train_loss = self.train_epoch(eval_every=VAL_EVERY_NSAMPLES//self.train_loader.batch_size)

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
            

    def prepare_input(self, batch):

        x, ages, subject_ids = self.get_tensors_from_batch(batch)
        max_ages             = self.model.get_max_ages_per_subject(ages, subject_ids)
        x, ages, subject_ids = self.model.insert_no_event_tokens(x, ages, subject_ids)
        x, ages, subject_ids = self.model.mask_tokens_after_age (x, ages, subject_ids, max_ages)
        x, ages, subject_ids = self.adjust_to_seqlen(x, ages, subject_ids, seqlen:=48)
        return x, ages, subject_ids


    def prepare_input_simplified(self):
        
        '''
        Sample only one input tensor from the training loader and reuse it
        for profiling purposes
        '''

        if not hasattr(self, "_sample_input_tensor"):
            sample_batch = next(iter(self.train_loader))
            self._sample_input_tensor = self.prepare_input(sample_batch) 
        return self._sample_input_tensor

    
    def shared_step(self, batch, batch_idx, epoch, log_loss_per_disease=False, return_logits=False, return_att=False, stage="training", add_prefix=None):

        model = self.model

        # x, ages, subject_ids = self.prepare_input_simplified()
        x, ages, subject_ids = self.prepare_input(batch)
        # x    = { k: v.detach() for k, v in x.items()}
        # ages = { k: v.detach() for k, v in ages.items()}
        # subject_ids = { k: v.detach() for k, v in subject_ids.items()}
        
        # events = EventSet(batch).\
        #    insert_no_event_tokens(rate=5, trim_right_padding=True).\
        #    adjust_to_seqlen(256)

        # x, ages, emb, subject_ids, domains = model.to_tensor(x, ages, emb := model.transformer.embed(x), subject_ids)

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
            f'{prefix}ce_loss':       loss_ce,            
            f'{prefix}time_loss':     time_loss,
            f'{prefix}ce_ema_loss':   torch.lerp(outputs[-1][f'{prefix}ce_ema_loss'], loss_ce, weight=0.002) if outputs else loss_ce,
            f'{prefix}time_ema_loss': torch.lerp(outputs[-1][f'{prefix}time_ema_loss'], time_loss, weight=0.002) if outputs else time_loss,
            f'{prefix}total':         loss_ce + time_loss,            
        })

        if log_loss_per_disease:
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

        return pd.concat([x['val_ce_loss_per_disease'] for x in loss_outputs]).\
            reset_index().\
            pivot(index="token_id", columns="batch_idx", values="log_p").\
            fillna(0).\
            sum(axis=1).\
            sort_values().\
            reset_index()
        

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
            disable=not self.use_tqdm
        )

        batch = next(iter(self.train_loader))
        for i, batch in enumerate(self.train_loader):
        # for i, _ in enumerate(range(len(self.train_loader))):
            
            self.optimizer.zero_grad()
            loss = self.shared_step(batch, batch_idx=i, epoch=self.current_epoch, stage="training", add_prefix="train")
            
            loss['train_total'].backward()#retain_graph=True)
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
            disable=not self.use_tqdm
        )

        with torch.no_grad():
            
            loss_outputs = []
            for i, batch in enumerate(self.valid_loader):

                loss = self.shared_step(batch, batch_idx=i, epoch=self.current_epoch, log_loss_per_disease=True, stage="validation", add_prefix="val")
                loss_outputs.append(loss)
            
                pbar.set_postfix({
                    "val_cce":      f"{loss['val_ce_loss'].item():.3f} ({loss['val_ce_ema_loss'].item():.3f})", 
                    "val_time_loss":    f"{loss['val_time_loss'].item():.3f} ({loss['val_time_ema_loss'].item():.3f})"
                })
            
                pbar.update(self.valid_loader.batch_size)

                if n_batches != "all" and i == n_batches:
                    break
        
        pbar.close()

        loss_per_disease_df = self.valid_epoch_end(loss_outputs)
        
        losses_per_epoch_file = self.LOSSES_PER_EPOCH_FILEPATTERN.format(current_epoch=self.current_epoch, val_step=self.val_step)

        self.logger.log_df_as_artifact(df=loss_per_disease_df, filename=losses_per_epoch_file, artifact_path="val_loss_per_disease")
        # loss_per_disease_df.to_csv(f"{odir}/loss_outputs_{self._validation_counter}.csv", index=False)

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


        domains = self.model.domains
    
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
    