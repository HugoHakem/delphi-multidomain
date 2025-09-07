import os, sys
import time
import math
import pickle as pkl
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

from ast import literal_eval

from pprint import pprint
from collections import defaultdict

from model import Delphi, DelphiConfig
from utils import get_p2i, get_batch

import hla_genes
from hla_genes import get_hla_protein_sequences

from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

import mlflow
import dagshub

from dagshub_helper import (
    sync_run_to_dagshub
)

import random

MLFLOW_LOG_INTERVAL = 1000
EVAL_INTERVAL = 1000

def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def estimate_loss(model, eval_iters, batch_size, block_size, train_data, val_data, train_p2i, val_p2i, no_event_token_rate, device, ctx):

    '''
    '''

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, 2)
        data = train_data if split == 'train' else val_data
        p2i = train_p2i if split == 'train' else val_p2i
        for k in range(eval_iters):
            ix = torch.randint(len(p2i), (batch_size,))
            X, A, Y, B = get_batch(ix, data, p2i, block_size=block_size,
                                   device=device, select='left', lifestyle_augmentations=True,
                                   no_event_token_rate=no_event_token_rate, 
                                   cut_batch=True)
            with ctx:
                logits, loss, _ = model(X, A, Y, B, validation_loss_mode=True)
            losses[k] = torch.stack([loss['loss_ce'], loss['loss_dt']])
        out[split] = losses.mean(0)
    model.train()   
    return out


def replace_config_items(global_variables, replacement_values):
    
    # new_local_variables = local_variables.copy()

    for key, val in replacement_values.items():
        if key in global_variables:
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(global_variables[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            global_variables[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
    
    return global_variables


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    
    '''
    Implements learning rate scheduler
    '''
    
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)



def main(args, replacement_values, code_to_exec):

    if args.experiment_name is None:
        args.experiment_name = "delphi" + ("" if args.no_hla else "-hla")
    elif args.no_hla:
        assert ("no-hla" in args.experiment_name) or ("hla" not in args.experiment_name), f'''The experiment {args.experiment_name} name might be wrong considering you are not including HLA allele information.'''

    if not args.dry_run:
        local_client = MlflowClient(tracking_uri="./mlruns")
        experiment_name = args.experiment_name
        experiment = local_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = local_client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        run = local_client.create_run(experiment_id)
        run_id = run.info.run_id

        mlflow.start_run(run_id=run_id)
        
        # logger = DualMLflowLogger(
            # local_uri="file:./mlruns",
            # remote_uri="https://dagshub.com/rbonazzola/delphi.mlflow",
            # experiment_name="Delphi-HLA",
            # artifact_size_limit_mb=10
        # )
    
    global seed, out_dir, log_interval, eval_iters, eval_only, always_save_checkpoint,\
        gradient_accumulation_steps, batch_size, block_size, init_from,\
        n_layer, n_head, n_embd, dropout, bias, vocab_size,\
        learning_rate, weight_decay, beta1, beta2,\
        max_iters, grad_clip, decay_lr, warmup_iters, lr_decay_iters, min_lr,\
        device, device_type, dtype, compile, token_dropout,\
        t_min, mask_ties, ignore_tokens, data_fraction, no_event_token_rate, patience,\
        eps, best_val_loss, patience_counter, max_steps
    
    out_dir = 'out'
    seed = 42

    # ──────────────────── OPTIMIZER ────────────────────
    log_interval = 100
    eval_iters = 1000
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = False  # if True, always save a checkpoint after each eval    
    max_steps = 1000000
    
    # ──────────────────── BATCHING ────────────────────
    gradient_accumulation_steps = 1  # used to simulate larger batch sizes
    batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 24
    
    # ──────────────────────── MODEL ──────────────────────────
    init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'    
    n_layer, n_head, n_embd = (12, 10, 120)
    dropout = 0.2  # for pretraining 0 is good, for finetuning try 0.1+
    bias = False  # do we use bias inside LayerNorm and Linear layers?
    vocab_size = 1270 # <---- can we set this automatically?
    
    # ──────────────────── ADAMW OPTIMIZER ────────────────────
    learning_rate, weight_decay, beta1, beta2 = 3e-4, 3e-2, 0.9, 0.99
    max_iters = 10000  # total number of training iterations
    grad_clip = 0.01  # clip gradients at this value, or disable if == 0.0
    
    # ──────────────── LEARNING RATE SCHEDULER ────────────────
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 5000  # how many steps to warm up for
    lr_decay_iters = 10000  # should be ~= max_iters per Chinchilla
    min_lr = learning_rate / 10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    
    # ──────────────────────── HARDWARE ───────────────────────
    device = 'cuda:0'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    device_type = 'cuda'
    dtype = 'float32'  # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True  # use PyTorch 2.0 to compile the model to be faster
    
    # ──────────────────── DELPHI TRAINING ────────────────────
    token_dropout = 0.0
    t_min = 0.0  # 365.25/12.
    mask_ties = True
    ignore_tokens = [0]
    data_fraction = 1.0
    no_event_token_rate = 5    
    # ─────────────────────────────────────────────────────────

    # ──────────────────── EARLY STOPPING ──────────────────────
    patience = 10
    eps = 1e-6
    best_val_loss = float('inf')
    patience_counter = 0

    # ──────────────────── OVERWRITE DEFAULT CONFIG ────────────────────
    code_to_exec = [code_to_exec] if isinstance(code_to_exec, str) else code_to_exec
    if code_to_exec[0]:
        print("\n──────── REPLACING VARIABLE BASED ON FILE... ────────")

    for code in code_to_exec:
        print("."*100)        
        exec(code, globals())
        print("──────────────────────────────────────────────────────────────────────────────────")
    del code, code_to_exec
            
    # _locals = 
    replace_config_items(global_variables=globals(), replacement_values=replacement_values)
    # max_steps = _locals['max_steps']
  
    MODEL_ARG_NAMES = ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "dropout", "token_dropout", "t_min", "mask_ties", "ignore_tokens"]
    model_args = { k: globals()[k] for k in MODEL_ARG_NAMES }
    
    set_global_seed(seed)
    # torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    full_config = model_args | { k: globals()[k] for k in ["seed", "mask_ties", "ignore_tokens", "data_fraction", "no_event_token_rate", "patience", "learning_rate", "batch_size"] }

    if args.dry_run or args.show_config:
        pprint(full_config)
        if args.dry_run: exit()
    
    gptconf = DelphiConfig(**model_args)
    model = Delphi(gptconf).to(device)
    
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # ───────────────────────── OPTIMIZER ─────────────────────────
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # ────────────────────────── DATASET ──────────────────────────
    
    if args.no_hla:
        dataset, file_prefix, data_type = 'ukb_real_data', "ukb_real_", "real-no-hla"
    else:
        dataset, file_prefix, data_type = 'ukb_real_data', "ukb_real_hla2d_", "real-hla-2digits"
        # dataset, file_prefix, data_type = 'ukb_real_data', "ukb_real_hla4d_", "real-hla-4digits"

    data_dir = os.path.join('data', dataset)
    
    load_data_from_bin = lambda datadir, file: np.memmap(os.path.join(data_dir, file), dtype=np.uint32, mode='r').reshape(-1, 3)

    train_filename = f'{file_prefix}train.bin'
    val_filename = f'{file_prefix}val.bin'
    train_data = load_data_from_bin(data_dir, train_filename)
    val_data   = load_data_from_bin(data_dir, train_filename)
    train_p2i, val_p2i = get_p2i(train_data), get_p2i(val_data)
    
    # ─────────────────────────────────────────────────────────────

    # {'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=(ptdtype := getattr(torch, dtype)))
    
    torch.set_default_dtype(ptdtype)
    
    ix = torch.randint(len(train_p2i), (batch_size,))
    X, A, Y, B = get_batch(ix, train_data, train_p2i, block_size=block_size, device=device,
                           padding='random', lifestyle_augmentations=True, select='left',
                           no_event_token_rate=no_event_token_rate)
    
    val_loss, step, iter_num, t0 = None, 0, 0, time.time()    
    
    for k, v in model_args.items():
        local_client.log_param(run_id, k, v)

    local_client.log_param(run_id, "learning_rate", learning_rate)
    local_client.log_param(run_id, "batch_size", batch_size)
    local_client.log_param(run_id, "seed", seed)
    local_client.log_param(run_id, "hla", str(not args.no_hla))
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    local_client.log_param(run_id, "num_parameters", num_params)
    local_client.log_param(run_id, "dtype", dtype)
    local_client.log_param(run_id, "data_type", data_type)
    local_client.log_param(run_id, "val_filename", val_filename)
    local_client.log_param(run_id, "train_filename", train_filename)

    metric_buffer = defaultdict(list)        

    print(f"{max_steps=}")

    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    print("\n──────── TRAINING STARTS ──────────────────────────────────────────────────────────────────────────")
    # ─────────────────────────────── TRAINING LOOP ───────────────────────────────
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr) if decay_lr else learning_rate
        if iter_num % MLFLOW_LOG_INTERVAL == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr            
                metric_buffer["instant_learning_rate"].append((step, lr))
        
        if iter_num % MLFLOW_LOG_INTERVAL == 0: # and mlflow.active_run():
            for metric_name, values in metric_buffer.items():
                metric_objs = [Metric(key=metric_name, value=v, timestamp=int(time.time()*1000), step=s) for s, v in values]
                local_client.log_batch(run_id, metrics=metric_objs)
            metric_buffer.clear()
                
        if step % EVAL_INTERVAL == 0 and iter_num > 0:
            
            losses = estimate_loss(model, eval_iters, batch_size, block_size, train_data, val_data, train_p2i, val_p2i, no_event_token_rate, device, ctx)

            # Raw losses
            val_loss_raw = losses['val'].sum().item()
            train_loss_raw = losses['train'].sum().item()
            
            # Smooth losses
            if val_loss is None:
                val_loss_unpooled = losses['val']
                train_loss_unpooled = losses['train']
            val_loss_unpooled = (gamma := 0.1) * losses['val'] + (1 - gamma) * val_loss_unpooled
            train_loss_unpooled = gamma * losses['train'] + (1 - gamma) * train_loss_unpooled
            
            val_loss = val_loss_unpooled.sum().item()
            train_loss = train_loss_unpooled.sum().item()
            
            # Logging
            local_client.log_metric(run_id, "val_loss", val_loss, step=step)
            local_client.log_metric(run_id, "val_loss_raw", val_loss_raw, step=step)
            local_client.log_metric(run_id, "train_loss", train_loss, step=step)
            local_client.log_metric(run_id, "train_loss_raw", train_loss_raw, step=step)

            # start_eval = time.time()
            # losses = estimate_loss(model, eval_iters, batch_size, block_size, train_data, val_data, train_p2i, val_p2i, no_event_token_rate, device, ctx)
            # print(f"[eval] time: {time.time() - start_eval:.2f}s")            
            # if val_loss is None:
            #    val_loss_unpooled = losses['val']
            # val_loss_unpooled = (gamma := 0.5) * losses['val'] + (1 - gamma) * val_loss_unpooled  # ie exponential decay
            # val_loss = val_loss_unpooled.sum().item()
            # local_client.log_metric(run_id, "val_loss", val_loss, step=step)

            #print(f"[eval step {step}]")
            #print(f"  val_loss_raw     = {val_loss_raw:.6f}")
            #print(f"  val_loss_smooth  = {val_loss:.6f}")
            #print(f"  train_loss_raw   = {train_loss_raw:.6f}")
            #print(f"  train_loss_smooth= {train_loss:.6f}")

            if val_loss is not None and (val_loss < best_val_loss * (1 - eps)):
                # best_val_loss = val_loss
                patience_counter = 0
            elif val_loss is not None:
                patience_counter += 1
                print(f"[early stopping] No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered. Best val loss {best_val_loss}")
                    break


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if iter_num > 0:
                    best_ckpt = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': val_loss,
                        'config': full_config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    best_iter_num = iter_num
                    best_ckpt_path = os.path.join(out_dir, ckpt_file := f'best_ckpt__{run_id}__{iter_num}.pt')
                    # torch.save(checkpoint, ckpt_path)
                    # mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

            if iter_num % 100_000 == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': full_config,
                }
                print(f"saving checkpoint to {out_dir}")
                ckpt_path = os.path.join(out_dir, ckpt_file := f'ckpt__{run_id}__{iter_num}.pt')
                torch.save(checkpoint, ckpt_path)
                local_client.log_artifact(run_id, ckpt_path, artifact_path="checkpoints")

        step += 1
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss, att = model(X, A, Y, B)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            ix = torch.randint(len(train_p2i), (batch_size,))
            
            X, A, Y, B = get_batch(ix, train_data, train_p2i, block_size=block_size, device=device,
                                   padding='random', lifestyle_augmentations=True, select='left',
                                   no_event_token_rate=no_event_token_rate, cut_batch=True)
    
            # backward pass, with gradient scaling if training in fp16
            loss = loss['loss_ce'] + loss['loss_dt']
            scaler.scale(loss).backward()
        
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
    
        # timing and logging
        dt = (t1 := time.time()) - t0
        t0 = t1
        if iter_num % log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    
        iter_num += 1
        if max_steps <= step:
            break
        
    torch.save(best_ckpt, best_ckpt_path)
    local_client.log_metric(run_id, "best_val_loss", best_val_loss)
    local_client.log_artifact(run_id, best_ckpt_path, artifact_path="checkpoints")
    # local_iter_num += 1
    
    
    mlflow.end_run()
    sync_run_to_dagshub("file:./mlruns", "https://dagshub.com/rbonazzola/Delphi-HLA.mlflow", run_id)


def parse_manual_args(manual_args):
    
    replacement_values = {}
    code_to_exec = []
    
    for arg in manual_args:

        if '=' not in arg and arg.startswith("--"):
            # This is already being handled by argparse
            continue
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--'), "Arguments with -- should be of the form --x=y unless handled by argparse"
            assert os.path.exists(arg), f"{arg} should be a file but it does not exist"
            config_file = arg
            print(config_file)
            with open(config_file) as f:
                code_to_exec.append(open(config_file).read())
                # print(f.read())
            # exec(open(config_file).read())
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            replacement_values[key] = attempt
    
    return replacement_values, code_to_exec


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser("Delphi CLI")
    parser.add_argument("--dry-run", "--dry_run", "--dryrun", dest="dry_run", action="store_true", default=False)
    parser.add_argument("--experiment-name", "--experiment_name", "-x", dest="experiment_name", default=None, help="Default: delphi-hla or delphi (if --no-hla is set)")
    parser.add_argument("--no-hla", "--no_hla", dest="no_hla", action="store_true", default=False)
    parser.add_argument("--show-config", "--show_config", dest="show_config", action="store_true", default=False)
    parser.add_argument("--exclude_subjects", default=None)
    parser.add_argument("--log_to_dagshub", "--dagshub", "-d", dest="log_to_dagshub", default=False, action="store_true", help="Whether to log to DagsHub or only locally")
    parser.add_argument("--interactive", "-i", dest="interactive", default=False, action="store_true", help="Placeholder argument (behaviour not yet implemented)")
    args, manual_args = parser.parse_known_args()
    
    replacement_values, code_to_exec = parse_manual_args(manual_args)

    if args.interactive:
        raise NotImplementedError
    
    main(args, replacement_values, code_to_exec)
    
