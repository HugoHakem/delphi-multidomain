import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import fields, is_dataclass
import inspect

import pandas as pd

from delphi.model.components import (
    CompetingExpHead,
    CrossEntropyHead,
    DelphiEmbedding,
    causal_attention_mask,
    target_mask,
    ties_adjusted_delta_t,
)
from delphi.model.components import DelphiConfig


def check_config(cls, args):
    """
    Validate a config dict or object against the expected config class.

    - If cls.__init__ expects a config object (e.g. DelphiConfig), the valid keys
      are taken from that config class (its dataclass fields if applicable).
    - If args is a dict: its keys are compared.
    - If args is an object: its attributes are compared.
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # Case 1: __init__ takes a config object (e.g. config: DelphiConfig)
    if len(params) == 2 and "config" in params:
        ann = params["config"].annotation
        if is_dataclass(ann):
            valid_keys = {f.name for f in fields(ann)}
        else:
            ann_sig = inspect.signature(ann.__init__)
            valid_keys = set(ann_sig.parameters.keys()) - {"self"}
    else:
        # Case 2: normal kwargs in __init__
        valid_keys = set(params.keys()) - {"self"}

    # Provided keys: handle dict vs object
    if isinstance(args, dict):
        provided_keys = set(args.keys())
    else:
        provided_keys = set(vars(args).keys())

    missing = valid_keys - provided_keys
    extra   = provided_keys - valid_keys

    if missing:
        print(f"[ERROR] Missing keys for {cls.__name__}: {sorted(missing)}")
    if extra:
        print(f"[WARN] Unused config keys for {cls.__name__}: {sorted(extra)}")

    # Return dict suitable for instantiation if args is dict
    if isinstance(args, dict):
        return {k: v for k, v in args.items() if k in valid_keys}
    else:
        return args


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att


def initialize_weights(model: torch.nn.Module, config: DelphiConfig):

    def _init_weights(module: torch.nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)
    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in model.named_parameters():
        if pn.endswith("c_proj.weight"):
            torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))


class Delphi(torch.nn.Module):
    
    model_type = "delphi-m4"

    def __init__(self, config: DelphiConfig):
        super().__init__()        
        self.config = config
        self.build_model(config)
        initialize_weights(self, config=config)


    def build_model(self, config: DelphiConfig):

        self.transformer = nn.ModuleDict(
            dict(
                embed=DelphiEmbedding(config),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        assert config.vocab_size is not None
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.embed.token_embedding.weight = self.lm_head.weight
        # self.transformer.embed.weight = self.lm_head.weight

        self.ce_head = CrossEntropyHead(config)
        self.dt_head = CompetingExpHead(
            n_input=config.n_embd,
            zero_inflate=config.zero_inflate,
            pi_head=config.zero_inflate_projector,
        )


    def is_not_padding(self, x):
        return x > 1

   
    def get_allowed_tokens_mask(self, targets, ignore_tokens):

        targets = targets.reshape(-1)
        pass_tokens = targets != -1 
        for k in ignore_tokens: # and gender
            pass_tokens *= targets != k

        return pass_tokens


    def set_valid_loss_mode(self, validation_loss_mode):

        self.validation_loss_mode = validation_loss_mode
   

    def build_attention_mask(self, idx, age, targets, targets_age, mask_ties):

        # causal self-attention mask, to ensure that attention is only applied to the left in the input sequence
        device = idx.device
        # Do not attend to padded positions
        attn_mask = (idx>0).view(idx.size(0), 1, 1, idx.size(1)) * (idx>0).view(idx.size(0), 1, idx.size(1), 1)  
        
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), device=device))[None,None,:,:] > 0
        
        # if targets is not None and self.config.mask_ties:
        if targets is not None and mask_ties:
            # Mask co-occuring tokens
            attn_mask *= ((age.view(idx.size(0),1,1,idx.size(1)) != targets_age.view(idx.size(0),1,idx.size(1),1))) 
            attn_mask += (attn_mask.sum(-1, keepdim=True)==0) * torch.diag(torch.ones(idx.size(1), device=device)) > 0
        
        # Except for padding
        attn_mask = attn_mask + (idx==0).view(idx.size(0), 1, 1, idx.size(1)) * torch.diag(torch.ones(idx.size(1), device=device)) > 0 
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), device=device))[None,None,:,:] > 0
        
        return attn_mask
    

    def cross_entropy_loss(self, logits, targets, pass_tokens, agg=None):
        '''
        Cross entropy loss for the next token prediction.
        Arguments:
            logits: Tensor, shape [batch_size, sequence_length, vocab_size]
            targets: Tensor, shape [batch_size, sequence_length]
            pass_tokens: Tensor of bools, batch_size * sequence_length
            agg: one of None, "mean" or "sum"
        '''

        print(f"{pass_tokens.shape=}")
        print(f"{logits.shape=}")
        print(f"{targets.shape=}")

        n_classes = logits.size(-1)            
        if agg == "per_token":
            log_softmax = F.log_softmax(logits.view(-1, n_classes)[pass_tokens], dim=-1)
            loss_ce_per_token = log_softmax[torch.arange(log_softmax.size(0)), targets.view(-1)[pass_tokens]]
            return loss_ce_per_token
        if agg == "per_disease":
            loss_ce_per_token = self.cross_entropy_loss(logits, targets, pass_tokens, agg="per_token")
            loss_ce_agg_per_disease = pd.DataFrame([loss_ce_per_token, targets.view(-1)[pass_tokens].cpu().numpy()]).T.\
                set_axis(["log_p", "token_id"], axis=1).\
                astype({"token_id": int}).\
                groupby("token_id").sum().\
                log_p.apply(lambda x: x.item())
            return loss_ce_agg_per_disease / pass_tokens.sum().item()
        elif agg is None:
            loss_ce = F.cross_entropy(
                logits.reshape(-1, n_classes)[pass_tokens], 
                targets.reshape(-1)[pass_tokens], 
                ignore_index=-1
            )
        else:
            raise ValueError("agg should be in [None, 'per_disease']")

        return loss_ce


    def time_to_event_loss(self, logits, time_to_next, pass_tokens, attn_mask, mask_ties, t_min, agg=None):       
        '''
        '''
        
        lse = torch.logsumexp(logits,-1) ## More forgiving than using torch.max() for the most likely next event
        lse = - torch.log(torch.exp(-lse) + t_min)

        dt  = torch.clamp(time_to_next, min=1.0)
        dd = dict(device=logits.device, dtype=torch.float32)
        block_size = attn_mask.size(-1)

        if mask_ties:
            # Use time from last untied token
            dt = torch.gather(
                dt, -1, (attn_mask * torch.arange(0, block_size, **dd).view(1, 1, 1, -1)).max(-1).indices.squeeze((1, 2))
            )  

        log_dt = - torch.log(dt + t_min).view(-1)        

        ## Exponential log-likelihood (real statistics, TM)
        loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - log_dt.reshape(-1))) 

        if agg is None:
            pass
        elif agg == "mean":
            loss_dt = (loss_dt[pass_tokens]).mean()
        elif agg == "sum":
            loss_dt = (loss_dt[pass_tokens]).sum()
        elif agg == "per_disease":
            raise NotImplementedError
          
        return loss_dt

    
    def blackout_ignored(self, logits, ignore_tokens):

        if self.validation_loss_mode:
            ignore_tokens += [1]
            logits[..., ignore_tokens] = -torch.inf

        return logits


    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        # modality: torch.Tensor,
        # biomarker: Optional[dict[str, torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
        targets_age: Optional[torch.Tensor] = None,
        validation_loss_mode: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]], torch.Tensor]:

        
        x = self.transformer.embed(x=idx, t=age) #, M=modality, biomarker_x=biomarker)
        x = self.transformer.drop(x)

        attn_mask = causal_attention_mask(
            pad=self.is_not_padding(idx), t1=targets_age, t0=age, mask_ties=self.config.mask_ties
        )

        self.set_valid_loss_mode(validation_loss_mode)

        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        if (targets is not None) and (targets_age is not None):
            logits = self.lm_head(x)

            logits_cp = logits.clone()
            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                logits_cp[..., ignored_tokens] = -torch.inf

            dt = ties_adjusted_delta_t(
                t0=age,
                t1=targets_age,
                attn_mask=attn_mask,
                mask_ties=self.config.mask_ties,
                eps=0.0 if self.config.zero_inflate else 1.0,
            )

            is_valid_target = target_mask(targets, ignore_tokens=ignored_tokens)
            loss_ce = self.ce_head(logits=logits_cp, targets=targets)
            loss_ce = torch.mean(loss_ce[is_valid_target])
            loss_dt = self.dt_head(logits=logits, delta_t=dt)
            loss_dt = torch.mean(loss_dt[is_valid_target])

            loss = {
                "loss_ce": loss_ce, # * self.config.ce_beta,
                "loss_dt": loss_dt, # * self.config.dt_beta,
                "loss": loss_ce * self.config.ce_beta + loss_dt * self.config.dt_beta,
            }
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, :, :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, att


    @classmethod
    def from_checkpoint(cls, ckpt_path, device=None):

        import inspect

        def safe_load_config(cls, args_dict):
            sig = inspect.signature(cls.__init__)
            valid_keys = set(sig.parameters.keys()) - {"self"}
            
            used = {k: v for k, v in args_dict.items() if k in valid_keys}
            unused = {k: v for k, v in args_dict.items() if k not in valid_keys}
            
            if unused:
                print(f"[WARN] Unused config keys for {cls.__name__}: {list(unused.keys())}")
            
            return cls(**used)
                        
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(ckpt_path, map_location=device)        

        # conf = DelphiConfig(**checkpoint['model_args'])        
        conf = safe_load_config(DelphiConfig, checkpoint["model_args"])

        def translate_keys(state_dict):

            mapping = {
                "transformer.wte.weight": "transformer.embed.token_embedding.weight",
                "transformer.wae.div_term": "transformer.embed.age_encoding.div_term",
                "transformer.wae.linear.weight": "transformer.embed.age_encoding.linear.weight",
            }
            new_state = {}
            for k, v in state_dict.items():
                new_key = mapping.get(k, k)  # usa el mapeo si existe, si no deja igual
                new_state[new_key] = v

            return new_state        
        
        check_config(Delphi, conf)
        model = Delphi(conf)
        
        state_dict = checkpoint['model']
        state_dict = { k.replace("_orig_mod.", ""): v for k, v in state_dict.items() }
        state_dict = translate_keys(state_dict)

        model.load_state_dict(state_dict)         
        model = model.to(device)

        return model