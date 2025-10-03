import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import fields, is_dataclass
import inspect

import pandas as pd

import math
from typing import Optional
from dataclasses import dataclass, field
from typing import Optional
import yaml

import logging
logger = logging.getLogger(__name__)
from pathlib import Path

# from delphi.model.components import (
#     CompetingExpHead,
#     CrossEntropyHead,
#     DelphiConfig,
#     DelphiEmbedding,
#     AgeEncoding,
#     causal_attention_mask,
#     target_mask,
#     ties_adjusted_delta_t,
# )

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

# ——————————————————————————————————————————————————————————————————————————————————————————————————

class AgeEncoding(nn.Module):

    def __init__(
        self, n_embd: int, norm_factor: float = 365.25, max_wavelen: float = 10000.0
    ):
        super().__init__()
        div_term = torch.exp(
            torch.arange(0, n_embd, 2) * (-math.log(max_wavelen) / n_embd)
        )
        self.register_buffer("div_term", div_term)
        self.n_embd = n_embd
        self.linear = torch.nn.Linear(n_embd, n_embd, bias=False)

        self.norm_factor = norm_factor

    def forward(self, age: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size]``
        """
        time_years = age / self.norm_factor
        print(age.shape)
        seq_len, batch_size = age.shape
        y = torch.zeros(seq_len, batch_size, self.n_embd, device=age.device)

        # .unsqueeze(-1) is added because with batch_size == 1 the last dimension gets contracted.
        # with this addition it works seamlessly for both batch_size == 1 and > 1.
        y[..., 0::2] = torch.sin(time_years.unsqueeze(-1) * self.div_term)  # * (1-self.div_term)
        y[..., 1::2] = torch.cos(time_years.unsqueeze(-1) * self.div_term)  # * (1-self.div_term)
        y = y.squeeze(1)
        y = self.linear(y)
        return y


class Time2Vec(nn.Module):

    def __init__(self, n_embd: int, norm_factor: float = 365.25):
        super().__init__()
        self.linear = torch.nn.Linear(1, n_embd, bias=True)
        self.norm_factor = norm_factor

    def forward(self, x: torch.Tensor):
        x = self.linear(x / self.norm_factor)
        return torch.cat([x[..., :1], torch.sin(x[..., 1:])], dim=-1)


class PiecewiseAgeEncoding(nn.Module):

    def __init__(
        self,
        n_embd: int,
        norm_factors: list[float] = [365.25, 1.0],
        max_wavelen: float = 10000.0,
    ):
        super().__init__()
        assert n_embd % len(norm_factors) == 0
        piece_n_embd = n_embd / len(norm_factors)
        div_term = torch.exp(
            torch.arange(0, piece_n_embd, 2) * (-math.log(max_wavelen) / piece_n_embd)
        )
        self.register_buffer("div_term", div_term)
        self.n_embd = n_embd
        self.linear = torch.nn.Linear(n_embd, n_embd, bias=False)

        self.norm_factors = norm_factors

    def forward(self, x: torch.Tensor):
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        for i, norm_factor in enumerate(self.norm_factors):
            piece_start = int(i / len(self.norm_factors) * self.n_embd)
            piece_end = int((i + 1) / len(self.norm_factors) * self.n_embd)
            y[..., piece_start:piece_end:2] = torch.sin(x / norm_factor * self.div_term)
            y[..., piece_start + 1 : piece_end : 2] = torch.cos(
                x / norm_factor * self.div_term
            )
            x = torch.remainder(x, norm_factor)
        y = self.linear(y)

        return y

# ———————————————————— CONFIG ————————————————————————————————————————————————————————————————

@dataclass
class EmbedConfig:
    projector:  str           = "linear"  # "linear", "mlp", or "embed"
    n_layers:   Optional[int] = None
    n_hidden:   Optional[int] = None
    input_size: Optional[int] = None
    pretrained_path: Optional[str] = None  # path to .pt/.npy/.pkl with lookup table
    freeze: bool = True                    # if previous lookup table is to be left fixed
    path: Optional[str] = None 
    predict: bool = False
    age_jitter: bool = False

@dataclass
class DelphiConfig:
    vocab_size: Optional[int] = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    bias: bool = True
    block_size: int = 64
    dropout: float = 0.1
    token_dropout: float = 0.1
    mask_ties: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])
    domains: dict[str, EmbedConfig] = field(default_factory=dict)
    modality_emb: bool = False
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    zero_inflate: bool = False
    zero_inflate_projector: str = "linear"


# ———————————————————— VALIDATE CONFIGS ——————————————————————————————————————————————————————————

def validate_model_config(config: DelphiConfig):
    assert config.mask_ties != config.zero_inflate, "mask_ties and zero_inflate cannot be both True or both False"


def validate_model_config_for_finetuning(
    finetune_config: DelphiConfig, pretrain_config: DelphiConfig
) -> None:

    assert (
        finetune_config.vocab_size == pretrain_config.vocab_size
        and finetune_config.n_layer == pretrain_config.n_layer
        and finetune_config.n_head == pretrain_config.n_head
        and finetune_config.n_embd == pretrain_config.n_embd
        and finetune_config.bias == pretrain_config.bias
    ), "model dimensions must match between finetune and pretrain configs"

    finetune_biomarkers = set(finetune_config.biomarkers.keys())
    pretrain_biomarkers = set(pretrain_config.biomarkers.keys())
    assert pretrain_biomarkers.issubset(
        finetune_biomarkers
    ), "finetune config must have all biomarkers from pretrain config"

    intersect_biomarkers = finetune_biomarkers.intersection(pretrain_biomarkers)
    for biomarker in intersect_biomarkers:
        finetune_bm = finetune_config.biomarkers[biomarker]
        pretrain_bm = pretrain_config.biomarkers[biomarker]

        assert (
            finetune_bm.projector == pretrain_bm.projector
            and finetune_bm.n_layers == pretrain_bm.n_layers
            and finetune_bm.n_hidden == pretrain_bm.n_hidden
            and finetune_bm.input_size == pretrain_bm.input_size
        ), f"biomarker {biomarker} embed configs must match between finetune and pretrain configs"


def parse_token_list(token_list: list[str]) -> list:
    if not token_list:
        return []

    parsed = []
    for token in token_list:
        if token.endswith(".yaml") or token.endswith(".yml"):
            with open(token, "r") as f:
                tokens = yaml.safe_load(f)
            if not isinstance(tokens, list):
                raise ValueError(f"Expected a list of tokens in {token}")
            parsed.extend(tokens)
        else:
            parsed.append(token)

    return parsed

# ———————————————————— EMBEDDINGS ————————————————————————————————————————————————————————————————

class DomainEmbedding(nn.Module):

    def __init__(self, config: EmbedConfig, n_embed: int) -> None:

        super().__init__()
        self.config = config
        # assert config.input_size is not None, "input_size must be specified"

        if config.input_size is None and config.projector.lower() != "pretrained":            
            tokenizer_file = Path(config.path) / "tokenizer.yaml"
            with open(tokenizer_file, "r") as f:
                self.config.input_size = len(yaml.load(f, Loader=yaml.FullLoader))

        elif config.projector.lower() == "pretrained":
            weights = torch.load(config.pretrained_path)  # Tensor [vocab_size, d_ext]
            self.config.input_size, d_ext = weights.shape

        if config.projector.lower() == "linear":
            self.projector = nn.Linear(config.input_size, n_embed, bias=False)

        elif config.projector.lower() == "mlp":
            self.projector = self.build_mlp_projector(config)
            
        elif config.projector.lower() == "embed":
            print(f"DomainEmbedding: Using nn.Embedding with input_size={config.input_size}, n_embed={n_embed}")
            self.projector = nn.Embedding(self.config.input_size, n_embed, padding_idx=0)

        elif config.projector.lower() == "pretrained":
            weights = torch.load(config.pretrained_path)  # Tensor [vocab_size, d_ext]
            vocab_size, d_ext = weights.shape
            logger.info(f"Loading pretrained embedding: vocab_size={vocab_size}, d_ext={d_ext}")
            self.embed = nn.Embedding(vocab_size, d_ext, padding_idx=0)
            self.embed.weight.data.copy_(weights)
            self.embed.weight.requires_grad = not config.freeze

            # To project onto Delphi embedding space
            self.projector = nn.Linear(d_ext, n_embed, bias=False)

        else:
            raise ValueError(f"unknown projector type: {config.projector}")


    def build_mlp_projector(self, config):

        assert config.n_layers is not None, "n_layers must be specified for mlp projector"
        assert config.n_hidden is not None, "n_hidden must be specified for mlp projector"

        I, L, H, E = config.input_size, config.n_layers, config.n_hidden, n_embed
        
        sizes = [I] + [H] * (L-1) + [E]
        isizes, osizes = sizes[:-1], sizes[1:]

        # build MLP
        layers = []
        for i in range(L):
            linear_layer = nn.Linear(isizes[i], osizes[i], bias=False)
            layers.append(linear_layer)
            if i < L-1:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        logger.debug(f"DomainEmbedding.forward | projector={self.config.projector} | input_shape={tuple(x.shape)}")

        if self.config.projector.lower() == "pretrained":
            x = self.embed(x)
            logger.debug(f"After pretrained lookup: {tuple(x.shape)}")
            out = self.projector(x)
            logger.debug(f"After projection: {tuple(out.shape)}")
            return out

        out = self.projector(x)
        logger.debug(f"After projector: {tuple(out.shape)}")
        return out


class DelphiEmbedding(nn.Module):

    def __init__(self, config: DelphiConfig) -> None:
        
        super().__init__()
        
        self.config = config
        # assert config.vocab_size is not None
         
        # self.age_encoding = AgeEncoding(n_embd=config.n_embd)
        self.token_drop   = nn.Dropout(config.token_dropout)        
        
        self.domain_embed = nn.ModuleDict()
        if len(config.domains) > 0:            
            for domain_name, domain_cfg in config.domains.items():
                self.domain_embed[domain_name] = DomainEmbedding(config=domain_cfg, n_embed=config.n_embd)
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)

        # domain_modalities = []
        # for domain_name, domain_cfg in config.domains.items():
            # domain_key = module_name(Modality[domain_name.upper()])
            # self.domain_embed[domain_key] = DomainEmbedding(config=domain_cfg, n_embed=config.n_embd)
            # domain_modalities.append(Modality[domain_name.upper()])

        # if config.modality_emb:
        #     max_modality_idx = (
        #         max([modality.value for modality in domain_modalities])
        #         if len(domain_modalities) > 0
        #         else 1
        #     )
        #     self.mod_embedding = nn.Embedding(max_modality_idx + 1, config.n_embd, padding_idx=0)


    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:

        if len(self.domain_embed) > 0:
            for domain_name in self.domain_embed:
                token_emb = self.domain_embed[domain_name](x[domain_name])
                x[domain_name] = token_emb
        else:
            token_emb = self.token_embedding(x)
            token_emb = self.token_drop(token_emb) * (1 - self.config.token_dropout)
        
        return x            
            

# ———————————————————— HEADS ————————————————————————————————————————————————————————————————

class CrossEntropyHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        logits = logits.permute(0, 2, 1)  # (b, l, n_vocab) -> (b, n_vocab, l)
        loss_ce = F.cross_entropy(logits, targets, reduction="none")

        return loss_ce


class CompetingExpHead(nn.Module):

    def __init__(self,
        n_input:      Optional[int] = None,
        zero_inflate: bool = False,
        pi_head:      Optional[str] = None,
    ):
        super().__init__()

        self.zero_inflate = zero_inflate
        if zero_inflate:
            assert n_input is not None
            assert pi_head is not None
            if pi_head == "linear":
                self.pi_head = nn.Linear(n_input, 1, bias=False)
            elif pi_head == "mlp":
                self.pi_head = nn.Sequential(
                    nn.Linear(n_input, 32, bias=False),
                    nn.ReLU(),
                    nn.Linear(32, 1, bias=False),
                )
            else:
                raise ValueError(f"Unknown pi_head: {pi_head}")

    def forward(self, logits: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:

        lse = torch.logsumexp(logits, -1)
        lse = -torch.log(torch.exp(-lse) + 1.0)
        t_min = 0.0 if self.zero_inflate else 1.0
        delta_t = torch.clamp(delta_t, min=t_min)
        ldt = -torch.log(delta_t + 1.0)
        exp_log_likelihood = lse - torch.exp(lse - ldt)

        if self.zero_inflate:
            pi = self.pi_head(logits).squeeze()
            zero_case_nll = -(F.softplus(-pi + lse) - F.softplus(-pi))
            nonzero_case_nll = -(exp_log_likelihood - pi - F.softplus(-pi))
            loss_dt = (
                zero_case_nll * (delta_t == 0).float()
                + nonzero_case_nll * (delta_t > 0).float()
            )
        else:
            loss_dt = -exp_log_likelihood

        return loss_dt


# ———————————————————— MASKS ————————————————————————————————————————————————————————————————

def causal_attention_mask(
    pad: torch.Tensor,
    mask_ties: bool = False,
    t0: Optional[torch.Tensor] = None,
    t1: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    b, l = pad.shape
    device = pad.device
    dd = {"device": device}

    lower_tri_mask = torch.tril(torch.ones((l, l), **dd))
    lower_tri_mask = lower_tri_mask.view(1, l, l)
    pad_mask = pad.view(b, 1, l).to(torch.int)
    attn_mask = pad_mask * lower_tri_mask

    if mask_ties:
        assert t0 is not None
        if t1 is not None:
            ties_mask = (t1.view(b, l, 1) != t0.view(b, 1, l)).to(torch.int)
            attn_mask *= ties_mask

    attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(
        torch.ones(l, **dd)
    ) > 0

    return attn_mask.unsqueeze(1)


def target_mask(x1: torch.Tensor, ignore_tokens: list[int]) -> torch.Tensor:

    is_valid_target = x1 != 0
    
    for k in ignore_tokens:
        is_valid_target *= x1 != k

    return is_valid_target


def ties_adjusted_delta_t(t0, t1, attn_mask, mask_ties: bool, eps: float = 1.0) -> torch.Tensor:

    delta_t = torch.clamp(t1-t0, min=eps)

    dd = dict(device=t0.device, dtype=torch.float32)
    
    if mask_ties:
        idx = ( attn_mask * torch.arange(0, t0.size(1), **dd).view(1, 1, 1, -1) ).max(-1).indices.squeeze((1, 2))
        delta_t = torch.gather(delta_t, -1, idx)

   #  if mask_ties:
   #      delta_t = torch.gather(delta_t, -1, (attn_mask * torch.arange(
   #                  0, t0.size(1), 
   #              ).view(1, 1, 1, -1)
   #          )
   #          .max(-1)
   #          .indices.squeeze((1, 2)),
   #      )

    return delta_t


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
    
    def __init__(self, config: DelphiConfig):
        super().__init__()        
        self.config = config
        self.build_model(config)
        initialize_weights(self, config=config)


    def build_model(self, config: DelphiConfig):

        self.transformer = nn.ModuleDict(
            dict(
                embed=DelphiEmbedding(config),
                age_embedding=AgeEncoding(n_embd=config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        assert config.vocab_size is not None
        
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.transformer.embed.token_embedding.weight = self.lm_head.weight
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
        dd = dict(device=idx.device)
        # Do not attend to padded positions
        attn_mask = (idx>0).view(idx.size(0), 1, 1, idx.size(1)) * (idx>0).view(idx.size(0), 1, idx.size(1), 1)  
        
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), **dd))[None,None,:,:] > 0
        
        # if targets is not None and self.config.mask_ties:
        if targets is not None and mask_ties:
            # Mask co-occuring tokens
            attn_mask *= ((age.view(idx.size(0),1,1,idx.size(1)) != targets_age.view(idx.size(0),1,idx.size(1),1))) 
            attn_mask += (attn_mask.sum(-1, keepdim=True)==0) * torch.diag(torch.ones(idx.size(1), **dd)) > 0
        
        # Except for padding
        attn_mask = attn_mask + (idx==0).view(idx.size(0), 1, 1, idx.size(1)) * torch.diag(torch.ones(idx.size(1), **dd)) > 0 
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), **dd))[None,None,:,:] > 0
        
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


    def forward(self, idx: torch.Tensor, age: torch.Tensor,
        # targets: Optional[torch.Tensor] = None,
        # targets_age: Optional[torch.Tensor] = None,
        validation_loss_mode: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]], torch.Tensor]:

        self.set_valid_loss_mode(validation_loss_mode)

        x = self.transformer.embed(x=idx) 
        # x = self.insert_no_event_tokens(x)
        
        for domain in age:
            age_emb = self.transformer.age_embedding(age[domain])
            x[domain] += age_emb

        # mask tokens in a domain-wise manner                
        x = self.transformer.drop(x)

        attn_mask = causal_attention_mask(
            pad=self.is_not_padding(idx), 
            t1=targets_age, t0=age, 
            mask_ties=self.config.mask_ties
        )
        
        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        logits = self.lm_head(
            x[:, :, :]
        )   # note: using list [-1] to preserve the time dim
    
        return logits, loss, att

        # if (targets is not None) and (targets_age is not None):
            # logits = self.lm_head(x)
# 
            # logits_cp = logits.clone()
            # ignored_tokens = self.config.ignore_tokens.copy()
            # if validation_loss_mode:
                # ignored_tokens += [1]
                # logits_cp[..., ignored_tokens] = -torch.inf
# 
            # dt = ties_adjusted_delta_t(
                # t0=age, t1=targets_age,
                # attn_mask=attn_mask,
                # mask_ties=self.config.mask_ties,
                # eps=0.0 if self.config.zero_inflate else 1.0,
            # )
# 
            # is_valid_target = target_mask(targets, ignore_tokens=ignored_tokens)
            # loss_ce = self.ce_head(logits=logits_cp, targets=targets)
            # loss_ce = torch.mean(loss_ce[is_valid_target])
            # loss_dt = self.dt_head(logits=logits, delta_t=dt)
            # loss_dt = torch.mean(loss_dt[is_valid_target])
# 
            # loss = {
                # "loss_ce": loss_ce,
                # "loss_dt": loss_dt, 
                # "loss": loss_ce * self.config.ce_beta + loss_dt * self.config.dt_beta,
            # }
        # else:
            # inference-time mini-optimization: only forward the lm_head on the very last position


    def compute_loss(self, logits, targets):

        loss = {
           "loss_ce": self.cross_entropy_loss(targets),
           "loss_dt": self.time_to_event_loss(targets, targets_age), 
           "loss": loss_ce * self.config.ce_beta + loss_dt * self.config.dt_beta
           # or "ce", "dt", "total"
        }

        return loss


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
                new_key = mapping.get(k, k)
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
