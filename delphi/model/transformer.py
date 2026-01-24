import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import fields, is_dataclass, dataclass, field, asdict

from pathlib import Path
import sys

if ( DELPHI_DIR := Path(__file__).resolve().parent.parent.parent ) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from typing import Optional, List, Tuple, Union

import inspect

import numpy as np
import pandas as pd
import yaml

import logging
logger = logging.getLogger(__name__)
from pathlib import Path

DAYS_PER_YEAR = 365.25

import warnings
from collections import defaultdict

from easydict import EasyDict

# —————————————————————————————————————————————————————————————————————————————————————————————————————
# TIPS TO HELP YOUR READING OF THIS FILE:
# - I make extensive use of separators like these  #————— to visually separate different classes or sections
# - I use the := operator (walrus) to assign and check values in a single line (e.g. if (x := compute_x()) > 0:).
#   This was introduced in Python 3.8, so you may want to understand what it does if you're not familiar with it (it's very simple).
# - 

# —————————————————————————————————————————————————————————————————————————————————————————————————————

class AttentionMaskBuilder(nn.Module):
    """
    Parses and builds attention masks from a string like:
        [hla_alleles,sex]:bidirectional,[disease,lifestyle,sex,death]:causal(mask_ties=True)
            which is the same as NoAttention([hla_alleles, sex]:bidirectional, [disease,lifestyle,sex,death]:causal(mask_ties=True))
        
    """

    def __init__(self, scheme_str: str):
        super().__init__()
        self.scheme = self._parse_scheme(scheme_str)

    # —-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—
    def _split_top_level(self, s: str, sep: str = ","):
        """
        Splits a string by sep but ignores separators inside [] or ().
        """
        parts, buf, depth_brack, depth_paren = [], "", 0, 0
        for ch in s:
            if ch == "[":
                depth_brack += 1
            elif ch == "]":
                depth_brack -= 1
            elif ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren -= 1

            if ch == sep and depth_brack == 0 and depth_paren == 0:
                parts.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            parts.append(buf.strip())
        return parts

    # —-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—-—
    def _parse_scheme(self, scheme_str: str):
        scheme = {}
        parts = self._split_top_level(scheme_str)

        for part in parts:
            if ":" not in part:
                raise ValueError(f"Invalid rule fragment: {part}")
            domain_part, rule_part = part.split(":", 1)
            domain_part, rule_part = domain_part.strip(), rule_part.strip()

            # domains
            if domain_part.startswith("[") and domain_part.endswith("]"):
                domains = [d.strip() for d in domain_part[1:-1].split(",")]
            else:
                domains = [domain_part]

            # rule type
            if rule_part.startswith("causal"):
                rule_type = "causal"
                mask_ties = "mask_ties=True" in rule_part
            elif rule_part.startswith("bidirectional"):
                rule_type = "bidirectional"
                mask_ties = False
            else:
                raise ValueError(f"Unknown rule type: {rule_part}")

            scheme[tuple(domains)] = {"type": rule_type, "mask_ties": mask_ties}

        return scheme


    def build(self, ages: torch.Tensor, domains: torch.Tensor, domain2id: dict):
        """
        Build a [B, L, L] attention mask from a declarative DSL.
        1 = attention allowed
        0 = attention blocked
        """
        B, L = ages.shape
        device = ages.device

        # Start with everything blocked
        mask = torch.zeros(B, L, L, device=device)

        # Always allow self-attention
        diag = torch.arange(L, device=device)
        mask[..., diag, diag] = 1

        # Precompute expanded views for causal tests
        age_row  = ages.unsqueeze(2)  # [B, L, 1]
        age_col  = ages.unsqueeze(1)  # [B, 1, L]

        for dom_names, cfg in self.scheme.items():
            # Gather domain ids
            dom_ids = torch.tensor([domain2id[d] for d in dom_names], device=device)

            # Boolean mask for tokens belonging to this rule
            dom_mask = torch.isin(domains, dom_ids)   # [B, L]

            # Pairs of tokens both belonging to allowed domains in this rule
            pair_mask = dom_mask.unsqueeze(2) & dom_mask.unsqueeze(1)   # [B, L, L]

            if cfg["type"] == "bidirectional":
                # Allow everything inside the domain pair
                mask[pair_mask] = 1

            elif cfg["type"] == "causal":
                if cfg.get("mask_ties", False):
                    # strict causal: do not allow ties
                    causal = age_row > age_col
                else:
                    # allow ties
                    causal = age_row >= age_col

                # Combine with the domain pair mask
                final = pair_mask & causal
                mask[final] = 1

            else:
                raise ValueError(f"Unknown attention type: {cfg['type']}")

        return mask

    def forward(self, ages, domains, domain2id):
        return self.build(ages, domains, domain2id)


# —————————————————————————————————————————————————————————————————————————————————————————————————————

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
    """
    Sinusoidal age encoding with a learnable linear projection.
    
    Input can be (seq_len, batch_size) or (seq_len,) or scalar.
    Output is always (seq_len, n_embd).
    """

    def __init__(self, n_embd: int, norm_factor: float = 365.25, max_wavelen: float = 10000.0):
        super().__init__()
        # frequency terms for sine/cosine
        div_term = torch.exp(
            torch.arange(0, n_embd, 2) * (-math.log(max_wavelen) / n_embd)
        )
        self.register_buffer("div_term", div_term)  # non-trainable buffer
        self.n_embd = n_embd
        self.linear = nn.Linear(n_embd, n_embd, bias=False)
        self.norm_factor = norm_factor

    def forward(self, age: torch.Tensor):
        """
        Args:
            age: Tensor with shape (seq_len, batch_size) or (seq_len,) or scalar.
        Returns:
            Tensor with shape (seq_len, n_embd).
        """
        # normalize ages
        age = age.float() / self.norm_factor

        # standardize input shapes
        if age.ndim == 0:        # scalar -> (1,1)
            age = age.view(1, 1)
        elif age.ndim == 1:      # (seq_len,) -> (seq_len,1)
            age = age.unsqueeze(1)

        seq_len, batch_size = age.shape

        # expand age and div_term to make broadcasting explicit
        # age_expanded: (seq_len, batch_size, 1)
        # div_term: (1, 1, n_embd/2)
        age_expanded = age.unsqueeze(-1)
        div_term = self.div_term.view(1, 1, -1)

        # compute sin/cos embeddings
        sin_part = torch.sin(age_expanded * div_term)
        cos_part = torch.cos(age_expanded * div_term)

        # interleave sin and cos along the last dimension
        # result: (seq_len, batch_size, n_embd)
        y = torch.zeros(seq_len, batch_size, self.n_embd, device=age.device)
        y[..., 0::2] = sin_part
        y[..., 1::2] = cos_part

        # squeeze batch dim if batch_size==1 to return (seq_len, n_embd)
        if batch_size == 1:
            y = y.squeeze(1)

        return self.linear(y)


# ———————————————————— CONFIG ————————————————————————————————————————————————————————————————

@dataclass
class EmbedConfig:
    projector:  str           = "linear"  # "linear", "mlp", or "embed"
    n_layers:   Optional[int] = None
    n_hidden:   Optional[int] = None
    input_size: Optional[int] = None
    pretrained_path: Optional[str] = None  # path to .pt/.npy/.pkl with lookup table
    freeze: bool = False                   # if previous lookup table is to be left fixed
    path: Optional[str] = None
    predict: bool = False
    age_jitter: bool = False
    type: str = "categorical"
    at_birth: bool = False


@dataclass
class DelphiConfig:
    # vocab_size: Optional[int] = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    attention_scheme:Union[str, List] = field(
        default_factory=lambda: "[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death]:causal(mask_ties=True)"
    )
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    bias: bool = True
    block_size: int = 64
    dropout: float = 0.1
    token_dropout: float = 0.1
    # ignore_tokens: list = field(default_factory=lambda: [0])
    domains: dict[str, EmbedConfig] = field(default_factory=dict)
    # modality_emb: bool = False
    # ce_beta: float = 1.0
    # dt_beta: float = 1.0
    zero_inflate: bool = False
    zero_inflate_projector: str = "linear"

    def items(self):        
        return asdict(self).items()
    

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


# ———————————————————— Embedding layer for a single domain —————————————————————————————————————————————

class DomainEmbedding(nn.Module):

    def __init__(self, config: EmbedConfig, domain_name, n_embed: int) -> None:

        super().__init__()
        self.config = config

        if domain_name == "padding":
            self.projector = nn.Embedding(2, n_embed)
            self.PADDING_TOKEN = 0
            self.PADDING_AGE = -10000
            self.NO_EVENT_TOKEN = 1
            return

        # ————————————————————————————————————————————————————————————————————————————————————

        if config.input_size is None and config.projector.lower() != "pretrained":            
            tokenizer_file = Path(config.path) / "tokenizer.yaml"
            with open(tokenizer_file, "r") as f:
                self.config.input_size = len(yaml.load(f, Loader=yaml.FullLoader))

        elif config.projector.lower() == "pretrained":
            weights = torch.load(config.pretrained_path)  # Tensor [vocab_size, d_ext]
            self.config.input_size, d_ext = weights.shape

        # ————————————————————————————————————————————————————————————————————————————————————

        if config.projector.lower() == "embed":
            # logging.info(f"DomainEmbedding for '{domain_name}': Using nn.Embedding with input_size={config.input_size}, n_embed={n_embed}")
            self.projector = nn.Embedding(config.input_size, n_embed)

        elif config.projector.lower() == "linear":
            self.projector = nn.Linear(config.input_size, n_embed, bias=False)

        elif config.projector.lower() == "mlp":
            self.projector = self.build_mlp_projector(config)                    

        elif config.projector.lower() == "pretrained":
            assert hasattr(config, "pretrained_path"), f"""
            You need to provide pretrained_path in the config if using config.projector == 'pretrained'
            """
            self.projector = self.get_from_pretrained(config.pretrained_path, n_embed, config.freeze)

        else:
            raise ValueError(f"unknown projector type: {config.projector}")
        
        # ————————————————————————————————————————————————————————————————————————————————————
    
    def get_from_pretrained(self, path, n_embed, freeze):

        weights = torch.load(path)
        vocab_size, d_ext = weights.shape

        logger.info(f"Loading pretrained embedding: vocab_size={vocab_size}, d_ext={d_ext}")

        projector = nn.Embedding(vocab_size, d_ext) #, padding_idx=0)
        projector.weight.data.copy_(weights)
        projector.weight.requires_grad = not freeze

        # To project onto Delphi embedding space
        return nn.Sequential(
            projector,
            nn.Linear(d_ext, n_embed, bias=False)
        )

    @property
    def weight(self):
        return self.projector.weight

    @property
    def vocab_len(self):
        return self.projector.weight.shape[0]

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
            h = self.embed(x)
            out = self.projector(h)
            return out

        out = self.projector(x)
        return out

# ———————————————————— Embedding layer for all domains ————————————————————————————————————————————————————————————————

class MultiDomainEmbedding(nn.Module):

    def __init__(self, config: DelphiConfig) -> None:
        
        super().__init__()
        self.config = config
        self.token_drop   = nn.Dropout(config.token_dropout)        
        
        self.domain_embed = nn.ModuleDict()
        
        if len(config.domains) > 0:            
            for domain_name, domain_cfg in config.domains.items():
                self.domain_embed[domain_name] = DomainEmbedding(config=domain_cfg, domain_name=domain_name, n_embed=config.n_embd)
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        
        if len(self.domain_embed) > 0:
            emb = {}
            for domain_name in self.domain_embed:                
                token_emb = self.domain_embed[domain_name](x[domain_name])
                emb[domain_name] = token_emb
        else:
            token_emb = self.token_embedding(x)
            token_emb = self.token_drop(token_emb) * (1 - self.config.token_dropout)
        
        return emb
    

    def __iter__(self):
        """Iterate over (domain_name, embedding_module) pairs."""
        return iter(self.domain_embed.items())

    def __getitem__(self, key):
        """Allow dict-like access: model['diseases']"""
        return self.domain_embed[key]

    def __len__(self):
        """Return the number of domain embeddings."""
        return len(self.domain_embed)
    
    def keys(self):
        return self.domain_embed.keys()


# ———————————————————— Final embedding to logits ——————————————————————————————————————————————————
class DomainwiseTiedLinear(nn.Module):
    
    def __init__(self, embedding_layer_dict):
        super().__init__()
        self.embedding_layer_dict = embedding_layer_dict


    def forward(self, x):
        return {
          dname: F.linear(x, embedding_layer.weight) 
          for dname, embedding_layer in self.embedding_layer_dict if dname != "padding"
        }

# ———————————————————— HEADS ————————————————————————————————————————————————————————————————

class CrossEntropyHead(nn.Module):

    def __init__(self): #, config):
        super().__init__()
        # self.config = config

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


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MaskedSelfAttention(nn.Module):

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


    def forward(self, x, attn_mask=None):

        B, T, C = x.size()
        # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

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
        self.attn = MaskedSelfAttention(config)
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
        self.config.attention_scheme = self.adapt_attention_scheme(self.config.attention_scheme)       
        self.build_model(config)
        
        self.block_size = config.block_size
        
        initialize_weights(self, config=config)
        
        self.PAD_TOKEN_ID = 0
        self.PAD_DOMAIN_ID = 0 
        self.PAD_AGE = -10000

    
    def adapt_attention_scheme(self, attention_scheme):

        if isinstance(attention_scheme, str):
            attention_scheme = [attention_scheme]
        if len(attention_scheme) == 1:
            attention_scheme = self.config.n_layer * attention_scheme
        
        return attention_scheme


    def build_model(self, config: DelphiConfig):

        self.transformer = nn.ModuleDict(dict(
            embed=MultiDomainEmbedding(config),
            age_embedding=AgeEncoding(n_embd=config.n_embd),
            drop=nn.Dropout(config.dropout),
            attn_mask_builder=nn.ModuleList([
                nn.ModuleList( [
                    AttentionMaskBuilder(config.attention_scheme[i]) 
                    for i in range(config.n_layer)
                ] ) for j in range(config.n_head) ]
            ),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias)
        ))

        self.embedding_to_logits = DomainwiseTiedLinear(self.transformer.embed)

    def set_block_size(self, block_size):
        self.block_size = block_size

    def set_valid_loss_mode(self, validation_loss_mode):
        self.validation_loss_mode = validation_loss_mode

    @property
    def domain_cfg(self):
        return self.config.domains

    @property
    def domains(self):
        return list(self.config.domains.keys())

    @property
    def predicted_domains(self):
        return [ dname for dname, config in self.config.domains.items() if config.predict ]

    #TODO: check this. There's a possible confusion here.
    #Do we want to use the space of all domains or only that of predicted domains to assign integers?
    @property
    def predicted_domains_as_int(self):
        return torch.tensor([ i for i, dname in enumerate(self.predicted_domains)]).to(self.device)

    @property
    def predicted_domains_to_int(self):
        return { dname: i for i, dname in enumerate(self.predicted_domains) }

    @property
    def domain_to_int(self):
        return { dname: i for i, dname in enumerate(self.transformer.embed.domain_embed.keys()) }

    @property
    def int_to_domain_name(self):
        return { v: k for k, v in self.domain_to_int.items() }

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def vocab_lens(self):
        if not hasattr(self, "_vocab_lens"):
            self._vocab_lens = { self.domain_to_int[k]: v.vocab_len for k, v in self.transformer.embed.domain_embed.items() }
        return self._vocab_lens


    #### LOSSES ########################################################################
    def cross_entropy_loss(self, logits, targets, agg=None):
        '''
        Cross entropy loss for the next token prediction.
        Arguments:
            logits: Tensor, shape [batch_size, sequence_length, vocab_size]
            targets: Tensor, shape [batch_size, sequence_length]
            pass_tokens: Tensor of bools, batch_size * sequence_length
            agg: one of None, "per_token" or "per_disease"
        '''

        n_classes = logits.size(-1)            
        if agg == "per_token":
            log_softmax = F.log_softmax(logits.view(-1, n_classes), dim=-1)
            loss_ce_per_token = log_softmax[torch.arange(log_softmax.size(0)), targets.view(-1)]
            return loss_ce_per_token
        if agg == "per_disease":
            loss_ce_per_token = self.cross_entropy_loss(logits, targets, agg="per_token")
            loss_ce_agg_per_disease = pd.DataFrame([loss_ce_per_token, targets.view(-1).cpu().numpy()]).T.\
                set_axis(["log_p", "token_id"], axis=1).\
                astype({"token_id": int}).\
                groupby("token_id").sum().\
                log_p.apply(lambda x: x.item())
            return loss_ce_agg_per_disease / len(logits)
        elif agg is None:
            loss_ce = F.cross_entropy(
                logits.reshape(-1, n_classes), 
                targets.reshape(-1), 
                ignore_index=-1
            )
        else:
            raise ValueError("agg should be in [None, 'per_disease']")

        return loss_ce


    def time_to_event_loss(self, logits, time_to_next, t_min, agg=None):

        '''
        '''
        
        lse = torch.logsumexp(logits,-1) ## More forgiving than using torch.max() for the most likely next event
        lse = - torch.log(torch.exp(-lse) + t_min)
        dt  = torch.clamp(time_to_next, min=1.0)
        log_dt = - torch.log(dt + t_min).view(-1) 
        loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - log_dt.reshape(-1))) 
    
        if agg is None: 
            pass
        elif agg == "mean":
            loss_dt = loss_dt.mean()
        elif agg == "sum":
            loss_dt = loss_dt.sum()
        elif agg == "per_disease":
            raise NotImplementedError
          
        return loss_dt
    

    def compute_loss(self, logits, targets, targets_age):

        loss = EasyDict({
           "loss_ce": self.cross_entropy_loss(targets),
           "loss_dt": self.time_to_event_loss(targets, targets_age), 
           "loss": loss_ce * self.config.ce_beta + loss_dt * self.config.dt_beta
        })

        return loss

    ########################################################################################
        
    def to_tensor(self, x, ages, embeddings, subject_ids):

        all_tokens, all_embeddings, all_ages, all_domains, all_subjects = [], [], [], [], []
    
        for domain_idx, dname in enumerate(x.keys()):
            t = x[dname]
            a = ages[dname]
            e = embeddings[dname]
            s = subject_ids[dname]
    
            n = t.shape[0]
            d = torch.full((n,), domain_idx, device=t.device, dtype=torch.long)
    
            all_tokens.append(t)
            all_ages.append(a)
            all_embeddings.append(e)
            all_domains.append(d)
            all_subjects.append(s)
    
        tokens_flat     = torch.cat(all_tokens)
        ages_flat       = torch.cat(all_ages)
        embeddings_flat = torch.cat(all_embeddings)
        domains_flat    = torch.cat(all_domains)
        subjects_flat   = torch.cat(all_subjects)
    
        # sort once by (subject, age)
        order = torch.argsort(ages_flat, stable=True)
        # 2) primaria: subject (estable, preserva el orden por age dentro de cada subject)
        order = order[torch.argsort(subjects_flat[order], stable=True)]
    
        tokens_flat     = tokens_flat[order]
        ages_flat       = ages_flat[order]
        embeddings_flat = embeddings_flat[order]
        domains_flat    = domains_flat[order]
        subjects_flat   = subjects_flat[order]
            
        # group by subject
        unique_subjects, counts = torch.unique_consecutive(
            subjects_flat, return_counts=True
        )
    
        # split
        batch_tokens     = torch.split(tokens_flat, counts.tolist())
        batch_ages       = torch.split(ages_flat, counts.tolist())
        batch_embeddings = torch.split(embeddings_flat, counts.tolist())
        batch_domains    = torch.split(domains_flat, counts.tolist())
    
        # stack → (B, T, *)
        batch_tokens     = torch.nn.utils.rnn.pad_sequence(batch_tokens, batch_first=True)
        batch_ages       = torch.nn.utils.rnn.pad_sequence(batch_ages, batch_first=True)
        batch_embeddings = torch.nn.utils.rnn.pad_sequence(batch_embeddings, batch_first=True)
        batch_domains    = torch.nn.utils.rnn.pad_sequence(batch_domains, batch_first=True)
    
        return (
            batch_tokens.int(),
            batch_ages,
            batch_embeddings,
            unique_subjects,
            batch_domains,
        )


    # Old and extremely slow on GPU, just kept temporarily for comparison purposes
    def _to_tensor_deprecated(self, x, ages, embeddings, subject_ids):

        '''
            input values:
              - x:          dict[domain, Tensor] where the tensor has data for batch_size subjects
              - ages:       dict[domain, Tensor]
              - subject_ids:   dict[domain, Tensor]
              - embeddings: dict[domain, Tensor]

            return values: batches of 
              - tokens
              - ages
              - embeddings
              - unique_subjects
              - domains
        '''

        all_tokens, all_embeddings, all_ages, all_domains, all_subjects = [], [], [], [], []
        
        for domain_idx, dname in enumerate(x.keys()):
            
            e = embeddings[dname]
            t = x[dname]
            a = ages[dname]
            s = subject_ids[dname]
    
            n = t.shape[0]
            
            # domain ids (move it from dict keys to a separate tensor)
            d = torch.full((n,), domain_idx, dtype=torch.long, device=t.device)
    
            all_tokens.append(t)
            all_ages.append(a)
            all_embeddings.append(e)
            all_domains.append(d)
            all_subjects.append(s)
        
        tokens_flat     = torch.cat(all_tokens)
        ages_flat       = torch.cat(all_ages)
        embeddings_flat = torch.cat(all_embeddings)
        domains_flat    = torch.cat(all_domains)
        subjects_flat   = torch.cat(all_subjects)
    
        unique_subjects = subjects_flat.unique(sorted=True)
        batch_tokens, batch_embeddings, batch_ages, batch_domains = [], [], [], []
        
        # 2) Process subject by subject        
        for subj in unique_subjects:
            mask = subjects_flat == subj
            a_subj = ages_flat[mask]
            a_subj = a_subj[ order:=torch.argsort(a_subj) ]
            t_subj = tokens_flat[mask][order]
            e_subj = embeddings_flat[mask][order]
            d_subj = domains_flat[mask][order]
            
            batch_tokens.append(t_subj.unsqueeze(0))
            batch_ages.append(a_subj.unsqueeze(0))
            batch_embeddings.append(e_subj.unsqueeze(0))
            batch_domains.append(d_subj.unsqueeze(0))
    
        batch_tokens = torch.stack(batch_tokens)
        batch_embeddings = torch.stack(batch_embeddings)
        batch_ages   = torch.stack(batch_ages)
        batch_domains = torch.stack(batch_domains)
    
        return \
            batch_tokens.int().squeeze(1),\
            batch_ages.squeeze(1),\
            batch_embeddings.squeeze(1),\
            unique_subjects,\
            batch_domains.squeeze(1)

    
    def _build_trace(self, x, ages, emb, subject_ids, domains):

        return dict(
            tokens=x.detach(),
            ages=ages.detach(),
            emb=emb.detach(),
            subject_ids=subject_ids.detach(),
            domains=domains.detach()
        )


    # ————————— FORWARD ————————————————————————————————————————————————————————————————————————————————————

    def forward(self, 
        x: torch.Tensor, ages: torch.Tensor, subject_ids: torch.Tensor, 
        validation_loss_mode: bool = False, 
        return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:

        
        x, ages, emb, subject_ids, domains = self.to_tensor(x, ages, emb := self.transformer.embed(x), subject_ids)

        self._trace = self._build_trace(x, ages, emb, subject_ids, domains)
                    
        emb += self.transformer.age_embedding(ages)
        
        single_mask = self.transformer.attn_mask_builder[0][0].build(ages, domains, self.domain_to_int)  # (B, L-1, L-1)
        
        attn_mask = single_mask.unsqueeze(1).unsqueeze(1).\
                expand(-1, self.config.n_layer, self.config.n_head, -1, -1).\
                    permute(1, 0, 2, 3, 4)        
        
        h, att = emb, []
        for i, transformer_block in enumerate(self.transformer.h):
            h, _att = transformer_block(h, attn_mask=attn_mask[i])
            att.append(_att)
        
        h = self.transformer.ln_f(h)        
        
        logits = self.embedding_to_logits(h)
        
        return logits,\
                (attention_matrices := torch.stack(att) if return_attention else None)

    # —————————— END FORWARD ————————————————————————————————————————————————————————————————————————————————

    def get_max_ages_per_subject(self, ages, subject_ids):
    
        device = ages['diseases'].device     

        n_subjects = torch.unique(subject_ids['diseases']).shape[0]

        # concatenar
        all_subjects = torch.cat([
            subject_ids['diseases'],
            subject_ids['death'],
        ]).long()

        unique_subjects, subject_ids_local = torch.unique(
            all_subjects,
            return_inverse=True
        )
    
        all_ages = torch.cat([
            ages['diseases'],
            ages['death'],
        ]).float()
    
        max_age = torch.full(
            (n_subjects,),
            -torch.inf,
            device=device,
            dtype=all_ages.dtype,
        )
    
        max_age.scatter_reduce_(
            0,
            subject_ids_local,
            all_ages,
            reduce="amax",
            include_self=True,
        )
    
        return max_age


    def _subject_allows_no_events(self, max_age: float, padding: str) -> bool:

        if padding in [None, "none"]:
            return False
        if not np.isfinite(max_age):
            return False
        return True


    def _generate_no_event_ages(
        self,
        max_age: float,
        padding: str,
        no_event_token_rate: float,
        device,
        gen=None,
    ):
        """
        Returns a 1D tensor of ages (float, days), or None if no no-events apply.
        """
    
        if padding == "regular":
            start = DAYS_PER_YEAR
            step = DAYS_PER_YEAR * no_event_token_rate
    
            # empty rank -> no no-events
            if max_age <= start or step <= 0:
                return None
    
            return torch.arange(
                start,
                max_age,
                step,
                device=device,
                dtype=torch.float,
            )
    
        elif padding == "random":
            step = DAYS_PER_YEAR * no_event_token_rate
            if step <= 0:
                return None
    
            n_pad = int(max_age // step)
            if n_pad <= 0:
                return None
    
            return torch.rand(
                n_pad,
                generator=gen,
                device=device,
            ) * max_age
    
        else:
            raise NotImplementedError(f"Unknown padding mode: {padding}")
    

    def insert_no_event_tokens(
        self,
        tokens,
        ages,
        subject_ids,
        max_ages,
        no_event_token_rate=5,
        padding="regular",
        gen=None,
    ):
    
        NO_EVENT_TOKEN = self.transformer.embed['padding'].NO_EVENT_TOKEN
        device = max_ages.device
    
        pad_tokens = []
        pad_ages = []
        pad_subjects = []
    
        warned = False
    
        batch_subjects = torch.unique(
            torch.cat([subject_ids[d] for d in subject_ids if d != "padding"]),
            sorted=True
        )
        
        # mapping: subject_id real → índice batch
        subject_to_batch = { int(s.item()): i for i, s in enumerate(batch_subjects) }       
        
        for subj_id in batch_subjects:          
            
            b = subject_to_batch[subj_id.int().item()]            
            
            max_age = float(max_ages[b].item())
    
            if not self._subject_allows_no_events(max_age, padding):
                continue
    
            pad = self._generate_no_event_ages(
                max_age=max_age,
                padding=padding,
                no_event_token_rate=no_event_token_rate,
                device=device,
                gen=gen,
            )
    
            if pad is None or pad.numel() == 0:
                if padding == "regular" and not warned:
                    warnings.warn(
                        "Some subjects have max_age too small for regular no-event padding. "
                        "No no-event tokens were inserted for them (expected behavior)."
                    )
                    warned = True
                continue
    
            pad_tokens.append(
                torch.full((pad.numel(),), NO_EVENT_TOKEN, device=device, dtype=torch.long)
            )
            pad_ages.append(pad)
            pad_subjects.append(
                torch.full((pad.numel(),), subj_id, device=device, dtype=torch.long)
            )
    
        if pad_tokens:
            tokens["padding"] = torch.cat(pad_tokens)
            ages["padding"] = torch.cat(pad_ages)
            subject_ids["padding"] = torch.cat(pad_subjects)
    
        if "padding" in subject_ids:
            pass


        return tokens, ages, subject_ids
   
     
    def run_inference(self, dataset, batch_size, block_size=None, return_token_df=False):
        
        if return_token_df:
            raise NotImplementedError

        from tqdm import tqdm
        from data.dataset import DelphiDataset, DelphiDataloader
        from data.event_set import EventSet

        dataloader = DelphiDataloader(dataset, batch_size=batch_size, shuffle=False)
        
        if block_size is not None:
            old_block_size = self.block_size
            self.set_block_size(block_size)

        all_logits = []
        for bi, batch in tqdm(enumerate(dataloader)):
            
            x, ages, subject_ids = self.prepare_input(batch)            
            logits_dict, _       = self(x, ages, subject_ids)            
            
            logits_all_domains = []
            for dom in self.predicted_domains:
                assert dom  in logits_dict, f"Domain '{dom}' not found in model output ({self.predicted_domains})."
                x = logits_dict[dom]   # [B, L, D]            
                logits_all_domains.append(x)
            logits_all_domains = torch.cat(logits_all_domains, dim=-1)  # [B * L, sum(Ds)]

            all_logits.append(logits_all_domains)
    
        logits = torch.cat(all_logits, dim=0)    
        
        # Restore old block size
        if block_size is not None:
            self.set_block_size(old_block_size)
        
        return logits


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

    
    def get_embedding_at_age(self, x, ages, readout_age):
        raise NotImplementedError


    def get_offset_per_domain(self, domains_of_interest=None):
        
        offsets_per_domain = np.array([0] + list(self.vocab_lens.values())).cumsum()[:-1]
        offsets_per_domain = torch.tensor(offsets_per_domain).to(self.device)
        return offsets_per_domain


    @property
    def offsets_per_domain(self):
        if not hasattr(self, "_offsets_per_domain"):
            self._offsets_per_domain = self.get_offset_per_domain()
        return self._offsets_per_domain

        
    def local_to_global_ids(self, domain_ids, local_ids):

        offsets = self.offsets_per_domain[domain_ids]
        global_ids = offsets + local_ids
        return global_ids


    def add_token_domain(self, new_token_domain):
        raise NotImplementedError("add_token_domain method not yet implemented.")
    

    # ——————————————————————————————————————————————————————————————————————————————

    def prepare_input(self, batch):

        x, ages, subject_ids = self.get_tensors_from_batch(batch)
        max_ages             = self.get_max_ages_per_subject(ages, subject_ids)
        x, ages, subject_ids = self.insert_no_event_tokens(x, ages, subject_ids, max_ages)
        x, ages, subject_ids = self.adjust_to_seqlen(x, ages, subject_ids, self.block_size)
        return x, ages, subject_ids
    

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
                
        # x  = deepcopy(x)
        # ages  = deepcopy(ages)
        # subject_ids  = deepcopy(subject_ids)

        domains = self.domains
    
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
    
