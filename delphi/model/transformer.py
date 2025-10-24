import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import fields, is_dataclass, dataclass, field
from typing import Optional, List, Tuple, Union
from copy import deepcopy

import inspect

import numpy as np
import pandas as pd
import yaml

import logging
logger = logging.getLogger(__name__)
from pathlib import Path

DAYS_PER_YEAR = 365.25

class AttentionMaskBuilder(nn.Module):
    """
    Parses and builds attention masks from a string like:
        [hla_alleles,sex]:bidirectional,[disease,lifestyle,sex,death]:causal(mask_ties=True)
            which is the same as NoAttention([hla_alleles, sex]:bidirectional, [disease,lifestyle,sex,death]:causal(mask_ties=True))
        
    """

    def __init__(self, scheme_str: str):
        super().__init__()
        self.scheme = self._parse_scheme(scheme_str)

    # --------------------------------------------------
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

    # --------------------------------------------------
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

    # --------------------------------------------------
    def build_slow(self, ages: torch.Tensor, domains: torch.Tensor, domain2id: dict):
        B, L = ages.shape
        dd = { "device": ages.device }
        mask = torch.ones(B, L, L, **dd)

        for dom_names, cfg in self.scheme.items():

            dom_ids = [domain2id[d] for d in dom_names]
            dom_mask = torch.isin(domains, torch.tensor(dom_ids, **dd))

            for b in range(B):
                idxs = torch.nonzero(dom_mask[b], as_tuple=True)[0]
                for i in idxs:
                    for j in idxs:
                        if cfg["type"] == "bidirectional":
                            continue
                        elif cfg["type"] == "causal":
                            if ages[b, i] < ages[b, j]:
                                mask[b, i, j] = 0
                            elif ages[b, i] == ages[b, j] and cfg["mask_ties"]:
                                mask[b, i, j] = 0

        return mask


    def build(self, ages: torch.Tensor, domains: torch.Tensor, domain2id: dict):

        B, L = ages.shape
        dd = { "device": ages.device }
        mask = torch.ones(B, L, L, **dd) # All True

        for dom_names, cfg in self.scheme.items():
            
            dom_ids = torch.tensor([domain2id[d] for d in dom_names], **dd)
            dom_mask = torch.isin(domains, dom_ids)  # [B, L]
    
            if cfg["type"] == "bidirectional":
                continue
    
            mask_i = dom_mask.unsqueeze(2)  # (B, L, 1)
            mask_j = dom_mask.unsqueeze(1)  # (B, 1, L)
            both = mask_i & mask_j
    
            age_i, age_j = ages.unsqueeze(2), ages.unsqueeze(1)
    
            causal = (age_i < age_j) | ((cfg["mask_ties"]) & (age_i == age_j))
            # mask = mask.masked_fill(both & causal, float("-inf"))
            mask = mask.masked_fill(both & causal, FALSE := 0)
    
        # Finally, let's remove the last line of input and the first line of output.
        mask = mask[:,1:,:-1]
        
        return mask


    def forward(self, ages, domains, domain2id):
        return self.build(ages, domains, domain2id)


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
    freeze: bool = False                    # if previous lookup table is to be left fixed
    path: Optional[str] = None
    predict: bool = False
    age_jitter: bool = False

@dataclass
class DelphiConfig:
    # vocab_size: Optional[int] = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    attention_scheme:Union[str, List] = field(default_factory=lambda: "[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death]:causal(mask_ties=True)")
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    bias: bool = True
    block_size: int = 64
    dropout: float = 0.1
    token_dropout: float = 0.1
    # mask_ties: bool = False
    # ignore_tokens: list = field(default_factory=lambda: [0])
    domains: dict[str, EmbedConfig] = field(default_factory=dict)
    # modality_emb: bool = False
    # ce_beta: float = 1.0
    # dt_beta: float = 1.0
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

    def __init__(self, config: EmbedConfig, domain_name, n_embed: int) -> None:

        super().__init__()
        self.config = config
        # assert config.input_size is not None, "input_size must be specified"

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
            print(f"DomainEmbedding: Using nn.Embedding with input_size={config.input_size}, n_embed={n_embed}")
            self.projector = nn.Embedding(config.input_size, n_embed)

        elif config.projector.lower() == "linear":
            self.projector = nn.Linear(config.input_size, n_embed, bias=False)

        elif config.projector.lower() == "mlp":
            self.projector = self.build_mlp_projector(config)                    

        elif config.projector.lower() == "pretrained":
            assert hasattr(config, "pretrained_path"), f"You need to provide pretrained_path in the config if using config.projector == 'pretrained'"
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
        # if self.config.lower() == "pretrained":
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
            logger.debug(f"After pretrained lookup: {tuple(x.shape)}")
            out = self.projector(h)
            logger.debug(f"After projection: {tuple(out.shape)}")
            return out

        out = self.projector(x)
        logger.debug(f"After projector: {tuple(out.shape)}")
        return out


class DomainwiseTiedLinear(nn.Module):
    
    def __init__(self, embedding_layer_dict):
        super().__init__()
        self.embedding_layer_dict = embedding_layer_dict


    def forward(self, x):
        return {
          dname: F.linear(x, embedding_layer.weight) 
          for dname, embedding_layer in self.embedding_layer_dict if dname != "padding"
        }


class DelphiEmbedding(nn.Module):

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
        
        # Add embedding for padding tokens
        # self.domain_embed["padding"] = nn.Embedding(2, config.n_embd)
        # self.PAD_TOKEN_ID = 0
        # self.NO_EVENT_TOKEN_ID = 1
                

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


# ———————————————————— MASKS ————————————————————————————————————————————————————————————————

# def causal_attention_mask(
#     pad: torch.Tensor,
#     mask_ties: bool = False,
#     t0: Optional[torch.Tensor] = None,
#     t1: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
# 
#     b, l = pad.shape
#     device = pad.device
#     dd = {"device": device}
# 
#     lower_tri_mask = torch.tril(torch.ones((l, l), **dd))
#     lower_tri_mask = lower_tri_mask.view(1, l, l)
#     pad_mask = pad.view(b, 1, l).to(torch.int)
#     attn_mask = pad_mask * lower_tri_mask
# 
#     if mask_ties:
#         assert t0 is not None
#         if t1 is not None:
#             ties_mask = (t1.view(b, l, 1) != t0.view(b, 1, l)).to(torch.int)
#             attn_mask *= ties_mask
# 
#     attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(
#         torch.ones(l, **dd)
#     ) > 0
# 
#     return attn_mask.unsqueeze(1)


def target_mask(x1: torch.Tensor, ignore_tokens: list[int]) -> torch.Tensor:

    is_valid_target = x1 != 0
    
    for k in ignore_tokens:
        is_valid_target *= x1 != k

    return is_valid_target


# TODO: make sure that this is not needed anymore.
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


# Rename to CustomSelfAttention or RestrictedSelfAttention
# there is nothing in this function that makes it "causal", 
# "causality" eventually comes from the attn_mask computed during forward
# TODO: Make sure that the above is correct
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

        # TODO: Take care of this for generalized attention patterns!!!
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

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
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
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
        
        if isinstance(self.config.attention_scheme, str):
            self.config.attention_scheme = [self.config.attention_scheme]
        if len(self.config.attention_scheme) == 1:
            self.config.attention_scheme = self.config.n_layer * self.config.attention_scheme

        self.build_model(config)
        initialize_weights(self, config=config)
        
        self.max_seq_len = 128
        
        self.PAD_TOKEN_ID = 0
        self.PAD_DOMAIN_ID = 0 
        self.PAD_AGE = -10000


    def build_model(self, config: DelphiConfig):

        self.transformer = nn.ModuleDict(dict(
            embed=DelphiEmbedding(config),
            age_embedding=AgeEncoding(n_embd=config.n_embd),
            drop=nn.Dropout(config.dropout),
            attn_mask_builder=nn.ModuleList([AttentionMaskBuilder(config.attention_scheme[i]) for i in range(config.n_layer)]),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias)
        ))

        self.embedding_to_logits = DomainwiseTiedLinear(self.transformer.embed)

        self.ce_head = CrossEntropyHead()
        
        self.dt_head = CompetingExpHead(
            n_input=config.n_embd, 
            zero_inflate=config.zero_inflate, 
            pi_head=config.zero_inflate_projector
        )


    def get_allowed_tokens_mask(self, targets, ignore_tokens):

        targets = targets.reshape(-1)
        pass_tokens = targets != -1 
        for k in ignore_tokens: # and gender
            pass_tokens *= targets != k

        return pass_tokens


    def set_max_seq_len(self, max_seq_len):
        self.max_seq_len = max_seq_len


    def set_valid_loss_mode(self, validation_loss_mode):
        self.validation_loss_mode = validation_loss_mode
   

    # def build_attention_mask(self, idx, age, targets, targets_age, mask_ties):
    # 
    # 
    #     # causal self-attention mask, to ensure that attention is only applied to the left in the input sequence
    #     dd = dict(device=idx.device)
    #     # Do not attend to padded positions
    #     attn_mask = (idx>0).view(idx.size(0), 1, 1, idx.size(1)) * (idx>0).view(idx.size(0), 1, idx.size(1), 1)  
    #     
    #     attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), **dd))[None,None,:,:] > 0
    #     
    #     # if targets is not None and self.config.mask_ties:
    #     if targets is not None and mask_ties:
    #         # Mask co-occuring tokens
    #         attn_mask *= ((age.view(idx.size(0),1,1,idx.size(1)) != targets_age.view(idx.size(0),1,idx.size(1),1))) 
    #         attn_mask += (attn_mask.sum(-1, keepdim=True)==0) * torch.diag(torch.ones(idx.size(1), **dd)) > 0
    #     
    #     # Except for padding
    #     attn_mask = attn_mask + (idx==0).view(idx.size(0), 1, 1, idx.size(1)) * torch.diag(torch.ones(idx.size(1), **dd)) > 0 
    #     attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), **dd))[None,None,:,:] > 0
    #     
    #     return attn_mask
    

    def cross_entropy_loss(self, logits, targets, agg=None): #, pass_tokens, agg=None):
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
            return loss_ce_agg_per_disease / len(logits) # pass_tokens.sum().item()
        elif agg is None:
            loss_ce = F.cross_entropy(
                logits.reshape(-1, n_classes), 
                targets.reshape(-1), 
                ignore_index=-1
            )
        else:
            raise ValueError("agg should be in [None, 'per_disease']")

        return loss_ce


    def time_to_event_loss(self, logits, time_to_next, t_min, agg=None): # , pass_tokens, attn_mask, mask_ties):       
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
    
        # dd = dict(device=logits.device, dtype=torch.float32)
        # block_size = attn_mask.size(-1)

        # if mask_ties:
            # Use time from last untied token
            # dt = torch.gather(
                # dt, -1, (attn_mask * torch.arange(0, block_size, **dd).view(1, 1, 1, -1)).max(-1).indices.squeeze((1, 2))
            # )  

        # log_dt = - torch.log(dt + t_min).view(-1)        

        ## Exponential log-likelihood (real statistics, TM)
        # loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - log_dt.reshape(-1))) 

        

    
    def blackout_ignored(self, logits, ignore_tokens):

        if self.validation_loss_mode:
            ignore_tokens += [1]
            logits[..., ignore_tokens] = -torch.inf

        return logits


    def to_tensor(self, x, ages, embeddings, subject_ids):

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
        
        tokens_flat   = torch.cat(all_tokens)
        ages_flat     = torch.cat(all_ages)
        embeddings_flat = torch.cat(all_embeddings)
        domains_flat  = torch.cat(all_domains)
        subjects_flat = torch.cat(all_subjects)
    
        unique_subjects = subjects_flat.unique(sorted=True)
        batch_tokens, batch_embeddings, batch_ages, batch_domains = [], [], [], []
        # 2) Process subject by subject
        
        for subj in unique_subjects:
            mask = subjects_flat == subj
            a_subj = ages_flat[mask]
            a_subj = a_subj[order:=torch.argsort(a_subj)]
            t_subj = tokens_flat[mask][order]
            e_subj = embeddings_flat[mask][order]
            d_subj = domains_flat[mask][order]
            
            # sort by age
            # t_subj = t_subj[order]
            # a_subj = a_subj[order]
            # e_subj = e_subj[order]
            # d_subj = d_subj[order]
    
            batch_tokens.append(t_subj.unsqueeze(0))
            batch_ages.append(a_subj.unsqueeze(0))
            batch_embeddings.append(e_subj.unsqueeze(0))
            batch_domains.append(d_subj.unsqueeze(0))
    
        batch_tokens = torch.stack(batch_tokens)
        batch_embeddings = torch.stack(batch_embeddings)
        batch_ages   = torch.stack(batch_ages)
        batch_domains = torch.stack(batch_domains)
    
        return batch_tokens, batch_ages, batch_embeddings, unique_subjects, batch_domains


    def forward(self, x: torch.Tensor, ages: torch.Tensor, subject_ids: torch.Tensor,
        validation_loss_mode: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]], torch.Tensor]:

        '''
        # self.set_valid_loss_mode(validation_loss_mode)
        
        max_ages = self.get_max_ages_per_subject(age, subject_ids)
        
        # get tokens with additional no-event tokens interleaved for the domains that need it (typically 'diseases')        
        x, age, subject_ids = self.insert_no_event_tokens(x, age, subject_ids)
        x, age, subject_ids = self.mask_tokens_after_age (x, age, subject_ids, max_ages)
                
        x = self.transformer.embed(x=x) 
        
        for domain in age:
            age_emb = self.transformer.age_embedding(age[domain])
            x[domain] += age_emb

        # mask tokens in a domain-wise manner                
        # x = self.transformer.drop(x)

        x, age, domains = self.build_seq_for_transformer(x, age, subject_ids)
        # input, target = x[:, :-1], x[:, 1:]
        # input_age, target_age = age[:, :-1], age[:, 1:]

        # attn_masks = self.build_attention_mask(x[:-1], input_age, x[1:], target_age, domains, True)#self.attention_scheme)
        # attn_masks = self.build_attention_mask(x[:,:-1], input_age, x[:,1:], target_age, True) #self.attention_scheme)

        # attn_mask = causal_attention_mask(
        #     pad=self.is_not_padding(idx), 
        #     t1=target_age, t0=input_age, 
        #     mask_ties=self.config.mask_ties
        # )                
        '''
        
        domain2id = { k: i for i, k in enumerate(x) }
        
        emb = self.transformer.embed(x)
        x, ages, emb, subject_ids, domains = self.to_tensor(x, ages, emb, subject_ids)
        
        
        # TODO: remove the need for these squeeze's
        ages    = ages.int().squeeze(1)
        x       = x.squeeze(1)
        emb     = emb.squeeze(1)
        domains = domains.squeeze(1)
        
        self._trace = dict(domains=domains.detach(), tokens=x.detach(), emb=emb.detach(), ages=ages.detach())
        
        # for domain in ages:
        emb += self.transformer.age_embedding(ages)

        #TODO: passing domain2id on every call doesn't seem right. Try to pass it in the constructor if possible.
        attn_mask = torch.stack([ 
            self.transformer.attn_mask_builder[i](ages, domains, domain2id) 
            for i, _ in enumerate(self.transformer.h) ]
        ) 
        attn_mask = attn_mask.permute(1, 0, 2, 3) # (N_LAYERS, BATCH_SIZE, ..., ...) -> (BATCH_SIZE, N_LAYERS, ..., ...)

        h, att = emb[:, :-1], []
        for i, transformer_block in enumerate(self.transformer.h):            
            h, _att = transformer_block(h, attn_mask)
            att.append(_att)
        att = torch.stack(att)

        h = self.transformer.ln_f(h)        
        logits = self.embedding_to_logits(h)
        
        return logits, att

        # self.lm_head(
            # h[:, :, :]
        # )   # note: using list [-1] to preserve the time dim
    

    def compute_loss(self, logits, targets, targets_age):

        loss = {
           "loss_ce": self.cross_entropy_loss(targets),
           "loss_dt": self.time_to_event_loss(targets, targets_age), 
           "loss": loss_ce * self.config.ce_beta + loss_dt * self.config.dt_beta
           # or "ce", "dt", "total"
        }

        return loss


    def get_max_ages_per_subject(self, ages, subject_ids):

        max_ages= {}
        for subj in np.unique(subject_ids.diseases.cpu().numpy()):
            subj = int(subj)
            death_age = ages.death[subject_ids.death == subj]
            if len(death_age):
                max_ages[subj] = death_age.item()
                continue
            max_ages[subj] = ages.diseases[subject_ids.diseases == subj][-1].item()
        return max_ages


    def mask_tokens_after_age(self, tokens, ages, subject_ids, max_ages):
 
        for dname in tokens:
            for subj, max_age in max_ages.items():
                tokens[dname] = tokens[dname].masked_fill(
                    (subject_ids[dname] == subj) & (ages[dname] > max_ages[subj]),
                    self.transformer.embed['padding'].PADDING_TOKEN
                ) 

                ages[dname] = ages[dname].masked_fill(
                    (subject_ids[dname] == subj) & (ages[dname] > max_ages[subj]),
                    self.transformer.embed['padding'].PADDING_AGE
                ) 

                # ages['padding']   = ages['padding'].masked_fill(ages['padding']   > max_age_in_years * DAYS_PER_YEAR, PADDING_AGE)           # MASKING_AGE

        return tokens, ages, subject_ids


    def insert_no_event_tokens(self, tokens, ages, subject_ids, no_event_token_rate=5, padding="regular", gen=None):

        """Insert synthetic 'no event' tokens at regular or random intervals."""

        NO_EVENT_TOKEN = self.transformer.embed['padding'].NO_EVENT_TOKEN

        device = tokens['diseases'].device

        if padding == "random" and gen is None:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(tokens.sum().item())
    
        unique_subject_ids = torch.from_numpy(np.unique(subject_ids.diseases.cpu().numpy()))

        no_event_per_subject = {'tokens': [], 'ages':[], 'subject_ids': []}
        for subj_id in unique_subject_ids:
            if padding in [None, "none"] or no_event_token_rate in [0, None]:
                pad = torch.ones(tokens.shape[0], 0)
            elif padding == "regular":
                pad = torch.arange(0, 100 * DAYS_PER_YEAR, DAYS_PER_YEAR * no_event_token_rate) * torch.ones(1) + 1
            elif padding == "random":
                pad = torch.randint(1, 100 * DAYS_PER_YEAR, (tokens.shape[0], int(100 / no_event_token_rate)), generator=gen)
            else:
                raise NotImplementedError(f"Unknown padding {padding}")
    
            no_event_per_subject['tokens'].append(NO_EVENT_TOKEN * torch.ones_like(pad, dtype=torch.int))
            no_event_per_subject['ages'].append(pad)
            no_event_per_subject['subject_ids'].append(torch.tensor([subj_id] * len(pad)))
            
        tokens['padding']      = torch.stack(no_event_per_subject['tokens']).reshape(-1).to(device)
        ages['padding']        = torch.stack(no_event_per_subject['ages']).reshape(-1).to(device)
        subject_ids['padding'] = torch.stack(no_event_per_subject['subject_ids']).reshape(-1).to(device)
            
        return tokens, ages, subject_ids
    

    def get_embedding_at_age(self, x, ages, readout_age):
        raise NotImplementedError


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
