import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# from delphi.model.config import EmbedConfig, DelphiConfig

from delphi.multimodal import Modality, module_name
from dataclasses import dataclass, field
from typing import Optional
import yaml

import logging
logger = logging.getLogger(__name__)


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

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        time_years = x / self.norm_factor
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(time_years * self.div_term)  # * (1-self.div_term)
        y[..., 1::2] = torch.cos(time_years * self.div_term)  # * (1-self.div_term)
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
            self.config.input_size = len(yaml.load(open(config.path + "/tokenizer.yaml"), Loader=yaml.FullLoader))            

        elif config.projector.lower() == "pretrained":
            weights = torch.load(config.pretrained_path)  # Tensor [vocab_size, d_ext]
            self.config.input_size, d_ext = weights.shape

        if config.projector.lower() == "linear":
            self.projector = nn.Linear(config.input_size, n_embed, bias=False)

        elif config.projector.lower() == "mlp":
            
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
            
            self.projector = nn.Sequential(*layers)

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

         
        self.age_encoding = AgeEncoding(n_embd=config.n_embd)
        self.token_drop = nn.Dropout(config.token_dropout)        
        
        print(config.domains)
        
        if len(config.domains) > 0:
            self.domain_embed = nn.ModuleDict()
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


    def forward(self, x: dict[str, torch.Tensor], t: dict[str, torch.Tensor]) -> torch.Tensor: # , M: torch.Tensor, biomarker_x: dict[Modality, torch.Tensor] = {},) -> torch.Tensor:

        import ipdb; ipdb.set_trace()

        if len(self.domain_embed) > 0:
            for domain_name in self.domain_embed:
                token_emb = self.domain_embed[domain_name](x[domain_name])
                age_emb   = self.age_encoding(t[domain_name].unsqueeze(-1))
                x[domain_name] = token_emb + age_emb
        else:
            print("KKKKKKKKKKKKKKKKKK")
            token_emb = self.token_embedding(x)
            token_emb = self.token_drop(token_emb) * (1 - self.config.token_dropout)
            age_emb = self.age_encoding(t.unsqueeze(-1))
            x = token_emb + age_emb
        
        return x            
            
        # token_emb = self.token_embedding(x)
        # token_emb = self.token_drop(token_emb) * (1 - self.config.token_dropout)
        # age_emb = self.age_encoding(t.unsqueeze(-1))
        # x = token_emb + age_emb

        # for modality in biomarker_x.keys():
        #     m_pos = torch.nonzero(M == modality.value)  # N * 2
        #     if m_pos.size == 0: continue
        #     # m_emb = self.domain_embed[module_name(modality)](biomarker_x[modality])  # N * H
        #     assert m_emb.shape[0] == m_pos.shape[0]
        #     token_emb[m_pos[:, 0], m_pos[:, 1], :] *= 0
        #     token_emb[m_pos[:, 0], m_pos[:, 1], :] += m_emb

        # if self.config.modality_emb:
            # mod_emb = self.mod_embedding(M)
            # x += mod_emb

        # return x

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
