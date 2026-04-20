import sys
from pathlib import Path

import pandas as pd
import yaml

DELPHI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DELPHI_DIR))

from delphi.model import DomainConfig


def load_domain_config(cfg_path, tokens_path):
    raw = yaml.safe_load(Path(cfg_path).read_text())
    cfg = {}
    for domain, params in raw.items():
        if domain == "padding":
            continue
        p = dict(params)
        if "path" in p:
            p["path"] = tokens_path / p["path"]
        cfg[domain] = DomainConfig(**p)
    cfg["padding"] = DomainConfig(projector="embed")
    return cfg


def read_ids(path, type=int):
    """
    Read UK Biobank IDs from a CSV file.
    Handles files with or without header; uses first column only.
    """
    s = pd.read_csv(path, dtype=str, comment="#").iloc[:, 0]
    return set(
        s.str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .dropna()
         .astype(type)
         .tolist()
    )


def get_top_counts(data, labels, top_n=200, ignored_tokens=[]):
    id_to_token = dict(zip(labels.index - 1, labels.name))
    counts = (
        pd.DataFrame(data, columns=["subject_id", "age", "token_id"])
        .query("token_id not in @ignored_tokens")
        .assign(token=lambda df: df.token_id.map(id_to_token))
        .token.value_counts(ascending=False)
        .head(top_n)
        .sort_values()
    )
    return counts


def get_domain_configs_from_string(s: str) -> dict:
    """Alias for parse_domains_param kept for backward compatibility."""
    from utils.mlflow_utils import parse_domains_param
    return parse_domains_param(s)
