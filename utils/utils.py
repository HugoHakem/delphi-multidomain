import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import pandas as pd
import yaml

DELPHI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DELPHI_DIR))

from delphi.model import DomainConfig

_DOMAIN_CONFIG_FIELDS = {f.name for f in dc_fields(DomainConfig)}


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


def apply_domain_overrides(domain_cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides to a loaded domain config dict.

    Each override must be a string of the form ``domain.field=value``.
    Values are parsed with ``yaml.safe_load`` so Python types are inferred
    correctly: ``True``/``False`` → bool, integers → int, floats → float,
    ``null`` → None, plain strings stay as str.

    Raises ``ValueError`` for unknown domains or unknown DomainConfig fields.
    """
    for override in overrides:
        if "=" not in override or "." not in override.split("=", 1)[0]:
            raise ValueError(
                f"Invalid override {override!r}: expected 'domain.field=value'"
            )
        lhs, value_str = override.split("=", 1)
        domain, field = lhs.split(".", 1)

        if domain not in domain_cfg:
            raise ValueError(
                f"Domain {domain!r} not in config. Available: {sorted(domain_cfg)}"
            )
        if field not in _DOMAIN_CONFIG_FIELDS:
            raise ValueError(
                f"Field {field!r} is not a valid DomainConfig field. "
                f"Valid fields: {sorted(_DOMAIN_CONFIG_FIELDS)}"
            )

        value = yaml.safe_load(value_str)
        setattr(domain_cfg[domain], field, value)

    return domain_cfg


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


