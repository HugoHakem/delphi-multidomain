import numpy as np
import yaml
from pathlib import Path


def infer_scheme(attn_scheme):
    if not isinstance(attn_scheme, str):
        return "Unknown"
    val = attn_scheme.lower()
    if "hla" in val:
        if "bidirectional" in val:
            return "HLA-bidirectional"
        elif "causal(mask_ties=true)" in val:
            return "HLA-causal"
        else:
            return "HLA-other"
    return "No HLA"


def exponential_moving_average(values, alpha=1):
    if len(values) == 0:
        return np.array([])

    values = np.asarray(values, dtype=float)
    ema = np.zeros_like(values)
    ema[0] = values[0]

    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

    return ema


def load_labels(path):
    path = Path(path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        return data.get("tokens", [])
    return data
