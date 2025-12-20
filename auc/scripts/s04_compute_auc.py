#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import mannwhitneyu
from tqdm import tqdm


# ================================================================
#                      AUC HELPERS
# ================================================================

def optimized_bootstrapped_auc_gpu(case, control, n_bootstrap=200):
    """Fast AUC bootstrap using CUDA."""
    if not torch.cuda.is_available():
        return None

    case = torch.tensor(case, dtype=torch.float32, device="cuda")
    control = torch.tensor(control, dtype=torch.float32, device="cuda")

    n_case = case.numel()
    n_ctrl = control.numel()
    total = n_case + n_ctrl

    boot_idx_case = torch.randint(0, n_case, (n_bootstrap, n_case), device="cuda")
    boot_idx_ctrl = torch.randint(0, n_ctrl, (n_bootstrap, n_ctrl), device="cuda")

    bc = case[boot_idx_case]
    bn = control[boot_idx_ctrl]

    combined = torch.cat([bc, bn], dim=1)

    mask = torch.zeros((n_bootstrap, total), device="cuda", dtype=torch.bool)
    mask[:, :n_case] = True

    ranks = combined.argsort(dim=1).argsort(dim=1)
    rank_sum = (ranks.float() * mask.float()).sum(dim=1)

    min_sum = n_case * (n_case - 1) / 2
    U = rank_sum - min_sum

    auc = U / (n_case * n_ctrl)
    return auc.cpu().numpy().tolist()


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)

    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j

    T2 = np.empty(N, dtype=np.float32)
    T2[J] = T + 1
    return T2


def fastDeLong(pred_sorted, m):
    """One-classifier DeLong."""
    n = pred_sorted.shape[1] - m
    pos = pred_sorted[:, :m]
    neg = pred_sorted[:, m:]

    tx = np.vstack([compute_midrank(pos[r]) for r in range(1)])
    ty = np.vstack([compute_midrank(neg[r]) for r in range(1)])
    tz = np.vstack([compute_midrank(pred_sorted[r]) for r in range(1)])

    auc = tz[:, :m].sum() / m / n - (m + 1) / (2 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return auc, cov


def delong_auc(case, ctrl):
    """Compute AUC + variance using DeLong."""
    if len(case) == 0 or len(ctrl) == 0:
        return None, None

    labels = np.array([1] * len(case) + [0] * len(ctrl))
    scores = np.concatenate([case, ctrl])

    order = (-labels).argsort()
    m = labels.sum()

    preds_sorted = scores[np.newaxis, order]
    auc, cov = fastDeLong(preds_sorted, m)

    return auc, cov # [0][0]


# ================================================================
#       MASTER FUNCTION FOR ALL STATS
# ================================================================

def compute_all_stats(case, ctrl, do_bootstrap=False, n_bootstrap=200):
    case = np.asarray(case, float)
    ctrl = np.asarray(ctrl, float)

    if len(case) == 0 or len(ctrl) == 0:
        return {
            "auc_delong": None,
            "auc_delong_var": None,
            "mann_u": None,
            "mann_p": None,
            "auc_bootstrap_mean": None,
            "auc_bootstrap_std": None,
        }

    auc_d, auc_var = delong_auc(case, ctrl)

    u, p = mannwhitneyu(case, ctrl, alternative="two-sided")

    if do_bootstrap and torch.cuda.is_available():
        boots = optimized_bootstrapped_auc_gpu(case, ctrl, n_bootstrap)
        auc_b_mean = float(np.mean(boots))
        auc_b_std = float(np.std(boots))
    else:
        auc_b_mean = None
        auc_b_std = None

    return {
        "auc_delong": auc_d,
        "auc_delong_var": auc_var,
        "mann_u": float(u),
        "mann_p": float(p),
        "auc_bootstrap_mean": auc_b_mean,
        "auc_bootstrap_std": auc_b_std,
    }


# ================================================================
#                       MAIN SCRIPT
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", required=True)
    parser.add_argument("--logits_parquet_dir", default="/hps/nobackup/birney/users/bonazzola/auc/logits_merged")
    parser.add_argument("--output_dir", default="auc_results")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n_bootstrap", type=int, default=200)
    args = parser.parse_args()

    runid = args.runid
    logits_path = Path(args.logits_parquet_dir) / f"logits_{runid}.parquet"

    print(f"[INFO] Loading: {logits_path}")
    df = pd.read_parquet(logits_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    results = []

    print("[INFO] Computing AUC / Mann–Whitney ...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):

        case = row["case_logits"]
        ctrl = row["ctrl_logits"]

        stats = compute_all_stats(
            case,
            ctrl,
            do_bootstrap=args.bootstrap,
            n_bootstrap=args.n_bootstrap,
        )

        results.append({
            "runid": runid,
            "token_id": row["token_id"],
            "domain": row["domain"],
            "sex": row["sex"],
            "age_start": row["age_start"],
            "age_end": row["age_end"],
            "n_case": row["n_case"],
            "n_ctrl": row["n_ctrl"],
            **stats
        })

    out_df = pd.DataFrame(results)
    out_path = out_dir / f"auc_{runid}.csv"
    out_df.to_csv(out_path, index=False)

    print(f"[OK] Saved AUC results → {out_path}")


if __name__ == "__main__":
    main()
