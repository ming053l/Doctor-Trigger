#!/usr/bin/env python3
"""Fit EVT calibration diagnostics from completed record_model_release score dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def threshold_metrics(y: np.ndarray, s: np.ndarray, target: float) -> Dict[str, float]:
    neg = s[y == 0]
    pos = s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return {"threshold": float("nan"), "empirical_fpr": float("nan"), "empirical_tpr": float("nan")}
    threshold = float(np.quantile(neg, max(0.0, min(1.0, 1.0 - target)), method="higher"))
    return {
        "threshold": threshold,
        "empirical_fpr": float(np.mean(neg >= threshold)),
        "empirical_tpr": float(np.mean(pos >= threshold)),
    }


def fit_gpd_tail(neg: np.ndarray, tail_q: float) -> Dict[str, float]:
    neg = np.asarray(neg, dtype=float)
    neg = neg[np.isfinite(neg)]
    if len(neg) < 20:
        return {"u": float("nan"), "xi": float("nan"), "sigma": float("nan"), "ks": float("nan"), "tail_n": int(len(neg))}
    u = float(np.quantile(neg, tail_q))
    excess = neg[neg > u] - u
    if len(excess) < 10 or np.allclose(excess, 0):
        return {"u": u, "xi": float("nan"), "sigma": float("nan"), "ks": float("nan"), "tail_n": int(len(excess))}
    try:
        from scipy import stats

        xi, loc, sigma = stats.genpareto.fit(excess, floc=0)
        ks = float(stats.kstest(excess, "genpareto", args=(xi, 0, sigma)).statistic)
        return {"u": u, "xi": float(xi), "sigma": float(sigma), "ks": ks, "tail_n": int(len(excess))}
    except Exception:
        return {"u": u, "xi": float("nan"), "sigma": float("nan"), "ks": float("nan"), "tail_n": int(len(excess))}


def collect_score_files(scores_dir: Path) -> List[Path]:
    return sorted(scores_dir.glob("record_model_*_scores.csv"))


def parse_score_path(path: Path) -> Dict[str, str]:
    stem = path.stem
    suffix = "_scores"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    parts = stem.split("_")
    seed_idx = next((i for i, part in enumerate(parts) if part.startswith("seed")), -1)
    if seed_idx < 1:
        return {"run_name": "unknown", "variant": "unknown", "seed": "", "eval_name": "unknown"}
    variant = "_".join(parts[seed_idx - 3 : seed_idx]) if seed_idx >= 3 and parts[seed_idx - 3].startswith("V") else parts[seed_idx - 1]
    run_name = "_".join(parts[: seed_idx - len(variant.split("_"))])
    eval_name = "_".join(parts[seed_idx + 1 :])
    return {"run_name": run_name, "variant": variant, "seed": parts[seed_idx].replace("seed", ""), "eval_name": eval_name}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, default=Path("results/record_model_release/scores"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/record_model_release/evt"))
    parser.add_argument("--tail-q", type=float, default=0.95)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    targets = {"S_fake": "y_fake", "S_rel": "y_rel", "S_key": "y_keycorr"}
    for path in collect_score_files(args.scores_dir):
        meta = parse_score_path(path)
        df = pd.read_csv(path)
        for score, target in targets.items():
            if score not in df.columns or target not in df.columns:
                continue
            y = pd.to_numeric(df[target], errors="coerce").fillna(0).to_numpy(dtype=int)
            s = pd.to_numeric(df[score], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(s)
            y = y[valid]
            s = s[valid]
            neg = s[y == 0]
            pos = s[y == 1]
            gpd = fit_gpd_tail(neg, args.tail_q)
            base = {
                **meta,
                "score": score,
                "target": target,
                "n": int(len(s)),
                "n_pos": int(len(pos)),
                "n_neg": int(len(neg)),
                "auc": safe_auc(y, s),
                "tail_q": args.tail_q,
                **gpd,
                "source_score_path": str(path.resolve()),
            }
            for fpr in [1e-2, 1e-3, 1e-4]:
                m = threshold_metrics(y, s, fpr)
                rows.append({**base, "target_fpr": fpr, **m})
    out_csv = args.out_dir / "record_model_release_evt_from_scores.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "output_csv": str(out_csv.resolve()),
        "rows": len(rows),
        "note": "Empirical 1e-4 remains unresolved unless n_neg >= 100000 or block-adjusted effective n supports it.",
    }
    (args.out_dir / "record_model_release_evt_from_scores_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
