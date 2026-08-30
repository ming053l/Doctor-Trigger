#!/usr/bin/env python3
"""Score-level fusion ablations from completed record_model_release score dumps.

This is a diagnostic fusion rerun over existing outputs. It does not replace a
feature-level detector/record-model fusion module, but it gives Sum/OR/MLP-style behavior
from the completed per-sample score dumps without rebuilding data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def auc(y: np.ndarray, s: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def eer(y: np.ndarray, s: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y, s)
        fnr = 1.0 - tpr
        idx = int(np.nanargmin(np.abs(fpr - fnr)))
        return float((fpr[idx] + fnr[idx]) / 2.0)
    except Exception:
        return float("nan")


def tpr_at(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
    neg = s[y == 0]
    pos = s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, max(0, min(1, 1.0 - target_fpr)), method="higher")
    return float(np.mean(pos >= thr))


def cv_logistic_score(x: np.ndarray, y: np.ndarray, seed: int, interaction: bool = False) -> np.ndarray:
    if interaction:
        x = np.column_stack([x, x[:, 0] * x[:, 1], np.abs(x[:, 0] - x[:, 1])])
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        out = np.zeros(len(y), dtype=float)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(x, y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_test = scaler.transform(x[test_idx])
            clf = LogisticRegression(max_iter=500, random_state=seed)
            clf.fit(x_train, y[train_idx])
            out[test_idx] = clf.predict_proba(x_test)[:, 1]
        return out
    except Exception:
        return np.nanmean(x, axis=1)


def parse_name(path: Path) -> Dict[str, str]:
    stem = path.stem.removesuffix("_scores")
    parts = stem.split("_")
    seed_idx = next((i for i, part in enumerate(parts) if part.startswith("seed")), -1)
    if seed_idx < 0:
        return {"run_name": "unknown", "variant": "unknown", "seed": "", "eval_name": "unknown"}
    variant_start = next((i for i in range(seed_idx - 1, -1, -1) if parts[i].startswith("V")), seed_idx - 1)
    return {
        "run_name": "_".join(parts[:variant_start]),
        "variant": "_".join(parts[variant_start:seed_idx]),
        "seed": parts[seed_idx].replace("seed", ""),
        "eval_name": "_".join(parts[seed_idx + 1 :]),
    }


def metric_row(meta: Dict[str, str], method: str, y: np.ndarray, s: np.ndarray, source: Path) -> Dict[str, object]:
    return {
        **meta,
        "fusion_method": method,
        "target": "y_fake",
        "auc": auc(y, s),
        "eer": eer(y, s),
        "tpr@1e-2": tpr_at(y, s, 1e-2),
        "tpr@1e-3": tpr_at(y, s, 1e-3),
        "tpr@1e-4": tpr_at(y, s, 1e-4),
        "n": int(len(y)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "source_score_path": str(source.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, default=Path("results/record_model_release/scores"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/record_model_release/fusion"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for path in sorted(args.scores_dir.glob("record_model_*_scores.csv")):
        df = pd.read_csv(path)
        if not {"S_fake", "S_rel", "y_fake"}.issubset(df.columns):
            continue
        meta = parse_name(path)
        y = pd.to_numeric(df["y_fake"], errors="coerce").fillna(0).to_numpy(dtype=int)
        detector_proxy = pd.to_numeric(df["S_fake"], errors="coerce").to_numpy(dtype=float)
        record_model_drop = 1.0 - pd.to_numeric(df["S_rel"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(detector_proxy) & np.isfinite(record_model_drop)
        y = y[valid]
        x = np.column_stack([detector_proxy[valid], record_model_drop[valid]])
        methods = {
            "DFH_proxy_S_fake": x[:, 0],
            "RecordModel_relation_drop": x[:, 1],
            "Sum_mean": np.mean(x, axis=1),
            "OR_max": np.max(x, axis=1),
            "MLP_3fold_score_level": cv_logistic_score(x, y, int(meta.get("seed") or 0), interaction=False),
            "CrossInteraction_3fold_score_level": cv_logistic_score(x, y, int(meta.get("seed") or 0), interaction=True),
        }
        for method, score in methods.items():
            rows.append(metric_row(meta, method, y, score, path))
    out = pd.DataFrame(rows)
    out_csv = args.out_dir / "record_model_release_score_level_fusion_ablation.csv"
    out.to_csv(out_csv, index=False)
    summary = (
        out.groupby(["run_name", "eval_name", "variant", "fusion_method"], as_index=False)[["auc", "eer", "tpr@1e-4"]]
        .mean()
        .sort_values(["run_name", "eval_name", "variant", "fusion_method"])
    )
    summary_csv = args.out_dir / "record_model_release_score_level_fusion_ablation_summary.csv"
    summary.to_csv(summary_csv, index=False)
    tex = ["% Rev9 score-level fusion diagnostic; not a replacement for feature-level fusion training."]
    for _, row in summary.iterrows():
        tex.append(
            f"{row['run_name']} & {row['eval_name']} & {row['variant']} & {row['fusion_method']} & "
            f"{row['auc']:.6f} & {row['eer']:.6f} & {row['tpr@1e-4']:.6f} \\\\"
        )
    (args.out_dir / "record_model_release_score_level_fusion_ablation_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(out), "output_csv": str(out_csv.resolve()), "summary_csv": str(summary_csv.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
