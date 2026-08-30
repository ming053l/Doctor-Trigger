#!/usr/bin/env python3
"""Train/evaluate record_model_release RecordModel variants on extracted feature tables."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.record_model import RecordModelConfig, RecordModelMLP, feature_columns_for_variant, multitask_loss, safe_auc, safe_eer, tpr_at_fpr


def load_config(path: Path) -> Dict[str, object]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-8] = 1.0
    out = (np.nan_to_num(x, nan=mean) - mean) / std
    return out.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def train_variant(df: pd.DataFrame, variant: str, seed: int, steps: int, hidden_dim: int, lr: float) -> Dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cols = feature_columns_for_variant(variant, list(df.columns))
    x_raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x, mean, std = standardize(x_raw)
    targets = {
        "y_fake": torch.tensor(df["y_fake"].to_numpy(dtype=np.float32)),
        "y_rel": torch.tensor(df["y_rel"].to_numpy(dtype=np.float32)),
        "y_keycorr": torch.tensor(df["y_keycorr"].to_numpy(dtype=np.float32)),
    }
    x_t = torch.tensor(x)
    feat_t = {}
    for name in ["G_s", "G_wrong_max"]:
        if name in df.columns:
            feat_t[name] = torch.tensor(pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    config = RecordModelConfig(variant=variant, input_dim=x.shape[1], hidden_dim=hidden_dim)
    model = RecordModelMLP(config)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    final_loss = float("nan")
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        out = model(x_t)
        loss = multitask_loss(out, targets, config, feat_t)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        out = model(x_t)
        scores = {name: torch.sigmoid(val).cpu().numpy() for name, val in out.items() if name.startswith("S_")}
    metrics: List[Dict[str, object]] = []
    metric_targets = {"S_fake": "y_fake", "S_rel": "y_rel", "S_key": "y_keycorr"}
    for score_name, target_name in metric_targets.items():
        labels = df[target_name].to_numpy(dtype=int)
        score = scores[score_name]
        metrics.append(
            {
                "variant": variant,
                "seed": seed,
                "score": score_name,
                "target": target_name,
                "auc": safe_auc(labels, score),
                "eer": safe_eer(labels, score),
                "tpr@1e-2": tpr_at_fpr(labels, score, 1e-2),
                "tpr@1e-3": tpr_at_fpr(labels, score, 1e-3),
                "tpr@1e-4": tpr_at_fpr(labels, score, 1e-4),
                "n": int(len(labels)),
                "n_pos": int(np.sum(labels == 1)),
                "n_neg": int(np.sum(labels == 0)),
                "final_loss": final_loss,
                "feature_columns": ",".join(cols),
            }
        )
    score_df = df[["sample_id", "dataset", "protocol", "y_fake", "y_rel", "y_keycorr"]].copy()
    for name, score in scores.items():
        score_df[f"{variant}_{name}"] = score
    return {
        "variant": variant,
        "seed": seed,
        "model": model,
        "state": {
            "state_dict": model.state_dict(),
            "variant": variant,
            "seed": seed,
            "feature_columns": cols,
            "mean": mean,
            "std": std,
        },
        "metrics": metrics,
        "scores": score_df,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / str(cfg["output_dir"])
    run_name = str(cfg.get("run_name", "smoke"))
    features_csv = out_dir / f"features/record_model_release_{run_name}_features.csv"
    if "features_csv" in cfg:
        features_csv = ROOT / str(cfg["features_csv"])
    checkpoints_dir = out_dir / "checkpoints"
    metrics_dir = out_dir / "metrics"
    scores_dir = out_dir / "scores"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    variants = list(cfg.get("variants", ["host_only", "trigger_only", "host_plus_trigger", "full_record_model"]))
    seeds = list(cfg.get("seeds", [0]))
    steps = int(cfg.get("train", {}).get("steps", 200))
    hidden_dim = int(cfg.get("train", {}).get("hidden_dim", 32))
    lr = float(cfg.get("train", {}).get("lr", 1e-3))
    all_metrics: List[Dict[str, object]] = []
    manifest: List[Dict[str, object]] = []
    merged_scores = None
    for seed in seeds:
        for variant in variants:
            result = train_variant(df, variant, int(seed), steps, hidden_dim, lr)
            ckpt_path = checkpoints_dir / f"{variant}_seed{seed}_{run_name}.pt"
            torch.save(result["state"], ckpt_path)
            score_path = scores_dir / f"{variant}_seed{seed}_{run_name}_scores.csv"
            result["scores"].to_csv(score_path, index=False)
            all_metrics.extend(result["metrics"])
            manifest.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "checkpoint_path": str(ckpt_path.resolve()),
                    "score_dump_path": str(score_path.resolve()),
                    "feature_csv": str(features_csv.resolve()),
                    "n_features": len(result["state"]["feature_columns"]),
                }
            )
            if merged_scores is None:
                merged_scores = result["scores"]
            else:
                score_cols = [c for c in result["scores"].columns if c.startswith(f"{variant}_")]
                merged_scores = merged_scores.merge(result["scores"][["sample_id", *score_cols]], on="sample_id", how="left")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_dir / f"record_model_release_{run_name}_metrics.csv", index=False)
    if merged_scores is not None:
        merged_scores.to_csv(scores_dir / f"record_model_release_{run_name}_scores_wide.csv", index=False)
    (checkpoints_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False))
    print(f"Wrote {metrics_dir / f'record_model_release_{run_name}_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
