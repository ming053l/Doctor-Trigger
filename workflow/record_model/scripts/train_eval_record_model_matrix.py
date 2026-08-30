#!/usr/bin/env python3
"""Train record_model_release RecordModel on one feature CSV and evaluate on multiple feature CSVs."""

from __future__ import annotations

import argparse
import json
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


def config_path(config_file: Path, value: object) -> Path:
    """Resolve a configured path relative to the YAML file, not a user home."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_file.resolve().parent / path).resolve()


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-6] = 1.0
    x = (np.nan_to_num(x, nan=mean) - mean) / std
    return x.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def fit_model(train_df: pd.DataFrame, variant: str, seed: int, steps: int, hidden_dim: int, lr: float, batch_size: int) -> Dict[str, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cols = feature_columns_for_variant(variant, list(train_df.columns))
    x_raw = train_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x, mean, std = standardize(x_raw)
    x_t = torch.tensor(x)
    targets = {
        "y_fake": torch.tensor(train_df["y_fake"].to_numpy(dtype=np.float32)),
        "y_rel": torch.tensor(train_df["y_rel"].to_numpy(dtype=np.float32)),
        "y_keycorr": torch.tensor(train_df["y_keycorr"].to_numpy(dtype=np.float32)),
    }
    feat_t = {}
    for name in ["G_s", "G_wrong_max"]:
        if name in train_df.columns:
            feat_t[name] = torch.tensor(pd.to_numeric(train_df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    config = RecordModelConfig(variant=variant, input_dim=x.shape[1], hidden_dim=hidden_dim)
    model = RecordModelMLP(config)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    final_loss = float("nan")
    for _ in range(int(steps)):
        if batch_size > 0 and batch_size < len(train_df):
            batch_idx = torch.tensor(rng.integers(0, len(train_df), size=batch_size), dtype=torch.long)
            x_batch = x_t[batch_idx]
            target_batch = {name: value[batch_idx] for name, value in targets.items()}
            feat_batch = {name: value[batch_idx] for name, value in feat_t.items()}
        else:
            x_batch = x_t
            target_batch = targets
            feat_batch = feat_t
        opt.zero_grad(set_to_none=True)
        out = model(x_batch)
        loss = multitask_loss(out, target_batch, config, feat_batch)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu().item())
    return {"model": model, "columns": cols, "mean": mean, "std": std, "final_loss": final_loss}


def predict(model: RecordModelMLP, df: pd.DataFrame, cols: List[str], mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    x_raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x = (np.nan_to_num(x_raw, nan=mean) - mean) / std
    with torch.no_grad():
        out = model(torch.tensor(x.astype(np.float32)))
    scores = df[["sample_id", "dataset", "protocol", "y_fake", "y_rel", "y_keycorr"]].copy()
    for key in ["S_fake", "S_rel", "S_key"]:
        scores[key] = torch.sigmoid(out[key]).numpy()
    return scores


def metric_rows(scores: pd.DataFrame, variant: str, seed: int, eval_name: str, final_loss: float) -> List[Dict[str, object]]:
    targets = {"S_fake": "y_fake", "S_rel": "y_rel", "S_key": "y_keycorr"}
    rows: List[Dict[str, object]] = []
    for score, target in targets.items():
        y = scores[target].to_numpy(dtype=int)
        s = scores[score].to_numpy(dtype=float)
        rows.append(
            {
                "eval_name": eval_name,
                "variant": variant,
                "seed": seed,
                "score": score,
                "target": target,
                "auc": safe_auc(y, s),
                "eer": safe_eer(y, s),
                "tpr@1e-2": tpr_at_fpr(y, s, 1e-2),
                "tpr@1e-3": tpr_at_fpr(y, s, 1e-3),
                "tpr@1e-4": tpr_at_fpr(y, s, 1e-4),
                "n": int(len(y)),
                "n_pos": int(np.sum(y == 1)),
                "n_neg": int(np.sum(y == 0)),
                "final_train_loss": final_loss,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_file = args.config.resolve()
    cfg = load_config(config_file)
    out_dir = config_path(config_file, cfg["output_dir"])
    run_name = str(cfg.get("run_name", "matrix"))
    matrix_dir = out_dir / "matrix"
    ckpt_dir = out_dir / "checkpoints"
    score_dir = out_dir / "scores"
    metric_dir = out_dir / "metrics"
    for d in [matrix_dir, ckpt_dir, score_dir, metric_dir]:
        d.mkdir(parents=True, exist_ok=True)
    train_csv = config_path(config_file, cfg["train_features_csv"])
    train_df = pd.read_csv(train_csv)
    eval_items = list(cfg["eval_feature_csvs"])
    variants = list(cfg.get("variants", ["trigger_only", "host_plus_trigger", "full_record_model"]))
    seeds = list(cfg.get("seeds", [0, 1, 2]))
    steps = int(cfg.get("train", {}).get("steps", 1000))
    hidden_dim = int(cfg.get("train", {}).get("hidden_dim", 64))
    lr = float(cfg.get("train", {}).get("lr", 0.001))
    batch_size = int(cfg.get("train", {}).get("batch_size", 4096))
    all_metrics: List[Dict[str, object]] = []
    manifest: List[Dict[str, object]] = []
    for seed in seeds:
        for variant in variants:
            print(f"[record_model_release_train_eval_matrix] train variant={variant} seed={seed} steps={steps} batch_size={batch_size}", flush=True)
            fit = fit_model(train_df, variant, int(seed), steps, hidden_dim, lr, batch_size)
            ckpt_path = ckpt_dir / f"{variant}_seed{seed}_{run_name}.pt"
            torch.save(
                {
                    "state_dict": fit["model"].state_dict(),
                    "variant": variant,
                    "seed": int(seed),
                    "feature_columns": fit["columns"],
                    "mean": fit["mean"],
                    "std": fit["std"],
                    "train_features_csv": str(train_csv),
                },
                ckpt_path,
            )
            for item in eval_items:
                eval_name = str(item["name"])
                eval_csv = config_path(config_file, item["path"])
                eval_df = pd.read_csv(eval_csv)
                print(f"[record_model_release_train_eval_matrix] eval variant={variant} seed={seed} eval={eval_name} n={len(eval_df)}", flush=True)
                scores = predict(fit["model"], eval_df, fit["columns"], fit["mean"], fit["std"])
                score_path = score_dir / f"{run_name}_{variant}_seed{seed}_{eval_name}_scores.csv"
                scores.to_csv(score_path, index=False)
                all_metrics.extend(metric_rows(scores, variant, int(seed), eval_name, float(fit["final_loss"])))
                manifest.append(
                    {
                        "run_name": run_name,
                        "variant": variant,
                        "seed": int(seed),
                        "eval_name": eval_name,
                        "checkpoint_path": str(ckpt_path.resolve()),
                        "score_path": str(score_path.resolve()),
                        "eval_features_csv": str(eval_csv.resolve()),
                    }
                )
    metrics = pd.DataFrame(all_metrics)
    metrics_path = metric_dir / f"record_model_release_{run_name}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    (matrix_dir / f"{run_name}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = metrics.groupby(["eval_name", "variant", "score"])[["auc", "eer", "tpr@1e-4"]].mean().reset_index()
    summary.to_csv(matrix_dir / f"{run_name}_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
