#!/usr/bin/env python3
"""Train and evaluate robust RecordModel variants from existing robust feature CSVs."""

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
    for step in range(int(steps)):
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
        if (step + 1) % 250 == 0 or step == 0:
            print(f"[record_model_release_robust_train_eval] variant={variant} seed={seed} step={step+1}/{steps} loss={final_loss:.6f}", flush=True)
    return {"model": model, "columns": cols, "mean": mean, "std": std, "final_loss": final_loss}


def predict(model: RecordModelMLP, df: pd.DataFrame, cols: List[str], mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    x_raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x = (np.nan_to_num(x_raw, nan=mean) - mean) / std
    with torch.no_grad():
        out = model(torch.tensor(x.astype(np.float32)))
    keep = [c for c in ["sample_id", "dataset", "protocol", "perturbation", "y_fake", "y_rel", "y_keycorr", "alpha", "beta"] if c in df.columns]
    scores = df[keep].copy()
    for key in ["S_fake", "S_rel", "S_key"]:
        scores[key] = torch.sigmoid(out[key]).numpy()
    return scores


def metric_rows(scores: pd.DataFrame, variant: str, seed: int, eval_name: str, final_loss: float, train_source: str, score_path: Path, ckpt_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    tasks = [
        ("S_fake", "y_fake", "fake_positive", 1.0),
        ("S_rel", "y_rel", "relation_preserved_positive", 1.0),
        ("S_rel", "y_rel", "relation_disrupted_positive", -1.0),
        ("S_key", "y_keycorr", "key_correct_positive", 1.0),
    ]
    group_cols = ["perturbation"] if "perturbation" in scores.columns else []
    groups = list(scores.groupby(group_cols, dropna=False)) if group_cols else [("ALL", scores)]
    for group_key, group in groups:
        perturb = group_key if isinstance(group_key, str) else (group_key[0] if isinstance(group_key, tuple) else str(group_key))
        for score, target, target_name, orient in tasks:
            if score not in group or target not in group:
                continue
            if target_name == "relation_disrupted_positive":
                y = 1 - group[target].to_numpy(dtype=int)
                s = -group[score].to_numpy(dtype=float)
            else:
                y = group[target].to_numpy(dtype=int)
                s = orient * group[score].to_numpy(dtype=float)
            rows.append(
                {
                    "eval_name": eval_name,
                    "perturbation": perturb,
                    "variant": variant,
                    "seed": seed,
                    "score": score,
                    "target": target_name,
                    "auc": safe_auc(y, s),
                    "eer": safe_eer(y, s),
                    "tpr@1e-2": tpr_at_fpr(y, s, 1e-2),
                    "tpr@1e-3": tpr_at_fpr(y, s, 1e-3),
                    "tpr@1e-4": tpr_at_fpr(y, s, 1e-4),
                    "n": int(len(y)),
                    "n_pos": int(np.sum(y == 1)),
                    "n_neg": int(np.sum(y == 0)),
                    "final_train_loss": final_loss,
                    "train_features_csv": train_source,
                    "score_dump_path": str(score_path.resolve()),
                    "checkpoint_path": str(ckpt_path.resolve()),
                }
            )
    # Overall across perturbations, useful for sanity only.
    for score, target, target_name, orient in tasks:
        if score not in scores or target not in scores:
            continue
        if target_name == "relation_disrupted_positive":
            y = 1 - scores[target].to_numpy(dtype=int)
            s = -scores[score].to_numpy(dtype=float)
        else:
            y = scores[target].to_numpy(dtype=int)
            s = orient * scores[score].to_numpy(dtype=float)
        rows.append(
            {
                "eval_name": eval_name,
                "perturbation": "ALL",
                "variant": variant,
                "seed": seed,
                "score": score,
                "target": target_name,
                "auc": safe_auc(y, s),
                "eer": safe_eer(y, s),
                "tpr@1e-2": tpr_at_fpr(y, s, 1e-2),
                "tpr@1e-3": tpr_at_fpr(y, s, 1e-3),
                "tpr@1e-4": tpr_at_fpr(y, s, 1e-4),
                "n": int(len(y)),
                "n_pos": int(np.sum(y == 1)),
                "n_neg": int(np.sum(y == 0)),
                "final_train_loss": final_loss,
                "train_features_csv": train_source,
                "score_dump_path": str(score_path.resolve()),
                "checkpoint_path": str(ckpt_path.resolve()),
            }
        )
    return rows


def parse_eval_items(values: List[str]) -> List[tuple[str, Path]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"eval item must be name=path, got {value}")
        name, path = value.split("=", 1)
        out.append((name, ROOT / path if not Path(path).is_absolute() else Path(path)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--eval", action="append", required=True, help="name=path")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["host_only", "trigger_only", "host_plus_trigger", "full_record_model"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    out_dir = args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    score_dir = out_dir / "scores"
    metric_dir = out_dir / "metrics"
    for d in [ckpt_dir, score_dir, metric_dir]:
        d.mkdir(parents=True, exist_ok=True)
    train_path = ROOT / args.train_features if not args.train_features.is_absolute() else args.train_features
    train_df = pd.read_csv(train_path)
    eval_items = parse_eval_items(args.eval)
    all_metrics: List[Dict[str, object]] = []
    manifest: List[Dict[str, object]] = []
    for seed in args.seeds:
        for variant in args.variants:
            print(f"[record_model_release_robust_train_eval] training variant={variant} seed={seed} train_rows={len(train_df)}", flush=True)
            fit = fit_model(train_df, variant, seed, args.steps, args.hidden_dim, args.lr, args.batch_size)
            ckpt_path = ckpt_dir / f"{variant}_seed{seed}_robust.pt"
            torch.save(
                {
                    "state_dict": fit["model"].state_dict(),
                    "variant": variant,
                    "seed": int(seed),
                    "feature_columns": fit["columns"],
                    "mean": fit["mean"],
                    "std": fit["std"],
                    "train_features_csv": str(train_path.resolve()),
                    "train_steps": int(args.steps),
                    "robust_train": True,
                },
                ckpt_path,
            )
            for eval_name, eval_path in eval_items:
                eval_df = pd.read_csv(eval_path)
                print(f"[record_model_release_robust_train_eval] eval variant={variant} seed={seed} eval={eval_name} rows={len(eval_df)}", flush=True)
                scores = predict(fit["model"], eval_df, fit["columns"], fit["mean"], fit["std"])
                score_path = score_dir / f"{variant}_seed{seed}_{eval_name}_scores.csv"
                scores.to_csv(score_path, index=False)
                all_metrics.extend(metric_rows(scores, variant, seed, eval_name, float(fit["final_loss"]), str(train_path.resolve()), score_path, ckpt_path))
                manifest.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "eval_name": eval_name,
                        "checkpoint_path": str(ckpt_path.resolve()),
                        "score_path": str(score_path.resolve()),
                        "eval_features_csv": str(eval_path.resolve()),
                    }
                )
    metrics = pd.DataFrame(all_metrics)
    metrics_path = metric_dir / "record_model_release_robust_train_eval_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = metrics.groupby(["eval_name", "perturbation", "variant", "score", "target"], dropna=False)[["auc", "eer", "tpr@1e-2", "tpr@1e-3", "tpr@1e-4"]].mean().reset_index()
    summary.to_csv(metric_dir / "record_model_release_robust_train_eval_summary.csv", index=False)
    (out_dir / "robust_train_eval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps({"metrics": str(metrics_path.resolve()), "rows": len(metrics)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
