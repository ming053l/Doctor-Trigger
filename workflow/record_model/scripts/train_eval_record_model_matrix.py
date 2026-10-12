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

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parents[1]))  # the doctor_trigger package

from doctor_trigger.metrics import equal_error_rate, roc_auc, tpr_at_fpr
from doctor_trigger.record_model import fit_record_model, save_checkpoint, score_table


def load_config(path: Path) -> Dict[str, object]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def config_path(config_file: Path, value: object) -> Path:
    """Resolve a configured path relative to the YAML file, not a user home."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_file.resolve().parent / path).resolve()


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
                "auc": roc_auc(y, s),
                "eer": equal_error_rate(y, s, rounding="fpr + fnr"),
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
            fitted = fit_record_model(train_df, variant, int(seed), steps, hidden_dim, lr, batch_size)
            ckpt_path = ckpt_dir / f"{variant}_seed{seed}_{run_name}.pt"
            save_checkpoint(fitted, ckpt_path, train_csv)
            for item in eval_items:
                eval_name = str(item["name"])
                eval_csv = config_path(config_file, item["path"])
                eval_df = pd.read_csv(eval_csv)
                print(f"[record_model_release_train_eval_matrix] eval variant={variant} seed={seed} eval={eval_name} n={len(eval_df)}", flush=True)
                scores = score_table(fitted, eval_df)
                score_path = score_dir / f"{run_name}_{variant}_seed{seed}_{eval_name}_scores.csv"
                scores.to_csv(score_path, index=False)
                all_metrics.extend(metric_rows(scores, variant, int(seed), eval_name, float(fitted.final_loss)))
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
