#!/usr/bin/env python
"""Joint relation-and-key verification with training-only calibration (Table 13, clean panels).

A query is confirmed only when both criteria pass. Two calibration rules, both fitted on the target
dataset's benign training queries (source videos disjoint from test):

  per-head   learned S_rel and S_key, each cut at its own 1% quantile
  matched    one common lower-tail quantile for both heads, the largest in [0, 0.01] whose joint
             rejection on the calibration queries stays within 1%; applied to the direct pair
             (full-ring C, release-normalized G_s) and to the learned pair (S_rel, S_key)

The learned heads are the FF++-trained record models (three seeds, weights fixed), read on the
record-scalar features. Reported rates are percentages per test query type (benign, swap,
wrong_key); learned rates are mean and std over seeds. The FF++ delivery panel of Table 13 is not
reproduced by this script.

Paths: config.yaml (outputs_root); env DT_FEATURES_CSV (record-scalar features), DT_CKPT_DIR (checkpoints
of the recordscalar_ffpp run). Writes $DT_OUTPUTS/joint_policy/{rates.csv, thresholds.csv}.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from doctor_trigger import config
from doctor_trigger.calibration import joint_rejection, joint_thresholds, lower_tail_threshold
from doctor_trigger.record_model import load_checkpoint, record_model_scores


def _required_env(name):
    value = os.environ.get(name)
    assert value, f"{name} unset: run through run.py, or export it (see the joint-policy entry of experiments/registry.yaml)"
    return value


OUT = Path(config.path("outputs_root")) / "joint_policy"
FRR = 0.01
"""Benign false-rejection budget on the calibration queries."""


def load_models():
    """(seed, fitted model) for every full record-model checkpoint in DT_CKPT_DIR."""
    models = []
    for path in sorted(Path(_required_env("DT_CKPT_DIR")).glob("full_record_model_seed*.pt")):
        fitted = load_checkpoint(path)
        models.append((fitted.seed, fitted))
    if not models:
        raise SystemExit("no full_record_model_seed*.pt in DT_CKPT_DIR")
    return models


def learned_scores(fitted, queries):
    """(S_rel, S_key) per query, one row each."""
    scores = record_model_scores(fitted, queries)
    return np.stack([scores["S_rel"], scores["S_key"]], axis=1)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(_required_env("DT_FEATURES_CSV"))
    models = load_models()
    rates, thresholds = [], []

    def add_policy(dataset, policy, method, seed, calibration, test, protocols, cut, quantile, calibration_rejection):
        """Record one policy's thresholds and its confirmed rate per test query type."""
        thresholds.append(dict(dataset=dataset, policy=policy, method=method, seed=seed, tau_relation=cut[0],
                               tau_key=cut[1], quantile=quantile, calibration_rejection=calibration_rejection,
                               n_calibration=len(calibration)))
        confirmed = np.all(test >= cut, axis=1)
        for protocol in np.unique(protocols):
            selected = protocols == protocol
            rates.append(dict(dataset=dataset, policy=policy, method=method, seed=seed, protocol=protocol,
                              n=int(selected.sum()), confirmed_pct=100 * float(confirmed[selected].mean())))

    for dataset, group in features.groupby("dataset"):
        calibration = group[(group.split == "train") & (group.protocol == "benign")]
        test = group[group.split == "test"]
        assert len(calibration) and len(test) and not set(calibration.source_group_id) & set(test.source_group_id)
        protocols = test.protocol.to_numpy()
        direct_calibration = calibration[["C", "G_s"]].to_numpy(np.float64)
        joint = joint_thresholds(direct_calibration, frr=FRR)
        add_policy(dataset, "matched", "direct", -1, direct_calibration, test[["C", "G_s"]].to_numpy(np.float64),
                   protocols, joint.thresholds, joint.quantile, joint.calibration_rejection)
        for seed, fitted in models:
            calibration_scores, test_scores = learned_scores(fitted, calibration), learned_scores(fitted, test)
            per_head = lower_tail_threshold(calibration_scores, FRR, axis=0)
            add_policy(dataset, "per_head", "learned", seed, calibration_scores, test_scores, protocols, per_head,
                       FRR, joint_rejection(calibration_scores, per_head))
            joint = joint_thresholds(calibration_scores, frr=FRR)
            add_policy(dataset, "matched", "learned", seed, calibration_scores, test_scores, protocols,
                       joint.thresholds, joint.quantile, joint.calibration_rejection)

    rates = pd.DataFrame(rates)
    rates.to_csv(OUT / "rates.csv", index=False)
    thresholds = pd.DataFrame(thresholds)
    thresholds.to_csv(OUT / "thresholds.csv", index=False)
    summary = (rates.groupby(["policy", "dataset", "method", "protocol"])
               .agg(n=("n", "first"), mean=("confirmed_pct", "mean"), std=("confirmed_pct", "std")).reset_index())
    table = summary.pivot_table(index=["policy", "dataset", "method"], columns="protocol", values=["mean", "std"])
    print(table.round(2).to_string())
    print("\ncalibration rejection (%):")
    print((thresholds.groupby(["policy", "dataset", "method"]).calibration_rejection.mean() * 100)
          .round(3).to_string())


if __name__ == "__main__":
    main()
