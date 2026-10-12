#!/usr/bin/env python
"""Calibration and record-model sensitivity beyond release strength (appendix Fig. 12).

(a) FRR target: tau from the train-benign quantile at target t, then test TPR and realized FPR on
    FF++ (closed-form original-reference agreement H, the deployed reading);
(b) calibration-set size: spread of the realized test FPR when tau is set from n train-benign
    samples (200 resamples per n), target 1%;
(c) record-model loss weight lambda_key: retrain the full record model (3 seeds) per value, report
    S_rel AUC (y_rel) and S_key AUC (y_keycorr) on FF++ test and Celeb-DF test.

Paths: config.yaml (canonical_root, outputs_root). Optional env DT_FEATURES_CSV (default: the
pipeline's calibrated_features.csv; the registry entry passes the record-scalar features) and
DT_OUT (output folder, default $DT_OUTPUTS/recordscalar_hyper). Writes <DT_OUT>/hyper_sensitivity.json;
fig-hyper draws Fig. 12 from it.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from doctor_trigger import config
from doctor_trigger.calibration import lower_tail_threshold
from doctor_trigger.metrics import roc_auc
from doctor_trigger.record_model import (FittedRecordModel, RecordModel, RecordModelConfig, feature_columns,
                                         feature_matrix, multitask_loss, record_model_scores,
                                         standardized_input)

STD_FLOOR = 1e-8
"""This sweep's standardization floor (the record-model trainer uses 1e-6)."""

PIPELINE_ROOT = config.path("canonical_root")
OUT = Path(os.environ.get("DT_OUT") or Path(config.path("outputs_root")) / "recordscalar_hyper")
features = pd.read_csv(os.environ.get("DT_FEATURES_CSV", f"{PIPELINE_ROOT}/features/calibrated_features.csv"))
results = {}

# ---------- (a)+(b): closed-form relation thresholding ----------
ffpp = features[features.dataset == "FF++"]
train_benign = ffpp[(ffpp.split == "train") & (ffpp.protocol == "benign")].H.to_numpy()
ffpp_test = ffpp[ffpp.split == "test"]
test_benign = ffpp_test[ffpp_test.protocol == "benign"].H.to_numpy()
test_swap = ffpp_test[ffpp_test.protocol == "swap"].H.to_numpy()
results["frr_target"] = []
for target in [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]:
    tau = float(lower_tail_threshold(train_benign, frr=target))
    results["frr_target"].append(dict(target=target, tau=tau, tpr=float((test_swap < tau).mean()),
                                      realized_fpr=float((test_benign < tau).mean())))
rng = np.random.default_rng(0)
results["cal_size"] = []
for n in [25, 50, 100, 200, 400]:
    fprs = []
    for _ in range(200):
        subset = rng.choice(train_benign, size=n, replace=False)
        tau = lower_tail_threshold(subset, frr=0.01)
        fprs.append(float((test_benign < tau).mean()))
    quantiles = np.quantile(fprs, [0.1, 0.25, 0.5, 0.75, 0.9])
    results["cal_size"].append(dict(n=n, q10=quantiles[0], q25=quantiles[1], med=quantiles[2],
                                    q75=quantiles[3], q90=quantiles[4]))
print("closed-form sweeps done", flush=True)

# ---------- (c): lambda_key retraining ----------
columns = feature_columns("full_record_model", list(features.columns))
train = features[(features.dataset == "FF++") & (features.split == "train")].reset_index(drop=True)
evaluations = {"FF++": features[(features.dataset == "FF++") & (features.split == "test")].reset_index(drop=True),
               "CelebDF": features[(features.dataset == "CelebDF") & (features.split == "test")].reset_index(drop=True)}
train_raw = feature_matrix(train, columns)
# training-set mean and std kept in float64 (record_model.standardize returns them as float32)
mean = np.nanmean(train_raw, axis=0); std = np.nanstd(train_raw, axis=0); std[std < STD_FLOOR] = 1.0
train_input = standardized_input(train_raw, mean, std)
targets = {name: torch.tensor(train[column].to_numpy(np.float32)) for name, column in
           [("y_fake", "y_fake"), ("y_rel", "y_rel"), ("y_key", "y_keycorr")]}
loss_features = {name: torch.tensor(pd.to_numeric(train[name], errors="coerce").fillna(0.0).to_numpy(np.float32))
                 for name in ["G_s", "G_wrong_max"] if name in train.columns}
inputs = torch.tensor(train_input)
results["lambda_key"] = []
for lambda_key in [0.0, 0.1, 0.3, 1.0, 3.0]:
    for seed in [0, 1, 2]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model_config = RecordModelConfig(variant="full_record_model", input_dim=train_input.shape[1],
                                         hidden_dim=64, lambda_key=lambda_key)
        model = RecordModel(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()
        for _ in range(1000):
            optimizer.zero_grad(set_to_none=True)
            loss = multitask_loss(model(inputs), targets, model_config, loss_features)
            loss.backward(); optimizer.step()
        model.eval()
        fitted = FittedRecordModel(model, "full_record_model", seed, columns, mean, std)
        for name, evaluation in evaluations.items():
            scores = record_model_scores(fitted, evaluation)
            results["lambda_key"].append(dict(lam=lambda_key, seed=seed, eval=name,
                                              rel_auc=roc_auc(evaluation.y_rel.to_numpy(int), scores["S_rel"]),
                                              key_auc=roc_auc(evaluation.y_keycorr.to_numpy(int), scores["S_key"])))
        print(f"lam={lambda_key} seed={seed} done", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "hyper_sensitivity.json").write_text(json.dumps(results, indent=1))
print("ALL_DONE")
