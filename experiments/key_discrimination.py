#!/usr/bin/env python
"""Key discrimination and learned record scores on the main test split (appendix Table 9).

Top of the table: correct-versus-wrong-key separation on paired queries. A pair is one released
query read with its own key (protocol benign) and with a wrong key (protocol wrong_key); the image
is identical within a pair, so only the key differs. For each direct reading the script reports AUC,
EER and the TPR at FPR 1e-2, 1e-3 and 1e-4. Bottom: the mean learned scores S_rel, S_key, S_fake per
query type, from the FF++-trained full record model averaged over its three seeds.

Paths: config.yaml (canonical_root, outputs_root). Writes $DT_OUTPUTS/key_pure_canonical{tag}.csv and
$DT_OUTPUTS/record_scores_by_query{tag}.csv and prints both tables. Optional DT_FEATURES_CSV,
DT_RECORD_MODEL_SCORES_DIR and DT_TAG point it at another feature extraction and its record models
(the registry entry uses the record-scalar features and runs).
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from doctor_trigger import config
from doctor_trigger.metrics import equal_error_rate, roc_auc, tpr_at_fpr

PIPELINE_ROOT = config.path("canonical_root")
OUT = Path(config.path("outputs_root"))
FEATURES = os.environ.get("DT_FEATURES_CSV", f"{PIPELINE_ROOT}/features/calibrated_features.csv")
SCORE_DIR = os.environ.get("DT_RECORD_MODEL_SCORES_DIR", f"{PIPELINE_ROOT}/results/record_model/scores")
TAG = os.environ.get("DT_TAG", "")
READINGS = ["H", "C", "G_s", "G_margin", "G_z", "G_wrong_mean", "G_wrong_max"]

features = pd.read_csv(FEATURES, usecols=["dataset", "split", "protocol", "enrollment_id"] + READINGS)
features = features[features.split == "test"]

rows = []
for dataset, group in features.groupby("dataset"):
    pairs = group[group.protocol == "benign"].drop_duplicates("enrollment_id").merge(
        group[group.protocol == "wrong_key"].drop_duplicates("enrollment_id"), on="enrollment_id",
        suffixes=("_c", "_w"))
    assert np.allclose(pairs.H_c, pairs.H_w), "pairs must share the query image"
    labels = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
    for reading in READINGS:
        scores = np.r_[pairs[reading + "_c"], pairs[reading + "_w"]]
        rows.append(dict(dataset=dataset, score=reading, auc=roc_auc(labels, scores),
                         eer=equal_error_rate(labels, scores, rounding="fpr + 1 - tpr"),
                         tpr_1e2=tpr_at_fpr(labels, scores, 1e-2), tpr_1e3=tpr_at_fpr(labels, scores, 1e-3),
                         tpr_1e4=tpr_at_fpr(labels, scores, 1e-4), n_pairs=len(pairs)))
key_table = pd.DataFrame(rows)
key_table.to_csv(OUT / f"key_pure_canonical{TAG}.csv", index=False)

score_files = sorted(glob.glob(f"{SCORE_DIR}/*full_record_model_seed*_ffpp_test_scores.csv"))
assert len(score_files) == 3, score_files
record_scores = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(score_files)])
by_query = record_scores.groupby("protocol")[["S_rel", "S_key", "S_fake"]].mean()
by_query["n_per_seed"] = record_scores[record_scores.seed == 0].protocol.value_counts()
by_query.to_csv(OUT / f"record_scores_by_query{TAG}.csv")

print(key_table.round(4).to_string(index=False)); print(by_query.round(3).to_string())
