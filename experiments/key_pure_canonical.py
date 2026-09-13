#!/usr/bin/env python
"""Paired key-pure audit and per-query-type record-model scores on the main test split.

Replaces the earlier-protocol numbers (FF++ 2,198 / Celeb-DF 7,352 pairs) in the appendix tables
tab:key_pure_operating and tab:trigger_signal. A pair is one released query scored with its own key
(protocol benign) and with a wrong key (protocol wrong_key); the image is identical within a pair.
Record-model means use the FF++-trained full model, averaged over its three seeds.

Env: DT_CANON_ROOT, DT_OUTPUTS. Writes $DT_OUTPUTS/key_pure_canonical{tag}.csv and
$DT_OUTPUTS/record_scores_by_query{tag}.csv, and prints the LaTeX rows. Optional DT_FEATURES_CSV,
DT_RECORD_MODEL_SCORES_DIR and DT_TAG point it at another feature extraction and its record models
(the registry entry uses the record-scalar features and runs).
"""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

R = os.environ["DT_CANON_ROOT"]; OUT = Path(os.environ["DT_OUTPUTS"])
FEATURES = os.environ.get("DT_FEATURES_CSV", f"{R}/features/calibrated_features.csv")
SCORE_DIR = os.environ.get("DT_RECORD_MODEL_SCORES_DIR", f"{R}/results/record_model/scores")
TAG = os.environ.get("DT_TAG", "")
SCORES = ["H", "C", "G_s", "G_margin", "G_z", "G_wrong_mean", "G_wrong_max"]
d = pd.read_csv(FEATURES,
                usecols=["dataset", "split", "protocol", "enrollment_id"] + SCORES)
d = d[d.split == "test"]


def tpr_at(y, s, t):
    fpr, tpr, _ = roc_curve(y, s); ok = fpr <= t
    return float(tpr[ok].max()) if ok.any() else 0.0


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)


rows = []
for ds, g in d.groupby("dataset"):
    m = g[g.protocol == "benign"].drop_duplicates("enrollment_id").merge(
        g[g.protocol == "wrong_key"].drop_duplicates("enrollment_id"), on="enrollment_id", suffixes=("_c", "_w"))
    assert np.allclose(m.H_c, m.H_w), "pairs must share the query image"
    y = np.r_[np.ones(len(m)), np.zeros(len(m))]
    for s in SCORES:
        sc = np.r_[m[s + "_c"], m[s + "_w"]]
        rows.append(dict(dataset=ds, score=s, auc=roc_auc_score(y, sc), eer=eer(y, sc), tpr_1e2=tpr_at(y, sc, 1e-2),
                         tpr_1e3=tpr_at(y, sc, 1e-3), tpr_1e4=tpr_at(y, sc, 1e-4), n_pairs=len(m)))
kp = pd.DataFrame(rows); kp.to_csv(OUT / f"key_pure_canonical{TAG}.csv", index=False)

files = sorted(glob.glob(f"{SCORE_DIR}/*full_record_model_seed*_ffpp_test_scores.csv"))
assert len(files) == 3, files
sc = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(files)])
by = sc.groupby("protocol")[["S_rel", "S_key", "S_fake"]].mean()
by["n_per_seed"] = sc[sc.seed == 0].protocol.value_counts()
by.to_csv(OUT / f"record_scores_by_query{TAG}.csv")

print(kp.round(4).to_string(index=False)); print(by.round(3).to_string())
