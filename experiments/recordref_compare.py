#!/usr/bin/env python
"""Old (released-PNG reference) vs new (stored-template reference) record-model inputs and scores.

Prints, for the paper items the reference change can move:
  1. Table 2 record-model S_fake row: 7 cells, mean AUC over 3 seeds, per pin.
  2. Per-feature test AUCs (manipulated vs benign, and correct vs wrong key) for the affected columns.
  3. Benign-query sanity: fraction of C and G_s pinned at 1.
  4. Mean S_rel / S_key / S_fake by query type, FF++-trained V6 on FF++ test (tab:trigger_signal bottom).
  5. V1/V2/V3/V6 variant AUCs from the FF++ pin.
Env: DT_CANON_ROOT, DT_OUTPUTS. Writes $DT_OUTPUTS/recordref_compare.json.
"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

R = os.environ["DT_CANON_ROOT"]; OUT = Path(os.environ["DT_OUTPUTS"])
V = os.environ.get("DT_RR_VARIANT", "recordref")   # recordref | recordscalar
NEW_F = OUT / {"recordref": "features_record_ref", "recordscalar": "features_record_scalar"}[V] / f"{V}_features.csv"
OLD_F = Path(f"{R}/features/calibrated_features.csv")
RUNS = Path(os.environ["DT_RECORD_MODEL_RUNS"])
METRICS = {
    "old": {"ffpp": Path(f"{R}/results/record_model/metrics/record_model_release_calibrated_ffpp_to_all_metrics.csv"),
            "dfdc": RUNS / "calibrated_dfdc/metrics/record_model_release_calibrated_dfdc_to_all_metrics.csv",
            "celebdf": RUNS / "calibrated_celebdf/metrics/record_model_release_calibrated_celebdf_to_all_metrics.csv"},
    "new": {p: RUNS / f"{V}_{p}/metrics/record_model_release_{V}_{p}_to_all_metrics.csv"
            for p in ("ffpp", "dfdc", "celebdf")},
}
SCORES = {"old": Path(f"{R}/results/record_model/scores"), "new": RUNS / f"{V}_ffpp/scores"}
# Table 2 column order: same dataset FF++, CDF, DFDC | from FF++ -> CDF, DFDC | from DFDC -> CDF, FF++
CELLS = [("ffpp", "ffpp_test"), ("celebdf", "celebdf_test"), ("dfdc", "dfdc_test"),
         ("ffpp", "celebdf_test"), ("ffpp", "dfdc_test"), ("dfdc", "celebdf_test"), ("dfdc", "ffpp_test")]
AFFECTED = ["C", "G_s", "G_wrong_mean", "G_wrong_max", "G_margin", "G_z"]
res = {}


def table2(tag):
    m = {p: pd.read_csv(f) for p, f in METRICS[tag].items() if f.exists()}
    row = []
    for pin, ev in CELLS:
        if pin not in m:
            row.append(None); continue
        g = m[pin]; g = g[(g.variant == "full_record_model") & (g.score == "S_fake") & (g.eval_name == ev)]
        row.append(round(float(g.auc.mean()), 4) if len(g) else None)
    return row


print("1. Table 2 S_fake row (FF++ CDF DFDC | FF++->CDF FF++->DFDC | DFDC->CDF DFDC->FF++)")
for tag in ("old", "new"):
    res[f"table2_{tag}"] = table2(tag)
    print(f"   {tag}: " + "  ".join("  n/a " if v is None else f"{v:.4f}" for v in res[f"table2_{tag}"]))

print("2. per-feature test AUC  [fake: manipulated vs benign]  [key: benign vs wrong_key]")
old = pd.read_csv(OLD_F, usecols=["sample_id", "dataset", "split", "protocol"] + AFFECTED + ["H"])
new = pd.read_csv(NEW_F, usecols=["sample_id", "dataset", "split", "protocol"] + AFFECTED + ["H"])
assert len(old) == len(new), (len(old), len(new))
res["features"] = {}
for ds in ("FF++", "DFDC", "CelebDF"):
    for tag, d in (("old", old), ("new", new)):
        t = d[(d.dataset == ds) & (d.split == "test")]
        ben, fak, wk = t[t.protocol == "benign"], t[t.protocol == "swap"], t[t.protocol == "wrong_key"]
        line = {}
        for c in ["H"] + AFFECTED:
            fa = roc_auc_score(np.r_[np.zeros(len(fak)), np.ones(len(ben))], np.r_[fak[c], ben[c]])
            ka = roc_auc_score(np.r_[np.ones(len(ben)), np.zeros(len(wk))], np.r_[ben[c], wk[c]])
            line[c] = (round(fa, 4), round(ka, 4))
        res["features"][f"{ds}_{tag}"] = line
        print(f"   {ds:7s} {tag}: " + " ".join(f"{c}={v[0]:.3f}/{v[1]:.3f}" for c, v in line.items()))

print("3. test benign pinned at 1 (C, G_s)")
for tag, d in (("old", old), ("new", new)):
    b = d[(d.protocol == "benign") & (d.split == "test")]
    res[f"pinned_{tag}"] = [float(np.mean(np.isclose(b.C, 1))), float(np.mean(np.isclose(b.G_s, 1)))]
    print(f"   {tag}: C {res[f'pinned_{tag}'][0]:.1%}  G_s {res[f'pinned_{tag}'][1]:.1%}  G_s median {b.G_s.median():.3f}")

print("4. mean record scores by query type (FF++-trained V6, FF++ test, 3 seeds)")
for tag in ("old", "new"):
    files = sorted(SCORES[tag].glob("*full_record_model_seed*_ffpp_test_scores.csv"))
    if len(files) != 3:
        print(f"   {tag}: {len(files)} score files, skipped"); continue
    sc = pd.concat([pd.read_csv(f) for f in files])
    by = sc.groupby("protocol")[["S_rel", "S_key", "S_fake"]].mean().round(4)
    res[f"by_query_{tag}"] = by.to_dict()
    print(f"   {tag}:\n" + "\n".join("     " + l for l in by.to_string().splitlines()))

print("5. FF++ pin variants, mean AUC over seeds (S_fake/S_rel/S_key)")
for tag in ("old", "new"):
    f = METRICS[tag]["ffpp"]
    if not f.exists():
        continue
    m = pd.read_csv(f)
    s = m.groupby(["eval_name", "variant", "score"]).auc.mean().unstack("score").round(4)
    res[f"variants_{tag}"] = {f"{a}|{b}": v for (a, b), v in s.to_dict("index").items()}
    print(f"   {tag}:\n" + "\n".join("     " + l for l in s.to_string().splitlines()))

(OUT / f"{V}_compare.json").write_text(json.dumps(res, indent=1))
print("=== COMPARE DONE ===")
