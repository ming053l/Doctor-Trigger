"""Joint relation-and-key verification with training-only calibration (Table 13, clean panels).

A query is confirmed only when both criteria pass. Two calibration rules, both fitted on the target
dataset's benign training queries (source videos disjoint from test):

  per-head   learned S_rel and S_key, each cut at its own 1% quantile (original policy)
  matched    one common lower-tail quantile for both heads, the largest in [0, 0.01] whose joint
             rejection on the calibration queries stays within 1%; applied to the direct pair
             (full-ring C, release-normalized G_s) and to the learned pair (S_rel, S_key)

The learned heads are the FF++-trained record models (three seeds, weights fixed), read on the
record-scalar features. Reported rates are percentages per test query type (benign, swap,
wrong_key); learned rates are mean and std over seeds. The FF++ delivery panel of Table 13 is not
reproduced by this script.

Env: DT_CANON_ROOT, DT_OUTPUTS, DT_FEATURES_CSV (record-scalar features), DT_CKPT_DIR (checkpoints
of the recordscalar_ffpp run). Writes $DT_OUTPUTS/joint_policy/{rates.csv, thresholds.csv}.
"""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R = _dt("DT_CANON_ROOT")
sys.path.insert(0, f"{R}/record_model")
from src.record_model import RecordModelConfig, RecordModelMLP

OUT = Path(_dt("DT_OUTPUTS")) / "joint_policy"
TARGET = 0.01


def calibrate_joint(scores, target=TARGET):
    """Equal marginal quantiles, the largest whose joint rejection stays within the budget."""
    lo, hi = 0.0, target
    for _ in range(60):
        qt = (lo + hi) / 2
        cut = np.quantile(scores, qt, axis=0, method="linear")
        if np.any(scores < cut, axis=1).mean() <= target:
            lo = qt
        else:
            hi = qt
    cut = np.quantile(scores, lo, axis=0, method="linear")
    return cut, lo, float(np.any(scores < cut, axis=1).mean())


def load_models():
    out = []
    for path in sorted(Path(_dt("DT_CKPT_DIR")).glob("full_record_model_seed*.pt")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = RecordModelMLP(RecordModelConfig(variant="full_record_model", input_dim=len(ck["feature_columns"]),
                                             hidden_dim=ck["state_dict"]["encoder.0.weight"].shape[0]))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        out.append((int(ck["seed"]), m, ck))
    if not out:
        raise SystemExit("no full_record_model_seed*.pt in DT_CKPT_DIR")
    return out


def predict(model, ck, df):
    x = df[ck["feature_columns"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    x = (np.nan_to_num(x, nan=ck["mean"]) - ck["mean"]) / ck["std"]
    with torch.no_grad():
        o = model(torch.tensor(x.astype(np.float32)))
    return np.stack([torch.sigmoid(o["S_rel"]).numpy(), torch.sigmoid(o["S_key"]).numpy()], axis=1)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(_dt("DT_FEATURES_CSV"))
    models = load_models()
    rates, thresholds = [], []

    def record(dataset, policy, method, seed, cal, test, protocols, cut, qt, cal_rej):
        thresholds.append(dict(dataset=dataset, policy=policy, method=method, seed=seed, tau_relation=cut[0],
                               tau_key=cut[1], quantile=qt, calibration_rejection=cal_rej, n_calibration=len(cal)))
        ok = np.all(test >= cut, axis=1)
        for p in np.unique(protocols):
            sel = protocols == p
            rates.append(dict(dataset=dataset, policy=policy, method=method, seed=seed, protocol=p,
                              n=int(sel.sum()), confirmed_pct=100 * float(ok[sel].mean())))

    for ds, g in feats.groupby("dataset"):
        cal = g[(g.split == "train") & (g.protocol == "benign")]
        test = g[g.split == "test"]
        assert len(cal) and len(test) and not set(cal.source_group_id) & set(test.source_group_id)
        prot = test.protocol.to_numpy()
        direct_cal = cal[["C", "G_s"]].to_numpy(np.float64)
        cut, qt, rej = calibrate_joint(direct_cal)
        record(ds, "matched", "direct", -1, direct_cal, test[["C", "G_s"]].to_numpy(np.float64), prot, cut, qt, rej)
        for seed, m, ck in models:
            lc, lt = predict(m, ck, cal), predict(m, ck, test)
            per_head = np.quantile(lc, TARGET, axis=0)
            record(ds, "per_head", "learned", seed, lc, lt, prot, per_head, TARGET,
                   float(np.any(lc < per_head, axis=1).mean()))
            cut, qt, rej = calibrate_joint(lc)
            record(ds, "matched", "learned", seed, lc, lt, prot, cut, qt, rej)

    rates = pd.DataFrame(rates)
    rates.to_csv(OUT / "rates.csv", index=False)
    th = pd.DataFrame(thresholds)
    th.to_csv(OUT / "thresholds.csv", index=False)
    summ = (rates.groupby(["policy", "dataset", "method", "protocol"])
            .agg(n=("n", "first"), mean=("confirmed_pct", "mean"), std=("confirmed_pct", "std")).reset_index())
    table = summ.pivot_table(index=["policy", "dataset", "method"], columns="protocol", values=["mean", "std"])
    print(table.round(2).to_string())
    print("\ncalibration rejection (%):")
    print((th.groupby(["policy", "dataset", "method"]).calibration_rejection.mean() * 100).round(3).to_string())


if __name__ == "__main__":
    main()
