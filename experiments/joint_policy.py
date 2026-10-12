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

Paths: config.yaml (outputs_root); env DT_FEATURES_CSV (record-scalar features), DT_CKPT_DIR (checkpoints
of the recordscalar_ffpp run). Writes $DT_OUTPUTS/joint_policy/{rates.csv, thresholds.csv}.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402
from doctor_trigger.calibration import joint_rejection, joint_thresholds, lower_tail_threshold  # noqa: E402
from doctor_trigger.record_model import load_checkpoint, record_model_scores  # noqa: E402


def _required_env(name):
    value = os.environ.get(name)
    assert value, f"{name} unset: run through run.py, or export it (see config.example.yaml)"
    return value


OUT = Path(config.path("outputs_root")) / "joint_policy"
TARGET = 0.01


def load_models():
    out = []
    for path in sorted(Path(_required_env("DT_CKPT_DIR")).glob("full_record_model_seed*.pt")):
        fitted = load_checkpoint(path)
        out.append((fitted.seed, fitted))
    if not out:
        raise SystemExit("no full_record_model_seed*.pt in DT_CKPT_DIR")
    return out


def predict(fitted, df):
    scores = record_model_scores(fitted, df)
    return np.stack([scores["S_rel"], scores["S_key"]], axis=1)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(_required_env("DT_FEATURES_CSV"))
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
        joint = joint_thresholds(direct_cal, frr=TARGET)
        record(ds, "matched", "direct", -1, direct_cal, test[["C", "G_s"]].to_numpy(np.float64), prot,
               joint.thresholds, joint.quantile, joint.calibration_rejection)
        for seed, fitted in models:
            lc, lt = predict(fitted, cal), predict(fitted, test)
            per_head = lower_tail_threshold(lc, TARGET, axis=0)
            record(ds, "per_head", "learned", seed, lc, lt, prot, per_head, TARGET, joint_rejection(lc, per_head))
            joint = joint_thresholds(lc, frr=TARGET)
            record(ds, "matched", "learned", seed, lc, lt, prot, joint.thresholds, joint.quantile,
                   joint.calibration_rejection)

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
