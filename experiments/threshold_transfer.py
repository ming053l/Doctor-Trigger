#!/usr/bin/env python
"""A threshold fitted on one deployment, reused unchanged on another (appendix transfer table).

The verifier is fixed: the deployed record model trained on FF++ (three seeds pooled). For each
corpus and each target false-alarm rate, the threshold is the quantile of S_rel on that corpus's
benign test queries; it is then applied unchanged to the other corpora and we report the false-alarm
rate actually realized. The diagonal meets the target by construction.

Paths: config.yaml (outputs_root, record_model_runs); optional DT_SCORES_DIR (default: the record-scalar FF++ pin),
DT_SCORE (default S_rel). Writes $DT_OUTPUTS/threshold_transfer.csv.
"""
import glob, os, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402
from doctor_trigger.calibration import lower_tail_threshold  # noqa: E402
from doctor_trigger.manifests import DATASET_TAGS  # noqa: E402

OUT = Path(config.path("outputs_root"))
SCORES = os.environ.get("DT_SCORES_DIR") or config.path("record_model_runs") + "/recordscalar_ffpp/scores"
KEY = os.environ.get("DT_SCORE", "S_rel")
DS = ["FF++", "CelebDF", "DFDC"]


def benign(ds):
    fs = sorted(glob.glob(f"{SCORES}/*full_record_model_seed*_{DATASET_TAGS[ds]}_test_scores.csv"))
    assert fs, f"no score files for {ds} in {SCORES}"
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d[d.protocol == "benign"][KEY].to_numpy()


scores = {ds: benign(ds) for ds in DS}
rows = []
for fit in DS:
    for target in (1e-2, 1e-3):
        tau = float(lower_tail_threshold(scores[fit], frr=target))
        row = dict(fitted_on=fit, target=target, tau=tau)
        for ev in DS:
            row[ev] = float((scores[ev] < tau).mean())
        row["n_per_corpus"] = {ds: len(v) // 3 for ds, v in scores.items()}[fit]
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT / "threshold_transfer.csv", index=False)
with pd.option_context("display.float_format", lambda v: f"{v:.3g}"):
    print(df.to_string(index=False))
print("=== THRESHOLD TRANSFER DONE ===")
