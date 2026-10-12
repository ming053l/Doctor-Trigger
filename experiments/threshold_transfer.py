#!/usr/bin/env python
"""A threshold fitted on one deployment, reused unchanged on another (appendix Table 14).

The verifier is fixed: the deployed record model trained on FF++ (three seeds pooled). For each
corpus and each target false-alarm rate, the threshold is the quantile of S_rel on that corpus's
benign test queries; it is then applied unchanged to the other corpora and we report the false-alarm
rate actually realized. The diagonal meets the target by construction.

Paths: config.yaml (outputs_root, record_model_runs); optional DT_SCORES_DIR (default: the scores of
the record-scalar FF++ run), DT_SCORE (default S_rel). Writes $DT_OUTPUTS/threshold_transfer.csv.
"""
import glob
import os
from pathlib import Path

import pandas as pd

from doctor_trigger import config
from doctor_trigger.calibration import lower_tail_threshold
from doctor_trigger.manifests import DATASET_TAGS

OUT = Path(config.path("outputs_root"))
SCORES = os.environ.get("DT_SCORES_DIR") or config.path("record_model_runs") + "/recordscalar_ffpp/scores"
SCORE = os.environ.get("DT_SCORE", "S_rel")
DATASETS = ["FF++", "CelebDF", "DFDC"]
SEED_COUNT = 3


def benign_scores(dataset):
    """The chosen score on the benign test queries of one corpus, all seeds pooled."""
    files = sorted(glob.glob(f"{SCORES}/*full_record_model_seed*_{DATASET_TAGS[dataset]}_test_scores.csv"))
    assert files, f"no score files for {dataset} in {SCORES}"
    scores = pd.concat([pd.read_csv(f) for f in files])
    return scores[scores.protocol == "benign"][SCORE].to_numpy()


scores = {dataset: benign_scores(dataset) for dataset in DATASETS}
rows = []
for fitted_on in DATASETS:
    for target in (1e-2, 1e-3):
        tau = float(lower_tail_threshold(scores[fitted_on], frr=target))
        row = dict(fitted_on=fitted_on, target=target, tau=tau)
        for evaluated_on in DATASETS:
            row[evaluated_on] = float((scores[evaluated_on] < tau).mean())
        row["n_per_corpus"] = len(scores[fitted_on]) // SEED_COUNT
        rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(OUT / "threshold_transfer.csv", index=False)
with pd.option_context("display.float_format", lambda v: f"{v:.3g}"):
    print(table.to_string(index=False))
print("=== THRESHOLD TRANSFER DONE ===")
