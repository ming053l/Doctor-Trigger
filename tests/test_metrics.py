"""doctor_trigger.metrics equals the metric helpers it replaces (plan Sec. 2.11), on the key-pure
pairs of experiments/key_pure_canonical.py (calibrated features, test split).

workflow/record_model/src/record_model.py is imported by file path; the experiment helpers run code
at import time, so they are copied verbatim below.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

from doctor_trigger.metrics import equal_error_rate, robust_z, roc_auc, tpr_at_fpr

REPO = Path(__file__).resolve().parents[1]
FEATURES = Path("/ssd2/ming/ttf_run/public_view/features/calibrated_features.csv")
pytestmark = pytest.mark.skipif(not FEATURES.exists(), reason="public_view not available")
SCORES = ["H", "C", "G_s", "G_margin", "G_z", "G_wrong_mean", "G_wrong_max"]


def load_by_path(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


OLD_RECORD_MODEL = load_by_path("old_record_model", REPO / "workflow/record_model/src/record_model.py")


# experiments/key_pure_canonical.py:tpr_at (= operating_points.py:tpr_at) and :eer
def key_pure_tpr_at(y, s, t):
    fpr, tpr, _ = roc_curve(y, s); ok = fpr <= t
    return float(tpr[ok].max()) if ok.any() else 0.0


def key_pure_eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)


# experiments/adaptive_attacks.py:rz (= diffusion_regeneration.py:rz, copy_ring_detector.py:rz)
def adaptive_rz(v, cal):
    cal = np.asarray(cal); med = np.median(cal); mad = np.median(np.abs(cal - med)) + 1e-6
    return (np.asarray(v) - med) / mad


def key_pure_pairs():
    d = pd.read_csv(FEATURES, usecols=["dataset", "split", "protocol", "enrollment_id"] + SCORES)
    d = d[d.split == "test"]
    for _, g in d.groupby("dataset"):
        m = g[g.protocol == "benign"].drop_duplicates("enrollment_id").merge(
            g[g.protocol == "wrong_key"].drop_duplicates("enrollment_id"), on="enrollment_id", suffixes=("_c", "_w"))
        y = np.r_[np.ones(len(m)), np.zeros(len(m))]
        for s in SCORES:
            yield y, np.r_[m[s + "_c"], m[s + "_w"]]


PAIRS = list(key_pure_pairs())


def test_roc_metrics_match_record_model_and_key_pure():
    for y, scores in PAIRS:
        assert roc_auc(y, scores) == roc_auc_score(y, scores) == OLD_RECORD_MODEL.safe_auc(y, scores)
        for max_fpr in (1e-2, 1e-3, 1e-4):
            assert tpr_at_fpr(y, scores, max_fpr) == key_pure_tpr_at(y, scores, max_fpr)
            assert tpr_at_fpr(y, scores, max_fpr) == OLD_RECORD_MODEL.tpr_at_fpr(y, scores, max_fpr)
        assert equal_error_rate(y, scores) == OLD_RECORD_MODEL.safe_eer(y, scores)


def test_equal_error_rate_matches_key_pure_eer():
    # key_pure_canonical.py writes (fpr + 1 - tpr) / 2; record_model.py writes (fpr + fnr) / 2.
    for y, scores in PAIRS:
        assert equal_error_rate(y, scores, rounding="fpr + 1 - tpr") == key_pure_eer(y, scores)


def test_table4_auc_of_lower_is_suspicious_scores():
    # experiments/table4_artifact.py:stage_collect.auc(neg, pos) scores -x, benign label 0
    rng = np.random.default_rng(0)
    neg, pos = rng.normal(0.95, 0.02, 300), rng.normal(0.85, 0.05, 300)
    expected = float(roc_auc_score([0] * len(neg) + [1] * len(pos), np.concatenate([-neg, -pos])))
    assert roc_auc(np.r_[np.zeros(len(neg)), np.ones(len(pos))], -np.concatenate([neg, pos])) == expected


def test_nonfinite_and_single_class():
    assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert np.isnan(equal_error_rate([0, 1], [np.nan, 0.5]))
    assert roc_auc([0, 1, 1], [0.1, np.nan, 0.9]) == 1.0


def test_robust_z_matches_adaptive_rz():
    rng = np.random.default_rng(2)
    calibration, values = rng.normal(0.9, 0.05, 200), rng.normal(0.8, 0.1, 50)
    assert robust_z(values, calibration).tobytes() == adaptive_rz(values, calibration).tobytes()
    assert robust_z(list(values), list(calibration)).tobytes() == adaptive_rz(values, calibration).tobytes()
