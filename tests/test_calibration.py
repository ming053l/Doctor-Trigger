"""doctor_trigger.calibration equals experiments/joint_policy.py:calibrate_joint (frozen outputs in
tests/golden/experiment_record_model.json) and the inline
``np.quantile(scores, 0.01)`` cutoffs, on the frozen record-scalar features (tests/data_roots.py)."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from doctor_trigger.calibration import JointCalibration, joint_rejection, joint_thresholds, lower_tail_threshold
from data_roots import OUTPUTS, requires, under
from golden_values import digest, load

FEATURES = under(OUTPUTS, "features_record_scalar/recordscalar_features.csv")
pytestmark = requires(FEATURES)


@pytest.fixture(scope="module")
def calibration_sets() -> dict[str, np.ndarray]:
    features = pd.read_csv(FEATURES, usecols=["dataset", "split", "protocol", "C", "G_s"])
    benign_train = features[(features.split == "train") & (features.protocol == "benign")]
    return {dataset: group[["C", "G_s"]].to_numpy(np.float64) for dataset, group in benign_train.groupby("dataset")}


def test_joint_thresholds_equal_calibrate_joint(calibration_sets):
    golden = load("experiment_record_model")["calibrate_joint"]
    learned_like = np.random.default_rng(0).random((500, 2)).astype(np.float32)  # (S_rel, S_key) are float32
    for name, scores in [*calibration_sets.items(), ("learned_like", learned_like)]:
        old = golden[name]
        result = joint_thresholds(scores)
        assert isinstance(result, JointCalibration)
        assert digest(result.thresholds) == old["thresholds"]
        assert result.quantile == old["quantile"] and result.calibration_rejection == old["calibration_rejection"]
        assert result.calibration_rejection <= 0.01
        assert joint_rejection(scores, result.thresholds) == old["calibration_rejection"]


def test_lower_tail_threshold_equals_inline_quantiles(calibration_sets):
    for scores in calibration_sets.values():
        for column in range(2):
            assert lower_tail_threshold(scores[:, column]) == np.quantile(scores[:, column], 0.01)
            assert lower_tail_threshold(list(scores[:, column]), frr=0.05) == np.quantile(list(scores[:, column]), 0.05)
        # per-head cutoffs (joint_policy.py: np.quantile(lc, TARGET, axis=0))
        assert lower_tail_threshold(scores, axis=0).tobytes() == np.quantile(scores, 0.01, axis=0).tobytes()
    float32_scores = np.random.default_rng(1).random(300).astype(np.float32)
    cut = lower_tail_threshold(float32_scores)
    assert cut.dtype == np.quantile(float32_scores, 0.01).dtype and cut == np.quantile(float32_scores, 0.01)


def test_pandas_quantile_sites_match(calibration_sets):
    # experiments/original_control.py and table4_artifact.py call Series.quantile / np.quantile on a Series.
    for scores in calibration_sets.values():
        series = pd.Series(scores[:, 0])
        assert lower_tail_threshold(series) == series.quantile(0.01)
        assert lower_tail_threshold(series.values) == float(np.quantile(series, 0.01))

