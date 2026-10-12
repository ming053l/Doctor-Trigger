"""doctor_trigger.calibration equals experiments/joint_policy.py:calibrate_joint and the inline
``np.quantile(scores, 0.01)`` cutoffs, on the frozen record-scalar features (public_view_outputs)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from doctor_trigger.calibration import JointCalibration, joint_rejection, joint_thresholds, lower_tail_threshold

REPO = Path(__file__).resolve().parents[1]
PUBLIC_VIEW = Path("/ssd2/ming/ttf_run/public_view")
OUTPUTS = Path("/ssd2/ming/ttf_run/public_view_outputs")
FEATURES = OUTPUTS / "features_record_scalar/recordscalar_features.csv"
pytestmark = pytest.mark.skipif(not FEATURES.exists(), reason="public_view not available")


@pytest.fixture(scope="module")
def old_joint_policy(tmp_path_factory):
    # joint_policy.py reads DT_CANON_ROOT / DT_OUTPUTS at import and imports src.record_model from the root.
    saved_env, saved_path = dict(os.environ), list(sys.path)
    os.environ["DT_CANON_ROOT"] = str(PUBLIC_VIEW)
    os.environ["DT_OUTPUTS"] = str(tmp_path_factory.mktemp("outputs"))
    try:
        module_spec = importlib.util.spec_from_file_location("old_joint_policy", REPO / "experiments/joint_policy.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        yield module
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        sys.path[:] = saved_path


@pytest.fixture(scope="module")
def calibration_sets() -> dict[str, np.ndarray]:
    features = pd.read_csv(FEATURES, usecols=["dataset", "split", "protocol", "C", "G_s"])
    benign_train = features[(features.split == "train") & (features.protocol == "benign")]
    return {dataset: group[["C", "G_s"]].to_numpy(np.float64) for dataset, group in benign_train.groupby("dataset")}


def test_joint_thresholds_equal_calibrate_joint(old_joint_policy, calibration_sets):
    rng = np.random.default_rng(0)
    learned_like = rng.random((500, 2)).astype(np.float32)  # (S_rel, S_key) are float32 sigmoid outputs
    for scores in [*calibration_sets.values(), learned_like]:
        cut, quantile, rejection = old_joint_policy.calibrate_joint(scores)
        result = joint_thresholds(scores)
        assert isinstance(result, JointCalibration)
        assert result.thresholds.dtype == cut.dtype and result.thresholds.tobytes() == cut.tobytes()
        assert result.quantile == quantile and result.calibration_rejection == rejection
        assert result.calibration_rejection <= 0.01
        assert joint_rejection(scores, result.thresholds) == rejection


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

