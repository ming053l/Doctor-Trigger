"""Evaluation metrics: ROC AUC, TPR at a fixed FPR, equal error rate, robust z-score.

Labels are 1 for the positive class (larger scores should mean positive). The ROC metrics drop
non-finite scores first and return NaN when only one class remains; on finite scores with both
classes they equal plain ``sklearn`` calls.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROBUST_SCALE_STABILIZER = 1e-6
"""Added to the median absolute deviation in ``robust_z``."""


def _finite_pairs(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return labels, scores


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve."""
    pairs = _finite_pairs(labels, scores)
    return float("nan") if pairs is None else float(roc_auc_score(*pairs))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Largest true-positive rate on the ROC curve at a false-positive rate of at most ``max_fpr``."""
    pairs = _finite_pairs(labels, scores)
    if pairs is None:
        return float("nan")
    fpr, tpr, _ = roc_curve(*pairs)
    keep = tpr[fpr <= float(max_fpr)]
    return float(np.max(keep)) if keep.size else 0.0


def equal_error_rate(
    labels: np.ndarray,
    scores: np.ndarray,
    rounding: Literal["fpr + fnr", "fpr + 1 - tpr"] = "fpr + fnr",
) -> float:
    """EER: (FPR + FNR) / 2 at the ROC point where |FPR - FNR| is smallest.

    ``rounding`` picks the float evaluation order of the sum, which can differ in the last bit:
    ``"fpr + fnr"`` (FNR = 1 - TPR first; the record-model trainer) or ``"fpr + 1 - tpr"``
    (``key_pure_canonical``).
    """
    pairs = _finite_pairs(labels, scores)
    if pairs is None:
        return float("nan")
    fpr, tpr, _ = roc_curve(*pairs)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    if rounding == "fpr + 1 - tpr":
        return float((fpr[index] + 1 - tpr[index]) / 2)
    if rounding == "fpr + fnr":
        return float((fpr[index] + fnr[index]) / 2.0)
    raise ValueError(f"unknown rounding: {rounding}")


def robust_z(values: np.ndarray, calibration: np.ndarray, stabilizer: float = ROBUST_SCALE_STABILIZER) -> np.ndarray:
    """(v - median) / (MAD + 1e-6), with median and median absolute deviation of the calibration scores."""
    calibration = np.asarray(calibration, dtype=np.float64)
    median = np.median(calibration)
    spread = np.median(np.abs(calibration - median)) + stabilizer
    return (np.asarray(values, dtype=np.float64) - median) / spread
