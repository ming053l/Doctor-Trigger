"""Thresholds from benign calibration scores (Sec. 3.3).

Calibration fixes the tolerance for normal variation on benign training queries. A relation-only
cutoff (tau_H, tau_C on H_B / C_B, or tau_rel, tau_key on the record-model scores) is the lower-tail
quantile at the target false-rejection rate (FRR, first percentile by default). The direct joint
policy (C >= tau_C and G_s^(release) >= tau_G) shares one lower-tail quantile between its criteria,
the largest that keeps the joint calibration rejection within the FRR budget.

Functions work on any score array, so clip-level scores calibrate the same way as image scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_FRR = 0.01
"""Target benign false-rejection rate: the first percentile."""
JOINT_BISECTION_STEPS = 60
"""Bisection steps for the shared quantile; 60 halvings of [0, 0.01] reach float64 resolution."""


def lower_tail_threshold(scores: np.ndarray, frr: float = DEFAULT_FRR, axis: int | None = None):
    """Cutoff tau that rejects a fraction ``frr`` of benign calibration scores (scores below tau fail).

    ``np.quantile`` with linear interpolation, returned unchanged (a NumPy scalar, or an array with
    ``axis``), so cutoffs written by the experiments keep their exact values. With ``axis=0`` each
    column gets its own cutoff (the per-head policy for (S_rel, S_key)).
    """
    return np.quantile(scores, frr, axis=axis)


@dataclass(frozen=True, eq=False)
class JointCalibration:
    """Shared-quantile cutoffs of a joint policy and how they behave on the calibration scores."""

    thresholds: np.ndarray
    """One cutoff per criterion, e.g. (tau_C, tau_G) or (tau_rel, tau_key)."""
    quantile: float
    """The shared lower-tail quantile the cutoffs were taken at."""
    calibration_rejection: float
    """Fraction of calibration queries failing at least one criterion (at most the FRR budget)."""


def joint_rejection(scores: np.ndarray, thresholds: np.ndarray) -> float:
    """Fraction of rows of ``scores`` (n, criteria) below the cutoff on at least one criterion."""
    return float(np.any(scores < thresholds, axis=1).mean())


def joint_thresholds(
    scores: np.ndarray,
    frr: float = DEFAULT_FRR,
    steps: int = JOINT_BISECTION_STEPS,
) -> JointCalibration:
    """Joint rule of Sec. 3.3: one lower-tail quantile q for every criterion, the largest q in [0, frr]
    whose joint calibration rejection stays within ``frr``, found by bisection.

    ``scores`` is (n, criteria) of benign calibration queries, e.g. columns (C, G_s^(release)).
    """
    low, high = 0.0, frr
    for _ in range(steps):
        middle = (low + high) / 2
        cut = np.quantile(scores, middle, axis=0, method="linear")
        if np.any(scores < cut, axis=1).mean() <= frr:
            low = middle
        else:
            high = middle
    thresholds = np.quantile(scores, low, axis=0, method="linear")
    return JointCalibration(thresholds, low, joint_rejection(scores, thresholds))
