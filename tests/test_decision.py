"""doctor_trigger.decision: Eq. (11) on the direct path, and the single-unit aggregation rule.

The expected statuses follow examples/quickstart.py:decide (C >= tau_C and G >= tau_G confirms;
missing record or query, or a size mismatch, is unavailable).
"""

from __future__ import annotations

import numpy as np
import pytest

from doctor_trigger.decision import AggregationRule, DirectThresholds, Status, aggregate_status, unit_status
from doctor_trigger.readings import UnitReading


def reading(c: float | None, g: float | None, reason: str | None = None) -> UnitReading:
    return UnitReading(None, None, c, g, 0.15, False, reason)


THRESHOLDS = DirectThresholds(tau_C=0.9, tau_G=0.5)


@pytest.mark.parametrize(
    ("c", "g", "expected"),
    [
        (0.95, 0.8, Status.CONFIRMED),
        (0.9, 0.5, Status.CONFIRMED),  # cutoffs are inclusive (>=)
        (0.89, 0.8, Status.UNCONFIRMED),
        (0.95, 0.49, Status.UNCONFIRMED),
        (0.1, -0.2, Status.UNCONFIRMED),
        (float("nan"), 0.8, Status.UNCONFIRMED),  # a NaN reading never meets a cutoff
    ],
)
def test_unit_status_direct_policy(c, g, expected):
    assert unit_status(reading(c, g), THRESHOLDS) is expected


def test_quickstart_decide_rule_agrees():
    rng = np.random.default_rng(0)
    for c, g in rng.normal([0.9, 0.5], [0.05, 0.3], size=(200, 2)):
        quickstart = "Confirmed" if c >= THRESHOLDS.tau_C and g >= THRESHOLDS.tau_G else "Unconfirmed"
        assert unit_status(reading(float(c), float(g)), THRESHOLDS).value == quickstart


def test_unavailable_regardless_of_scores():
    assert unit_status(reading(None, None, "record not resolved"), THRESHOLDS) is Status.UNAVAILABLE
    assert Status.UNAVAILABLE.value == "Unavailable"


def test_aggregate_status_single_unit():
    for status in Status:
        assert aggregate_status([status]) is status
        assert aggregate_status(iter([status]), AggregationRule.SINGLE_UNIT) is status
    with pytest.raises(ValueError):
        aggregate_status([])
    with pytest.raises(ValueError):
        aggregate_status([Status.CONFIRMED, Status.CONFIRMED])
