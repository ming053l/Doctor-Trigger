"""Verification status of a unit and of a set of units (Eq. 11, direct path).

status(J) is ``Confirmed`` when the query is eligible and C >= tau_C and G_s^(release) >= tau_G,
``Unconfirmed`` when it is eligible and either criterion fails, and ``Unavailable`` when it is not
eligible (record unresolved, face geometry invalid, channel unsupported), regardless of scores.

``aggregate_status`` turns the statuses of several units into one verdict. Images have a single
unit, so the only rule today is ``AggregationRule.SINGLE_UNIT``; a clip-level rule for video
(for example, a fraction of confirmed frames) is added to ``AggregationRule`` and
``aggregate_status`` here, calibrated on clip-level benign scores.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .readings import UnitReading


class Status(Enum):
    """The three outcomes of Eq. (11)."""

    CONFIRMED = "Confirmed"
    UNCONFIRMED = "Unconfirmed"
    UNAVAILABLE = "Unavailable"


@dataclass(frozen=True)
class DirectThresholds:
    """Cutoffs of the direct policy: tau_C on spectral agreement, tau_G on the key projection."""

    tau_C: float
    tau_G: float


def unit_status(reading: UnitReading, thresholds: DirectThresholds) -> Status:
    """status(J) of one unit on the direct path: both criteria must meet their cutoffs.

    The thresholds must be calibrated on the same reading (C or C_B; G_s^(release) or unnormalized).
    """
    if not reading.available:
        return Status.UNAVAILABLE
    if reading.spectral_agreement >= thresholds.tau_C and reading.key_projection >= thresholds.tau_G:
        return Status.CONFIRMED
    return Status.UNCONFIRMED


class AggregationRule(Enum):
    """How the statuses of several units combine into one verdict."""

    SINGLE_UNIT = "single_unit"
    """Exactly one unit (an image); its status is the verdict."""


def aggregate_status(statuses: Iterable[Status], rule: AggregationRule = AggregationRule.SINGLE_UNIT) -> Status:
    """One verdict from the statuses of a unit set under ``rule``."""
    statuses = list(statuses)
    if rule is AggregationRule.SINGLE_UNIT:
        if len(statuses) != 1:
            raise ValueError(f"single-unit rule needs exactly one status, got {len(statuses)}")
        return statuses[0]
    raise ValueError(f"unknown aggregation rule: {rule}")
