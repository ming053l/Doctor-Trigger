"""doctor_trigger.record reads and writes the enrollment workflow's NPZ records byte for byte."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_roots import ENROLLMENTS, PIPELINE_ROOT, enrollment_subset, requires, under
from doctor_trigger.key import Ring
from doctor_trigger.record import (
    RecordStore,
    UnitId,
    enrollment_id,
    image_unit_path,
    load_record,
    save_record,
)

PUBLIC_VIEW_TEMPLATES = under(PIPELINE_ROOT, "private/templates")

ALL_SUBSET_ROWS = enrollment_subset()
ROWS = enrollment_subset({"FF++": 3, "DFDC": 3, "CelebDF": 2})
ROW_LIST = list(ROWS.itertuples(index=False))


@pytest.mark.parametrize("row", ROW_LIST, ids=lambda row: row.enrollment_id)
def test_load_record_matches_feature_row_reading(row):
    record = load_record(row.template_artifact)
    # The reading block of workflow/scripts/extract_features.py:feature_row.
    with np.load(row.template_artifact) as stored:
        assert record.template_phase.tobytes() == stored["template_phase"].astype(np.float64).tobytes()
        assert record.original_phase.tobytes() == stored["host_phase"].astype(np.float64).tobytes()
        assert record.displacement.tobytes() == stored["delta"].astype(np.float64).tobytes()
        assert record.amplitude_weights.tobytes() == stored["host_amplitude"].astype(np.float64).tobytes()
        assert np.array_equal(record.active_bins, stored["support"].astype(bool))
        assert np.array_equal(record.sign_pattern, stored["code"])
        assert record.key == int(stored["key"][0])
        assert record.alpha == float(stored["alpha"][0])
        assert record.beta == float(stored["beta"][0])
        assert record.ring == Ring(float(stored["bands"][0]), float(stored["bands"][1]))
        assert record.amplitude_floor_ratio == float(stored["amplitude_floor_ratio"][0])
    assert record.shape == record.template_phase.shape
    assert not record.template_phase.flags.writeable


@pytest.mark.parametrize("row", ROW_LIST, ids=lambda row: row.enrollment_id)
def test_save_record_round_trip_is_byte_identical(row, tmp_path):
    record = load_record(row.template_artifact)
    written = tmp_path / "record.npz"
    save_record(written, record)
    assert written.read_bytes() == Path(row.template_artifact).read_bytes()
    again = load_record(written)
    for name in ("template_phase", "original_phase", "displacement", "amplitude_weights", "active_bins", "sign_pattern"):
        assert getattr(again, name).tobytes() == getattr(record, name).tobytes()
    for name in ("key", "alpha", "beta", "ring", "amplitude_floor_ratio"):
        assert getattr(again, name) == getattr(record, name)


@requires(ENROLLMENTS)
def test_enrollment_id_matches_manifest():
    for row in ALL_SUBSET_ROWS.itertuples(index=False):
        assert enrollment_id(str(row.sample_id)) == row.enrollment_id


@requires(ENROLLMENTS, PUBLIC_VIEW_TEMPLATES)
def test_record_store_matches_today_layout():
    units = [(UnitId(str(row.sample_id)), row.dataset, row.split) for row in ALL_SUBSET_ROWS.itertuples(index=False)]
    canonical_root = Path(ALL_SUBSET_ROWS.iloc[0].template_artifact).parents[2]
    canonical = RecordStore.for_images(canonical_root, units)
    public = RecordStore.for_images(PUBLIC_VIEW_TEMPLATES, units)
    assert len(canonical) == len(ALL_SUBSET_ROWS)
    for (unit, _, _), row in zip(units, ALL_SUBSET_ROWS.itertuples(index=False)):
        assert unit in canonical
        assert canonical.path(unit) == Path(row.template_artifact)
        assert public.path(unit).is_file()
        assert public.path(unit).read_bytes() == Path(row.template_artifact).read_bytes()
    released_root = Path(ALL_SUBSET_ROWS.iloc[0].released_path).parents[2]
    for unit, dataset, split in units[:8]:
        row = ALL_SUBSET_ROWS[ALL_SUBSET_ROWS.sample_id == unit.asset_id].iloc[0]
        assert image_unit_path(released_root, dataset, split, unit, ".png") == Path(row.released_path)
    first_unit = units[0][0]
    assert canonical.load(first_unit).key == load_record(canonical.path(first_unit)).key


def test_image_layout_rejects_frames():
    with pytest.raises(ValueError):
        image_unit_path("root", "FF++", "test", UnitId("asset", frame_index=3), ".npz")
