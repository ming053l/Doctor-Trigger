"""doctor_trigger.enrollment reproduces workflow/scripts/enroll.py and the canonical files.

The old script's helpers (stable_key, embed, psnr, alpha_candidates and its alpha loop) are
compared through their frozen outputs in tests/golden/enroll_script.json (see golden_values.py).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from skimage.metrics import structural_similarity

from doctor_trigger.enrollment import (
    DEFAULT_ALPHA_LADDER,
    EnrollmentSettings,
    FidelityRejection,
    alpha_ladder,
    encode_release,
    enroll,
    enroll_many,
    passes_fidelity_gate,
    psnr,
    ssim,
)
from doctor_trigger.key import Ring
from doctor_trigger.record import UnitId, save_record, stored_arrays
from data_roots import ENROLLMENTS, enrollment_subset, requires
from golden_values import digest, load

REPO = Path(__file__).resolve().parents[1]


OLD_ENROLL = load("enroll_script")

NEEDS_ENROLLMENTS = requires(ENROLLMENTS)
ALL_SUBSET_ROWS = enrollment_subset()
ROWS = enrollment_subset({"FF++": 3, "DFDC": 3, "CelebDF": 2})
ROW_LIST = list(ROWS.itertuples(index=False))
SETTINGS = EnrollmentSettings()


def original_of(row) -> np.ndarray:
    return np.asarray(Image.open(row.aligned_path).convert("RGB"), dtype=np.uint8)


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def test_settings_match_enroll_cli_defaults():
    assert SETTINGS == EnrollmentSettings(17, 0.15, Ring(0.10, 0.45), 1e-6, 35.0, 0.98)
    for dataset in ("FF++", "DFDC", "CelebDF", "other"):
        assert list(alpha_ladder(dataset)) == OLD_ENROLL["alpha_candidates"][dataset]


@pytest.mark.parametrize("alpha", [0.05, 0.015, 0.5])
@pytest.mark.parametrize("row", ROW_LIST[:2] + ROW_LIST[3:4], ids=lambda row: getattr(row, "enrollment_id", row))
def test_encode_release_matches_embed(row, alpha, tmp_path):
    original = original_of(row)
    key = OLD_ENROLL["stable_key"][str(row.sample_id)]
    old = OLD_ENROLL["embed"][f"{row.enrollment_id}|{alpha}"]
    released, record = encode_release(original, key, alpha, 0.15, Ring(0.10, 0.45), 1e-6)
    assert released.dtype == np.uint8 and digest(released) == old["released"]
    new_arrays = stored_arrays(record)
    for name, (dtype, array_digest, f_contiguous) in old["record"].items():
        assert new_arrays[name].dtype.str == dtype and digest(new_arrays[name]) == array_digest, name
        # Memory layout is written into the NPZ header, so it must match too.
        assert new_arrays[name].flags.f_contiguous == f_contiguous, name


@pytest.mark.parametrize("row", list(ALL_SUBSET_ROWS.itertuples(index=False)), ids=lambda row: row.enrollment_id)
def test_enroll_reproduces_canonical_files(row, tmp_path):
    original = original_of(row)
    result = enroll(original, UnitId(str(row.sample_id)), alpha_ladder(row.dataset), SETTINGS)
    assert png_bytes(result.released) == Path(row.released_path).read_bytes()
    written = tmp_path / "record.npz"
    save_record(written, result.record)
    assert written.read_bytes() == Path(row.template_artifact).read_bytes()
    metadata = json.loads(Path(row.template_artifact).with_suffix(".json").read_text())
    assert result.alpha == metadata["alpha"]
    assert result.psnr == metadata["persisted_psnr"]
    assert result.ssim == metadata["persisted_ssim"]
    # The old script re-measures fidelity after the PNG round trip; lossless PNG gives the same.
    reread = np.asarray(Image.open(io.BytesIO(png_bytes(result.released))).convert("RGB"), dtype=np.uint8)
    assert result.psnr == OLD_ENROLL["persisted_psnr"][row.enrollment_id]
    assert result.ssim == float(structural_similarity(original, reread, channel_axis=2, data_range=255))


@NEEDS_ENROLLMENTS
def test_ladder_steps_down_like_enroll_script(tmp_path):
    # The alpha loop of workflow/scripts/enroll.py:process_enrollment on the ladder below.
    row = ROW_LIST[0]
    original = original_of(row)
    ladder = (0.9, 0.5, *DEFAULT_ALPHA_LADDER)
    old = OLD_ENROLL["ladder"]
    assert old["alpha"] != ladder[0]
    result = enroll(original, UnitId(str(row.sample_id)), ladder, SETTINGS)
    assert result.alpha == old["alpha"] and result.psnr == old["psnr"] and result.ssim == old["ssim"]
    assert digest(result.released) == old["released"]
    new_arrays = stored_arrays(result.record)
    for name, array_digest in old["record"].items():
        assert digest(new_arrays[name]) == array_digest, name


@NEEDS_ENROLLMENTS
def test_rejection_reason_matches_enroll_script():
    row = ROW_LIST[3]
    original = original_of(row)
    strict = EnrollmentSettings(minimum_psnr=99.0)
    with pytest.raises(ValueError) as raised:
        enroll(original, UnitId(str(row.sample_id)), alpha_ladder(row.dataset), strict)
    assert f"{type(raised.value).__name__}:{raised.value}" == OLD_ENROLL["rejection"]


@NEEDS_ENROLLMENTS
def test_enroll_many_streams_outcomes():
    rows = ROW_LIST[:2]
    requests = ((UnitId(str(row.sample_id)), original_of(row), alpha_ladder(row.dataset)) for row in rows)
    strict = EnrollmentSettings(minimum_psnr=99.0)
    outcomes = list(enroll_many(requests, strict))
    assert all(isinstance(outcome, FidelityRejection) for outcome in outcomes)
    assert [outcome.unit.asset_id for outcome in outcomes] == [str(row.sample_id) for row in rows]
    requests = [(UnitId(str(row.sample_id)), original_of(row), alpha_ladder(row.dataset)) for row in rows]
    for (unit, original, alphas), outcome in zip(requests, enroll_many(iter(requests))):
        single = enroll(original, unit, alphas)
        assert outcome.unit == unit and outcome.released.tobytes() == single.released.tobytes()


@NEEDS_ENROLLMENTS
def test_fidelity_gate_and_metrics():
    assert passes_fidelity_gate(35.0, 0.98, SETTINGS)
    assert not passes_fidelity_gate(34.999, 0.99, SETTINGS)
    assert not passes_fidelity_gate(40.0, 0.979, SETTINGS)
    image = original_of(ROW_LIST[0])
    assert psnr(image, image) == OLD_ENROLL["psnr_identical"] == 100.0
    noisy = np.clip(image.astype(int) + 3, 0, 255).astype(np.uint8)
    assert psnr(image, noisy) == OLD_ENROLL["psnr_noisy"]
    assert ssim(image, noisy) == float(structural_similarity(image, noisy, channel_axis=2, data_range=255))


@NEEDS_ENROLLMENTS
def test_video_units_are_not_enrolled():
    with pytest.raises(ValueError):
        enroll(original_of(ROW_LIST[0]), UnitId("asset", frame_index=0), DEFAULT_ALPHA_LADDER)
