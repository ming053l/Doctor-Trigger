"""doctor_trigger.enrollment reproduces workflow/scripts/enroll.py and the canonical files."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
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

REPO = Path(__file__).resolve().parents[1]
CANONICAL_ENROLLMENTS = Path(
    "/ssd2/huanghungkai/Doctor_Trigger_handoff_v2/workflow_v1/manifests/full_v1_enrollments.csv"
)
SUBSET_ENROLLMENTS = Path("/ssd2/ming/refactor_baseline/subset_enrollments.csv")


def load_script(name: str):
    path = REPO / "workflow" / "scripts" / f"{name}.py"
    module_spec = importlib.util.spec_from_file_location(f"old_{name}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


OLD_ENROLL = load_script("enroll")

subset = pd.read_csv(SUBSET_ENROLLMENTS)
ALL_SUBSET_ROWS = pd.read_csv(CANONICAL_ENROLLMENTS).merge(subset[["enrollment_id"]], on="enrollment_id")
ROWS = pd.concat([ALL_SUBSET_ROWS[ALL_SUBSET_ROWS.dataset == name].head(count) for name, count in {"FF++": 3, "DFDC": 3, "CelebDF": 2}.items()])
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
        assert list(alpha_ladder(dataset)) == OLD_ENROLL.alpha_candidates(dataset)


@pytest.mark.parametrize("alpha", [0.05, 0.015, 0.5])
@pytest.mark.parametrize("row", ROW_LIST[:2] + ROW_LIST[3:4], ids=lambda row: getattr(row, "enrollment_id", row))
def test_encode_release_matches_embed(row, alpha, tmp_path):
    original = original_of(row)
    key = OLD_ENROLL.stable_key(17, str(row.sample_id))
    old_released, old_record = OLD_ENROLL.embed(original, key, alpha, 0.15, (0.10, 0.45), 1e-6)
    released, record = encode_release(original, key, alpha, 0.15, Ring(0.10, 0.45), 1e-6)
    assert released.dtype == np.uint8 and released.tobytes() == old_released.tobytes()
    new_arrays = stored_arrays(record)
    for name, array in old_record.items():
        assert new_arrays[name].dtype == array.dtype and new_arrays[name].tobytes() == array.tobytes(), name
        # Memory layout is written into the NPZ header, so it must match too.
        assert new_arrays[name].flags.f_contiguous == array.flags.f_contiguous, name


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
    assert result.psnr == OLD_ENROLL.psnr(original, reread)
    assert result.ssim == float(structural_similarity(original, reread, channel_axis=2, data_range=255))


def old_ladder(original, key, alphas, minimum_psnr, minimum_ssim):
    """The alpha loop of workflow/scripts/enroll.py:process_enrollment."""
    attempts = []
    for alpha in alphas:
        released, template = OLD_ENROLL.embed(original, key, alpha, 0.15, (0.10, 0.45), 1e-6)
        candidate_psnr = OLD_ENROLL.psnr(original, released)
        candidate_ssim = float(structural_similarity(original, released, channel_axis=2, data_range=255))
        attempts.append({"alpha": alpha, "psnr": candidate_psnr, "ssim": candidate_ssim})
        if candidate_psnr >= minimum_psnr and candidate_ssim >= minimum_ssim:
            return alpha, released, template, candidate_psnr, candidate_ssim
    return f"quality_gate_failed:{attempts}"


def test_ladder_steps_down_like_enroll_script(tmp_path):
    row = ROW_LIST[0]
    original = original_of(row)
    key = OLD_ENROLL.stable_key(17, str(row.sample_id))
    ladder = (0.9, 0.5, *DEFAULT_ALPHA_LADDER)
    alpha, released, template, old_psnr, old_ssim = old_ladder(original, key, ladder, 35.0, 0.98)
    assert alpha != ladder[0]
    result = enroll(original, UnitId(str(row.sample_id)), ladder, SETTINGS)
    assert result.alpha == alpha and result.psnr == old_psnr and result.ssim == old_ssim
    assert result.released.tobytes() == released.tobytes()
    new_arrays = stored_arrays(result.record)
    for name, array in template.items():
        assert new_arrays[name].tobytes() == array.tobytes(), name


def test_rejection_reason_matches_enroll_script():
    row = ROW_LIST[3]
    original = original_of(row)
    key = OLD_ENROLL.stable_key(17, str(row.sample_id))
    strict = EnrollmentSettings(minimum_psnr=99.0)
    expected = old_ladder(original, key, alpha_ladder(row.dataset), 99.0, 0.98)
    with pytest.raises(ValueError) as raised:
        enroll(original, UnitId(str(row.sample_id)), alpha_ladder(row.dataset), strict)
    assert f"{type(raised.value).__name__}:{raised.value}" == f"ValueError:{expected}"


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


def test_fidelity_gate_and_metrics():
    assert passes_fidelity_gate(35.0, 0.98, SETTINGS)
    assert not passes_fidelity_gate(34.999, 0.99, SETTINGS)
    assert not passes_fidelity_gate(40.0, 0.979, SETTINGS)
    image = original_of(ROW_LIST[0])
    assert psnr(image, image) == OLD_ENROLL.psnr(image, image) == 100.0
    noisy = np.clip(image.astype(int) + 3, 0, 255).astype(np.uint8)
    assert psnr(image, noisy) == OLD_ENROLL.psnr(image, noisy)
    assert ssim(image, noisy) == float(structural_similarity(image, noisy, channel_axis=2, data_range=255))


def test_video_units_are_not_enrolled():
    with pytest.raises(ValueError):
        enroll(original_of(ROW_LIST[0]), UnitId("asset", frame_index=0), DEFAULT_ALPHA_LADDER)
