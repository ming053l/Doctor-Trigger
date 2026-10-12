"""doctor_trigger.key is bit-identical to the workflow scripts it replaces."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from doctor_trigger import key as keymod
from doctor_trigger.spectrum import kappa_grid, luminance_spectrum

REPO = Path(__file__).resolve().parents[1]
CANONICAL_ENROLLMENTS = Path(
    "/ssd2/huanghungkai/Doctor_Trigger_handoff_v2/workflow_v1/manifests/full_v1_enrollments.csv"
)
SUBSET_ENROLLMENTS = Path("/ssd2/ming/refactor_baseline/subset_enrollments.csv")
SHAPES = [(256, 256), (268, 330), (569, 512), (7, 5), (1, 4), (2, 2)]
KEYS = [0, 1, 17, 2**63 + 12345, 2**64 - 1, 0x9E3779B97F4A7C15]


def load_script(name: str):
    path = REPO / "workflow" / "scripts" / f"{name}.py"
    module_spec = importlib.util.spec_from_file_location(f"old_{name}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


OLD_ENROLL = load_script("enroll")
OLD_FEATURES = load_script("extract_features")

subset = pd.read_csv(SUBSET_ENROLLMENTS)
ROWS = pd.read_csv(CANONICAL_ENROLLMENTS).merge(subset[["enrollment_id"]], on="enrollment_id")
ROWS = pd.concat([ROWS[ROWS.dataset == name].head(count) for name, count in {"FF++": 3, "DFDC": 3, "CelebDF": 2}.items()])
ROW_LIST = list(ROWS.itertuples(index=False))


@pytest.mark.parametrize("row", ROW_LIST, ids=lambda row: row.enrollment_id)
def test_derive_key_matches_stable_key_and_record(row):
    key = keymod.derive_key(17, str(row.sample_id))
    assert key == OLD_ENROLL.stable_key(17, str(row.sample_id))
    with np.load(row.template_artifact) as record:
        assert key == int(record["key"][0])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("key", KEYS)
def test_sign_pattern_matches_both_copies(shape, key):
    expected = OLD_ENROLL.signed_hermitian_code(shape, key)
    assert expected.tobytes() == OLD_FEATURES.signed_hermitian_code(shape, key).tobytes()
    pattern = keymod.sign_pattern(shape, key)
    assert pattern.dtype == np.float64 and pattern.tobytes() == expected.tobytes()
    assert keymod.sign_pattern(shape, key, centered=True).tobytes() == np.fft.fftshift(expected).tobytes()


@pytest.mark.parametrize("row", ROW_LIST, ids=lambda row: row.enrollment_id)
def test_active_bins_phase_challenge_displacement_match_embed(row):
    # Reproduces the intermediate arrays of workflow/scripts/enroll.py:embed line by line.
    image = np.asarray(Image.open(row.aligned_path).convert("RGB"), dtype=np.uint8)
    shape = image.shape[:2]
    key = OLD_ENROLL.stable_key(17, str(row.sample_id))
    alpha, beta, bands, floor_ratio = float(row.alpha), 0.15, (0.10, 0.45), 1e-6
    code = OLD_ENROLL.signed_hermitian_code(shape, key)
    radius = OLD_ENROLL.frequency_radius(shape)
    host_amplitude = np.abs(np.fft.fft2(OLD_ENROLL.rgb_to_y(image.astype(np.float64))))
    amplitude_floor = float(np.max(host_amplitude) * floor_ratio)
    support = (radius >= bands[0]) & (radius <= bands[1]) & (host_amplitude > amplitude_floor) & (code != 0)
    preblend_shift = alpha * np.pi * code * support
    effective_delta = np.angle((1.0 - beta) + beta * np.exp(1j * preblend_shift))

    amplitude = np.abs(luminance_spectrum(image))
    active = keymod.active_bins(kappa_grid(shape, centered=False), amplitude, keymod.sign_pattern(shape, key),
                                keymod.Ring(*bands), floor_ratio)
    assert np.array_equal(active, support)
    challenge = keymod.phase_challenge(code, alpha, active)
    assert challenge.tobytes() == preblend_shift.tobytes()
    delta = keymod.displacement(code, alpha, beta, active)
    assert delta.tobytes() == effective_delta.tobytes()


@pytest.mark.parametrize("row", ROW_LIST, ids=lambda row: row.enrollment_id)
def test_displacement_matches_wrong_delta(row):
    with np.load(row.template_artifact) as record:
        support = record["support"].astype(bool)
        record_key = int(record["key"][0])
        alpha, beta = float(record["alpha"][0]), float(record["beta"][0])
    for candidate in [record_key, *OLD_FEATURES.wrong_keys(record_key, 8)]:
        expected = OLD_FEATURES.wrong_delta(candidate, support.shape, alpha, beta, support)
        pattern = keymod.sign_pattern(support.shape, candidate, centered=True)
        assert keymod.displacement(pattern, alpha, beta, support).tobytes() == expected.tobytes()


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("m", [1, 4, 8])
def test_wrong_keys_match(key, m):
    assert keymod.wrong_keys(key, m) == OLD_FEATURES.wrong_keys(key, m)
    assert keymod.wrong_keys(key) == OLD_FEATURES.wrong_keys(key, 8)


@pytest.mark.parametrize("key", KEYS)
def test_wrong_key_query_key_matches_extract_features(key):
    # extract_features.rows_for_query, benign branch.
    expected = int(hashlib.sha256(f"wrong-key\x1f{key}".encode()).hexdigest()[:16], 16)
    assert keymod.wrong_key_query_key(key) == expected
