"""doctor_trigger.key is bit-identical to the workflow scripts it replaces.

The old helpers (enroll.py stable_key / signed_hermitian_code / frequency_radius / rgb_to_y,
extract_features.py wrong_keys / wrong_delta) are compared through their frozen outputs in
tests/golden/ (see golden_values.py).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from doctor_trigger import key as keymod
from doctor_trigger.spectrum import kappa_grid, luminance_spectrum
from data_roots import enrollment_subset
from golden_values import digest, load

REPO = Path(__file__).resolve().parents[1]
SHAPES = [(256, 256), (268, 330), (569, 512), (7, 5), (1, 4), (2, 2)]
KEYS = [0, 1, 17, 2**63 + 12345, 2**64 - 1, 0x9E3779B97F4A7C15]


OLD_ENROLL = load("enroll_script")
OLD_FEATURES = load("extract_features_script")

ROWS = enrollment_subset({"FF++": 3, "DFDC": 3, "CelebDF": 2})
ROW_LIST = list(ROWS.itertuples(index=False))


@pytest.mark.parametrize("row", ROW_LIST, ids=lambda row: row.enrollment_id)
def test_derive_key_matches_stable_key_and_record(row):
    key = keymod.derive_key(17, str(row.sample_id))
    assert key == OLD_ENROLL["stable_key"][str(row.sample_id)]
    with np.load(row.template_artifact) as record:
        assert key == int(record["key"][0])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("key", KEYS)
def test_sign_pattern_matches_both_copies(shape, key):
    # enroll.py and extract_features.py had byte-identical copies; both gave these digests.
    expected, expected_centered = OLD_ENROLL["sign_pattern"][f"{shape}|{key}"]
    pattern = keymod.sign_pattern(shape, key)
    assert pattern.dtype == np.float64 and digest(pattern) == expected
    assert digest(keymod.sign_pattern(shape, key, centered=True)) == expected_centered


@pytest.mark.parametrize("row", ROW_LIST, ids=lambda row: row.enrollment_id)
def test_active_bins_phase_challenge_displacement_match_embed(row):
    # The intermediate arrays of workflow/scripts/enroll.py:embed (support, preblend_shift, effective_delta).
    expected = OLD_ENROLL["embed_arrays"][row.enrollment_id]
    image = np.asarray(Image.open(row.aligned_path).convert("RGB"), dtype=np.uint8)
    shape = image.shape[:2]
    key = OLD_ENROLL["stable_key"][str(row.sample_id)]
    alpha, beta, bands, floor_ratio = float(row.alpha), 0.15, (0.10, 0.45), 1e-6
    code = keymod.sign_pattern(shape, key)
    amplitude = np.abs(luminance_spectrum(image))
    active = keymod.active_bins(kappa_grid(shape, centered=False), amplitude, code, keymod.Ring(*bands), floor_ratio)
    assert digest(active) == expected["support"]
    challenge = keymod.phase_challenge(code, alpha, active)
    assert digest(challenge) == expected["preblend_shift"]
    delta = keymod.displacement(code, alpha, beta, active)
    assert digest(delta) == expected["effective_delta"]


@pytest.mark.parametrize("row", ROW_LIST, ids=lambda row: row.enrollment_id)
def test_displacement_matches_wrong_delta(row):
    with np.load(row.template_artifact) as record:
        support = record["support"].astype(bool)
        record_key = int(record["key"][0])
        alpha, beta = float(record["alpha"][0]), float(record["beta"][0])
    expected = OLD_FEATURES["wrong_delta"][row.enrollment_id]
    assert [candidate for candidate, _ in expected] == [record_key, *keymod.wrong_keys(record_key, 8)]
    for candidate, expected_digest in expected:
        pattern = keymod.sign_pattern(support.shape, candidate, centered=True)
        assert digest(keymod.displacement(pattern, alpha, beta, support)) == expected_digest


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("m", [1, 4, 8])
def test_wrong_keys_match(key, m):
    assert keymod.wrong_keys(key, m) == OLD_FEATURES["wrong_keys"][f"{key}|{m}"]
    assert keymod.wrong_keys(key) == OLD_FEATURES["wrong_keys"][f"{key}|8"]


@pytest.mark.parametrize("key", KEYS)
def test_wrong_key_query_key_matches_extract_features(key):
    # extract_features.rows_for_query, benign branch.
    expected = int(hashlib.sha256(f"wrong-key\x1f{key}".encode()).hexdigest()[:16], 16)
    assert keymod.wrong_key_query_key(key) == expected
