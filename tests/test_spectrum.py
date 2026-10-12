"""doctor_trigger.spectrum is bit-identical to the workflow scripts it replaces.

The old helpers (enroll.py rgb_to_y / wrap_phase / frequency_radius, extract_features.py
query_phase / wrap_phase) are compared through their frozen outputs in tests/golden/
(see golden_values.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from doctor_trigger import spectrum
from golden_values import digest, load

REPO = Path(__file__).resolve().parents[1]
CANONICAL_ENROLLMENTS = Path(
    "/ssd2/huanghungkai/Doctor_Trigger_handoff_v2/workflow_v1/manifests/full_v1_enrollments.csv"
)
SUBSET_ENROLLMENTS = Path("/ssd2/ming/refactor_baseline/subset_enrollments.csv")
SHAPES = [(256, 256), (268, 330), (569, 512), (7, 5), (1, 4)]


OLD_ENROLL = load("enroll_script")
OLD_FEATURES = load("extract_features_script")


def subset_rows(per_dataset: dict[str, int]) -> pd.DataFrame:
    subset = pd.read_csv(SUBSET_ENROLLMENTS)
    rows = pd.read_csv(CANONICAL_ENROLLMENTS).merge(subset[["enrollment_id"]], on="enrollment_id")
    return pd.concat([rows[rows.dataset == name].head(count) for name, count in per_dataset.items()])


ROWS = subset_rows({"FF++": 3, "DFDC": 3, "CelebDF": 2})


@pytest.fixture(scope="module", params=list(ROWS.itertuples(index=False)), ids=lambda row: row.enrollment_id)
def row(request):
    return request.param


def test_luminance_matches_rgb_to_y(row):
    image = np.asarray(Image.open(row.aligned_path).convert("RGB"), dtype=np.uint8)
    expected = OLD_ENROLL["rgb_to_y"][row.enrollment_id]
    assert digest(spectrum.luminance(image)) == expected
    assert digest(spectrum.luminance(image.astype(np.float64))) == expected


def test_luminance_spectrum_matches_enroll(row):
    image = np.asarray(Image.open(row.aligned_path).convert("RGB"), dtype=np.uint8)
    assert digest(spectrum.luminance_spectrum(image)) == OLD_ENROLL["luminance_fft"][row.enrollment_id]


def test_luminance_phase_matches_query_phase(row):
    released = np.asarray(Image.open(row.released_path).convert("RGB"))
    centered = spectrum.luminance_phase(released, centered=True)
    assert digest(centered) == OLD_FEATURES["query_phase_released"][row.enrollment_id]
    assert np.array_equal(spectrum.luminance_phase(released, centered=False), np.fft.ifftshift(centered))


def test_wrap_phase_matches_both_copies():
    values = np.concatenate(
        [
            np.random.default_rng(0).uniform(-20.0, 20.0, 10000),
            np.array([-np.pi, np.pi, 0.0, -0.0, 3 * np.pi, -3 * np.pi, 1e-300, np.nextafter(np.pi, 0)]),
        ]
    )
    result = spectrum.wrap_phase(values)
    # enroll.py and extract_features.py had byte-identical copies; both gave this digest.
    assert digest(result) == OLD_ENROLL["wrap_phase"]


@pytest.mark.parametrize("shape", SHAPES)
def test_kappa_grid_matches_frequency_radius(shape):
    expected = OLD_ENROLL["frequency_radius"][str(shape)]
    assert digest(spectrum.kappa_grid(shape, centered=False)) == expected["radius"]
    assert digest(spectrum.kappa_grid(shape, centered=True)) == expected["radius_centered"]


@pytest.mark.parametrize("shape", SHAPES)
def test_kappa_grid_centered_matches_shifted_frequencies(shape):
    # The experiments' centered variant shifts each frequency axis, then takes the radius.
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0])).reshape(-1, 1)
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1])).reshape(1, -1)
    expected = np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5)
    assert spectrum.kappa_grid(shape, centered=True).tobytes() == expected.tobytes()


@pytest.mark.parametrize("shape", SHAPES)
def test_subband_b(shape):
    expected = OLD_ENROLL["frequency_radius"][str(shape)]
    assert digest(spectrum.subband_b(shape, centered=False)) == expected["subband"]
    assert digest(spectrum.subband_b(shape, centered=True)) == expected["subband_centered"]
    assert spectrum.SUBBAND_B_KAPPA_MAX == 0.15
