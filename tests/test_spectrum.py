"""doctor_trigger.spectrum is bit-identical to the workflow scripts it replaces."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from doctor_trigger import spectrum

REPO = Path(__file__).resolve().parents[1]
CANONICAL_ENROLLMENTS = Path(
    "/ssd2/huanghungkai/Doctor_Trigger_handoff_v2/workflow_v1/manifests/full_v1_enrollments.csv"
)
SUBSET_ENROLLMENTS = Path("/ssd2/ming/refactor_baseline/subset_enrollments.csv")
SHAPES = [(256, 256), (268, 330), (569, 512), (7, 5), (1, 4)]


def load_script(name: str):
    path = REPO / "workflow" / "scripts" / f"{name}.py"
    module_spec = importlib.util.spec_from_file_location(f"old_{name}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


OLD_ENROLL = load_script("enroll")
OLD_FEATURES = load_script("extract_features")


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
    expected = OLD_ENROLL.rgb_to_y(image.astype(np.float64))
    assert np.array_equal(spectrum.luminance(image), expected)
    assert np.array_equal(spectrum.luminance(image.astype(np.float64)), expected)


def test_luminance_spectrum_matches_enroll(row):
    image = np.asarray(Image.open(row.aligned_path).convert("RGB"), dtype=np.uint8)
    expected = np.fft.fft2(OLD_ENROLL.rgb_to_y(image.astype(np.float64)))
    assert np.array_equal(spectrum.luminance_spectrum(image), expected)


def test_luminance_phase_matches_query_phase(row):
    expected = OLD_FEATURES.query_phase(Path(row.released_path))
    released = np.asarray(Image.open(row.released_path).convert("RGB"))
    centered = spectrum.luminance_phase(released, centered=True)
    assert np.array_equal(centered, expected)
    assert np.array_equal(spectrum.luminance_phase(released, centered=False), np.fft.ifftshift(expected))


def test_wrap_phase_matches_both_copies():
    values = np.concatenate(
        [
            np.random.default_rng(0).uniform(-20.0, 20.0, 10000),
            np.array([-np.pi, np.pi, 0.0, -0.0, 3 * np.pi, -3 * np.pi, 1e-300, np.nextafter(np.pi, 0)]),
        ]
    )
    result = spectrum.wrap_phase(values)
    assert result.tobytes() == OLD_ENROLL.wrap_phase(values).tobytes()
    assert result.tobytes() == OLD_FEATURES.wrap_phase(values).tobytes()


@pytest.mark.parametrize("shape", SHAPES)
def test_kappa_grid_matches_frequency_radius(shape):
    expected = OLD_ENROLL.frequency_radius(shape)
    assert spectrum.kappa_grid(shape, centered=False).tobytes() == expected.tobytes()
    assert spectrum.kappa_grid(shape, centered=True).tobytes() == np.fft.fftshift(expected).tobytes()


@pytest.mark.parametrize("shape", SHAPES)
def test_kappa_grid_centered_matches_shifted_frequencies(shape):
    # The experiments' centered variant shifts each frequency axis, then takes the radius.
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0])).reshape(-1, 1)
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1])).reshape(1, -1)
    expected = np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5)
    assert spectrum.kappa_grid(shape, centered=True).tobytes() == expected.tobytes()


@pytest.mark.parametrize("shape", SHAPES)
def test_subband_b(shape):
    expected = OLD_ENROLL.frequency_radius(shape) <= 0.15
    assert np.array_equal(spectrum.subband_b(shape, centered=False), expected)
    assert np.array_equal(spectrum.subband_b(shape, centered=True), np.fft.fftshift(expected))
    assert spectrum.SUBBAND_B_KAPPA_MAX == 0.15
