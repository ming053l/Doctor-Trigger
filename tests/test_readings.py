"""doctor_trigger.readings reproduces the H / C / G_s readings of the code it replaced.

Old helpers are compared through their frozen outputs (see golden_values.py): those of
workflow/scripts/extract_features.py (feature_row, masked_mean, projection, query_phase,
radial_values, wrong_delta) in tests/golden/extract_features_script.json, and those of the
removed trigger_conditioned.py (compute_template_score, compute_trigger_projection,
compute_host_score) in tests/golden/trigger_conditioned.json. Experiment scripts run at
import time, so their one-line readers are copied below with their file and function name; ``wrap``
below is the wrap expression those readers use. Those readers summed over the whole grid with the
weights zeroed outside the bins; readings has one summation form (the feature pipeline's), which
agrees with them up to summation order.
Data: the 512-query regression subset (tests/data/subset_queries.csv); see tests/data_roots.py.
"""

from __future__ import annotations

import multiprocessing

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from doctor_trigger import readings
from doctor_trigger.key import wrong_key_query_key, wrong_keys
from doctor_trigger.record import UnitId, load_record
from doctor_trigger.spectrum import luminance_phase, subband_b
from data_roots import PIPELINE_ROOT, SUBSET_QUERIES, skip_module_unless, under
from golden_values import digest, features_digest, load

QUERIES = under(PIPELINE_ROOT, "manifests/queries.csv")
skip_module_unless(QUERIES)


OLD_FEATURES = load("extract_features_script")
OLD_TRIGGER = load("trigger_conditioned")


def subset_queries() -> pd.DataFrame:
    queries = pd.read_csv(QUERIES)
    return queries[queries.status == "pass"].merge(pd.read_csv(SUBSET_QUERIES), on="query_id")


def wrap(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def image_phase(path: str) -> np.ndarray:
    """extract_features.query_phase: centered luminance phase of an image file."""
    return luminance_phase(np.asarray(Image.open(path).convert("RGB"), dtype=np.float64), centered=True)


# ----------------------------------------------------------------------------- record_features
SUBSET_LIST = list(subset_queries().itertuples(index=False))


def _compare_query(index: int) -> list[str]:
    """Rows of one query as workflow/scripts/extract_features.py:rows_for_query built them."""
    query = SUBSET_LIST[index]
    reference = image_phase(query.released_path)
    phase = reference if query.query_path == query.released_path else image_phase(query.query_path)
    record = load_record(query.template_artifact)
    keys = [None] + ([wrong_key_query_key(record.key)] if query.protocol == "benign" else [])
    expected = OLD_FEATURES["feature_rows"][query.query_id]
    if [row["projection_key"] for row in expected] != keys:
        return [f"{query.query_id}: projection keys differ"]
    problems = []
    for projection_key, old in zip(keys, expected):
        new = readings.record_features(phase, record, reference, projection_key=projection_key)
        if old["key"] != record.key or list(new) != OLD_FEATURES["feature_columns"]:
            problems.append(f"{query.query_id}/{projection_key}: columns or key differ")
        if features_digest(new) != old["features"]:
            problems.append(f"{query.query_id}/{projection_key}: values differ")
    return problems


def test_record_features_equal_feature_row_on_the_512_query_subset():
    assert len(SUBSET_LIST) == 512
    with multiprocessing.get_context("fork").Pool(16) as pool:
        problems = [problem for result in pool.map(_compare_query, range(len(SUBSET_LIST))) for problem in result]
    assert problems == []


def test_feature_columns_are_the_csv_columns():
    assert readings.FEATURE_COLUMNS == tuple(OLD_FEATURES["feature_columns"])
    assert len(readings.FEATURE_COLUMNS) == 17 + 8 * 9


def test_record_features_without_reference_uses_the_template():
    query = subset_queries().iloc[1]
    phase = image_phase(query.query_path)
    new = readings.record_features(phase, load_record(query.template_artifact))
    assert features_digest(new) == OLD_FEATURES["without_reference"]


def test_record_features_shape_mismatch_raises():
    record = load_record(subset_queries().iloc[0].template_artifact)
    with pytest.raises(ValueError, match="shape mismatch"):
        readings.record_features(np.zeros((3, 3)), record)


# ----------------------------------------------------------------------------- per-unit pieces
SAMPLE = subset_queries().groupby("dataset", group_keys=False).head(4)
SAMPLE_LIST = list(SAMPLE.itertuples(index=False))


def rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def stored(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as record:
        return {name: record[name] for name in record.files}


@pytest.mark.parametrize("query", SAMPLE_LIST, ids=lambda query: query.query_id)
def test_wrong_key_displacement_and_pixel_radius(query):
    record = load_record(query.template_artifact)
    expected = OLD_FEATURES["sample"][query.query_id]
    for candidate, old in zip(wrong_keys(record.key, 3), expected["wrong_delta"], strict=True):
        assert digest(readings.wrong_key_displacement(record, candidate)) == old
    assert digest(readings.normalized_pixel_radius(record.shape)) == expected["radial_values"]


@pytest.mark.parametrize("query", SAMPLE_LIST, ids=lambda query: query.query_id)
def test_pipeline_kernels_match_extract_features(query):
    record = load_record(query.template_artifact)
    expected = OLD_FEATURES["sample"][query.query_id]
    phase = image_phase(query.query_path)
    host, weights, active = record.original_phase, record.amplitude_weights, record.active_bins
    drift = readings.phase_drift(phase, host)
    assert digest(drift) == expected["drift"]
    assert readings.spectral_agreement(phase, host, weights, active) == expected["masked_mean_cos_drift"]
    assert readings.key_projection(drift, record.displacement, weights, active) == expected["projection"]
    assert np.isnan(readings.amplitude_weighted_mean(drift, weights, np.zeros_like(active)))


SUMMATION_ORDER_TOLERANCE = 1e-12
"""Relative difference allowed between a reading and the same sum taken in another order."""


def close(new: float, old: float, tolerance: float = SUMMATION_ORDER_TOLERANCE) -> bool:
    return abs(new - old) <= tolerance * abs(old)


@pytest.mark.parametrize("query", SAMPLE_LIST, ids=lambda query: query.query_id)
def test_readings_agree_with_former_experiment_forms(query):
    """The experiments' former whole-grid sums give the same readings up to summation order.

    Their denominator stabilizers are set to eps_0 = 1e-12 here; the 1e-8 some agreements added
    moves them by about 2e-9 relative (sum w is 7 to 30), which no printed value resolves.
    """
    record = load_record(query.template_artifact)
    arrays = stored(query.template_artifact)
    host = arrays["host_phase"].astype(np.float64)
    tmpl = arrays["template_phase"].astype(np.float64)
    sup = arrays["support"].astype(bool)
    w = arrays["host_amplitude"].astype(np.float64)
    delta = arrays["delta"].astype(np.float64)
    pil = Image.open(query.query_path).convert("RGB")
    phase = readings.query_phase(pil, record.shape)
    weights = record.amplitude_weights
    drift = readings.phase_drift(phase, record.original_phase)

    # full ring and sub-band B, whole-grid weight map (benign_channels, other_manipulations,
    # reference_baselines, table4_artifact, operating_points, subband_table1)
    band = np.sqrt(np.fft.fftshift(np.fft.fftfreq(sup.shape[0])).reshape(-1, 1) ** 2
                   + np.fft.fftshift(np.fft.fftfreq(sup.shape[1])).reshape(1, -1) ** 2) / np.sqrt(0.5) <= 0.15
    for bins, ww in ((record.active_bins, np.where(sup, w, 0.0)),
                     (readings.subband_bins(record), np.where(sup & band, w, 0.0))):
        for reference, record_reference in ((host, record.original_phase), (tmpl, record.template_phase)):
            old = float(np.sum(np.cos(wrap(phase - reference)) * ww) / (np.sum(ww) + 1e-12))
            assert close(readings.spectral_agreement(phase, record_reference, weights, bins), old)
    bins = readings.subband_bins(record)
    # the same in the uncentered layout (operating_points, subband_table1)
    uncentered_phase = readings.query_phase(pil, record.shape, centered=False)
    uncentered_bins = np.fft.ifftshift(record.active_bins) & subband_b(record.shape, centered=False)
    assert close(readings.spectral_agreement(uncentered_phase, np.fft.ifftshift(record.template_phase),
                                             np.fft.ifftshift(weights), uncentered_bins),
                 readings.spectral_agreement(phase, record.template_phase, weights, bins))

    # G_s over the whole grid (record_codec, build_fig_keymatch)
    ww_ring = np.where(sup, w, 0.0)
    old = float(np.sum(ww_ring * delta * np.sin(wrap(phase - host)))) / (float(np.sum(ww_ring * delta * delta)) + 1e-12)
    assert close(readings.key_projection(drift, record.displacement, weights, record.active_bins), old)
    # table4_artifact g_raw_sub and build_fig_signal wrote the denominator w * delta ** 2
    old = float(np.sum(w[bins] * delta[bins] * np.sin(wrap(phase - host)[bins]))
                / (np.sum(w[bins] * delta[bins] ** 2) + 1e-12))
    assert close(readings.key_projection(drift, record.displacement, weights, bins), old)

    # examples/quickstart.py, formerly through trigger_conditioned (frozen values): C_B against the
    # template, G_s with unit weights, and H with a zero-filled weight map
    former = OLD_TRIGGER[query.query_id]
    assert close(readings.spectral_agreement(phase, record.template_phase, weights, bins),
                 former["compute_template_score"], 1e-9)
    assert close(readings.key_projection(drift, record.displacement, np.ones(record.shape), record.active_bins),
                 former["compute_trigger_projection"], 1e-9)
    assert close(readings.spectral_agreement(phase, record.original_phase, weights, record.active_bins),
                 former["compute_host_score"], 1e-9)


@pytest.mark.parametrize("query", SAMPLE_LIST, ids=lambda query: query.query_id)
def test_original_control_template_normalization(query):
    # experiments/original_control.py: G_s^(template) via extract_features.projection (frozen values)
    record = load_record(query.template_artifact)
    ph = readings.query_phase(Image.open(query.query_path), record.shape)
    weights = record.amplitude_weights
    expected = OLD_FEATURES["sample"][query.query_id]
    template_scale = max(abs(expected["projection_template_stored"]), 1e-6)
    drift = readings.phase_drift(ph, record.original_phase)
    template_drift = readings.phase_drift(record.template_phase, record.original_phase)
    assert readings.release_normalized(
        readings.key_projection(drift, record.displacement, weights, record.active_bins),
        readings.key_projection(template_drift, record.displacement, weights, record.active_bins),
    ) == expected["projection_query_stored"] / template_scale


@pytest.mark.parametrize("query", SAMPLE_LIST[:4], ids=lambda query: query.query_id)
def test_query_phase_size_mismatch_policies(query):
    record = load_record(query.template_artifact)
    pil = Image.open(query.query_path).convert("RGB")
    small = pil.resize((pil.size[0] - 3, pil.size[1] - 2))
    with pytest.raises(ValueError, match="mismatch"):
        readings.query_phase(small, record.shape)
    assert readings.query_phase(small, record.shape, readings.SizeMismatch.UNAVAILABLE) is None
    # experiments/benign_canonical.py:H_of resizes with PIL's default; table4_artifact.py:phase_resized with BICUBIC
    for old_resized in (small.resize((record.shape[1], record.shape[0])),
                        small.resize((record.shape[1], record.shape[0]), Image.BICUBIC)):
        expected = np.fft.fftshift(np.angle(np.fft.fft2(
            0.299 * rgb_of(old_resized)[:, :, 0] + 0.587 * rgb_of(old_resized)[:, :, 1] + 0.114 * rgb_of(old_resized)[:, :, 2])))
        assert readings.query_phase(small, record.shape, readings.SizeMismatch.RESIZE).tobytes() == expected.tobytes()


def rgb_of(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float64)


def test_wrong_key_statistics_match_feature_row_block():
    rng = np.random.default_rng(3)
    correct, alternatives = float(rng.normal()), rng.normal(size=8)
    stats = readings.wrong_key_statistics(correct, alternatives)
    assert stats == readings.WrongKeyStats(
        mean=float(np.mean(alternatives)), std=float(np.std(alternatives)), max=float(np.max(alternatives)),
        margin=float(correct - np.max(alternatives)),
        z=float((correct - np.mean(alternatives)) / (np.std(alternatives) + 1e-8)))


# ----------------------------------------------------------------------------- read_unit
def test_read_unit_released_query_and_unavailable_cases():
    query = SAMPLE.iloc[0]
    record = load_record(query.template_artifact)
    released = np.asarray(Image.open(query.released_path).convert("RGB"))
    reading = readings.read_unit(released, record, unit=UnitId(query.sample_id), release_rgb=released)
    assert reading.available and reading.release_normalized and reading.subband_kappa_max == 0.15
    # A released crop read against its own release projection is 1 by construction.
    assert reading.key_projection == 1.0
    assert reading.spectral_agreement == OLD_FEATURES["read_unit"]["subband_template"]
    assert readings.read_unit(released, None).unavailable_reason == "record not resolved"
    assert not readings.read_unit(None, record).available
    assert not readings.read_unit(released[1:], record).available
    full_ring = readings.read_unit(released, record, subband_kappa_max=None)
    assert full_ring.original_reference_agreement == OLD_FEATURES["read_unit"]["full_ring_original"]
