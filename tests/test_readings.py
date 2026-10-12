"""doctor_trigger.readings is bit-identical to the H / C / G_s copies it replaces (plan Sec. 2.6-2.7).

workflow/record_model/src/trigger_conditioned.py is imported by file path. The old helpers of
workflow/scripts/extract_features.py (feature_row, masked_mean, projection, query_phase,
radial_values, wrong_delta) are compared through their frozen outputs in
tests/golden/extract_features_script.json (see golden_values.py). Experiment scripts run at
import time, so their one-line readers are copied verbatim below with their file and function name;
``wrap`` below is the wrap expression those readers use.
Data: the 512-query regression subset (/ssd2/ming/refactor_baseline/subset_queries.csv) on public_view.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from doctor_trigger import readings
from doctor_trigger.key import wrong_key_query_key, wrong_keys
from doctor_trigger.record import UnitId, load_record
from doctor_trigger.spectrum import luminance_phase, subband_b
from golden_values import digest, features_digest, load

REPO = Path(__file__).resolve().parents[1]
QUERIES = Path("/ssd2/ming/ttf_run/public_view/manifests/queries.csv")
SUBSET_QUERIES = Path("/ssd2/ming/refactor_baseline/subset_queries.csv")
pytestmark = pytest.mark.skipif(not (QUERIES.exists() and SUBSET_QUERIES.exists()), reason="public_view not available")


def load_by_path(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module  # dataclasses in the module look themselves up here
    module_spec.loader.exec_module(module)
    return module


OLD_FEATURES = load("extract_features_script")
OLD_TRIGGER = load_by_path("old_trigger_conditioned", REPO / "workflow/record_model/src/trigger_conditioned.py")


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


@pytest.mark.parametrize("query", SAMPLE_LIST, ids=lambda query: query.query_id)
def test_weight_map_forms_match_experiment_copies(query):
    path, template_file = query.query_path, query.template_artifact
    record = load_record(template_file)
    arrays = stored(template_file)
    host = arrays["host_phase"].astype(np.float64)
    tmpl = arrays["template_phase"].astype(np.float64)
    sup = arrays["support"].astype(bool)
    w = arrays["host_amplitude"].astype(np.float64)
    delta = arrays["delta"].astype(np.float64)
    pil = Image.open(path).convert("RGB")
    centered_phase = readings.query_phase(pil, record.shape)
    weights = record.amplitude_weights

    # experiments/benign_canonical.py:H_of and engines_canonical.py:H_of (EPS 1e-12)
    r = np.cos((centered_phase - host + np.pi) % (2 * np.pi) - np.pi)
    ww = np.where(sup, w, 0.0)
    assert readings.spectral_agreement(centered_phase, record.original_phase, weights, record.active_bins,
                                       stabilizer=1e-12, form=readings.SumForm.WEIGHT_MAP) == float(np.sum(r * ww) / (np.sum(ww) + 1e-12))
    # experiments/reference_baselines.py:H_against (EPS 1e-8), against the template
    r = np.cos((centered_phase - tmpl + np.pi) % (2 * np.pi) - np.pi)
    assert readings.spectral_agreement(centered_phase, record.template_phase, weights, record.active_bins,
                                       stabilizer=1e-8, form=readings.SumForm.WEIGHT_MAP) == float(np.sum(r * ww) / (np.sum(ww) + 1e-8))

    # experiments/table4_artifact.py:h_host_sub / c_tmpl_sub (centered, sub-band B, EPS 1e-12, inline product)
    band = np.sqrt(np.fft.fftshift(np.fft.fftfreq(sup.shape[0])).reshape(-1, 1) ** 2
                   + np.fft.fftshift(np.fft.fftfreq(sup.shape[1])).reshape(1, -1) ** 2) / np.sqrt(0.5) <= 0.15
    ww = np.where(sup & band, w, 0.0)
    bins = readings.subband_bins(record)
    for reference, record_reference in ((host, record.original_phase), (tmpl, record.template_phase)):
        old = float(np.sum(np.cos(wrap(centered_phase - reference)) * ww) / (np.sum(ww) + 1e-12))
        assert readings.spectral_agreement(centered_phase, record_reference, weights, bins, stabilizer=1e-12,
                                           form=readings.SumForm.WEIGHT_MAP_VALUE_LAYOUT) == old

    # experiments/operating_points.py:rel_err (uncentered, B, 1e-8, inline cosine-times-weights)
    tm, sup_u, amp = (np.fft.ifftshift(tmpl), np.fft.ifftshift(sup), np.fft.ifftshift(w))
    fy = np.fft.fftfreq(tm.shape[0])[:, None]; fx = np.fft.fftfreq(tm.shape[1])[None, :]
    w_sub = np.where(sup_u & (np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5) <= 0.15), amp, 0.0)
    ph = np.angle(np.fft.fft2(0.299 * rgb(path)[:, :, 0] + 0.587 * rgb(path)[:, :, 1] + 0.114 * rgb(path)[:, :, 2]))
    old = 1.0 - float(np.sum(np.cos((ph - tm + np.pi) % (2 * np.pi) - np.pi) * w_sub) / (np.sum(w_sub) + 1e-8))
    uncentered_phase = readings.query_phase(pil, record.shape, centered=False)
    uncentered_bins = np.fft.ifftshift(record.active_bins) & subband_b(record.shape, centered=False)
    new = readings.spectral_agreement(uncentered_phase, np.fft.ifftshift(record.template_phase),
                                      np.fft.ifftshift(weights), uncentered_bins, stabilizer=1e-8,
                                      form=readings.SumForm.WEIGHT_MAP_VALUE_LAYOUT)
    assert 1.0 - new == old

    # experiments/subband_table1.py:H_sub, copy_ring_canonical.py:h_sub (uncentered, B, 1e-8, named cosine)
    r = np.cos((ph - tm + np.pi) % (2 * np.pi) - np.pi)
    assert readings.spectral_agreement(uncentered_phase, np.fft.ifftshift(record.template_phase),
                                       np.fft.ifftshift(weights), uncentered_bins, stabilizer=1e-8,
                                       form=readings.SumForm.WEIGHT_MAP) == float(np.sum(r * w_sub) / (np.sum(w_sub) + 1e-8))

    # examples/quickstart.py:read_copy -> trigger_conditioned.compute_template_score / compute_trigger_projection
    assert readings.spectral_agreement(uncentered_phase, np.fft.ifftshift(record.template_phase),
                                       np.fft.ifftshift(weights), uncentered_bins, form=readings.SumForm.WEIGHT_MAP) \
        == OLD_TRIGGER.compute_template_score(ph, tm, weights=w_sub)
    host_u, delta_u = np.fft.ifftshift(host), np.fft.ifftshift(delta)
    assert readings.key_projection(readings.phase_drift(uncentered_phase, np.fft.ifftshift(record.original_phase)),
                                   np.fft.ifftshift(record.displacement), np.ones(record.shape), sup_u,
                                   form=readings.SumForm.WEIGHT_MAP) \
        == OLD_TRIGGER.compute_trigger_projection(ph, host_u, delta_u, support=sup_u)
    # trigger_conditioned.compute_host_score with a zero-filled weight map
    assert readings.spectral_agreement(ph, host_u, np.fft.ifftshift(weights), sup_u, form=readings.SumForm.WEIGHT_MAP) \
        == OLD_TRIGGER.compute_host_score(ph, host_u, weights=np.where(sup_u, amp, 0.0))

    # experiments/sensitivity_release.py:G_of (uncentered, zero-filled, +1e-6)
    resid = (ph - host_u + np.pi) % (2 * np.pi) - np.pi
    w_ring = np.where(sup_u, amp, 0.0)
    old = float(np.sum(w_ring * delta_u * np.sin(resid))) / (float(np.sum(w_ring * delta_u * delta_u)) + 1e-6)
    assert readings.key_projection(readings.phase_drift(ph, host_u), delta_u, amp, sup_u,
                                   stabilizer=1e-6, form=readings.SumForm.WEIGHT_MAP) == old


@pytest.mark.parametrize("query", SAMPLE_LIST, ids=lambda query: query.query_id)
def test_gathered_forms_match_experiment_copies(query):
    record = load_record(query.template_artifact)
    arrays = stored(query.template_artifact)
    hp, tp = arrays["host_phase"].astype(np.float64), arrays["template_phase"].astype(np.float64)
    sup, w = arrays["support"].astype(bool), arrays["host_amplitude"].astype(np.float64)
    delta = arrays["delta"].astype(np.float64)
    ph = readings.query_phase(Image.open(query.query_path), record.shape)
    weights = record.amplitude_weights

    # experiments/original_control.py:main (gathered, +1e-12)
    assert readings.spectral_agreement(ph, record.original_phase, weights, record.active_bins, stabilizer=1e-12) \
        == float(np.sum(w[sup] * np.cos(wrap(ph - hp))[sup]) / (w[sup].sum() + 1e-12))
    assert readings.spectral_agreement(ph, record.template_phase, weights, record.active_bins, stabilizer=1e-12) \
        == float(np.sum(w[sup] * np.cos(wrap(ph - tp))[sup]) / (w[sup].sum() + 1e-12))
    # experiments/original_control.py: G_s^(template) via extract_features.projection (frozen values)
    expected = OLD_FEATURES["sample"][query.query_id]
    template_scale = max(abs(expected["projection_template_stored"]), 1e-6)
    drift = readings.phase_drift(ph, record.original_phase)
    template_drift = readings.phase_drift(record.template_phase, record.original_phase)
    assert readings.release_normalized(
        readings.key_projection(drift, record.displacement, weights, record.active_bins),
        readings.key_projection(template_drift, record.displacement, weights, record.active_bins),
    ) == expected["projection_query_stored"] / template_scale

    # experiments/record_codec.py (gathered on the active bins, G with +1e-6)
    resid = (ph[sup] - hp[sup] + np.pi) % (2 * np.pi) - np.pi
    a, d = w[sup], delta[sup]
    old = float(np.sum(a * d * np.sin(resid)) / (np.sum(a * d * d) + 1e-6))
    assert readings.key_projection(drift, record.displacement, weights, record.active_bins, stabilizer=1e-6) == old

    # experiments/table4_artifact.py:g_raw_sub squares delta first (w * dl ** 2), a different rounding
    # order from Eq. (6) as written; it is not reproduced by key_projection (see the plan's Decisions).


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
