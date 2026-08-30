#!/usr/bin/env python3
"""Extract calibrated RecordModel features from explicit private template records."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow.paths import require_input_path, require_output_path


EPS = 1e-12


def wrap_phase(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def signed_hermitian_code(shape: tuple[int, int], key: int) -> np.ndarray:
    rng = np.random.default_rng(key)
    raw = rng.standard_normal(shape)
    rows = np.mod(-np.arange(shape[0]), shape[0])
    cols = np.mod(-np.arange(shape[1]), shape[1])
    partner = raw[np.ix_(rows, cols)]
    code = np.sign(raw - partner)
    self_rows = np.arange(shape[0]) == rows
    self_cols = np.arange(shape[1]) == cols
    code[np.ix_(self_rows, self_cols)] = 0.0
    return code.astype(np.float64)


def wrong_keys(key: int, count: int) -> list[int]:
    rng = np.random.default_rng((key ^ 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
    result = []
    while len(result) < count:
        candidate = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        if candidate != key and candidate not in result:
            result.append(candidate)
    return result


def masked_mean(values: np.ndarray, support: np.ndarray, weights: np.ndarray) -> float:
    mask = support & np.isfinite(values) & np.isfinite(weights)
    total = float(np.sum(weights[mask]))
    if total <= EPS:
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / total)


def projection(
    query_phase: np.ndarray,
    host_phase: np.ndarray,
    delta: np.ndarray,
    support: np.ndarray,
    weights: np.ndarray,
) -> float:
    residual = wrap_phase(query_phase - host_phase)
    mask = support & np.isfinite(residual) & np.isfinite(delta) & np.isfinite(weights)
    numerator = float(np.sum(weights[mask] * delta[mask] * np.sin(residual[mask])))
    denominator = float(np.sum(weights[mask] * delta[mask] * delta[mask]))
    return numerator / (denominator + EPS)


def query_phase(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.fft.fftshift(np.angle(np.fft.fft2(luminance)))


def wrong_delta(key: int, shape: tuple[int, int], alpha: float, beta: float, support: np.ndarray) -> np.ndarray:
    code = np.fft.fftshift(signed_hermitian_code(shape, key))
    preblend = alpha * np.pi * code * support
    return np.angle((1.0 - beta) + beta * np.exp(1j * preblend)) * support


def radial_values(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return radius / (float(np.max(radius)) + EPS)


def feature_row(
    query: Path,
    template_path: Path,
    reference_path: Path | None = None,
    projection_key: int | None = None,
    n_wrong: int = 8,
    n_bands: int = 8,
    phase_override: np.ndarray | None = None,
    reference_phase_override: np.ndarray | None = None,
) -> tuple[dict[str, float], int]:
    with np.load(template_path) as record:
        template_phase = record["template_phase"].astype(np.float64)
        host_phase = record["host_phase"].astype(np.float64)
        enrolled_delta = record["delta"].astype(np.float64)
        weights = record["host_amplitude"].astype(np.float64)
        support = record["support"].astype(bool)
        original_key = int(record["key"][0])
        alpha = float(record["alpha"][0])
        beta = float(record["beta"][0])
    phase = phase_override if phase_override is not None else query_phase(query)
    if phase.shape != template_phase.shape:
        raise ValueError(f"query/template shape mismatch: {phase.shape} != {template_phase.shape}")
    if reference_phase_override is not None:
        reference_phase = reference_phase_override
    else:
        reference_phase = query_phase(reference_path) if reference_path is not None else template_phase
    if reference_phase.shape != template_phase.shape:
        raise ValueError(f"reference/template shape mismatch: {reference_phase.shape} != {template_phase.shape}")
    key = original_key if projection_key is None else projection_key
    delta = enrolled_delta if key == original_key else wrong_delta(key, phase.shape, alpha, beta, support)
    residual = wrap_phase(phase - host_phase)
    reference_projection = projection(reference_phase, host_phase, enrolled_delta, support, weights)
    projection_scale = max(abs(reference_projection), 1e-6)
    values: dict[str, float] = {
        "H": masked_mean(np.cos(residual), support, weights),
        "C": masked_mean(np.cos(wrap_phase(phase - reference_phase)), support, weights),
        "residual_mean": masked_mean(residual, support, weights),
        "residual_std": float(np.std(residual[support])),
        "sin_projection_mean": masked_mean(delta * np.sin(residual), support, weights),
        "cos_consistency_mean": masked_mean(np.cos(residual), support, weights),
        "valid_count": int(np.sum(support)),
        "valid_fraction": float(np.mean(support)),
        "delta_abs_mean": float(np.mean(np.abs(delta[support]))),
        "delta_normalized_mean": float(np.mean(np.abs(delta[support])) / max(alpha * beta, EPS)),
    }
    correct = projection(phase, host_phase, delta, support, weights) / projection_scale
    alternatives = []
    for candidate in wrong_keys(key, n_wrong):
        candidate_delta = wrong_delta(candidate, phase.shape, alpha, beta, support)
        alternatives.append(projection(phase, host_phase, candidate_delta, support, weights) / projection_scale)
    alternative_array = np.asarray(alternatives, dtype=np.float64)
    values.update(
        {
            "G_s": correct,
            "G_wrong_mean": float(np.mean(alternative_array)),
            "G_wrong_std": float(np.std(alternative_array)),
            "G_wrong_max": float(np.max(alternative_array)),
            "G_margin": float(correct - np.max(alternative_array)),
            "G_z": float((correct - np.mean(alternative_array)) / (np.std(alternative_array) + 1e-8)),
            "n_wrong": n_wrong,
        }
    )
    radius = radial_values(phase.shape)
    supported_radius = radius[support]
    edges = np.linspace(float(np.min(supported_radius)), float(np.max(supported_radius)) + 1e-12, n_bands + 1)
    for index in range(n_bands):
        band = support & (radius >= edges[index]) & (radius < edges[index + 1])
        band_reference = projection(reference_phase, host_phase, enrolled_delta, band, weights)
        band_scale = max(abs(band_reference), 1e-6)
        values[f"H_b{index}"] = masked_mean(np.cos(residual), band, weights)
        values[f"C_b{index}"] = masked_mean(np.cos(wrap_phase(phase - reference_phase)), band, weights)
        values[f"G_b{index}"] = projection(phase, host_phase, delta, band, weights) / band_scale
        values[f"G_margin_b{index}"] = values[f"G_b{index}"]
        values[f"residual_mean_b{index}"] = masked_mean(residual, band, weights)
        values[f"residual_std_b{index}"] = float(np.std(residual[band])) if np.any(band) else float("nan")
        values[f"sin_projection_mean_b{index}"] = masked_mean(delta * np.sin(residual), band, weights)
        values[f"cos_consistency_mean_b{index}"] = masked_mean(np.cos(residual), band, weights)
        values[f"valid_count_b{index}"] = int(np.sum(band))
    return values, original_key


def rows_for_query(query: Any, n_wrong: int, n_bands: int) -> list[dict[str, Any]]:
    path = require_input_path(query.query_path)
    template = require_input_path(query.template_artifact)
    reference = require_input_path(query.released_path)
    reference_phase = query_phase(reference)
    phase = reference_phase if path == reference else query_phase(path)
    features, original_key = feature_row(
        path,
        template,
        reference_path=reference,
        n_wrong=n_wrong,
        n_bands=n_bands,
        phase_override=phase,
        reference_phase_override=reference_phase,
    )
    rows = [
        {
            "sample_id": query.query_id,
            "query_id": query.query_id,
            "enrollment_id": query.enrollment_id,
            "source_group_id": query.source_group_id,
            "dataset": query.dataset,
            "split": query.split,
            "protocol": query.protocol,
            "source_path": str(path),
            "template_id": str(template),
            "key_id": original_key,
            "original_key_id": original_key,
            "host_recovery_key_id": original_key,
            "code_mode": "signed_hermitian_pm_pi",
            "y_fake": query.y_fake,
            "y_rel": query.y_rel,
            "y_keycorr": query.y_keycorr,
            **features,
        }
    ]
    if query.protocol == "benign":
        wrong_key = int(hashlib.sha256(f"wrong-key\x1f{original_key}".encode()).hexdigest()[:16], 16)
        wrong_features, _ = feature_row(
            path,
            template,
            reference_path=reference,
            projection_key=wrong_key,
            n_wrong=n_wrong,
            n_bands=n_bands,
            phase_override=phase,
            reference_phase_override=reference_phase,
        )
        rows.append(
            {
                "sample_id": f"{query.query_id}_wrong_key",
                "query_id": query.query_id,
                "enrollment_id": query.enrollment_id,
                "source_group_id": query.source_group_id,
                "dataset": query.dataset,
                "split": query.split,
                "protocol": "wrong_key",
                "source_path": str(path),
                "template_id": str(template),
                "key_id": wrong_key,
                "original_key_id": original_key,
                "host_recovery_key_id": original_key,
                "code_mode": "signed_hermitian_pm_pi",
                "y_fake": 0,
                "y_rel": 1,
                "y_keycorr": 0,
                **wrong_features,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "features/features.csv")
    parser.add_argument("--n-wrong", type=int, default=8)
    parser.add_argument("--n-bands", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    queries = pd.read_csv(require_input_path(args.queries))
    queries = queries[queries["status"] == "pass"].copy()
    output = require_output_path(args.output)
    rows: list[dict[str, Any]] = []
    query_rows = list(queries.itertuples(index=False))
    worker_count = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        iterator = executor.map(
            lambda query: rows_for_query(query, args.n_wrong, args.n_bands),
            query_rows,
        )
        for count, produced in enumerate(iterator, start=1):
            rows.extend(produced)
            if count % 100 == 0:
                print(f"processed={count}/{len(queries)} output_rows={len(rows)} workers={worker_count}", flush=True)

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    for (dataset, split), part in result.groupby(["dataset", "split"]):
        split_path = output.with_name(f"{output.stem}_{str(dataset).lower().replace('+', 'p')}_{split}{output.suffix}")
        part.to_csv(split_path, index=False)
    print(f"Wrote {output} rows={len(result)} columns={len(result.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
