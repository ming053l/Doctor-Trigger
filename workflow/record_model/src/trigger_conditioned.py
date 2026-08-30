"""Trigger-conditioned phase-relation features for record_model_release.

This module is intentionally independent from trigger synthesis.  It consumes
existing templates/frames/keys and never embeds or releases new images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class RingSpec:
    shape: Tuple[int, int]
    bands: Tuple[float, float]
    half_plane: bool = True


@dataclass(frozen=True)
class DeltaResult:
    delta: np.ndarray
    code_phase: np.ndarray
    support: np.ndarray
    code_mode: str
    key: int
    alpha: float
    beta: float


def wrap_phase(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians into [-pi, pi)."""
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def radial_values(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy = h / 2.0
    cx = w / 2.0
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return dist / (float(np.max(dist)) + EPS)


def radial_mask(shape: Tuple[int, int], bands: Tuple[float, float]) -> np.ndarray:
    radius = radial_values(shape)
    return np.logical_and(radius >= float(bands[0]), radius <= float(bands[1]))


def hermitian_half_plane_mask(shape: Tuple[int, int]) -> np.ndarray:
    """Return one representative from each conjugate FFT pair.

    The synthesis path writes a full random field, but the record_model_release diagnostic must
    not double count conjugate duplicates when estimating keyed evidence.
    """
    h, w = shape
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    fx = np.fft.fftfreq(w).reshape(1, -1)
    return np.logical_or(fy > 0.0, np.logical_and(fy == 0.0, fx >= 0.0))


def effective_ring_mask(ring: RingSpec) -> np.ndarray:
    mask = radial_mask(ring.shape, ring.bands)
    if ring.half_plane:
        mask = np.logical_and(mask, hermitian_half_plane_mask(ring.shape))
    return mask


def _code_phase(shape: Tuple[int, int], key: int, code_mode: str) -> np.ndarray:
    rng = np.random.default_rng(int(key))
    code = rng.choice([-1.0, 1.0], size=shape).astype(np.float64)
    if code_mode == "zero_pi":
        return (np.pi * (1.0 - code) / 2.0).astype(np.float64)
    if code_mode == "pm_pi":
        return (np.pi * code).astype(np.float64)
    if code_mode == "pm_one_scaled":
        return code.astype(np.float64)
    raise ValueError(f"Unsupported code_mode: {code_mode}")


def compute_delta_s(
    key: int,
    alpha: float,
    beta: float,
    ring: RingSpec | Mapping[str, object] | Tuple[int, int],
    code_mode: str = "zero_pi",
    bands: Tuple[float, float] | None = None,
) -> DeltaResult:
    """Generate the actual first-order trigger direction delta_s(k).

    delta_s,c(k) = arg((1-beta) + beta * exp(i * alpha * Delta_phi_c(k)))
    """
    if isinstance(ring, RingSpec):
        spec = ring
    elif isinstance(ring, Mapping):
        shape = tuple(ring["shape"])  # type: ignore[arg-type]
        spec = RingSpec(shape=(int(shape[0]), int(shape[1])), bands=tuple(ring["bands"]))  # type: ignore[arg-type]
    else:
        if bands is None:
            raise ValueError("bands must be provided when ring is a shape tuple")
        spec = RingSpec(shape=(int(ring[0]), int(ring[1])), bands=bands)

    support = effective_ring_mask(spec)
    code_phase = _code_phase(spec.shape, int(key), code_mode)
    complex_mix = (1.0 - float(beta)) + float(beta) * np.exp(1j * float(alpha) * code_phase)
    delta = np.angle(complex_mix).astype(np.float64)
    delta = np.where(support, delta, 0.0)
    code_phase = np.where(support, code_phase, 0.0)
    return DeltaResult(
        delta=delta,
        code_phase=code_phase,
        support=support,
        code_mode=code_mode,
        key=int(key),
        alpha=float(alpha),
        beta=float(beta),
    )


def recover_host_phase(template_phase: np.ndarray, delta_s: np.ndarray | DeltaResult) -> np.ndarray:
    delta = delta_s.delta if isinstance(delta_s, DeltaResult) else delta_s
    return wrap_phase(template_phase - delta)


def _masked_mean(values: np.ndarray, support: np.ndarray | None = None, weights: np.ndarray | None = None) -> float:
    mask = np.ones_like(values, dtype=bool) if support is None else support.astype(bool)
    finite = np.isfinite(values)
    mask = np.logical_and(mask, finite)
    if weights is None:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(values[mask]))
    w = np.where(mask, weights, 0.0).astype(np.float64)
    denom = float(np.sum(w))
    if denom <= EPS:
        return float("nan")
    return float(np.sum(values * w) / denom)


def compute_host_score(
    phi_j: np.ndarray,
    phi_hat_i: np.ndarray,
    support: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> float:
    return _masked_mean(np.cos(wrap_phase(phi_j - phi_hat_i)), support, weights)


def compute_template_score(
    phi_j: np.ndarray,
    template_phase: np.ndarray,
    support: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> float:
    return _masked_mean(np.cos(wrap_phase(phi_j - template_phase)), support, weights)


def compute_trigger_projection(
    phi_j: np.ndarray,
    phi_hat_i: np.ndarray,
    delta_s: np.ndarray | DeltaResult,
    support: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    eps: float = 1e-12,
    return_details: bool = False,
) -> float | Dict[str, float]:
    delta = delta_s.delta if isinstance(delta_s, DeltaResult) else np.asarray(delta_s, dtype=np.float64)
    base_support = delta_s.support if isinstance(delta_s, DeltaResult) else np.ones_like(delta, dtype=bool)
    if support is not None:
        base_support = np.logical_and(base_support, support.astype(bool))
    w = np.ones_like(delta, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    mask = np.logical_and(base_support, np.isfinite(phi_j) & np.isfinite(phi_hat_i) & np.isfinite(delta))
    w = np.where(mask, w, 0.0)
    residual = wrap_phase(phi_j - phi_hat_i)
    numerator = float(np.sum(w * delta * np.sin(residual)))
    denominator = float(np.sum(w * delta * delta))
    valid = denominator > eps
    value = numerator / (denominator + eps) if valid else 0.0
    if return_details:
        return {
            "G_s": float(value),
            "numerator": numerator,
            "denominator": denominator,
            "valid": float(valid),
            "valid_count": int(np.sum(mask)),
        }
    return float(value)


def deterministic_wrong_keys(key: int, n_wrong: int = 4) -> List[int]:
    rng = np.random.default_rng((int(key) ^ 0x9E3779B9) & 0xFFFFFFFF)
    out: List[int] = []
    while len(out) < int(n_wrong):
        candidate = int(rng.integers(0, 2**31 - 1))
        if candidate != int(key) and candidate not in out:
            out.append(candidate)
    return out


def compute_wrong_key_projection(
    phi_j: np.ndarray,
    phi_hat_i: np.ndarray,
    key: int,
    alpha: float,
    beta: float,
    ring: RingSpec,
    code_mode: str = "zero_pi",
    n_wrong: int = 4,
    weights: np.ndarray | None = None,
) -> Dict[str, float]:
    correct = compute_delta_s(key, alpha, beta, ring, code_mode)
    g_correct = float(compute_trigger_projection(phi_j, phi_hat_i, correct, weights=weights))
    wrong_values = []
    for wrong_key in deterministic_wrong_keys(key, n_wrong=n_wrong):
        wrong = compute_delta_s(wrong_key, alpha, beta, ring, code_mode)
        wrong_values.append(float(compute_trigger_projection(phi_j, phi_hat_i, wrong, weights=weights)))
    wrong_arr = np.asarray(wrong_values, dtype=np.float64)
    wrong_mean = float(np.mean(wrong_arr)) if wrong_arr.size else float("nan")
    wrong_std = float(np.std(wrong_arr)) if wrong_arr.size else float("nan")
    wrong_max = float(np.max(wrong_arr)) if wrong_arr.size else float("nan")
    return {
        "G_s": g_correct,
        "G_wrong_mean": wrong_mean,
        "G_wrong_std": wrong_std,
        "G_wrong_max": wrong_max,
        "G_margin": float(g_correct - wrong_max),
        "G_z": float((g_correct - wrong_mean) / (wrong_std + 1e-8)),
        "n_wrong": int(n_wrong),
    }


def compute_band_features(
    phi_j: np.ndarray,
    template_phase: np.ndarray,
    phi_hat_i: np.ndarray,
    delta_s: DeltaResult,
    n_bands: int = 4,
    weights: np.ndarray | None = None,
) -> Dict[str, float]:
    radius = radial_values(delta_s.delta.shape)
    support_radius = radius[delta_s.support]
    if support_radius.size == 0:
        return {}
    low = float(np.min(support_radius))
    high = float(np.max(support_radius))
    edges = np.linspace(low, high + 1e-12, int(n_bands) + 1)
    residual = wrap_phase(phi_j - phi_hat_i)
    out: Dict[str, float] = {}
    for idx in range(int(n_bands)):
        band = np.logical_and(delta_s.support, np.logical_and(radius >= edges[idx], radius < edges[idx + 1]))
        out[f"H_b{idx}"] = compute_host_score(phi_j, phi_hat_i, band, weights)
        out[f"C_b{idx}"] = compute_template_score(phi_j, template_phase, band, weights)
        out[f"G_b{idx}"] = float(compute_trigger_projection(phi_j, phi_hat_i, delta_s, support=band, weights=weights))
        out[f"G_margin_b{idx}"] = out[f"G_b{idx}"]
        out[f"residual_mean_b{idx}"] = _masked_mean(residual, band, weights)
        out[f"residual_std_b{idx}"] = float(np.nanstd(residual[band])) if np.any(band) else float("nan")
        out[f"sin_projection_mean_b{idx}"] = _masked_mean(delta_s.delta * np.sin(residual), band, weights)
        out[f"cos_consistency_mean_b{idx}"] = _masked_mean(np.cos(residual), band, weights)
        out[f"valid_count_b{idx}"] = int(np.sum(band))
    return out


def compute_trigger_maps(
    phi_j: np.ndarray,
    template_phase: np.ndarray,
    phi_hat_i: np.ndarray,
    delta_s: np.ndarray | DeltaResult,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    delta = delta_s.delta if isinstance(delta_s, DeltaResult) else np.asarray(delta_s, dtype=np.float64)
    r_host = wrap_phase(phi_j - phi_hat_i)
    r_template = wrap_phase(phi_j - template_phase)
    r_trigger = delta * np.sin(r_host)
    norm = float(np.sqrt(np.mean(delta * delta)) + eps)
    return {
        "R_host_map": r_host,
        "R_template_map": r_template,
        "R_trigger_map": r_trigger,
        "R_trigger_norm_map": r_trigger / norm,
    }


def feature_vector(
    phi_j: np.ndarray,
    template_phase: np.ndarray,
    key: int,
    alpha: float,
    beta: float,
    bands: Tuple[float, float],
    code_mode: str = "zero_pi",
    n_wrong: int = 4,
    n_bands: int = 4,
    weights: np.ndarray | None = None,
    host_key: int | None = None,
) -> Dict[str, float]:
    ring = RingSpec(shape=template_phase.shape, bands=bands, half_plane=True)
    delta = compute_delta_s(key, alpha, beta, ring, code_mode=code_mode)
    host_delta = delta if host_key is None or int(host_key) == int(key) else compute_delta_s(int(host_key), alpha, beta, ring, code_mode=code_mode)
    phi_hat = recover_host_phase(template_phase, host_delta)
    residual = wrap_phase(phi_j - phi_hat)
    row: Dict[str, float] = {
        "H": compute_host_score(phi_j, phi_hat, delta.support, weights),
        "C": compute_template_score(phi_j, template_phase, delta.support, weights),
        "residual_mean": _masked_mean(residual, delta.support, weights),
        "residual_std": float(np.nanstd(residual[delta.support])) if np.any(delta.support) else float("nan"),
        "sin_projection_mean": _masked_mean(delta.delta * np.sin(residual), delta.support, weights),
        "cos_consistency_mean": _masked_mean(np.cos(residual), delta.support, weights),
        "valid_count": int(np.sum(delta.support)),
        "delta_abs_mean": float(np.mean(np.abs(delta.delta[delta.support]))) if np.any(delta.support) else 0.0,
    }
    row.update(compute_wrong_key_projection(phi_j, phi_hat, key, alpha, beta, ring, code_mode, n_wrong, weights))
    row.update(compute_band_features(phi_j, template_phase, phi_hat, delta, n_bands=n_bands, weights=weights))
    return row


def face_context_features(
    face_features: Mapping[str, float],
    context_features: Mapping[str, float] | None,
) -> Dict[str, float | str]:
    if not context_features:
        return {"context_available": "false"}
    g_face = float(face_features.get("G_s", np.nan))
    g_context = float(context_features.get("G_s", np.nan))
    h_face = float(face_features.get("H", np.nan))
    h_context = float(context_features.get("H", np.nan))
    c_face = float(face_features.get("C", np.nan))
    c_context = float(context_features.get("C", np.nan))
    return {
        "context_available": "true",
        "G_face": g_face,
        "G_context": g_context,
        "D_face_context": g_context - g_face,
        "H_face": h_face,
        "H_context": h_context,
        "C_face": c_face,
        "C_context": c_context,
    }
