"""Enrollment: turn an original crop I into a released crop I' and its private record R.

Sec. 3.1 of the paper. For each colour channel the active Fourier coefficients are multiplied
by e^{i alpha pi h_s(k)}; the edited image is blended with the original,
I~' = (1 - beta) I + beta F^-1(F~), clipped to [0, 255] and truncated to uint8. The phase
strength alpha is lowered along a fixed ladder until the encoded release passes the fidelity
gate (PSNR >= 35 dB and SSIM >= 0.98 against the original).

This module does no file I/O; callers decode images and write releases and records.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
import math

import numpy as np
from skimage.metrics import structural_similarity

from .key import Ring, active_bins, derive_key, displacement, phase_challenge, sign_pattern
from .record import Record, UnitId, record_from_stored_arrays
from .spectrum import kappa_grid, luminance_spectrum, wrap_phase

ALPHA_LADDER: dict[str, tuple[float, ...]] = {"DFDC": (0.015, 0.010, 0.005)}
"""Per-dataset alpha ladders, tried in order; datasets not listed use ``DEFAULT_ALPHA_LADDER``."""
DEFAULT_ALPHA_LADDER: tuple[float, ...] = (0.050, 0.030, 0.015)

AMPLITUDE_WEIGHT_MIN_PEAK = 1e-12
"""Lower bound on max A_I when normalizing the amplitude weights (guards an all-black crop)."""
PSNR_OF_IDENTICAL_IMAGES = 100.0


@dataclass(frozen=True)
class EnrollmentSettings:
    """Enrollment parameters shared by every unit; defaults are the paper's (Sec. 3.1, App. D.1)."""

    master_seed: int = 17
    beta: float = 0.15
    ring: Ring = field(default_factory=lambda: Ring(0.10, 0.45))
    amplitude_floor_ratio: float = 1e-6
    minimum_psnr: float = 35.0
    minimum_ssim: float = 0.98


@dataclass(frozen=True, eq=False)
class Enrollment:
    """Outcome of enrolling one unit: the released crop I', its record R and its fidelity."""

    unit: UnitId
    released: np.ndarray
    """Released crop I' (uint8, H x W x 3)."""
    record: Record
    alpha: float
    """Phase strength alpha that passed the fidelity gate (unrounded; the record stores float32)."""
    psnr: float
    ssim: float


@dataclass(frozen=True)
class FidelityAttempt:
    """One rung of the alpha ladder and the fidelity it reached."""

    alpha: float
    psnr: float
    ssim: float


@dataclass(frozen=True)
class FidelityRejection:
    """A unit for which no alpha on the ladder passed the fidelity gate."""

    unit: UnitId
    attempts: tuple[FidelityAttempt, ...]

    @property
    def reason(self) -> str:
        """Rejection text as written to the enrollment quarantine manifest."""
        attempts = [{"alpha": a.alpha, "psnr": a.psnr, "ssim": a.ssim} for a in self.attempts]
        return f"quality_gate_failed:{attempts}"


def alpha_ladder(dataset: str) -> tuple[float, ...]:
    """Alpha values tried for ``dataset``, strongest first."""
    return ALPHA_LADDER.get(dataset, DEFAULT_ALPHA_LADDER)


def psnr(original: np.ndarray, released: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB on the 0..255 scale (100 for identical images)."""
    mse = float(np.mean((original.astype(np.float64) - released.astype(np.float64)) ** 2))
    return PSNR_OF_IDENTICAL_IMAGES if mse <= 1e-12 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def ssim(original: np.ndarray, released: np.ndarray) -> float:
    """Structural similarity of two uint8 RGB crops (skimage, per channel, data range 255)."""
    return float(structural_similarity(original, released, channel_axis=2, data_range=255))


def passes_fidelity_gate(psnr_db: float, ssim_value: float, settings: EnrollmentSettings) -> bool:
    """Fidelity gate: PSNR >= minimum_psnr and SSIM >= minimum_ssim."""
    return psnr_db >= settings.minimum_psnr and ssim_value >= settings.minimum_ssim


@dataclass(frozen=True, eq=False)
class _EncodingBasis:
    # Everything encode_release needs that does not depend on alpha: computed once per unit and
    # shared by every rung of the alpha ladder.
    original: np.ndarray
    key: int
    ring: Ring
    amplitude_floor_ratio: float
    pattern: np.ndarray
    spectrum: np.ndarray
    active: np.ndarray
    channel_amplitudes: tuple[np.ndarray, ...]
    channel_phases: tuple[np.ndarray, ...]
    stored_constants: dict[str, np.ndarray]


def _encoding_basis(original: np.ndarray, key: int, ring: Ring, amplitude_floor_ratio: float) -> _EncodingBasis:
    height, width = original.shape[:2]
    pattern = sign_pattern((height, width), key)
    kappa = kappa_grid((height, width), centered=False)
    spectrum = luminance_spectrum(original)
    amplitude = np.abs(spectrum)
    active = active_bins(kappa, amplitude, pattern, ring, amplitude_floor_ratio)
    channel_spectra = [np.fft.fft2(original[:, :, channel].astype(np.float64)) for channel in range(3)]
    stored_constants = {
        "host_phase": np.fft.fftshift(np.angle(spectrum)).astype(np.float16),
        "host_amplitude": np.fft.fftshift(
            amplitude / max(float(np.max(amplitude)), AMPLITUDE_WEIGHT_MIN_PEAK)
        ).astype(np.float16),
        "support": np.fft.fftshift(active).astype(np.uint8),
        "code": np.fft.fftshift(pattern).astype(np.int8),
    }
    return _EncodingBasis(
        original=original,
        key=key,
        ring=ring,
        amplitude_floor_ratio=amplitude_floor_ratio,
        pattern=pattern,
        spectrum=spectrum,
        active=active,
        channel_amplitudes=tuple(np.abs(channel_spectrum) for channel_spectrum in channel_spectra),
        channel_phases=tuple(np.angle(channel_spectrum) for channel_spectrum in channel_spectra),
        stored_constants=stored_constants,
    )


def _encode_at(basis: _EncodingBasis, alpha: float, beta: float) -> tuple[np.ndarray, Record]:
    challenge = phase_challenge(basis.pattern, alpha, basis.active)
    edited_channels = []
    for channel_amplitude, channel_phase in zip(basis.channel_amplitudes, basis.channel_phases):
        # phi + challenge, added into a copy of the channel phase so the sum keeps its memory layout.
        edited_phase = channel_phase.copy(order="K")
        edited_phase += challenge
        edited = channel_amplitude * np.exp(1j * edited_phase)
        edited_channels.append(np.fft.ifft2(edited).real)
    edited_rgb = np.stack(edited_channels, axis=2)
    blended = (1.0 - beta) * basis.original.astype(np.float64) + beta * edited_rgb
    released = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    delta = displacement(basis.pattern, alpha, beta, basis.active)
    # T_I = wrap(phi_I + delta_s). Keep np.angle(...) as an unnamed temporary: NumPy then adds
    # in place into it, the template inherits the FFT's Fortran layout, and the NPZ bytes match.
    template_phase = wrap_phase(np.angle(basis.spectrum) + delta)
    constants = basis.stored_constants
    stored = {
        "template_phase": np.fft.fftshift(template_phase).astype(np.float16),
        "host_phase": constants["host_phase"],
        "delta": np.fft.fftshift(delta).astype(np.float16),
        "host_amplitude": constants["host_amplitude"],
        "support": constants["support"],
        "code": constants["code"],
        "key": np.asarray([basis.key], dtype=np.uint64),
        "alpha": np.asarray([alpha], dtype=np.float32),
        "beta": np.asarray([beta], dtype=np.float32),
        "bands": np.asarray([basis.ring.kappa_min, basis.ring.kappa_max], dtype=np.float32),
        "amplitude_floor_ratio": np.asarray([basis.amplitude_floor_ratio], dtype=np.float32),
    }
    return released, record_from_stored_arrays(stored)


def encode_release(
    original: np.ndarray,
    key: int,
    alpha: float,
    beta: float,
    ring: Ring,
    amplitude_floor_ratio: float,
) -> tuple[np.ndarray, Record]:
    """Encode the released crop I' and record R of ``original`` at one phase strength alpha.

    ``original`` is the aligned crop I (uint8, H x W x 3). K_eff and h_s come from the luminance
    spectrum and are applied unchanged to all three colour channels: each channel's active
    coefficients get the phase challenge alpha pi h_s(k), and the edit is blended in with weight beta.
    """
    return _encode_at(_encoding_basis(original, key, ring, amplitude_floor_ratio), alpha, beta)


def _climb_alpha_ladder(
    original: np.ndarray,
    unit: UnitId,
    alphas: Sequence[float],
    settings: EnrollmentSettings,
) -> Enrollment | FidelityRejection:
    if unit.frame_index is not None:
        raise ValueError(f"key derivation for video frames is not defined: {unit}")
    key = derive_key(settings.master_seed, unit.asset_id)
    basis = _encoding_basis(original, key, settings.ring, settings.amplitude_floor_ratio)
    attempts = []
    for alpha in alphas:
        released, record = _encode_at(basis, alpha, settings.beta)
        released_psnr = psnr(original, released)
        released_ssim = ssim(original, released)
        attempts.append(FidelityAttempt(alpha, released_psnr, released_ssim))
        if passes_fidelity_gate(released_psnr, released_ssim, settings):
            return Enrollment(unit, released, record, alpha, released_psnr, released_ssim)
    return FidelityRejection(unit, tuple(attempts))


def enroll(
    original: np.ndarray,
    unit: UnitId,
    alphas: Sequence[float],
    settings: EnrollmentSettings = EnrollmentSettings(),
) -> Enrollment:
    """Enroll one image unit: try each alpha in ``alphas`` until the fidelity gate passes.

    The key is s = derive_key(master_seed, unit.asset_id). Raises ``ValueError`` with the
    quarantine reason if no alpha passes.
    """
    outcome = _climb_alpha_ladder(original, unit, alphas, settings)
    if isinstance(outcome, FidelityRejection):
        raise ValueError(outcome.reason)
    return outcome


def enroll_many(
    requests: Iterable[tuple[UnitId, np.ndarray, Sequence[float]]],
    settings: EnrollmentSettings = EnrollmentSettings(),
) -> Iterator[Enrollment | FidelityRejection]:
    """Enroll a stream of (unit, original crop, alpha ladder); yield one outcome per request.

    Units that fail the fidelity gate are yielded as ``FidelityRejection`` so one rejection does
    not stop the stream. Runs serially; worker pools belong to the caller.
    """
    for unit, original, alphas in requests:
        yield _climb_alpha_ladder(original, unit, alphas, settings)
