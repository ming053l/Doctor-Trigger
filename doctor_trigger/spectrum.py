"""Luminance Fourier spectrum, phase wrapping and the normalized frequency radius.

Everything Doctor Trigger compares lives on the luminance spectrum of a crop (Sec. 3.1):
Y = 0.299 R + 0.587 G + 0.114 B, its 2-D DFT F = A e^{i phi}, and the normalized radius
kappa(k) of each Fourier bin k.

Layout convention. NumPy's ``fft2`` puts the zero frequency at index (0, 0) ("uncentered").
Records store their arrays after ``np.fft.fftshift`` ("centered", zero frequency in the middle).
Functions that can return either take ``centered=`` explicitly; nothing guesses.
"""

from __future__ import annotations

import math

import numpy as np

SUBBAND_B_KAPPA_MAX = 0.15
"""Upper edge of the retained sub-band B = {k : kappa(k) <= 0.15} used by H_B and C_B."""


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Luminance Y = 0.299 R + 0.587 G + 0.114 B of an (H, W, 3) RGB array, in float64.

    The input is cast to float64 first (uint8 pixels stay on the 0..255 scale).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def luminance_spectrum(rgb: np.ndarray) -> np.ndarray:
    """Uncentered 2-D DFT F of the luminance Y (complex, zero frequency at index (0, 0))."""
    return np.fft.fft2(luminance(rgb))


def luminance_phase(rgb: np.ndarray, centered: bool) -> np.ndarray:
    """Luminance phase phi_I (original) or phi_J (query): the angle of F, in radians.

    ``centered=True`` applies ``np.fft.fftshift`` so the result lines up with record arrays.
    """
    phase = np.angle(luminance_spectrum(rgb))
    return np.fft.fftshift(phase) if centered else phase


def wrap_phase(angle: np.ndarray) -> np.ndarray:
    """wrap(x) = (x + pi) mod 2 pi - pi, mapping angles to [-pi, pi) (Eq. 4)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def kappa_grid(shape: tuple[int, int], centered: bool) -> np.ndarray:
    """Normalized radius kappa(k) = sqrt(fx^2 + fy^2) / sqrt(0.5^2 + 0.5^2) of every bin.

    fx, fy are in cycles per pixel, so kappa is 0 at DC and 1 at the (0.5, 0.5) corner.
    """
    fy = np.fft.fftfreq(shape[0]).reshape(-1, 1)
    fx = np.fft.fftfreq(shape[1]).reshape(1, -1)
    kappa = np.sqrt(fy * fy + fx * fx) / math.sqrt(0.5**2 + 0.5**2)
    return np.fft.fftshift(kappa) if centered else kappa


def subband_b(shape: tuple[int, int], centered: bool, kappa_max: float = SUBBAND_B_KAPPA_MAX) -> np.ndarray:
    """Boolean mask of the retained sub-band B = {k : kappa(k) <= kappa_max} (default 0.15)."""
    return kappa_grid(shape, centered=centered) <= kappa_max
