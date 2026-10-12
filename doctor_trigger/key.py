"""The secret key s and everything derived from it: sign pattern, active bins, displacement.

Sec. 3.1 of the paper: a key s seeds a standard-normal field u_s over the Fourier grid; the
sign pattern h_s(k) = sign(u_s(k) - u_s(-k)) is +1 or -1 on every bin paired with a distinct
conjugate bin and 0 on self-conjugate bins. h_s is one luminance pattern, applied unchanged to
all three colour channels; it is binary, never a continuous value.

Hash salts below (``doctor-trigger-v1``, ``wrong-key``, ``0x9E3779B97F4A7C15``) define every
key ever enrolled and must never change.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

KEY_SALT = "doctor-trigger-v1"
WRONG_KEY_SEED_MIX = 0x9E3779B97F4A7C15
WRONG_KEY_QUERY_SALT = "wrong-key"
WRONG_KEY_COUNT = 8
"""m = 8 wrong keys per record (Sec. 3.3, key-contrast features)."""


def derive_key(master_seed: int, sample_id: str) -> int:
    """Secret key s of one enrolled face: the first 8 bytes of SHA-256 over seed and sample id.

    Returns an unsigned 64-bit integer (stored in the record as ``key``).
    """
    payload = f"{KEY_SALT}\x1f{master_seed}\x1f{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def sign_pattern(shape: tuple[int, int], key: int, centered: bool = False) -> np.ndarray:
    """Sign pattern h_s(k) = sign(u_s(k) - u_s(-k)) as a float64 array of -1, 0, +1.

    u_s is ``np.random.default_rng(key).standard_normal(shape)``; -k is taken modulo the grid
    dimensions, so h_s(-k) = -h_s(k) and the edited spectrum stays Hermitian. Self-conjugate bins
    (k = -k) get 0. ``centered=True`` returns the pattern in record layout (``fftshift``).
    """
    standard_normal = np.random.default_rng(key).standard_normal(shape)
    conjugate_rows = np.mod(-np.arange(shape[0]), shape[0])
    conjugate_cols = np.mod(-np.arange(shape[1]), shape[1])
    conjugate = standard_normal[np.ix_(conjugate_rows, conjugate_cols)]
    pattern = np.sign(standard_normal - conjugate)
    self_conjugate_rows = np.arange(shape[0]) == conjugate_rows
    self_conjugate_cols = np.arange(shape[1]) == conjugate_cols
    pattern[np.ix_(self_conjugate_rows, self_conjugate_cols)] = 0.0
    pattern = pattern.astype(np.float64)
    return np.fft.fftshift(pattern) if centered else pattern


@dataclass(frozen=True)
class Ring:
    """Frequency ring [kappa_min, kappa_max] that carries the keyed signal (Eq. 2)."""

    kappa_min: float
    kappa_max: float


def active_bins(
    kappa: np.ndarray,
    amplitude: np.ndarray,
    pattern: np.ndarray,
    ring: Ring,
    amplitude_floor_ratio: float,
) -> np.ndarray:
    """Active bins K_eff = {k : kappa_min <= kappa(k) <= kappa_max, A_I(k) > A_min, h_s(k) != 0}.

    ``kappa``, ``amplitude`` (A_I, original luminance magnitude) and ``pattern`` (h_s) must share
    one layout. The amplitude floor is A_min = amplitude_floor_ratio * max_k A_I(k).
    Returns a boolean mask.
    """
    amplitude_floor = float(np.max(amplitude) * amplitude_floor_ratio)
    return (kappa >= ring.kappa_min) & (kappa <= ring.kappa_max) & (amplitude > amplitude_floor) & (pattern != 0)


def phase_challenge(pattern: np.ndarray, alpha: float, active: np.ndarray) -> np.ndarray:
    """Pre-blend phase challenge Delta phi_s(k) = alpha pi h_s(k) on active bins, 0 elsewhere."""
    return alpha * np.pi * pattern * active


def displacement(pattern: np.ndarray, alpha: float, beta: float, active: np.ndarray) -> np.ndarray:
    """Signed keyed displacement delta_s(k) = arg{(1 - beta) + beta e^{i alpha pi h_s(k)}} (Eq. 3).

    This is the phase shift the blend leaves on each active bin (+-delta_eff); it is 0 outside
    ``active`` because the challenge is 0 there. Used for the enrolled key at enrollment and for
    wrong keys at verification. The result keeps the memory layout NumPy gives this expression
    (the enrollment template inherits it, and the NPZ records its Fortran/C order).
    """
    challenge = phase_challenge(pattern, alpha, active)
    return np.angle((1.0 - beta) + beta * np.exp(1j * challenge))


def wrong_keys(key: int, m: int = WRONG_KEY_COUNT) -> list[int]:
    """The m wrong keys s^-_1..s^-_m compared against the enrolled key s (Sec. 3.3).

    Drawn from a PCG64 generator seeded with ``key XOR 0x9E3779B97F4A7C15``; each differs from
    ``key`` and from the others. Deterministic per key.
    """
    generator = np.random.default_rng((key ^ WRONG_KEY_SEED_MIX) & ((1 << 64) - 1))
    keys: list[int] = []
    while len(keys) < m:
        candidate = int(generator.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        if candidate != key and candidate not in keys:
            keys.append(candidate)
    return keys


def wrong_key_query_key(key: int) -> int:
    """Projection key of the benign wrong-key query built from a record whose key is ``key``.

    The first 16 hex digits of SHA-256 over ``"wrong-key\\x1f<key>"``, as an integer.
    """
    return int(hashlib.sha256(f"{WRONG_KEY_QUERY_SALT}\x1f{key}".encode()).hexdigest()[:16], 16)
