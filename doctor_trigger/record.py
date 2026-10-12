"""The private record R of one enrolled unit, its NPZ file, and where records live.

A record (Sec. 3.1, Eq. 4) holds, per Fourier bin of the luminance spectrum: the template T_I,
the original phase phi-hat_I, the displacement delta_s, the amplitude weights w_k, the active
bins K_eff and the sign pattern h_s; plus the key s, strengths alpha and beta, the ring and the
amplitude-floor ratio.

On disk (NPZ, never renamed): ``template_phase``, ``host_phase``, ``delta``, ``host_amplitude``,
``support``, ``code``, ``key``, ``alpha``, ``beta``, ``bands``, ``amplitude_floor_ratio``.
Arrays are stored centered (``fftshift``); phases and weights as float16, which quantizes them
to about 0.002 rad. In memory a ``Record`` holds exactly the stored values: float arrays are
widened to float64 and ``support`` to bool (both exact), scalars come back from their float32 /
uint64 storage. Saving a ``Record`` therefore writes the same bytes it was loaded from.

A unit is one enrolled crop. Images are units with ``frame_index=None``; a video module may
later enroll one unit per frame.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .key import Ring

ENROLLMENT_ID_SALT = "enrollment-v1"


@dataclass(frozen=True)
class UnitId:
    """Identity of one enrolled unit: the asset (an image's ``sample_id``) and, for video, a frame."""

    asset_id: str
    frame_index: int | None = None


@dataclass(frozen=True, eq=False)
class Record:
    """Private record R of one unit. Arrays are centered (record layout) and read-only."""

    template_phase: np.ndarray
    """T_I = wrap(phi_I + delta_s), the luminance phase after the ideal edit (float64)."""
    original_phase: np.ndarray
    """phi-hat_I, the original luminance phase (float64; NPZ key ``host_phase``)."""
    displacement: np.ndarray
    """delta_s, the signed keyed displacement, 0 outside K_eff (float64; NPZ key ``delta``)."""
    amplitude_weights: np.ndarray
    """w_k = A_I(k) / max A_I, original luminance amplitude weights (float64; NPZ key ``host_amplitude``)."""
    active_bins: np.ndarray
    """K_eff as a boolean mask (NPZ key ``support``)."""
    sign_pattern: np.ndarray
    """h_s in {-1, 0, +1} (float64; NPZ key ``code``, int8)."""
    key: int
    """Secret key s (uint64)."""
    alpha: float
    """Phase strength alpha, in units of pi radians (stored as float32)."""
    beta: float
    """Blend weight beta (stored as float32)."""
    ring: Ring
    """Frequency ring [kappa_min, kappa_max] (NPZ key ``bands``, float32)."""
    amplitude_floor_ratio: float
    """A_min / max A_I (stored as float32)."""

    def __post_init__(self) -> None:
        for name in ("template_phase", "original_phase", "displacement", "amplitude_weights", "active_bins", "sign_pattern"):
            getattr(self, name).flags.writeable = False

    @property
    def shape(self) -> tuple[int, int]:
        """Fourier grid shape (crop height, crop width)."""
        return self.template_phase.shape


def record_from_stored_arrays(arrays: Mapping[str, np.ndarray]) -> Record:
    """Build a ``Record`` from arrays keyed and typed as in the NPZ file."""
    bands = arrays["bands"]
    return Record(
        template_phase=np.asarray(arrays["template_phase"]).astype(np.float64),
        original_phase=np.asarray(arrays["host_phase"]).astype(np.float64),
        displacement=np.asarray(arrays["delta"]).astype(np.float64),
        amplitude_weights=np.asarray(arrays["host_amplitude"]).astype(np.float64),
        active_bins=np.asarray(arrays["support"]).astype(bool),
        sign_pattern=np.asarray(arrays["code"]).astype(np.float64),
        key=int(arrays["key"][0]),
        alpha=float(arrays["alpha"][0]),
        beta=float(arrays["beta"][0]),
        ring=Ring(float(bands[0]), float(bands[1])),
        amplitude_floor_ratio=float(arrays["amplitude_floor_ratio"][0]),
    )


def stored_arrays(record: Record) -> dict[str, np.ndarray]:
    """The record as NPZ arrays: on-disk keys, dtypes and key order."""
    return {
        "template_phase": record.template_phase.astype(np.float16),
        "host_phase": record.original_phase.astype(np.float16),
        "delta": record.displacement.astype(np.float16),
        "host_amplitude": record.amplitude_weights.astype(np.float16),
        "support": record.active_bins.astype(np.uint8),
        "code": record.sign_pattern.astype(np.int8),
        "key": np.asarray([record.key], dtype=np.uint64),
        "alpha": np.asarray([record.alpha], dtype=np.float32),
        "beta": np.asarray([record.beta], dtype=np.float32),
        "bands": np.asarray([record.ring.kappa_min, record.ring.kappa_max], dtype=np.float32),
        "amplitude_floor_ratio": np.asarray([record.amplitude_floor_ratio], dtype=np.float32),
    }


def load_record(path: Path | str) -> Record:
    """Read one record NPZ."""
    with np.load(path) as arrays:
        return record_from_stored_arrays(arrays)


def save_record(path: Path | str, record: Record) -> None:
    """Write one record NPZ (compressed), byte-identical to the enrollment workflow's files."""
    np.savez_compressed(path, **stored_arrays(record))


def enrollment_id(sample_id: str) -> str:
    """Enrollment id of an image: first 24 hex digits of SHA-256 over ``"enrollment-v1\\x1f<sample_id>"``."""
    return hashlib.sha256(f"{ENROLLMENT_ID_SALT}\x1f{sample_id}".encode("utf-8")).hexdigest()[:24]


def image_unit_path(root: Path | str, dataset: str, split: str, unit: UnitId, suffix: str) -> Path:
    """Path of an image unit's file: ``<root>/<dataset>/<split>/<enrollment_id><suffix>``.

    The same layout holds records (``.npz``, ``.json``) and released crops (``.png``).
    """
    if unit.frame_index is not None:
        raise ValueError(f"image layout has no frame index: {unit}")
    return Path(root) / dataset / split / f"{enrollment_id(unit.asset_id)}{suffix}"


class RecordStore:
    """Maps each ``UnitId`` to its record file and loads records one at a time, on request."""

    def __init__(self, paths: Mapping[UnitId, Path]) -> None:
        self._paths = dict(paths)

    @classmethod
    def for_images(cls, root: Path | str, units: Iterable[tuple[UnitId, str, str]]) -> RecordStore:
        """Store over today's image layout; ``units`` yields (unit, dataset, split)."""
        return cls({unit: image_unit_path(root, dataset, split, unit, ".npz") for unit, dataset, split in units})

    def path(self, unit: UnitId) -> Path:
        """Record file of ``unit``."""
        return self._paths[unit]

    def load(self, unit: UnitId) -> Record:
        """Read the record of ``unit`` from disk (not cached)."""
        return load_record(self._paths[unit])

    def __contains__(self, unit: object) -> bool:
        return unit in self._paths

    def __iter__(self) -> Iterator[UnitId]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)
