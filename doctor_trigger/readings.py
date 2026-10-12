"""Readings of a query J against a record R: phase drift, spectral agreement, key projection.

Sec. 3.2-3.3 of the paper. On the luminance phase phi_J of a query and the record's arrays:

- phase drift          r_k(J) = wrap(phi_J(k) - phi-hat_I(k))
- spectral agreement   the amplitude-weighted mean of cos(phi_J - reference) over a bin set (Eq. 5):
                       against the template T_I it is C(J; R), against the original phase phi-hat_I
                       it is H(J; R) (original-reference agreement); restricted to K_eff and the
                       retained sub-band B it is C_B / H_B
- key projection       G_s(J; R) = sum w delta_s sin r / (sum w delta_s^2 + eps_0) (Eq. 6), and its
                       release-normalized form G_s^(release) = G_s(J) / max(|G_s(I')|, 1e-6)
- wrong-key statistics (G-bar^-, G^-_max, G_Delta, G_z) over m = 8 wrong keys (Eq. 7)
- feature map          x = Phi(J, R), the 75 record-model inputs (Eq. 8, App. D.1)

Summation forms. The feature pipeline sums over the selected bins only (a boolean gather) and returns
NaN when their total weight is at most 1e-12. Most experiments instead sum over the whole grid with
the weights zeroed outside the bins (a weight map) and add a stabilizer (1e-12, 1e-8 or 1e-6) to the
denominator. The forms agree mathematically but not in the last bits, and a whole-grid sum also
depends on the memory layout of the product it sums, so the kernels take ``form`` (``SumForm``) and
``stabilizer`` explicitly; the defaults are the feature pipeline's. The kernels work in whatever
layout (centered or not, full grid or a vector of gathered bins) the caller passes; record arrays
should be passed as loaded (T_I, phi-hat_I and w are Fortran-ordered on disk).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np
from PIL import Image

from .key import WRONG_KEY_COUNT, displacement, sign_pattern, wrong_keys
from .record import Record, UnitId
from .spectrum import SUBBAND_B_KAPPA_MAX, luminance_phase, subband_b, wrap_phase

EMPTY_WEIGHT_LIMIT = 1e-12
"""A gathered amplitude-weighted mean whose total weight is at most this is NaN (no bins to read)."""
PROJECTION_STABILIZER = 1e-12
"""eps_0, the key-projection denominator stabilizer of Eq. (6)."""
RELEASE_PROJECTION_FLOOR = 1e-6
"""Floor on |G_s(I')| in the release normalization G_s(J) / max(|G_s(I')|, 1e-6) (Sec. 3.2)."""
WRONG_KEY_SPREAD_STABILIZER = 1e-8
"""Added to std_j G^-_j in the denominator of G_z (Eq. 7)."""
STRENGTH_PRODUCT_FLOOR = 1e-12
"""Floor on alpha * beta in the ``delta_normalized_mean`` feature."""
FEATURE_BAND_COUNT = 8
"""Number of radial bands l = 1..8 in Phi (Eq. 8)."""
PIXEL_RADIUS_STABILIZER = 1e-12
"""Added to the largest pixel radius when normalizing it, and to the top band edge so the
outermost active bin falls inside the last band."""
SIGN_PATTERN_CACHE_SIZE = 64
"""Sign patterns kept per process; a record's wrong keys recur across its benign and swap queries."""


# ----------------------------------------------------------------------------- query phase
class SizeMismatch(Enum):
    """What to do when a query's size differs from the record's Fourier grid."""

    RAISE = "raise"
    """Raise ``ValueError`` (the feature pipeline)."""
    UNAVAILABLE = "unavailable"
    """Return ``None``: the geometry is unresolved, so the reading is unavailable (Eq. 11)."""
    RESIZE = "resize"
    """Resize the query to the record's size with bicubic resampling, then read it."""


def query_phase(
    image: Image.Image,
    shape: tuple[int, int],
    size_mismatch: SizeMismatch = SizeMismatch.RAISE,
    centered: bool = True,
) -> np.ndarray | None:
    """Luminance phase phi_J of a query image on a record's Fourier grid of ``shape``.

    ``centered=True`` gives record layout. A size mismatch is handled as ``size_mismatch`` says.
    """
    image = image.convert("RGB")
    if (image.size[1], image.size[0]) != tuple(shape):
        if size_mismatch is SizeMismatch.RAISE:
            raise ValueError(f"query/record shape mismatch: {(image.size[1], image.size[0])} != {tuple(shape)}")
        if size_mismatch is SizeMismatch.UNAVAILABLE:
            return None
        image = image.resize((shape[1], shape[0]), Image.BICUBIC)
    return luminance_phase(np.asarray(image, dtype=np.float64), centered=centered)


# ----------------------------------------------------------------------------- kernels
class SumForm(Enum):
    """How a reading sums over its bins (see the module docstring)."""

    GATHERED = "gathered"
    """Sum the gathered bins, skipping non-finite entries (``record_features``, ``original_control``)."""
    WEIGHT_MAP = "weight_map"
    """Sum the whole grid with weights 0 outside the bins, ``np.sum(values * where(bins, w, 0))``;
    the product takes NumPy's default layout (scripts that name the cosine first, ``r = np.cos(...)``)."""
    WEIGHT_MAP_VALUE_LAYOUT = "weight_map_value_layout"
    """As ``WEIGHT_MAP``, but the product keeps the layout of the values. This is what an inline
    ``np.sum(np.cos(...) * w)`` gives, because NumPy then reuses the cosine's temporary buffer for
    arrays of at least 256 KiB (``operating_points``, ``table4_artifact``, ``diffusion_regeneration``)."""


NUMPY_TEMPORARY_REUSE_BYTES = 256 * 1024
"""Smallest array NumPy reuses as the output of ``temporary * other`` (NPY_MIN_ELIDE_BYTES)."""


def _weight_map_product(values: np.ndarray, masked_weights: np.ndarray, form: SumForm) -> np.ndarray:
    if form is SumForm.WEIGHT_MAP_VALUE_LAYOUT and values.nbytes >= NUMPY_TEMPORARY_REUSE_BYTES:
        return np.multiply(values, masked_weights, out=np.empty_like(values))
    return values * masked_weights


def phase_drift(query_phase: np.ndarray, original_phase: np.ndarray) -> np.ndarray:
    """Wrapped phase drift r_k(J) = wrap(phi_J(k) - phi-hat_I(k)) of the query (Sec. 3.2)."""
    return wrap_phase(query_phase - original_phase)


def amplitude_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    *,
    stabilizer: float | None = None,
    form: SumForm = SumForm.GATHERED,
) -> float:
    """sum_{k in bins} w_k v_k / sum_{k in bins} w_k, the amplitude-weighted mean of App. D.1.

    With ``stabilizer=None`` a total weight at or below 1e-12 gives NaN; otherwise the stabilizer is
    added to the denominator. The ``GATHERED`` form skips bins where ``values`` or ``weights`` are
    not finite; in the weight-map forms a non-finite value propagates, as in the scripts they replace.
    """
    if form is SumForm.GATHERED:
        mask = bins & np.isfinite(values) & np.isfinite(weights)
        total = float(np.sum(weights[mask]))
        weighted_sum = np.sum(values[mask] * weights[mask])
    else:
        masked_weights = np.where(bins, weights, 0.0)
        total = float(np.sum(masked_weights))
        weighted_sum = np.sum(_weight_map_product(values, masked_weights, form))
    if stabilizer is None:
        if total <= EMPTY_WEIGHT_LIMIT:
            return float("nan")
        return float(weighted_sum / total)
    return float(weighted_sum / (total + stabilizer))


def spectral_agreement(
    query_phase: np.ndarray,
    reference_phase: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    *,
    stabilizer: float | None = None,
    form: SumForm = SumForm.GATHERED,
) -> float:
    """Spectral agreement (Eq. 5): amplitude-weighted mean of cos wrap(phi_J - reference) over ``bins``.

    The reference names the reading: the template T_I gives C(J; R), the original phase phi-hat_I
    gives H(J; R), the original-reference agreement. ``bins`` = K_eff gives the full-ring reading,
    K_eff and sub-band B (``subband_bins``) gives C_B / H_B. The feature pipeline's ``C`` column uses
    the released crop's phase as the reference. ``stabilizer`` and ``form`` choose the summation
    (``amplitude_weighted_mean``).
    """
    return amplitude_weighted_mean(
        np.cos(wrap_phase(query_phase - reference_phase)),
        weights,
        bins,
        stabilizer=stabilizer,
        form=form,
    )


def key_projection(
    drift: np.ndarray,
    displacement: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    *,
    stabilizer: float = PROJECTION_STABILIZER,
    form: SumForm = SumForm.GATHERED,
) -> float:
    """Key projection G_s (Eq. 6): sum w delta sin r / (sum w delta^2 + eps_0) over ``bins``.

    ``drift`` is r_k(J) (``phase_drift``) and ``displacement`` the key's delta_s. ``stabilizer`` is
    eps_0 (1e-12 in the paper; some experiments use 1e-6). The ``GATHERED`` form skips bins where
    drift, displacement or weights are not finite; ``WEIGHT_MAP`` sums the whole grid with weights
    zeroed outside the bins (the products then keep NumPy's own layout, as in the scripts).
    """
    if form is SumForm.GATHERED:
        mask = bins & np.isfinite(drift) & np.isfinite(displacement) & np.isfinite(weights)
        numerator = float(np.sum(weights[mask] * displacement[mask] * np.sin(drift[mask])))
        denominator = float(np.sum(weights[mask] * displacement[mask] * displacement[mask]))
    elif form is SumForm.WEIGHT_MAP:
        masked_weights = np.where(bins, weights, 0.0)
        numerator = float(np.sum(masked_weights * displacement * np.sin(drift)))
        denominator = float(np.sum(masked_weights * displacement * displacement))
    else:
        raise ValueError(f"key_projection has no {form} form")
    return numerator / (denominator + stabilizer)


def release_normalized(projection: float, release_projection: float, floor: float = RELEASE_PROJECTION_FLOOR) -> float:
    """G_s^(release) = G_s(J) / max(|G_s(I')|, 1e-6): the projection relative to the release's own (Sec. 3.2).

    With the template's projection as ``release_projection`` this is G_s^(template).
    """
    return projection / max(abs(release_projection), floor)


@dataclass(frozen=True)
class WrongKeyStats:
    """Key-contrast features of Eq. (7) from the enrolled-key and m wrong-key projections."""

    mean: float
    """G-bar^-, the mean wrong-key projection (CSV ``G_wrong_mean``)."""
    std: float
    """std_j G^-_j with divisor m (CSV ``G_wrong_std``)."""
    max: float
    """G^-_max (CSV ``G_wrong_max``)."""
    margin: float
    """G_Delta = G_s - G^-_max (CSV ``G_margin``)."""
    z: float
    """G_z = (G_s - G-bar^-) / (std_j G^-_j + 1e-8) (CSV ``G_z``)."""


def wrong_key_statistics(enrolled: float, wrong: np.ndarray) -> WrongKeyStats:
    """Eq. (7): mean, spread, maximum, margin and standardized separation of wrong-key projections."""
    wrong = np.asarray(wrong, dtype=np.float64)
    return WrongKeyStats(
        mean=float(np.mean(wrong)),
        std=float(np.std(wrong)),
        max=float(np.max(wrong)),
        margin=float(enrolled - np.max(wrong)),
        z=float((enrolled - np.mean(wrong)) / (np.std(wrong) + WRONG_KEY_SPREAD_STABILIZER)),
    )


# ----------------------------------------------------------------------------- record helpers
def subband_bins(record: Record, kappa_max: float = SUBBAND_B_KAPPA_MAX) -> np.ndarray:
    """K_eff and the retained sub-band B = {kappa <= kappa_max}, in record (centered) layout."""
    return record.active_bins & subband_b(record.shape, centered=True, kappa_max=kappa_max)


@lru_cache(maxsize=SIGN_PATTERN_CACHE_SIZE)
def _centered_sign_pattern(shape: tuple[int, int], key: int) -> np.ndarray:
    pattern = sign_pattern(shape, key, centered=True)
    pattern.flags.writeable = False
    return pattern


def wrong_key_displacement(record: Record, key: int) -> np.ndarray:
    """delta_s for a key other than the record's own, on the record's active bins (0 elsewhere).

    Uses the record's alpha, beta and K_eff, as the verifier does for each wrong key s^-_j.
    """
    return displacement(_centered_sign_pattern(record.shape, key), record.alpha, record.beta, record.active_bins)


@lru_cache(maxsize=8)
def normalized_pixel_radius(shape: tuple[int, int]) -> np.ndarray:
    """Euclidean distance of each bin from the centre of the shifted array, divided by its maximum.

    This is the radius of the eight MLP bands (App. D.1); it is not kappa (pixel units, not cycles).
    """
    y, x = np.ogrid[: shape[0], : shape[1]]
    centre_y, centre_x = shape[0] / 2.0, shape[1] / 2.0
    radius = np.sqrt((y - centre_y) ** 2 + (x - centre_x) ** 2)
    radius = radius / (float(np.max(radius)) + PIXEL_RADIUS_STABILIZER)
    radius.flags.writeable = False
    return radius


def feature_band_edges(active_radius: np.ndarray, band_count: int = FEATURE_BAND_COUNT) -> np.ndarray:
    """Edges of the radial bands of Phi: ``band_count`` equal intervals over the active bins' radius."""
    return np.linspace(
        float(np.min(active_radius)),
        float(np.max(active_radius)) + PIXEL_RADIUS_STABILIZER,
        band_count + 1,
    )


# ----------------------------------------------------------------------------- feature map
GLOBAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "H", "C", "residual_mean", "residual_std", "sin_projection_mean", "cos_consistency_mean",
    "valid_count", "valid_fraction", "delta_abs_mean", "delta_normalized_mean",
    "G_s", "G_wrong_mean", "G_wrong_std", "G_wrong_max", "G_margin", "G_z", "n_wrong",
)
BAND_FEATURE_STEMS: tuple[str, ...] = (
    "H", "C", "G", "G_margin", "residual_mean", "residual_std", "sin_projection_mean",
    "cos_consistency_mean", "valid_count",
)


def feature_columns(band_count: int = FEATURE_BAND_COUNT) -> list[str]:
    """CSV column names written by ``record_features``, in order (on-disk names, never renamed)."""
    return list(GLOBAL_FEATURE_COLUMNS) + [
        f"{stem}_b{band}" for band in range(band_count) for stem in BAND_FEATURE_STEMS
    ]


FEATURE_COLUMNS: tuple[str, ...] = tuple(feature_columns())


def record_features(
    query_phase: np.ndarray,
    record: Record,
    reference_phase: np.ndarray | None = None,
    projection_key: int | None = None,
    wrong_key_count: int = WRONG_KEY_COUNT,
    band_count: int = FEATURE_BAND_COUNT,
) -> dict[str, float]:
    """Feature map x = Phi(J, R) (Eq. 8, App. D.1), keyed by the calibrated-features CSV columns.

    ``query_phase`` is phi_J in record layout. ``reference_phase`` is the phase that normalizes the
    projections and that ``C`` is read against: the released crop's phase in the pipeline
    (G_s^(release)), the template T_I when omitted (G_s^(template)). ``projection_key`` replaces
    the enrolled key for a wrong-key query (``key.wrong_key_query_key``); H, C and the residual
    statistics do not depend on it. Columns, in order, are ``feature_columns(band_count)``:

    full ring  H; C; amplitude-weighted mean and plain std of r; weighted mean of delta sin r
               (``sin_projection_mean``) and of cos r (``cos_consistency_mean``, equal to H); active
               bin count and fraction; mean |delta| and mean |delta| / (alpha beta); G_s; Eq. (7)
               statistics over ``wrong_key_count`` wrong keys; ``n_wrong``.
    band l     the same readings on K_eff restricted to radial band l. ``G_margin_b<l>`` equals
               ``G_b<l>`` (the band key-distance column, kept because checkpoints read it).

    Every reading is computed on the vector of active bins (gathered once in C order), which is
    the order the gathered sums over the full grid use, so values equal the full-grid computation
    bit for bit.
    """
    if query_phase.shape != record.shape:
        raise ValueError(f"query/template shape mismatch: {query_phase.shape} != {record.shape}")
    if reference_phase is None:
        reference_phase = record.template_phase
    if reference_phase.shape != record.shape:
        raise ValueError(f"reference/template shape mismatch: {reference_phase.shape} != {record.shape}")
    key = record.key if projection_key is None else projection_key
    active = record.active_bins
    every_bin = np.ones(int(np.count_nonzero(active)), dtype=bool)

    query = query_phase[active]
    reference = reference_phase[active]
    original = record.original_phase[active]
    weights = record.amplitude_weights[active]
    enrolled_displacement = record.displacement[active]

    def active_displacement(other_key: int) -> np.ndarray:
        # delta_s of another key, evaluated on the active bins only (elementwise, so identical to
        # wrong_key_displacement(record, other_key)[active]).
        pattern = _centered_sign_pattern(record.shape, other_key)[active]
        return displacement(pattern, record.alpha, record.beta, every_bin)

    projection_displacement = enrolled_displacement if key == record.key else active_displacement(key)
    drift = phase_drift(query, original)
    reference_drift = phase_drift(reference, original)
    cos_drift = np.cos(drift)
    cos_reference = np.cos(wrap_phase(query - reference))
    sin_projection = projection_displacement * np.sin(drift)
    wrong_displacements = [active_displacement(candidate) for candidate in wrong_keys(key, wrong_key_count)]

    def projections(bins: np.ndarray) -> tuple[float, list[float]]:
        scale = max(abs(key_projection(reference_drift, enrolled_displacement, weights, bins)), RELEASE_PROJECTION_FLOOR)
        enrolled = key_projection(drift, projection_displacement, weights, bins) / scale
        return enrolled, [key_projection(drift, wrong, weights, bins) / scale for wrong in wrong_displacements]

    values: dict[str, float] = {
        "H": amplitude_weighted_mean(cos_drift, weights, every_bin),
        "C": amplitude_weighted_mean(cos_reference, weights, every_bin),
        "residual_mean": amplitude_weighted_mean(drift, weights, every_bin),
        "residual_std": float(np.std(drift)),
        "sin_projection_mean": amplitude_weighted_mean(sin_projection, weights, every_bin),
        "cos_consistency_mean": amplitude_weighted_mean(cos_drift, weights, every_bin),
        "valid_count": int(np.sum(active)),
        "valid_fraction": float(np.mean(active)),
        "delta_abs_mean": float(np.mean(np.abs(projection_displacement))),
        "delta_normalized_mean": float(
            np.mean(np.abs(projection_displacement)) / max(record.alpha * record.beta, STRENGTH_PRODUCT_FLOOR)
        ),
    }
    enrolled, wrong = projections(every_bin)
    stats = wrong_key_statistics(enrolled, np.asarray(wrong, dtype=np.float64))
    values.update(
        {
            "G_s": enrolled,
            "G_wrong_mean": stats.mean,
            "G_wrong_std": stats.std,
            "G_wrong_max": stats.max,
            "G_margin": stats.margin,
            "G_z": stats.z,
            "n_wrong": wrong_key_count,
        }
    )
    radius = normalized_pixel_radius(record.shape)[active]
    edges = feature_band_edges(radius, band_count)
    for band_index in range(band_count):
        band = (radius >= edges[band_index]) & (radius < edges[band_index + 1])
        scale = max(abs(key_projection(reference_drift, enrolled_displacement, weights, band)), RELEASE_PROJECTION_FLOOR)
        band_projection = key_projection(drift, projection_displacement, weights, band) / scale
        values[f"H_b{band_index}"] = amplitude_weighted_mean(cos_drift, weights, band)
        values[f"C_b{band_index}"] = amplitude_weighted_mean(cos_reference, weights, band)
        values[f"G_b{band_index}"] = band_projection
        values[f"G_margin_b{band_index}"] = band_projection
        values[f"residual_mean_b{band_index}"] = amplitude_weighted_mean(drift, weights, band)
        values[f"residual_std_b{band_index}"] = float(np.std(drift[band])) if np.any(band) else float("nan")
        values[f"sin_projection_mean_b{band_index}"] = amplitude_weighted_mean(sin_projection, weights, band)
        values[f"cos_consistency_mean_b{band_index}"] = amplitude_weighted_mean(cos_drift, weights, band)
        values[f"valid_count_b{band_index}"] = int(np.sum(band))
    return values


# ----------------------------------------------------------------------------- one unit
@dataclass(frozen=True)
class UnitReading:
    """Direct-verification readings of one unit (one image today; one frame of a clip later).

    Readings are ``None`` when the unit is unavailable (no record, or the query does not match the
    record's geometry); ``unavailable_reason`` then says why.
    """

    unit: UnitId | None
    original_reference_agreement: float | None
    """H, or H_B when ``subband_kappa_max`` is set."""
    spectral_agreement: float | None
    """C against the template T_I, or C_B when ``subband_kappa_max`` is set."""
    key_projection: float | None
    """G_s^(release) when the release was given, else the unnormalized G_s of Eq. (6)."""
    subband_kappa_max: float | None
    """Upper kappa of sub-band B the agreements were restricted to; ``None`` for the full ring."""
    release_normalized: bool
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """eligible(J) of Eq. (11), as far as the readings can tell: record and geometry resolved."""
        return self.unavailable_reason is None


def read_unit(
    query_rgb: np.ndarray | None,
    record: Record | None,
    *,
    unit: UnitId | None = None,
    release_rgb: np.ndarray | None = None,
    subband_kappa_max: float | None = SUBBAND_B_KAPPA_MAX,
) -> UnitReading:
    """Read H (H_B), C (C_B) and G_s of a query crop against its record (gathered summation form).

    ``query_rgb`` and ``release_rgb`` are (H, W, 3) arrays. With the released crop, G_s is
    release-normalized. ``subband_kappa_max=None`` reads the agreements over the full ring K_eff.
    A missing query or record, or a size mismatch, gives an unavailable reading.
    """
    def unavailable(reason: str) -> UnitReading:
        return UnitReading(unit, None, None, None, subband_kappa_max, release_rgb is not None, reason)

    if record is None:
        return unavailable("record not resolved")
    if query_rgb is None:
        return unavailable("query missing")
    if query_rgb.shape[:2] != record.shape:
        return unavailable(f"query size {query_rgb.shape[:2]} differs from the record's {record.shape}")
    phase = luminance_phase(query_rgb, centered=True)
    bins = record.active_bins if subband_kappa_max is None else subband_bins(record, subband_kappa_max)
    weights = record.amplitude_weights
    projection = key_projection(phase_drift(phase, record.original_phase), record.displacement, weights, record.active_bins)
    if release_rgb is not None:
        if release_rgb.shape[:2] != record.shape:
            raise ValueError(f"released crop size {release_rgb.shape[:2]} differs from the record's {record.shape}")
        release_drift = phase_drift(luminance_phase(release_rgb, centered=True), record.original_phase)
        projection = release_normalized(
            projection, key_projection(release_drift, record.displacement, weights, record.active_bins)
        )
    return UnitReading(
        unit=unit,
        original_reference_agreement=spectral_agreement(phase, record.original_phase, weights, bins),
        spectral_agreement=spectral_agreement(phase, record.template_phase, weights, bins),
        key_projection=projection,
        subband_kappa_max=subband_kappa_max,
        release_normalized=release_rgb is not None,
    )
