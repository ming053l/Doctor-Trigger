"""Deployment cost of the enrolled path (Table 16): CPU timings, parameter counts, record size.

Timing: 20 earlier trigger renders (config key ``legacy_trigger_glob``, not shipped) are given a
keyed phase nudge (one FFT, the displacement delta_s added on the support, one inverse FFT) and
then verified by the original-reference agreement H (one FFT, one cosine mean over the support).
No network runs on this path. The timings are indicative CPU measurements; the Table 16 timings
come from a later benchmark that is not part of this release.

Support. ``per_asset_template_bins`` counts the bins one asset stores. It is the ring
0.10 <= kappa <= 0.45 of the renders, evaluated on the centered kappa grid and applied to the
uncentered spectrum, keeping one bin of each conjugate pair (k and -k carry the same phase up to
sign). This is the support the earlier renders were timed with, so the count (9,849 bins for a
256 x 256 crop) is unchanged; the full ring with both conjugate bins has twice as many.

Sizes: the record-model parameter count follows its definition (75 inputs, hidden width 64, two
8-dimensional embeddings); the image-only detector is built from its definition and its deployed
checkpoint (FF++, augmented) is measured on disk.
"""
import glob
import json
import os
import time

import numpy as np
from PIL import Image

from doctor_trigger import config
from doctor_trigger.key import displacement, sign_pattern
from doctor_trigger.readings import spectral_agreement
from doctor_trigger.spectrum import kappa_grid, luminance_phase, luminance_spectrum, wrap_phase
from experiments.shared.image_only_detector import build_image_only_detector, checkpoint_path

CROP = 256
TIMED_IMAGES = 20
WARM_UP = 3
RECORD_MODEL_PARAMETERS = 75 * 64 + 64 + 64 * 64 + 64 + 3 * (64 + 1) + 2 * (64 * 8 + 8)


def one_bin_per_conjugate_pair(shape):
    """Bins with fy > 0, or fy == 0 and fx >= 0 (uncentered layout): one of each pair k, -k."""
    fy = np.fft.fftfreq(shape[0]).reshape(-1, 1)
    fx = np.fft.fftfreq(shape[1]).reshape(1, -1)
    return (fy > 0.0) | ((fy == 0.0) & (fx >= 0.0))


def timing_support(shape, kappa_min, kappa_max):
    kappa = kappa_grid(shape, centered=True)
    return (kappa >= kappa_min) & (kappa <= kappa_max) & one_bin_per_conjugate_pair(shape)


renders = [json.load(open(path)) for path in sorted(glob.glob(config.path("legacy_trigger_glob"), recursive=True))[:25]]
renders = [render for render in renders if os.path.exists(render["trigger_path"])]
images = [np.asarray(Image.open(render["trigger_path"]).convert("RGB").resize((CROP, CROP)))
          for render in renders[:TIMED_IMAGES]]
first = renders[0]
support = timing_support((CROP, CROP), *first["bands"])
delta = displacement(sign_pattern((CROP, CROP), int(first["phase_seed"])), float(first["alpha"]),
                     float(first["beta"]), support)
uniform_weights = np.ones((CROP, CROP))


def enroll(image):
    """Keyed phase nudge of the luminance spectrum; returns the edited luminance and its phase (the template)."""
    spectrum = luminance_spectrum(image)
    phase = np.angle(spectrum)
    template = phase.copy()
    template[support] = wrap_phase(phase[support] + delta[support])
    return np.real(np.fft.ifft2(np.abs(spectrum) * np.exp(1j * template))), template


def verify(image, template):
    """H: unweighted mean of cos(phi_J - phi-hat_I) over the support, phi-hat_I = wrap(T_I - delta)."""
    original_phase = wrap_phase(template - delta)
    return spectral_agreement(luminance_phase(image, centered=False), original_phase, uniform_weights, support)


for image in images[:WARM_UP]:
    verify(image, enroll(image)[1])
t0 = time.time()
templates = [enroll(image)[1] for image in images]
t1 = time.time()
for image, template in zip(images, templates):
    verify(image, template)
t2 = time.time()

image_detector = build_image_only_detector()
image_detector_ckpt = checkpoint_path("ffpp", augmented=True)   # every dataset's detector has the same size
bins = int(support.sum())
print(json.dumps({
    "enroll_ms_cpu": (t1 - t0) / len(images) * 1000,
    "relation_verify_ms_cpu": (t2 - t1) / len(images) * 1000,
    "relation_params": 0,
    "record_model_params": RECORD_MODEL_PARAMETERS,
    "image_detector_params": sum(p.numel() for p in image_detector.parameters()),
    "image_detector_ckpt_MiB": os.path.getsize(image_detector_ckpt) / 2**20 if os.path.exists(image_detector_ckpt) else None,
    "per_asset_template_bins": bins,
    "per_asset_KiB_fp32": bins * 4 / 1024,
    "per_asset_KiB_fp16": bins * 2 / 1024,
}, indent=1))
