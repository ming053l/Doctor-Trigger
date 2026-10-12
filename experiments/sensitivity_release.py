"""Release-mechanism sensitivity (Figs 13 and 14, Table 15): one (dataset, setting) job.

Usage: sensitivity_release.py <FF++|DFDC|CelebDF> <alpha:A|band:LO-HI>
  alpha:A      release strength alpha = A on the deployed ring 0.10 <= kappa <= 0.45
  band:LO-HI   ring LO <= kappa <= HI at the deployed alpha (0.015 for DFDC, 0.05 otherwise)
Re-encodes the first N train enrollments (by enrollment_id) with the deployed encoder
(doctor_trigger.enrollment.encode_release, beta = 0.15, each enrollment's own key and the deployed
amplitude floor), re-renders each release with SimSwap, and scores five queries per asset: the
release, the release after JPEG q90, the unedited original, the SimSwap re-rendering and the
re-rendering after JPEG q90. Readings: H against the original phase and the key projection G_s
(Eq. 6, stabilizer 1e-6) on K_eff, summed over the uncentered grid. The projections of 32 wrong
keys on the release are stored per asset (columns Gw0..Gw31) for the key axis and the m sweep.
An enrollment whose SimSwap re-rendering fails is skipped.
Writes <outputs_root>/sensitivity/<DS>_<setting>.csv (one row per asset; build_fig_sensitivity.py
draws the figures) and prints the median readings and the mean time per wrong-key projection.
Optional env DT_SENS_N (default 200).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from doctor_trigger import config
from doctor_trigger.channels import jpeg
from doctor_trigger.enrollment import encode_release
from doctor_trigger.key import Ring, displacement, sign_pattern, wrong_keys
from doctor_trigger.manifests import read_cases
from doctor_trigger.readings import SumForm, key_projection, phase_drift, spectral_agreement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase
from experiments.shared.swap_engines import ENGINES

DATASET, SETTING = sys.argv[1], sys.argv[2]
N = int(os.environ.get("DT_SENS_N", "200"))
BETA = 0.15
WRONG_KEYS = 32
if SETTING.startswith("alpha:"):
    ALPHA, RING = float(SETTING.split(":")[1]), (0.10, 0.45)
else:
    low, high = SETTING.split(":")[1].split("-")
    RING = (float(low), float(high))
    ALPHA = 0.015 if DATASET == "DFDC" else 0.05

train = read_cases(dataset=DATASET, split="train")
benign = train[train.protocol == "benign"].drop_duplicates("enrollment_id")[
    ["enrollment_id", "enrollment_input_path", "doctor_trigger_template_path"]]
swaps = train[train.protocol == "swap"].drop_duplicates("enrollment_id")[["enrollment_id", "attack_source_path"]]
pool = benign.merge(swaps, on="enrollment_id").sort_values("enrollment_id").head(N)

AMPLITUDE_FLOOR_RATIO = load_record(pool.iloc[0].doctor_trigger_template_path).amplitude_floor_ratio

swap = ENGINES["simswap"]()


def original_reference_agreement(image, original_phase, weights, active):
    """H against the original phase over K_eff (uncentered weight map, stabilizer 1e-8)."""
    return spectral_agreement(luminance_phase(image, centered=False), original_phase, weights, active,
                              stabilizer=1e-8, form=SumForm.WEIGHT_MAP)


def projection(image, original_phase, weights, active, delta):
    """G_s of Eq. (6) for displacement ``delta`` (uncentered weight map, stabilizer 1e-6)."""
    drift = phase_drift(luminance_phase(image, centered=False), original_phase)
    return key_projection(drift, delta, weights, active, stabilizer=1e-6, form=SumForm.WEIGHT_MAP)


rows = []
seconds_per_wrong_key = []
for _, enrollment in pool.iterrows():
    original = np.asarray(Image.open(enrollment.enrollment_input_path).convert("RGB"))
    key = load_record(enrollment.doctor_trigger_template_path).key
    released, record = encode_release(original, key, ALPHA, BETA, Ring(*RING), AMPLITUDE_FLOOR_RATIO)
    original_phase = luminance_phase(original, centered=False)
    active = np.fft.ifftshift(record.active_bins)
    weights = np.fft.ifftshift(record.amplitude_weights)
    delta = np.fft.ifftshift(record.displacement)
    released_pil = Image.fromarray(released)
    try:
        swapped_pil = swap(released_pil, Image.open(enrollment.attack_source_path).convert("RGB"))
    except Exception:
        continue
    mse = np.mean((original.astype(np.float64) - released.astype(np.float64))**2)
    row = {"enrollment_id": enrollment.enrollment_id, "psnr": 10*np.log10(255.0**2/(mse+1e-12))}
    queries = {"ben": np.asarray(released_pil), "ben_j90": jpeg(np.asarray(released_pil), 90),
               "orig": original, "swap": np.asarray(swapped_pil), "swap_j90": jpeg(np.asarray(swapped_pil), 90)}
    for tag, image in queries.items():
        row[f"H_{tag}"] = original_reference_agreement(image, original_phase, weights, active)
        row[f"G_{tag}"] = projection(image, original_phase, weights, active, delta)
    t0 = time.time()
    for j, wrong_key in enumerate(wrong_keys(key, WRONG_KEYS)):
        wrong_delta = displacement(sign_pattern(delta.shape, wrong_key, centered=True), ALPHA, BETA,
                                   np.fft.fftshift(active))
        row[f"Gw{j}"] = projection(queries["ben"], original_phase, weights, active, np.fft.ifftshift(wrong_delta))
    seconds_per_wrong_key.append((time.time() - t0) / WRONG_KEYS)
    rows.append(row)

out_dir = Path(config.path("outputs_root")) / "sensitivity"
out_dir.mkdir(parents=True, exist_ok=True)
tag = SETTING.replace(":", "_").replace(".", "p").replace("-", "_")
results = pd.DataFrame(rows)
results.to_csv(out_dir / f"{DATASET}_{tag}.csv", index=False)
print(f"{DATASET} {SETTING}: n={len(results)} psnr={results.psnr.mean():.2f} "
      f"G_ben={results.G_ben.median():.2f} G_orig={results.G_orig.median():.3f} "
      f"proj_ms={np.mean(seconds_per_wrong_key)*1000:.1f}")
