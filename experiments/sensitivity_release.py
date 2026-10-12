"""Release-mechanism sensitivity: one (dataset, setting) job.

Usage: sensitivity_release.py <FF++|DFDC|CelebDF> <alpha:0.02|band:0.05-0.45>
Re-encodes N train assets at the requested setting with the deployed encoder
(doctor_trigger.enrollment.encode_release, each asset's own key), then scores
training-free readings on five query types (benign, jpeg90 benign, unedited
original, SimSwap re-render, jpeg90 re-render): H against the original phase and
the key projection G_s (Eq. 6, stabilizer 1e-6). Wrong-key projections (m=32)
are stored per asset for the key axis and the m-sweep. Writes one per-asset CSV
<outputs_root>/sensitivity/<DS>_<setting>.csv; figures aggregate across jobs.
Optional env DT_SENS_N (default 200).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from doctor_trigger import config  # noqa: E402
from doctor_trigger.channels import jpeg  # noqa: E402
from doctor_trigger.enrollment import encode_release  # noqa: E402
from doctor_trigger.key import Ring, displacement, sign_pattern, wrong_keys  # noqa: E402
from doctor_trigger.manifests import read_cases  # noqa: E402
from doctor_trigger.readings import SumForm, key_projection, phase_drift, spectral_agreement  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from doctor_trigger.spectrum import luminance_phase  # noqa: E402
from experiments.shared.swap_engines import ENGINES  # noqa: E402

DS, SPEC = sys.argv[1], sys.argv[2]
N = int(os.environ.get("DT_SENS_N", "200"))
BETA = 0.15
WRONG_KEYS = 32
if SPEC.startswith("alpha:"):
    ALPHA, BANDS = float(SPEC.split(":")[1]), (0.10, 0.45)
else:
    lo, hi = SPEC.split(":")[1].split("-")
    BANDS = (float(lo), float(hi))
    ALPHA = 0.015 if DS == "DFDC" else 0.05

tr = read_cases(dataset=DS, split="train")
ben = tr[tr.protocol == "benign"].drop_duplicates("enrollment_id")[
    ["enrollment_id", "enrollment_input_path", "doctor_trigger_template_path"]]
swaps = tr[tr.protocol == "swap"].drop_duplicates("enrollment_id")[["enrollment_id", "attack_source_path"]]
pool = ben.merge(swaps, on="enrollment_id").sort_values("enrollment_id").head(N)

AMP_FLOOR = load_record(pool.iloc[0].doctor_trigger_template_path).amplitude_floor_ratio   # deployed ratio

swap = ENGINES["simswap"]()


def H_of(arr, rec):
    """H against the original phase over K_eff (uncentered weight map, stabilizer 1e-8)."""
    return spectral_agreement(luminance_phase(arr, centered=False), rec["_host"], rec["_amp"], rec["_sup"],
                              stabilizer=1e-8, form=SumForm.WEIGHT_MAP)


def G_of(arr, rec, delta):
    """G_s of Eq. (6) for displacement ``delta`` (uncentered weight map, stabilizer 1e-6)."""
    drift = phase_drift(luminance_phase(arr, centered=False), rec["_host"])
    return key_projection(drift, delta, rec["_amp"], rec["_sup"], stabilizer=1e-6, form=SumForm.WEIGHT_MAP)


rows = []
t_proj = []
for _, q in pool.iterrows():
    orig = np.asarray(Image.open(q.enrollment_input_path).convert("RGB"))
    key = load_record(q.doctor_trigger_template_path).key
    released_u8, record = encode_release(orig, key, ALPHA, BETA, Ring(*BANDS), AMP_FLOOR)
    rec = {"_host": luminance_phase(orig, centered=False),
           "_sup": np.fft.ifftshift(record.active_bins),
           "_amp": np.fft.ifftshift(record.amplitude_weights)}
    delta = np.fft.ifftshift(record.displacement)
    rel_pil = Image.fromarray(released_u8)
    try:
        sw_pil = swap(rel_pil, Image.open(q.attack_source_path).convert("RGB"))
    except Exception:
        continue
    mse = np.mean((orig.astype(np.float64) - released_u8.astype(np.float64))**2)
    r = {"enrollment_id": q.enrollment_id, "psnr": 10*np.log10(255.0**2/(mse+1e-12))}
    variants = {"ben": np.asarray(rel_pil), "ben_j90": jpeg(np.asarray(rel_pil), 90),
                "orig": orig, "swap": np.asarray(sw_pil), "swap_j90": jpeg(np.asarray(sw_pil), 90)}
    for tag, arr in variants.items():
        r[f"H_{tag}"] = H_of(arr, rec)
        r[f"G_{tag}"] = G_of(arr, rec, delta)
    t0 = time.time()
    shape = delta.shape
    for j, wk in enumerate(wrong_keys(key, WRONG_KEYS)):
        wd = displacement(sign_pattern(shape, wk, centered=True), ALPHA, BETA, np.fft.fftshift(rec["_sup"]))
        r[f"Gw{j}"] = G_of(variants["ben"], rec, np.fft.ifftshift(wd))
    t_proj.append((time.time() - t0) / WRONG_KEYS)
    rows.append(r)

out = Path(config.path("outputs_root") + "/sensitivity")
out.mkdir(exist_ok=True)
tag = SPEC.replace(":", "_").replace(".", "p").replace("-", "_")
d = pd.DataFrame(rows)
d.to_csv(out / f"{DS}_{tag}.csv", index=False)
print(f"{DS} {SPEC}: n={len(d)} psnr={d.psnr.mean():.2f} "
      f"G_ben={d.G_ben.median():.2f} G_orig={d.G_orig.median():.3f} "
      f"proj_ms={np.mean(t_proj)*1000:.1f}")
