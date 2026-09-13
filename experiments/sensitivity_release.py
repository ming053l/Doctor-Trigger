"""Release-mechanism sensitivity: one (dataset, setting) job.

Usage: sensitivity_release.py <FF++|DFDC|CelebDF> <alpha:0.02|band:0.05-0.45>
Embeds N train assets at the requested setting with the audited embed(), then
scores training-free readings on five query types (benign, jpeg90 benign,
untriggered original, SimSwap re-render, jpeg90 re-render). Wrong-key
projections (m=32) are stored per asset for the key axis and the m-sweep.
Writes one per-asset CSV; figures aggregate across jobs.
"""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest

import importlib.util, io, sys, time
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image

R = _dt("DT_CANON_ROOT")
sys.path.insert(0, R)                       # for workflow.paths
sys.path.insert(0, _dt("DT_PEERS_ROOT"))    # for engines_api deps

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

enroll = load_mod("hk_enroll", f"{R}/scripts/enroll.py")
feats = load_mod("hk_features", f"{R}/scripts/extract_features.py")

DS, SPEC = sys.argv[1], sys.argv[2]
N = int(_dtos.environ.get("DT_SENS_N", "200"))
AMP_FLOOR = 0.0     # deployed: support uses amplitude floor ratio; read from her NPZ below
BETA = 0.15
if SPEC.startswith("alpha:"):
    ALPHA, BANDS = float(SPEC.split(":")[1]), (0.10, 0.45)
else:
    lo, hi = SPEC.split(":")[1].split("-")
    BANDS = (float(lo), float(hi))
    ALPHA = 0.015 if DS == "DFDC" else 0.05

cases = pd.read_csv(f"{R}/common/cases.csv")
tr = cases[(cases.dataset == DS) & (cases.split == "train")]
ben = tr[tr.protocol == "benign"].drop_duplicates("enrollment_id")[
    ["enrollment_id", "enrollment_input_path", "doctor_trigger_template_path"]]
swaps = tr[tr.protocol == "swap"].drop_duplicates("enrollment_id")[["enrollment_id", "attack_source_path"]]
pool = ben.merge(swaps, on="enrollment_id").sort_values("enrollment_id").head(N)

with np.load(pool.iloc[0].doctor_trigger_template_path) as rec0:
    AMP_FLOOR = float(rec0["amplitude_floor_ratio"][0]) if "amplitude_floor_ratio" in rec0 else 0.0

from engines_api import ENGINES
swap = ENGINES["simswap"]()

def jpeg90(pil):
    b = io.BytesIO(); pil.save(b, "JPEG", quality=90); b.seek(0)
    return Image.open(b).convert("RGB")

def phase_of(arr):
    lum = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    return np.angle(np.fft.fft2(lum.astype(np.float64)))

def H_of(arr, rec):
    ph = phase_of(arr)
    r = np.cos((ph - rec["_host"] + np.pi) % (2*np.pi) - np.pi)
    w = np.where(rec["_sup"], rec["_amp"], 0.0)
    return float(np.sum(r*w) / (np.sum(w) + 1e-8))

def G_of(arr, rec, delta):
    ph = phase_of(arr)
    resid = (ph - rec["_host"] + np.pi) % (2*np.pi) - np.pi
    w = np.where(rec["_sup"], rec["_amp"], 0.0)
    num = float(np.sum(w * delta * np.sin(resid)))
    den = float(np.sum(w * delta * delta)) + 1e-6
    return num / den

rows = []
t_proj = []
for _, q in pool.iterrows():
    orig = np.asarray(Image.open(q.enrollment_input_path).convert("RGB"))
    with np.load(q.doctor_trigger_template_path) as rec:
        key = int(rec["key"][0])
    released_u8, record = enroll.embed(orig, key, ALPHA, BETA, BANDS, AMP_FLOOR)
    rec = {"_host": np.angle(np.fft.fft2((0.299*orig[:,:,0]+0.587*orig[:,:,1]+0.114*orig[:,:,2]).astype(np.float64))),
           "_sup": np.fft.ifftshift(record["support"].astype(bool)),
           "_amp": np.fft.ifftshift(record["host_amplitude"].astype(np.float64))}
    delta = np.fft.ifftshift(record["delta"].astype(np.float64))
    rel_pil = Image.fromarray(released_u8)
    try:
        sw_pil = swap(rel_pil, Image.open(q.attack_source_path).convert("RGB"))
    except Exception:
        continue
    mse = np.mean((orig.astype(np.float64) - released_u8.astype(np.float64))**2)
    r = {"enrollment_id": q.enrollment_id, "psnr": 10*np.log10(255.0**2/(mse+1e-12))}
    variants = {"ben": np.asarray(rel_pil), "ben_j90": np.asarray(jpeg90(rel_pil)),
                "orig": orig, "swap": np.asarray(sw_pil), "swap_j90": np.asarray(jpeg90(sw_pil))}
    for tag, arr in variants.items():
        r[f"H_{tag}"] = H_of(arr, rec)
        r[f"G_{tag}"] = G_of(arr, rec, delta)
    t0 = time.time()
    shape = delta.shape
    for j, wk in enumerate(feats.wrong_keys(key, 32)):
        wd = feats.wrong_delta(wk, shape, ALPHA, BETA, np.fft.fftshift(rec["_sup"]))
        r[f"Gw{j}"] = G_of(variants["ben"], rec, np.fft.ifftshift(wd))
    t_proj.append((time.time() - t0) / 32)
    rows.append(r)

out = Path(_dt("DT_OUTPUTS", "/sensitivity"))
out.mkdir(exist_ok=True)
tag = SPEC.replace(":", "_").replace(".", "p").replace("-", "_")
d = pd.DataFrame(rows)
d.attrs = {}
d.to_csv(out / f"{DS}_{tag}.csv", index=False)
print(f"{DS} {SPEC}: n={len(d)} psnr={d.psnr.mean():.2f} "
      f"G_ben={d.G_ben.median():.2f} G_orig={d.G_orig.median():.3f} "
      f"proj_ms={np.mean(t_proj)*1000:.1f}")
