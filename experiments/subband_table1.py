"""All eleven Table-1 cells for the declared sub-band reading (kappa <= 0.15).

Same protocol as the deployed row: benign families are severity means over the
full FF++ test split with the exact seeds of the deployed measurement; dataset
cells are clean SimSwap detection on each canonical test split; engine cells
re-score the cached engine swaps. CPU-only. Usage: subband_table1.py <part>
where part is benign | datasets | engines."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest
import sys
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from scripts.benign_channels import transform_image
R = _dt("DT_CANON_ROOT")
CUT = 0.15
PART = sys.argv[1]

def phase_of(pil):
    a = np.asarray(pil, dtype=np.float64)
    lum = 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
    return np.angle(np.fft.fft2(lum))

_rad_cache = {}
def radius_grid(shape):
    if shape not in _rad_cache:
        fy = np.fft.fftfreq(shape[0]).reshape(-1, 1)
        fx = np.fft.fftfreq(shape[1]).reshape(1, -1)
        _rad_cache[shape] = np.sqrt(fy*fy + fx*fx) / np.sqrt(0.5**2 + 0.5**2)
    return _rad_cache[shape]

def H_sub(pil, tp):
    with np.load(tp) as rec:
        tmpl = np.fft.ifftshift(rec["template_phase"].astype(np.float64))
        sup = np.fft.ifftshift(rec["support"].astype(bool))
        amp = np.fft.ifftshift(rec["host_amplitude"].astype(np.float64))
    ph = phase_of(pil)
    if ph.shape != tmpl.shape:
        pil = pil.resize((tmpl.shape[1], tmpl.shape[0])); ph = phase_of(pil)
    w = np.where(sup & (radius_grid(tmpl.shape) <= CUT), amp, 0.0)
    r = np.cos((ph - tmpl + np.pi) % (2*np.pi) - np.pi)
    return float(np.sum(r*w) / (np.sum(w) + 1e-8))

cases = pd.read_csv(f"{R}/common/cases.csv")

if PART == "benign":
    te = cases[(cases.dataset == "FF++") & (cases.split == "test")]
    PERTS = ["jpeg_q90", "jpeg_q70", "jpeg_q50", "blur_k7", "x264_crf30", "x264_crf35", "x264_crf40", "scale_0.5"]
    FAM = {"JPEG": ["jpeg_q90", "jpeg_q70", "jpeg_q50"], "blur": ["blur_k7"],
           "codec": ["x264_crf30", "x264_crf35", "x264_crf40"], "pixel": ["scale_0.5"]}
    res = {p: {"s": [], "y": []} for p in PERTS}
    for k, (_, q) in enumerate(te.iterrows()):
        img = Image.open(q.input_path).convert("RGB")
        for p in PERTS:
            res[p]["s"].append(1 - H_sub(transform_image(img, p, seed=(k+17)*1009), q.doctor_trigger_template_path))
            res[p]["y"].append(q.y_fake)
    for fam, ps in FAM.items():
        aucs = [roc_auc_score(res[p]["y"], res[p]["s"]) for p in ps]
        print(f"SUBBAND {fam}: {np.mean(aucs):.3f}  (per-sev: {' '.join(f'{a:.3f}' for a in aucs)})")

elif PART == "datasets":
    te = cases[cases.split == "test"]
    for ds, g in te.groupby("dataset"):
        ss = [1 - H_sub(Image.open(q.input_path).convert("RGB"), q.doctor_trigger_template_path)
              for _, q in g.iterrows()]
        print(f"SUBBAND {ds}: {roc_auc_score(g.y_fake.values, ss):.3f}")

elif PART == "engines":
    from pathlib import Path
    CACHE = _dt("DT_PEERS_ROOT", "/engine_cache")
    te = cases[(cases.dataset == "FF++") & (cases.split == "test")]
    ben = te[te.protocol == "benign"].drop_duplicates("enrollment_id")
    for eng in ["stargan", "uniface", "inswapper", "sdinpaint"]:
        ys, ss = [], []
        for _, q in ben.iterrows():
            cf = Path(f"{CACHE}/{eng}/{q.enrollment_id}.png")
            if not cf.exists(): continue
            ss.append(1 - H_sub(Image.open(q.doctor_trigger_released_path).convert("RGB"), q.doctor_trigger_template_path)); ys.append(0)
            ss.append(1 - H_sub(Image.open(cf).convert("RGB"), q.doctor_trigger_template_path)); ys.append(1)
        print(f"SUBBAND {eng}: {roc_auc_score(ys, ss):.3f}  n={len(ys)//2}")
