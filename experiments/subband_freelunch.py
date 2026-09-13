"""Free-lunch search for the lossy channels: hard radial cuts vs a measured
per-channel radial profile.

Hard cut: read only kappa <= cut. Profile: from train genuine copies, measure
the mean cosine agreement per radial band under the channel, and weight the
reading by amp * max(rho_band, 0)^p. The profile costs eight numbers per
declared channel, calibrated exactly like the threshold."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest
import importlib.util, sys
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
R = _dt("DT_CANON_ROOT")
sys.path.insert(0, R)
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from scripts.benign_channels import transform_image

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
feats = load_mod("hk_features", f"{R}/scripts/extract_features.py")

N_TEST, N_CAL = 150, 100
CUTS = [0.12, 0.15, 0.18, 0.20, 0.25, 0.45]
CHANNELS = ["clean", "jpeg_q90", "jpeg_q50", "scale_0.5", "blur_k7", "x264_crf35"]
BANDS = np.linspace(0.10, 0.45, 9)

def phase_of(pil):
    a = np.asarray(pil, dtype=np.float64)
    lum = 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
    return np.angle(np.fft.fft2(lum))

def radius_grid(shape):
    fy = np.fft.fftfreq(shape[0]).reshape(-1, 1)
    fx = np.fft.fftfreq(shape[1]).reshape(1, -1)
    return np.sqrt(fy*fy + fx*fx) / np.sqrt(0.5**2 + 0.5**2)

def tload(q):
    with np.load(q.doctor_trigger_template_path) as rec:
        return {k: np.fft.ifftshift(rec[k].astype(np.float64 if k != "support" else np.uint8))
                for k in ["template_phase", "host_phase", "support", "host_amplitude", "delta"]} | \
               {"key": int(rec["key"][0]), "alpha": float(rec["alpha"][0]), "beta": float(rec["beta"][0])}

cases = pd.read_csv(f"{R}/common/cases.csv")
ff = cases[cases.dataset == "FF++"]
te_b = ff[(ff.split == "test") & (ff.protocol == "benign")].drop_duplicates("enrollment_id")
te_f = ff[(ff.split == "test") & (ff.protocol == "swap")].drop_duplicates("enrollment_id")[["enrollment_id", "input_path"]]
pool = te_b.merge(te_f, on="enrollment_id", suffixes=("", "_swap")).head(N_TEST)
cal = ff[(ff.split == "train") & (ff.protocol == "benign")].drop_duplicates("enrollment_id").head(N_CAL)

# --- calibrate radial profiles on train genuine ---
prof = {ch: np.zeros(len(BANDS) - 1) for ch in CHANNELS}
cnt = {ch: np.zeros(len(BANDS) - 1) for ch in CHANNELS}
for _, q in cal.iterrows():
    t = tload(q)
    rad = radius_grid(t["template_phase"].shape)
    rel = Image.open(q.doctor_trigger_released_path).convert("RGB")
    for ci, ch in enumerate(CHANNELS):
        img = rel if ch == "clean" else transform_image(rel, ch, seed=31 + ci)
        r = np.cos((phase_of(img) - t["template_phase"] + np.pi) % (2*np.pi) - np.pi)
        for b in range(len(BANDS) - 1):
            m = t["support"].astype(bool) & (rad >= BANDS[b]) & (rad < BANDS[b+1])
            w = t["host_amplitude"][m]
            if w.sum() > 0:
                prof[ch][b] += float(np.sum(r[m]*w)); cnt[ch][b] += float(np.sum(w))
RHO = {ch: np.clip(prof[ch]/np.maximum(cnt[ch], 1e-8), 0, None) for ch in CHANNELS}
print("radial profiles rho_c (bands 0.10->0.45):")
for ch in CHANNELS:
    print(f"  {ch:11s} " + " ".join(f"{v:.2f}" for v in RHO[ch]))

# --- evaluate readings on test ---
def weights(t, rad, mode, ch):
    sup, amp = t["support"].astype(bool), t["host_amplitude"]
    if mode == "full":  return np.where(sup, amp, 0.0)
    if mode.startswith("cut"):
        return np.where(sup & (rad <= float(mode[3:])), amp, 0.0)
    p = int(mode[-1])
    rho = RHO[ch]
    rw = np.zeros_like(rad)
    for b in range(len(BANDS) - 1):
        rw[(rad >= BANDS[b]) & (rad < BANDS[b+1])] = rho[b] ** p
    return np.where(sup, amp * rw, 0.0)

MODES = ["full"] + [f"cut{c}" for c in CUTS[:-1]] + ["rho1", "rho2"]
res = {(ch, m): {"b": [], "f": [], "gb": [], "gw": []} for ch in CHANNELS for m in MODES}
for _, q in pool.iterrows():
    t = tload(q)
    rad = radius_grid(t["template_phase"].shape)
    rel = Image.open(q.doctor_trigger_released_path).convert("RGB")
    fake = Image.open(q.input_path_swap).convert("RGB")
    wrong = [np.fft.ifftshift(feats.wrong_delta(wk, t["template_phase"].shape, t["alpha"], t["beta"],
             np.fft.fftshift(t["support"].astype(bool)))) for wk in feats.wrong_keys(t["key"], 4)]
    for ci, ch in enumerate(CHANNELS):
        ib = rel if ch == "clean" else transform_image(rel, ch, seed=41 + ci)
        jf = fake if ch == "clean" else transform_image(fake, ch, seed=47 + ci)
        pb, pf = phase_of(ib), phase_of(jf)
        rb = np.cos((pb - t["template_phase"] + np.pi) % (2*np.pi) - np.pi)
        rf = np.cos((pf - t["template_phase"] + np.pi) % (2*np.pi) - np.pi)
        resid = (pb - t["host_phase"] + np.pi) % (2*np.pi) - np.pi
        for m in MODES:
            w = weights(t, rad, m, ch)
            z = np.sum(w) + 1e-8
            res[(ch, m)]["b"].append(float(np.sum(rb*w)/z))
            res[(ch, m)]["f"].append(float(np.sum(rf*w)/z))
            res[(ch, m)]["gb"].append(float(np.sum(w*t["delta"]*np.sin(resid)) / (np.sum(w*t["delta"]**2) + 1e-6)))
            for wd in wrong:
                res[(ch, m)]["gw"].append(float(np.sum(w*wd*np.sin(resid)) / (np.sum(w*wd*wd) + 1e-6)))

hdr = f"{'channel':11s}" + "".join(f" {m:>8s}" for m in MODES)
print("\ndetection AUC (1-H):"); print(hdr)
for ch in CHANNELS:
    line = f"{ch:11s}"
    for m in MODES:
        r_ = res[(ch, m)]
        y = np.r_[np.zeros(len(r_["b"])), np.ones(len(r_["f"]))]
        line += f" {roc_auc_score(y, np.r_[1-np.array(r_['b']), 1-np.array(r_['f'])]):8.3f}"
    print(line)
print("\nkey AUC (correct vs wrong, under channel):"); print(hdr)
for ch in CHANNELS:
    line = f"{ch:11s}"
    for m in MODES:
        r_ = res[(ch, m)]
        pos, neg = np.array(r_["gb"]), np.array(r_["gw"])
        line += f" {roc_auc_score(np.r_[np.ones_like(pos), np.zeros_like(neg)], np.r_[pos, neg]):8.3f}"
    print(line)
