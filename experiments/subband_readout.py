"""Channel-adaptive sub-band readout: can a declared lossy channel be partially
recovered by reading only the part of the ring it preserves?

Blur and codec are low-pass: they attenuate the upper end of the ring. The
record already stores amplitudes and support, so restricting the readout to
radius <= kappa_cut is a per-channel declaration, training-free. AUC per cut on
the explicit reading, FF++ test."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest
import sys, numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from scripts.benign_channels import transform_image
R = _dt("DT_CANON_ROOT")
N = 150

def phase_of(pil):
    a = np.asarray(pil, dtype=np.float64)
    lum = 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
    return np.angle(np.fft.fft2(lum))

def radius_grid(shape):
    fy = np.fft.fftfreq(shape[0]).reshape(-1, 1)
    fx = np.fft.fftfreq(shape[1]).reshape(1, -1)
    return np.sqrt(fy*fy + fx*fx) / np.sqrt(0.5**2 + 0.5**2)

cases = pd.read_csv(f"{R}/common/cases.csv")
te = cases[(cases.dataset == "FF++") & (cases.split == "test")]
ben = te[te.protocol == "benign"].drop_duplicates("enrollment_id")[
    ["enrollment_id", "doctor_trigger_template_path", "doctor_trigger_released_path"]]
swaps = te[te.protocol == "swap"].drop_duplicates("enrollment_id")[["enrollment_id", "input_path"]]
pool = ben.merge(swaps, on="enrollment_id").head(N)

CUTS = [0.15, 0.20, 0.25, 0.30, 0.45]
CHANNELS = ["blur_k7", "x264_crf35", "jpeg_q50"]
scores = {(c, k): {"b": [], "f": []} for c in CHANNELS for k in CUTS}
for _, q in pool.iterrows():
    with np.load(q.doctor_trigger_template_path) as rec:
        tmpl = np.fft.ifftshift(rec["template_phase"].astype(np.float64))
        sup = np.fft.ifftshift(rec["support"].astype(bool))
        amp = np.fft.ifftshift(rec["host_amplitude"].astype(np.float64))
    rad = radius_grid(tmpl.shape)
    rel = Image.open(q.doctor_trigger_released_path).convert("RGB")
    fake = Image.open(q.input_path).convert("RGB")
    for ci, ch in enumerate(CHANNELS):
        pb = phase_of(transform_image(rel, ch, seed=11 + ci))
        pf = phase_of(transform_image(fake, ch, seed=17 + ci))
        rb = np.cos((pb - tmpl + np.pi) % (2*np.pi) - np.pi)
        rf = np.cos((pf - tmpl + np.pi) % (2*np.pi) - np.pi)
        for k in CUTS:
            w = np.where(sup & (rad <= k), amp, 0.0)
            z = np.sum(w) + 1e-8
            scores[(ch, k)]["b"].append(float(np.sum(rb*w)/z))
            scores[(ch, k)]["f"].append(float(np.sum(rf*w)/z))

print(f"{'channel':11s}" + "".join(f"  cut<={k:.2f}" for k in CUTS) + "   (AUC, benign-H mean)")
for ch in CHANNELS:
    line = f"{ch:11s}"
    for k in CUTS:
        s_ = scores[(ch, k)]
        y = np.r_[np.zeros(len(s_["b"])), np.ones(len(s_["f"]))]
        auc = roc_auc_score(y, np.r_[1-np.array(s_["b"]), 1-np.array(s_["f"])])
        line += f"  {auc:.3f}/{np.mean(s_['b']):.2f}"
    print(line)
