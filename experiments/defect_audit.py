"""Audit: reproduce the old protocol's blur/codec optimism by re-introducing
the implicit-template defect on the audited data.

Old protocol: the template was read from the same file as the query, so a
distortion entered both sides and cancelled; benign H was pinned near cos(delta)
regardless of channel. Explicit protocol: the template is the enrollment-time
NPZ, so distortion enters only the query. Same assets, same channels, both
readings side by side."""
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

def wcos(dphi, sup, amp):
    r = np.cos((dphi + np.pi) % (2*np.pi) - np.pi)
    w = np.where(sup, amp, 0.0)
    return float(np.sum(r*w) / (np.sum(w) + 1e-8))

cases = pd.read_csv(f"{R}/common/cases.csv")
te = cases[(cases.dataset == "FF++") & (cases.split == "test")]
ben = te[te.protocol == "benign"].drop_duplicates("enrollment_id")[
    ["enrollment_id", "input_path", "doctor_trigger_template_path", "doctor_trigger_released_path"]]
swaps = te[te.protocol == "swap"].drop_duplicates("enrollment_id")[["enrollment_id", "input_path"]]
pool = ben.merge(swaps, on="enrollment_id", suffixes=("_ben", "_swap")).head(N)

CHANNELS = ["jpeg_q50", "blur_k7", "x264_crf35"]
res = {c: {"exp_b": [], "exp_f": [], "imp_b": [], "imp_f": []} for c in CHANNELS}
for _, q in pool.iterrows():
    with np.load(q.doctor_trigger_template_path) as rec:
        tmpl = np.fft.ifftshift(rec["template_phase"].astype(np.float64))
        sup = np.fft.ifftshift(rec["support"].astype(bool))
        amp = np.fft.ifftshift(rec["host_amplitude"].astype(np.float64))
    rel = Image.open(q.doctor_trigger_released_path).convert("RGB")
    fake = Image.open(q.input_path_swap).convert("RGB")
    for ci, ch in enumerate(CHANNELS):
        db = transform_image(rel, ch, seed=1009 + ci)
        df_ = transform_image(fake, ch, seed=2009 + ci)
        pb, pf = phase_of(db), phase_of(df_)
        # explicit: enrollment-time NPZ template on both sides of the comparison
        res[ch]["exp_b"].append(wcos(pb - tmpl, sup, amp))
        res[ch]["exp_f"].append(wcos(pf - tmpl, sup, amp))
        # implicit (the defect): template re-read from the same distorted file as the benign query
        res[ch]["imp_b"].append(wcos(pb - phase_of(db), sup, amp))
        res[ch]["imp_f"].append(wcos(pf - phase_of(transform_image(rel, ch, seed=1009 + ci)), sup, amp))

print(f"{'channel':11s} {'explicit benign H':>18s} {'explicit AUC':>13s} {'implicit benign H':>18s} {'implicit AUC':>13s}")
for ch in CHANNELS:
    r = res[ch]
    y = np.r_[np.zeros(len(r["exp_b"])), np.ones(len(r["exp_f"]))]
    a_exp = roc_auc_score(y, np.r_[1-np.array(r["exp_b"]), 1-np.array(r["exp_f"])])
    a_imp = roc_auc_score(y, np.r_[1-np.array(r["imp_b"]), 1-np.array(r["imp_f"])])
    print(f"{ch:11s} {np.mean(r['exp_b']):18.4f} {a_exp:13.4f} {np.mean(r['imp_b']):18.4f} {a_imp:13.4f}")
