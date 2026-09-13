"""Crop channel, forward-modeled reading (Table 5 crop row).

The old harness resized the cropped copy back to canvas size, which rescales
every frequency and misregisters the record reading (benign H ~ 0.15: a harness
artifact, not a method limit). A crop keeps interior pixels exact; the verifier
zero-pads the returning copy into the canvas -- equivalent to multiplying the
original by a known mask -- and predicts the same masked spectrum from the
record's ring alone (host amplitude x template phase on the support). Measured
vs predicted phase is compared on the support; placement is searched +/-2 px.
Uses only what the record stores.

Outputs: strict-rate, declared-tau preserved/flagged (tau at 1% FRR on 100
train-benign crops under the same reading), and AUC of the relation score.
"""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env); assert v, f"{env} unset"; return v + rest
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
R = _dt("DT_CANON_ROOT")
EPS = 1e-12; N = 200; N_CAL = 100; TAU_STRICT = 0.9821
cases = pd.read_csv(f"{R}/common/cases.csv")
te = cases[(cases.split == "test") & (cases.dataset == "FF++")]
enr = te.groupby("enrollment_id").first().head(N)
tr = cases[(cases.dataset == "FF++") & (cases.split == "train") & (cases.protocol == "benign")]\
    .drop_duplicates("enrollment_id").head(N_CAL)

def lum_of(img):
    a = np.asarray(img.convert("RGB"), np.float64)
    return 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
def wrap(v): return (v + np.pi) % (2*np.pi) - np.pi
def crop90(img):
    wc, hc = int(img.width*0.9), int(img.height*0.9)
    return img.crop(((img.width-wc)//2, (img.height-hc)//2, (img.width-wc)//2+wc, (img.height-hc)//2+hc))

def forward_H(img, tp):
    with np.load(tp) as rec:
        sup = rec["support"].astype(bool); amp = rec["host_amplitude"].astype(np.float64)
        tpz = rec["template_phase"].astype(np.float64)
    Hh, Wh = sup.shape; lum = lum_of(img); hc, wc = lum.shape
    spat = np.fft.ifft2(np.fft.ifftshift(np.where(sup, amp, 0.0) * np.exp(1j*tpz))).real
    ww = np.where(sup, amp, 0.0); sww = np.sum(ww) + EPS
    oy0, ox0 = (Hh-hc)//2, (Wh-wc)//2
    best = -2.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            oy, ox = oy0+dy, ox0+dx
            if oy < 0 or ox < 0 or oy+hc > Hh or ox+wc > Wh: continue
            canvas = np.zeros((Hh, Wh)); canvas[oy:oy+hc, ox:ox+wc] = lum - lum.mean()
            mask = np.zeros((Hh, Wh)); mask[oy:oy+hc, ox:ox+wc] = 1.0
            F_meas = np.fft.fftshift(np.fft.fft2(canvas))
            P_rel = np.fft.fftshift(np.fft.fft2(spat * mask))
            H = float(np.sum(np.cos(wrap(np.angle(F_meas) - np.angle(P_rel)))*ww) / sww)
            best = max(best, H)
    return best

cal = np.array([forward_H(crop90(Image.open(q.input_path)), q.doctor_trigger_template_path)
                for _, q in tr.iterrows()])
tau_ch = float(np.quantile(cal, 0.01))
print(f"declared tau (1% FRR, n={N_CAL}): {tau_ch:.4f}", flush=True)

gH, fH = [], []
for k, row in enumerate(enr.itertuples()):
    tp = row.doctor_trigger_template_path
    gH.append(forward_H(crop90(Image.open(te[(te.enrollment_id == row.Index) & (te.protocol == "benign")].input_path.iloc[0])), tp))
    fH.append(forward_H(crop90(Image.open(te[(te.enrollment_id == row.Index) & (te.protocol == "swap")].input_path.iloc[0])), tp))
    if (k+1) % 50 == 0: print(f"{k+1}/{N}", flush=True)
gH, fH = np.array(gH), np.array(fH)
print(f"benign H {gH.mean():.3f}+-{gH.std():.3f} | swap H {fH.mean():.3f}+-{fH.std():.3f}")
print(f"strict-rate (tau={TAU_STRICT}): {np.mean(gH >= TAU_STRICT):.1%}")
print(f"preserved (declared tau): {np.mean(gH >= tau_ch):.1%} | flagged: {np.mean(fH < tau_ch):.1%}")
print(f"AUC(relation): {roc_auc_score([0]*N+[1]*N, np.concatenate([-gH, -fH])):.3f}")
