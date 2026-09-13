"""Side effects of the declared sub-band reading (kappa <= cut) on everything
that is NOT swap detection: key separation, post-hoc ASR, release-evidence
margin, and same-person attribution. FF++ test, explicit reading."""
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

N = 150
CUTS = {"full": 0.45, "sub": 0.15}
CH = "blur_k7"

def phase_of(pil):
    a = np.asarray(pil, dtype=np.float64)
    lum = 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
    return np.angle(np.fft.fft2(lum))

def radius_grid(shape):
    fy = np.fft.fftfreq(shape[0]).reshape(-1, 1)
    fx = np.fft.fftfreq(shape[1]).reshape(1, -1)
    return np.sqrt(fy*fy + fx*fx) / np.sqrt(0.5**2 + 0.5**2)

def H_G(ph, tmpl, host, delta, w):
    z = np.sum(w) + 1e-8
    H = float(np.sum(np.cos((ph - tmpl + np.pi) % (2*np.pi) - np.pi) * w) / z)
    resid = (ph - host + np.pi) % (2*np.pi) - np.pi
    G = float(np.sum(w * delta * np.sin(resid)) / (np.sum(w * delta * delta) + 1e-6))
    return H, G

cases = pd.read_csv(f"{R}/common/cases.csv")
te = cases[(cases.dataset == "FF++") & (cases.split == "test")]
ben = te[te.protocol == "benign"].drop_duplicates("enrollment_id")
pool = ben.head(N)
sib = ben.groupby("source_group_id").filter(lambda g: len(g) >= 2)
sib_pairs = {gid: g.head(2) for gid, g in sib.groupby("source_group_id")}

out = {c: {"G_ben": [], "Gw": [], "G_orig": [], "H_own": [], "H_sib": []} for c in CUTS}
for _, q in pool.iterrows():
    with np.load(q.doctor_trigger_template_path) as rec:
        tmpl = np.fft.ifftshift(rec["template_phase"].astype(np.float64))
        host = np.fft.ifftshift(rec["host_phase"].astype(np.float64))
        sup = np.fft.ifftshift(rec["support"].astype(bool))
        amp = np.fft.ifftshift(rec["host_amplitude"].astype(np.float64))
        delta = np.fft.ifftshift(rec["delta"].astype(np.float64))
        key = int(rec["key"][0]); alpha = float(rec["alpha"][0]); beta = float(rec["beta"][0])
    rad = radius_grid(tmpl.shape)
    rel = Image.open(q.doctor_trigger_released_path).convert("RGB")
    ph_bch = phase_of(transform_image(rel, CH, seed=23))
    ph_orig = phase_of(Image.open(q.enrollment_input_path).convert("RGB"))
    wrong = [np.fft.ifftshift(feats.wrong_delta(wk, tmpl.shape, alpha, beta, np.fft.fftshift(sup)))
             for wk in feats.wrong_keys(key, 8)]
    for cname, cut in CUTS.items():
        w = np.where(sup & (rad <= cut), amp, 0.0)
        _, g_b = H_G(ph_bch, tmpl, host, delta, w)
        _, g_o = H_G(ph_orig, tmpl, host, delta, w)
        out[cname]["G_ben"].append(g_b)
        out[cname]["G_orig"].append(g_o)
        for wd in wrong:
            resid = (ph_bch - host + np.pi) % (2*np.pi) - np.pi
            out[cname]["Gw"].append(float(np.sum(w * wd * np.sin(resid)) / (np.sum(w * wd * wd) + 1e-6)))

for gid, g in list(sib_pairs.items()):
    a, b = g.iloc[0], g.iloc[1]
    with np.load(a.doctor_trigger_template_path) as rec:
        tmpl = np.fft.ifftshift(rec["template_phase"].astype(np.float64))
        sup = np.fft.ifftshift(rec["support"].astype(bool))
        amp = np.fft.ifftshift(rec["host_amplitude"].astype(np.float64))
    rad = radius_grid(tmpl.shape)
    ph_own = phase_of(Image.open(a.input_path).convert("RGB"))
    ph_sib = phase_of(Image.open(b.input_path).convert("RGB"))
    for cname, cut in CUTS.items():
        w = np.where(sup & (rad <= cut), amp, 0.0)
        z = np.sum(w) + 1e-8
        out[cname]["H_own"].append(float(np.sum(np.cos((ph_own - tmpl + np.pi) % (2*np.pi) - np.pi) * w) / z))
        out[cname]["H_sib"].append(float(np.sum(np.cos((ph_sib - tmpl + np.pi) % (2*np.pi) - np.pi) * w) / z))

print(f"{'reading':6s} {'keyAUC(blur)':>12s} {'posthocASR':>11s} {'G_ben med':>10s} {'G_orig max':>11s} {'attribAUC':>10s} {'H_sib med':>10s}")
for cname in CUTS:
    o = out[cname]
    pos, neg = np.array(o["G_ben"]), np.array(o["Gw"])
    key_auc = roc_auc_score(np.r_[np.ones_like(pos), np.zeros_like(neg)], np.r_[pos, neg])
    tau = np.quantile(pos, 0.01)
    asr = float((np.array(o["G_orig"]) >= tau).mean())
    own, sb = np.array(o["H_own"]), np.array(o["H_sib"])
    att = roc_auc_score(np.r_[np.ones_like(own), np.zeros_like(sb)], np.r_[own, sb])
    print(f"{cname:6s} {key_auc:12.3f} {asr:11.1%} {np.median(pos):10.2f} {np.max(o['G_orig']):11.3f} {att:10.3f} {np.median(sb):10.3f}")
