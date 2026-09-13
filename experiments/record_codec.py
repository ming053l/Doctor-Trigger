"""How small can the per-face record get?

Variants, each verified on FF++ test (n=150) for H, G_s, and wrong-key AUC:
  full   : the deployed NPZ as shipped (baseline bytes)
  derive : drop every derivable field (delta, code, host phase); keep active-bin
           template phase + amplitude as float16 + a band bitmask + metadata
  q84    : derive + phase quantized to 8 bits, log-amplitude to 4 bits
  sub    : q84 restricted to the declared sub-band (kappa <= 0.15)
Bytes are zlib-compressed serialized payloads."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest
import io, sys, zlib
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
R = _dt("DT_CANON_ROOT")
sys.path.insert(0, R)
import importlib.util
spec = importlib.util.spec_from_file_location("hk_features", f"{R}/scripts/extract_features.py")
feats = importlib.util.module_from_spec(spec); spec.loader.exec_module(feats)
N = 150

def phase_of(pil):
    a = np.asarray(pil, dtype=np.float64)
    lum = 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
    return np.angle(np.fft.fft2(lum))

def radius_grid(shape):
    fy = np.fft.fftfreq(shape[0]).reshape(-1, 1)
    fx = np.fft.fftfreq(shape[1]).reshape(1, -1)
    return np.sqrt(fy*fy + fx*fx) / np.sqrt(0.5)

def payload_bytes(arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return len(zlib.compress(buf.getvalue(), 9))

def enc(rec, rad, mode):
    sup = rec["support"]
    band = (rad >= 0.10) & (rad <= (0.15 if mode == "sub" else 0.45))
    act = sup & band
    tmpl = rec["template_phase"][act]
    amp = rec["host_amplitude"][act]
    mask = np.packbits(act[band])
    if mode == "derive":
        return {"t": tmpl.astype(np.float16), "a": amp.astype(np.float16), "m": mask}, act
    tq = np.round((tmpl + np.pi) / (2*np.pi) * 255).astype(np.uint8)
    la = np.log10(np.maximum(amp, 1e-6))
    lo, hi = la.min(), la.max()
    aq = np.round((la - lo) / max(hi - lo, 1e-9) * 15).astype(np.uint8)
    return {"t": tq, "a": aq, "m": mask, "r": np.array([lo, hi], np.float32)}, act

def dec_phase_amp(enc_d, mode):
    if mode == "derive":
        return enc_d["t"].astype(np.float64), enc_d["a"].astype(np.float64)
    t = enc_d["t"].astype(np.float64) / 255 * 2*np.pi - np.pi
    lo, hi = enc_d["r"]
    a = 10 ** (enc_d["a"].astype(np.float64) / 15 * (hi - lo) + lo)
    return t, a

cases = pd.read_csv(f"{R}/common/cases.csv")
te = cases[(cases.dataset == "FF++") & (cases.split == "test")]
ben = te[te.protocol == "benign"].drop_duplicates("enrollment_id")
swaps = te[te.protocol == "swap"].drop_duplicates("enrollment_id")[["enrollment_id", "input_path"]]
pool = ben.merge(swaps, on="enrollment_id", suffixes=("", "_swap")).head(N)

MODES = ["derive", "q84", "sub"]
sizes = {m: [] for m in MODES + ["full"]}
res = {m: {"Hb": [], "Hf": [], "G": [], "Gw": []} for m in MODES}
import os
for _, q in pool.iterrows():
    sizes["full"].append(os.path.getsize(q.doctor_trigger_template_path))
    with np.load(q.doctor_trigger_template_path) as z:
        rec = {k: np.fft.ifftshift(z[k].astype(np.float64 if k != "support" else np.uint8)) for k in
               ["template_phase", "host_phase", "support", "host_amplitude", "delta"]}
        rec["support"] = rec["support"].astype(bool)
        key, alpha, beta = int(z["key"][0]), float(z["alpha"][0]), float(z["beta"][0])
    rad = radius_grid(rec["template_phase"].shape)
    pb = phase_of(Image.open(q.input_path).convert("RGB"))
    pf = phase_of(Image.open(q.input_path_swap).convert("RGB"))
    # regenerate delta from key (the derivable field)
    delta_full = np.fft.ifftshift(feats.wrong_delta(key, rec["template_phase"].shape, alpha, beta,
                                                    np.fft.fftshift(rec["support"])))
    wrongs = [np.fft.ifftshift(feats.wrong_delta(wk, rec["template_phase"].shape, alpha, beta,
              np.fft.fftshift(rec["support"]))) for wk in feats.wrong_keys(key, 4)]
    for m in MODES:
        e, act = enc(rec, rad, m)
        sizes[m].append(payload_bytes(e))
        t, a = dec_phase_amp(e, m)
        host = t - delta_full[act]
        for tag, ph in [("Hb", pb), ("Hf", pf)]:
            r = np.cos((ph[act] - t + np.pi) % (2*np.pi) - np.pi)
            res[m][tag].append(float(np.sum(r*a) / (np.sum(a) + 1e-8)))
        resid = (pb[act] - host + np.pi) % (2*np.pi) - np.pi
        d = delta_full[act]
        res[m]["G"].append(float(np.sum(a*d*np.sin(resid)) / (np.sum(a*d*d) + 1e-6)))
        for wd in wrongs:
            w_ = wd[act]
            res[m]["Gw"].append(float(np.sum(a*w_*np.sin(resid)) / (np.sum(a*w_*w_) + 1e-6)))

print(f"{'variant':8s} {'KiB':>7s} {'det AUC':>8s} {'benH':>7s} {'keyAUC':>7s}")
print(f"{'full':8s} {np.mean(sizes['full'])/1024:7.1f}    (shipped NPZ)")
for m in MODES:
    y = np.r_[np.zeros(N), np.ones(N)]
    det = roc_auc_score(y, np.r_[1-np.array(res[m]['Hb']), 1-np.array(res[m]['Hf'])])
    pos, neg = np.array(res[m]["G"]), np.array(res[m]["Gw"])
    key_auc = roc_auc_score(np.r_[np.ones_like(pos), np.zeros_like(neg)], np.r_[pos, neg])
    print(f"{m:8s} {np.mean(sizes[m])/1024:7.1f} {det:8.3f} {np.mean(res[m]['Hb']):7.3f} {key_auc:7.3f}")
