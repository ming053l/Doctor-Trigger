"""How small can the per-face record get?

Variants, each verified on FF++ test (n=150) for H, G_s, and wrong-key AUC:
  full   : the deployed NPZ as shipped (baseline bytes)
  derive : drop every derivable field (delta, code, host phase); keep active-bin
           template phase + amplitude as float16 + a band bitmask + metadata
  q84    : derive + phase quantized to 8 bits, log-amplitude to 4 bits
  sub    : q84 restricted to the declared sub-band (kappa <= 0.15)
Bytes are zlib-compressed serialized payloads."""
import io, os, sys, zlib
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger.key import wrong_keys  # noqa: E402
from doctor_trigger.manifests import read_cases  # noqa: E402
from doctor_trigger.metrics import roc_auc  # noqa: E402
from doctor_trigger.readings import SumForm, key_projection, phase_drift, spectral_agreement, wrong_key_displacement  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from doctor_trigger.spectrum import kappa_grid, luminance_phase  # noqa: E402

N = 150
AGREEMENT_STABILIZER = 1e-8
PROJECTION_STABILIZER = 1e-6


def phase_of(path):
    """Unshifted luminance phase of an image file."""
    return luminance_phase(np.asarray(Image.open(path).convert("RGB"), dtype=np.float64), centered=False)

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

te = read_cases(dataset="FF++", split="test")
ben = te[te.protocol == "benign"].drop_duplicates("enrollment_id")
swaps = te[te.protocol == "swap"].drop_duplicates("enrollment_id")[["enrollment_id", "input_path"]]
pool = ben.merge(swaps, on="enrollment_id", suffixes=("", "_swap")).head(N)

MODES = ["derive", "q84", "sub"]
sizes = {m: [] for m in MODES + ["full"]}
res = {m: {"Hb": [], "Hf": [], "G": [], "Gw": []} for m in MODES}
for _, q in pool.iterrows():
    sizes["full"].append(os.path.getsize(q.doctor_trigger_template_path))
    record = load_record(q.doctor_trigger_template_path)
    rec = {"template_phase": np.fft.ifftshift(record.template_phase),
           "support": np.fft.ifftshift(record.active_bins),
           "host_amplitude": np.fft.ifftshift(record.amplitude_weights)}
    rad = kappa_grid(record.shape, centered=False)
    pb = phase_of(q.input_path)
    pf = phase_of(q.input_path_swap)
    # regenerate delta from key (the derivable field)
    delta_full = np.fft.ifftshift(wrong_key_displacement(record, record.key))
    wrongs = [np.fft.ifftshift(wrong_key_displacement(record, wk)) for wk in wrong_keys(record.key, 4)]
    for m in MODES:
        e, act = enc(rec, rad, m)
        sizes[m].append(payload_bytes(e))
        t, a = dec_phase_amp(e, m)
        every_bin = np.ones(t.shape, dtype=bool)
        host = t - delta_full[act]
        for tag, ph in [("Hb", pb), ("Hf", pf)]:
            res[m][tag].append(spectral_agreement(ph[act], t, a, every_bin, stabilizer=AGREEMENT_STABILIZER,
                                                  form=SumForm.WEIGHT_MAP))
        resid = phase_drift(pb[act], host)
        d = delta_full[act]
        res[m]["G"].append(key_projection(resid, d, a, every_bin, stabilizer=PROJECTION_STABILIZER))
        for wd in wrongs:
            res[m]["Gw"].append(key_projection(resid, wd[act], a, every_bin, stabilizer=PROJECTION_STABILIZER))

print(f"{'variant':8s} {'KiB':>7s} {'det AUC':>8s} {'benH':>7s} {'keyAUC':>7s}")
print(f"{'full':8s} {np.mean(sizes['full'])/1024:7.1f}    (shipped NPZ)")
for m in MODES:
    y = np.r_[np.zeros(N), np.ones(N)]
    det = roc_auc(y, np.r_[1-np.array(res[m]['Hb']), 1-np.array(res[m]['Hf'])])
    pos, neg = np.array(res[m]["G"]), np.array(res[m]["Gw"])
    key_auc = roc_auc(np.r_[np.ones_like(pos), np.zeros_like(neg)], np.r_[pos, neg])
    print(f"{m:8s} {np.mean(sizes[m])/1024:7.1f} {det:8.3f} {np.mean(res[m]['Hb']):7.3f} {key_auc:7.3f}")
