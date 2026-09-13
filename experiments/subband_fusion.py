"""Fusion row of Table 1 recomputed over the sub-band record reading:
score = max(image-path probability, 1 - H_subband), per query, per cell."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest
import sys
import numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score
import torchvision.transforms as T
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from scripts.benign_channels import transform_image as _ti_canon
import io as _io
def transform_image(img, pert, seed):
    if pert.startswith("webp_q"):
        b = _io.BytesIO(); img.save(b, "WEBP", quality=int(pert.split("q")[1])); b.seek(0)
        return Image.open(b).convert("RGB")
    return _ti_canon(img, pert, seed)
from src import models
import yaml
R = _dt("DT_CANON_ROOT")
CUT = 0.15
dev = "cuda"

cfg = yaml.safe_load(open(_dt("DT_MODEL_SNAPSHOT", "/config_ffpp_std_dfd_resnet50.yaml")))
image_detector = models.build_model(cfg).to(dev)
image_detector.load_state_dict(torch.load(_dt("DT_IMAGE_DETECTOR_AUG", "/ffpp/best_val_auc.pt"), map_location=dev, weights_only=False))
image_detector.eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
@torch.no_grad()
def prob(pil):
    o = image_detector(tf(pil).unsqueeze(0).to(dev)); lg = o["logits"] if isinstance(o, dict) else o
    return float(torch.softmax(lg, 1)[0, 1]) if lg.ndim > 1 and lg.shape[1] == 2 else float(torch.sigmoid(lg.reshape(-1))[0])

def phase_of(pil):
    a = np.asarray(pil, dtype=np.float64)
    lum = 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
    return np.angle(np.fft.fft2(lum))

_rc = {}
def radius_grid(shape):
    if shape not in _rc:
        fy = np.fft.fftfreq(shape[0]).reshape(-1, 1)
        fx = np.fft.fftfreq(shape[1]).reshape(1, -1)
        _rc[shape] = np.sqrt(fy*fy + fx*fx) / np.sqrt(0.5**2 + 0.5**2)
    return _rc[shape]

def H_both(pil, tp):
    with np.load(tp) as rec:
        tmpl = np.fft.ifftshift(rec["template_phase"].astype(np.float64))
        sup = np.fft.ifftshift(rec["support"].astype(bool))
        amp = np.fft.ifftshift(rec["host_amplitude"].astype(np.float64))
    ph = phase_of(pil)
    if ph.shape != tmpl.shape:
        pil2 = pil.resize((tmpl.shape[1], tmpl.shape[0])); ph = phase_of(pil2)
    r = np.cos((ph - tmpl + np.pi) % (2*np.pi) - np.pi)
    w1 = np.where(sup & (radius_grid(tmpl.shape) <= CUT), amp, 0.0)
    w2 = np.where(sup, amp, 0.0)
    return (float(np.sum(r*w1) / (np.sum(w1) + 1e-8)),
            float(np.sum(r*w2) / (np.sum(w2) + 1e-8)))

def H_sub(pil, tp):
    return H_both(pil, tp)[0]

cases = pd.read_csv(f"{R}/common/cases.csv")
PART = sys.argv[1] if len(sys.argv) > 1 else "all"
N_CAL = 200

def rz(x, cal):
    cal = np.asarray(cal, dtype=np.float64)
    med = np.median(cal)
    mad = np.median(np.abs(cal - med)) + 1e-6
    return (np.asarray(x, dtype=np.float64) - med) / mad

def fused_all(det, hs, det_cal, hs_cal):
    det, hs = np.asarray(det, np.float64), np.asarray(hs, np.float64)
    zd, zh = rz(det, det_cal), rz(hs, hs_cal)
    out = {"raw": np.maximum(det, hs), "z": np.maximum(zd, zh)}
    for c in (4, 8, 16):
        out[f"zdc{c}"] = np.maximum(zh, np.minimum(zd, c))
    return out

# --- dataset cells (FF++-pin, clean): image_detector probs from CSV, sub-H recomputed ---
if PART in ("all", "datasets"):
  d = pd.read_csv(_dt("DT_OUTPUTS", "/image_detector_aug.csv"))
  te = cases[cases.split == "test"].merge(d[["query_id", "image_detector_ffpp"]], on="query_id")
  for ds, g in te.groupby("dataset"):
    hs = np.array([1 - H_sub(Image.open(q.input_path).convert("RGB"), q.doctor_trigger_template_path) for _, q in g.iterrows()])
    tr = cases[(cases.dataset == ds) & (cases.split == "train") & (cases.protocol == "benign")].drop_duplicates("enrollment_id").head(N_CAL)
    hs_cal, det_cal = [], []
    for _, q in tr.iterrows():
        img = Image.open(q.input_path).convert("RGB")
        hs_cal.append(1 - H_sub(img, q.doctor_trigger_template_path))
        det_cal.append(prob(img))
    for tag, fus in fused_all(g.image_detector_ffpp.values, hs, det_cal, hs_cal).items():
        print(f"FUS[{tag}] {ds}: {roc_auc_score(g.y_fake.values, fus):.3f}", flush=True)

# --- benign family cells (FF++) ---
ff = cases[(cases.dataset == "FF++") & (cases.split == "test")]
if PART.startswith("benign:"):
    PERTS = PART.split(":")[1].split(",")
    tr = cases[(cases.dataset == "FF++") & (cases.split == "train") & (cases.protocol == "benign")].drop_duplicates("enrollment_id").head(N_CAL)
    cal = {p: {"h": [], "d": []} for p in PERTS}
    for k, (_, q) in enumerate(tr.iterrows()):
        img = Image.open(q.input_path).convert("RGB")
        for p in PERTS:
            t = transform_image(img, p, seed=(k+31)*811)
            cal[p]["h"].append(1 - H_sub(t, q.doctor_trigger_template_path))
            cal[p]["d"].append(prob(t))
    res = {p: {"h": [], "hf": [], "d": [], "y": []} for p in PERTS}
    for k, (_, q) in enumerate(ff.iterrows()):
        img = Image.open(q.input_path).convert("RGB")
        for p in PERTS:
            t = transform_image(img, p, seed=(k+17)*1009)
            hs, hfull = H_both(t, q.doctor_trigger_template_path)
            res[p]["h"].append(1 - hs); res[p]["hf"].append(1 - hfull)
            res[p]["d"].append(prob(t))
            res[p]["y"].append(q.y_fake)
    for p in PERTS:
        y = res[p]["y"]
        print(f"PATH[rel_sub]-SEV {p}: {roc_auc_score(y, res[p]['h']):.4f}", flush=True)
        print(f"PATH[rel_full]-SEV {p}: {roc_auc_score(y, res[p]['hf']):.4f}", flush=True)
        print(f"PATH[det]-SEV {p}: {roc_auc_score(y, res[p]['d']):.4f}", flush=True)
        for tag, fus in fused_all(res[p]["d"], res[p]["h"], cal[p]["d"], cal[p]["h"]).items():
            print(f"FUS[{tag}]-SEV {p}: {roc_auc_score(y, fus):.4f}", flush=True)

# --- engine cells ---
CACHE = _dt("DT_PEERS_ROOT", "/engine_cache")
ben = ff[ff.protocol == "benign"].drop_duplicates("enrollment_id")
ENGS = PART.split(":")[1].split(",") if PART.startswith("engines:") else []
if ENGS:
    tr = cases[(cases.dataset == "FF++") & (cases.split == "train") & (cases.protocol == "benign")].drop_duplicates("enrollment_id").head(N_CAL)
    hs_cal, det_cal = [], []
    for _, q in tr.iterrows():
        img = Image.open(q.doctor_trigger_released_path).convert("RGB")
        hs_cal.append(1 - H_sub(img, q.doctor_trigger_template_path))
        det_cal.append(prob(img))
for eng in ENGS:
    ys, ss = [], []
    for _, q in ben.iterrows():
        cf = Path(f"{CACHE}/{eng}/{q.enrollment_id}.png")
        if not cf.exists(): continue
        rel = Image.open(q.doctor_trigger_released_path).convert("RGB")
        sw = Image.open(cf).convert("RGB")
        ss.append((prob(rel), 1 - H_sub(rel, q.doctor_trigger_template_path))); ys.append(0)
        ss.append((prob(sw), 1 - H_sub(sw, q.doctor_trigger_template_path))); ys.append(1)
    det, hs = [a for a, b in ss], [b for a, b in ss]
    for tag, fus in fused_all(det, hs, det_cal, hs_cal).items():
        print(f"FUS[{tag}] {eng}: {roc_auc_score(ys, fus):.3f}", flush=True)
