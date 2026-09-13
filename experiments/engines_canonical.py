import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
"""StarGAN-v2 / UniFace attacks on the canonical releases, scored against the NPZ
templates with her math. Extends the recorded-attack set; SimSwap is already canonical."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, _dt("DT_PEERS_ROOT"))
from PIL import Image
from sklearn.metrics import roc_auc_score
from engines_api import ENGINES
ENGINE = sys.argv[1]
R = _dt("DT_CANON_ROOT")
EPS = 1e-12
cases = pd.read_csv(f"{R}/common/cases.csv")
cases = cases[(cases.split == "test") & (cases.dataset == "FF++")]

def phase_of(pil):
    rgb = np.asarray(pil, dtype=np.float64)
    lum = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    return np.fft.fftshift(np.angle(np.fft.fft2(lum)))

def H_of(pil, tp):
    with np.load(tp) as rec:
        host = rec["host_phase"].astype(np.float64); sup = rec["support"].astype(bool)
        w = rec["host_amplitude"].astype(np.float64)
    if (pil.size[1], pil.size[0]) != host.shape: pil = pil.resize((host.shape[1], host.shape[0]))
    r = np.cos((phase_of(pil) - host + np.pi) % (2*np.pi) - np.pi)
    ww = np.where(sup, w, 0.0); return float(np.sum(r*ww) / (np.sum(ww) + EPS))

import yaml
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from src import models
import torchvision.transforms as T
cfg = yaml.safe_load(open(_dt("DT_MODEL_SNAPSHOT", "/config_ffpp_std_dfd_resnet50.yaml")))
image_detector = models.build_model(cfg).to("cuda")
image_detector.load_state_dict(torch.load(__import__("os").environ.get("IMAGE_DETECTOR_CKPT", _dt("DT_IMAGE_DETECTOR_LEGACY")), map_location="cuda", weights_only=False)); image_detector.eval()
tfm = T.Compose([T.Resize((224,224)), T.ToTensor()])
@torch.no_grad()
def dprob(pil):
    o = image_detector(tfm(pil).unsqueeze(0).to("cuda")); lg = o["logits"] if isinstance(o, dict) else o
    return float(torch.softmax(lg,1)[0,1]) if lg.ndim>1 and lg.shape[1]==2 else float(torch.sigmoid(lg.reshape(-1))[0])

eng = ENGINES[ENGINE]()
ys, ss, dd, drop = [], [], [], 0
for eid, g in cases.groupby("enrollment_id"):
    rel = g.doctor_trigger_released_path.iloc[0]; tp = g.doctor_trigger_template_path.iloc[0]
    src = g[g.protocol=="swap"].attack_source_path.iloc[0]
    _cache = Path(_dt("DT_ENGINE_CACHE") + f"/{ENGINE}"); _cache.mkdir(parents=True, exist_ok=True)
    _cf = _cache / f"{eid}.png"
    try:
        pil = Image.open(rel).convert("RGB")
        hb = H_of(pil, tp)
        if _cf.exists():
            sw = Image.open(_cf).convert("RGB")
        else:
            sw = eng(pil, Image.open(src).convert("RGB"))
            sw.save(_cf)
        hs = H_of(sw, tp)
    except Exception: drop += 1; continue
    ys += [0,1]; ss += [1-hb, 1-hs]; dd += [dprob(pil), dprob(sw)]
fus = [max(a,b) for a,b in zip(dd,ss)]
print(f"{ENGINE}: AUC(1-H)={roc_auc_score(ys,ss):.4f} AUC(image-only detector)={roc_auc_score(ys,dd):.4f} AUC(Fus)={roc_auc_score(ys,fus):.4f} n={len(ys)//2} dropped={drop} "
      f"benignH={1-np.mean([s for s,y in zip(ss,ys) if y==0]):.4f} "
      f"attackH={1-np.mean([s for s,y in zip(ss,ys) if y==1]):.4f}")
