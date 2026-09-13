import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
"""Benign channels under the canonical templates: distort the canonical query
bytes, score closed-form H against the NPZ template (her math: luminance phase,
fftshift, host-amplitude weights) and the augmented image-only detector."""
import sys, yaml
import os as _os
import numpy as np, pandas as pd, torch
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from scripts.benign_channels import transform_image
from src import models

R = _dt("DT_CANON_ROOT")
dev = "cuda"; EPS = 1e-12
cases = pd.read_csv(f"{R}/common/cases.csv")
import sys as _s
DS = _s.argv[1] if len(_s.argv) > 1 else "FF++"
cases = cases[(cases.split == "test") & (cases.dataset == DS)]

def phase_of(pil):
    rgb = np.asarray(pil, dtype=np.float64)
    lum = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    return np.fft.fftshift(np.angle(np.fft.fft2(lum)))

def H_of(pil, tp):
    with np.load(tp) as rec:
        host = rec["host_phase"].astype(np.float64); sup = rec["support"].astype(bool)
        w = rec["host_amplitude"].astype(np.float64)
    ph = phase_of(pil)
    if ph.shape != host.shape: pil = pil.resize((host.shape[1], host.shape[0])); ph = phase_of(pil)
    r = np.cos((ph - host + np.pi) % (2*np.pi) - np.pi)
    ww = np.where(sup, w, 0.0); return float(np.sum(r*ww) / (np.sum(ww) + EPS))

cfg = yaml.safe_load(open(_dt("DT_MODEL_SNAPSHOT", "/config_ffpp_std_dfd_resnet50.yaml")))
image_detector = models.build_model(cfg).to(dev)
image_detector.load_state_dict(torch.load(_os.environ.get("IMAGE_DETECTOR_CKPT", _dt("DT_IMAGE_DETECTOR_LEGACY")), map_location=dev, weights_only=False)); image_detector.eval()
tf = T.Compose([T.Resize((224,224)), T.ToTensor()])
@torch.no_grad()
def prob(pil):
    o = image_detector(tf(pil).unsqueeze(0).to(dev)); lg = o["logits"] if isinstance(o, dict) else o
    return float(torch.softmax(lg,1)[0,1]) if lg.ndim>1 and lg.shape[1]==2 else float(torch.sigmoid(lg.reshape(-1))[0])

PERTS = ["clean","jpeg_q90","jpeg_q70","jpeg_q50","blur_k7","x264_crf30","x264_crf35","x264_crf40","scale_0.5"]
FAM = {"JPEG":["jpeg_q90","jpeg_q70","jpeg_q50"],"blur":["blur_k7"],
       "codec":["x264_crf30","x264_crf35","x264_crf40"],"pixel":["scale_0.5"]}
res = {}
for p in PERTS:
    ys, hs, ds_ = [], [], []
    for k, q in enumerate(cases.itertuples()):
        try:
            img = transform_image(Image.open(q.input_path).convert("RGB"), p, seed=(k+17)*1009)
            h = H_of(img, q.doctor_trigger_template_path); dp = prob(img)
        except Exception: continue
        ys.append(q.y_fake); hs.append(1-h); ds_.append(dp)
    res[p] = (roc_auc_score(ys, hs), roc_auc_score(ys, ds_),
              roc_auc_score(ys, np.maximum(ds_, hs)), len(ys))
print(f"{'family':8s} {'H(1-)':>8s} {'image-only detector':>8s} {'Fusion':>8s}   n")
for fam, ps in FAM.items():
    a=[res[p][0] for p in ps]; b=[res[p][1] for p in ps]; c=[res[p][2] for p in ps]
    print(f"{fam:8s} {np.mean(a):8.4f} {np.mean(b):8.4f} {np.mean(c):8.4f}  {res[ps[0]][3]}")
print(f"{'clean':8s} {res['clean'][0]:8.4f} {res['clean'][1]:8.4f} {res['clean'][2]:8.4f}")
