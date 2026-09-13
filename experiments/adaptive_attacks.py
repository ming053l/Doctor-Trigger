#!/usr/bin/env python
"""Adaptive attacks against the deployed readings (Table: stress tests).

Three access levels, all at L-infinity budget 8/255 on images in [0,1], on FF++ test pairs
(released image as the negative, its SimSwap re-rendering as the positive):

  white_box      PGD on the deployed scores themselves (relation, detector, or both).
  score_only     NES with 24 score queries per image, one signed step.
  surrogate      PGD on a proxy image detector trained on another corpus, transferred.

Each attack pushes every image toward the wrong verdict (benign down, re-rendered up), and we report
AUC afterwards for the record path (1 - H over the declared sub-band), the image-only detector, and
the standardized alarm. Chance is 0.500; 0.000 means the ranking is fully inverted.

Env: DT_CANON_ROOT, DT_OUTPUTS, DT_MODEL_SNAPSHOT, DT_IMAGE_DETECTOR_AUG; optional DT_ADAPT_N (default 200).
Writes $DT_OUTPUTS/adaptive/{per_query.csv,summary.csv}.
"""
import os, sys
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn.functional as Fn, yaml
from PIL import Image
from sklearn.metrics import roc_auc_score


def _dt(env, rest=""):
    v = os.environ.get(env); assert v, f"{env} unset"; return v + rest


R = _dt("DT_CANON_ROOT"); OUT = Path(_dt("DT_OUTPUTS")) / "adaptive"
N = int(os.environ.get("DT_ADAPT_N", "200")); EPS = 8 / 255; STEPS = 40; CUT = 0.15
dev = "cuda"
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from src import models  # noqa: E402

cfg = yaml.safe_load(open(_dt("DT_MODEL_SNAPSHOT", "/config_ffpp_std_dfd_resnet50.yaml")))


def load_image_detector(tag):
    m = models.build_model(cfg).to(dev)
    m.load_state_dict(torch.load(_dt("DT_IMAGE_DETECTOR_AUG", f"/{tag}/best_val_auc.pt"), map_location=dev, weights_only=False))
    return m.eval()


image_detector, proxy = load_image_detector("ffpp"), load_image_detector("dfdc")   # deployed detector and the attacker's proxy
for p in list(image_detector.parameters()) + list(proxy.parameters()):
    p.requires_grad_(False)


def det_logit(model, x):                      # x: (1,3,H,W) in [0,1]
    o = model(Fn.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False))
    lg = o["logits"] if isinstance(o, dict) else o
    return lg[:, 1] - lg[:, 0] if lg.ndim > 1 and lg.shape[1] == 2 else lg.reshape(-1)


def rec(tp):
    with np.load(tp) as z:
        tm = np.fft.ifftshift(z["template_phase"].astype(np.float64))
        sup = np.fft.ifftshift(z["support"].astype(bool))
        amp = np.fft.ifftshift(z["host_amplitude"].astype(np.float64))
    fy = np.fft.fftfreq(tm.shape[0])[:, None]; fx = np.fft.fftfreq(tm.shape[1])[None, :]
    w = np.where(sup & (np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5) <= CUT), amp, 0.0)
    return (torch.tensor(tm, device=dev, dtype=torch.float32),
            torch.tensor(w, device=dev, dtype=torch.float32))


def rel_err(x, tm, w):                        # 1 - H over the sub-band, differentiable
    lum = 0.299 * x[0, 0] + 0.587 * x[0, 1] + 0.114 * x[0, 2]
    ph = torch.angle(torch.fft.fft2(lum))
    d = ph - tm
    return 1.0 - (torch.cos(torch.atan2(torch.sin(d), torch.cos(d))) * w).sum() / (w.sum() + 1e-8)


def objective(x, tm, w, fake, kind):
    """Raise the loss on the true label: benign toward 'fake', re-rendered toward 'benign'."""
    s = 0.0
    if kind in ("white_box_record", "white_box_joint"):
        e = rel_err(x, tm, w); s = s + (e if not fake else -e) * 100.0
    if kind in ("white_box_image", "white_box_joint"):
        g = det_logit(image_detector, x); s = s + (g if not fake else -g)
    if kind == "surrogate":
        g = det_logit(proxy, x); s = s + (g if not fake else -g)
    return s


def pgd(x0, tm, w, fake, kind):
    d = torch.zeros_like(x0).uniform_(-EPS, EPS).requires_grad_(True)
    for _ in range(STEPS):
        loss = objective((x0 + d).clamp(0, 1), tm, w, fake, kind)
        (grad,) = torch.autograd.grad(loss, d)
        d = (d + (EPS / 8) * grad.sign()).clamp(-EPS, EPS).detach().requires_grad_(True)
    return (x0 + d).clamp(0, 1).detach()


@torch.no_grad()
def nes(x0, tm, w, fake, queries=24, sigma=2 / 255):
    g = torch.zeros_like(x0)
    for _ in range(queries // 2):                      # antithetic pairs, score queries only
        u = torch.randn_like(x0)
        sp = objective((x0 + sigma * u).clamp(0, 1), tm, w, fake, "white_box_joint")
        sm = objective((x0 - sigma * u).clamp(0, 1), tm, w, fake, "white_box_joint")
        g = g + (sp - sm) * u
    return (x0 + EPS * g.sign()).clamp(0, 1)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.tensor(a, device=dev).permute(2, 0, 1)[None]


@torch.no_grad()
def scores(x, tm, w):
    return float(rel_err(x, tm, w)), float(torch.sigmoid(det_logit(image_detector, x)))


def rz(v, cal):
    cal = np.asarray(cal); med = np.median(cal); mad = np.median(np.abs(cal - med)) + 1e-6
    return (np.asarray(v) - med) / mad


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(f"{R}/manifests/queries.csv"); man = man[man.status == "pass"]
    ff = man[(man.dataset == "FF++") & (man.split == "test")]
    pos = ff[ff.protocol == "swap"].head(N)
    neg = ff[ff.protocol == "benign"].set_index("enrollment_id")
    train = man[(man.dataset == "FF++") & (man.split == "train") & (man.protocol == "benign")].head(200)

    cal_h, cal_d = [], []
    for q in train.itertuples():
        tm, w = rec(q.template_artifact); h, d = scores(load(q.released_path), tm, w)
        cal_h.append(h); cal_d.append(d)

    rows = []
    for k, q in enumerate(pos.itertuples(), 1):
        if q.enrollment_id not in neg.index: continue
        tm, w = rec(q.template_artifact)
        pair = [("benign", load(neg.loc[q.enrollment_id].released_path), False),
                ("re-rendered", load(q.query_path), True)]
        for label, x0, fake in pair:
            h, d = scores(x0, tm, w)
            rows.append(dict(query_id=q.query_id, label=label, attack="none", rel_err=h, S_df=d))
            for kind in ("white_box_record", "white_box_image", "white_box_joint", "surrogate"):
                h2, d2 = scores(pgd(x0, tm, w, fake, kind), tm, w)
                rows.append(dict(query_id=q.query_id, label=label, attack=kind, rel_err=h2, S_df=d2))
            h3, d3 = scores(nes(x0, tm, w, fake), tm, w)
            rows.append(dict(query_id=q.query_id, label=label, attack="score_only_nes", rel_err=h3, S_df=d3))
        if k % 25 == 0: print(f"{k}/{len(pos)} pairs", flush=True)

    df = pd.DataFrame(rows)
    df["alarm"] = np.maximum(rz(df.rel_err, cal_h), np.minimum(rz(df.S_df, cal_d), 4))
    df.to_csv(OUT / "per_query.csv", index=False)
    out = []
    for atk, g in df.groupby("attack"):
        y = (g.label == "re-rendered").astype(int)
        out.append(dict(attack=atk, n_pairs=len(g) // 2,
                        record_only=roc_auc_score(y, g.rel_err), image_only=roc_auc_score(y, g.S_df),
                        alarm=roc_auc_score(y, g.alarm)))
    s = pd.DataFrame(out).set_index("attack").reindex(
        ["none", "white_box_record", "white_box_image", "white_box_joint", "score_only_nes", "surrogate"])
    s.to_csv(OUT / "summary.csv"); print(s.round(3).to_string())
    print("=== ADAPTIVE DONE ===")


if __name__ == "__main__":
    main()
