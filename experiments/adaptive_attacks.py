#!/usr/bin/env python
"""Targeted attacks against the deployed scores (Table 17, PGD / NES / transfer rows).

Three access levels, all at L-infinity budget 8/255 on images in [0,1], on FF++ test pairs
(released image as the negative, its SimSwap re-rendering as the positive):

  white_box      PGD on the deployed scores themselves (relation, detector, or both).
  score_only     NES with 24 score queries per image, one signed step.
  surrogate      PGD on a proxy image detector trained on another corpus, transferred.

Each attack pushes every image toward the wrong verdict (benign up, re-rendered down in
suspiciousness), and AUC is reported afterwards for the template mismatch 1 - C_B (column
``rel_err``: agreement with the template T_I on K_eff and sub-band B), the image-only detector S_df
and the standardized alarm max(z(1 - C_B), min(z(S_df), 4)), with robust z against 200 FF++ train
releases. Chance is 0.500; 0.000 means the ranking is fully inverted. The record-model column of
Table 17 is not computed here.

The attacks are random: the PGD start and the NES directions are drawn from torch's unseeded
global generator, as in the run behind Table 17, so a rerun reproduces the table up to sampling
noise, not bit for bit.

Paths from config.yaml (doctor_trigger.config): queries_csv, outputs_root, checkpoints.image_detector_aug;
optional env DT_ADAPT_N (default 200). Writes <outputs_root>/adaptive/{per_query.csv,summary.csv}.
"""
import os
from pathlib import Path

import numpy as np, pandas as pd, torch
from PIL import Image

from doctor_trigger import config
from doctor_trigger.manifests import read_queries
from doctor_trigger.metrics import robust_z, roc_auc
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import subband_b
from experiments.shared.image_only_detector import checkpoint_path, image_only_logit, load_image_only_detector

OUT = Path(config.path("outputs_root")) / "adaptive"
N = int(os.environ.get("DT_ADAPT_N", "200")); EPS = 8 / 255; STEPS = 40; CUT = 0.15
dev = "cuda"

# deployed detector and the attacker's proxy
image_detector = load_image_only_detector(checkpoint_path("ffpp"), dev)
proxy = load_image_only_detector(checkpoint_path("dfdc"), dev)
for p in list(image_detector.parameters()) + list(proxy.parameters()):
    p.requires_grad_(False)


def record_tensors(tp):
    """Template T_I and sub-band-B weights (K_eff, kappa <= CUT) of a record, uncentered, on the GPU."""
    record = load_record(tp)
    tm = np.fft.ifftshift(record.template_phase)
    w = np.where(np.fft.ifftshift(record.active_bins) & subband_b(tm.shape, centered=False, kappa_max=CUT),
                 np.fft.ifftshift(record.amplitude_weights), 0.0)
    return (torch.tensor(tm, device=dev, dtype=torch.float32),
            torch.tensor(w, device=dev, dtype=torch.float32))


def template_mismatch(x, tm, w):
    """1 - C_B of an image batch x in [0, 1] (differentiable; weights w are zero outside the bins)."""
    lum = 0.299 * x[0, 0] + 0.587 * x[0, 1] + 0.114 * x[0, 2]
    ph = torch.angle(torch.fft.fft2(lum))
    d = ph - tm
    return 1.0 - (torch.cos(torch.atan2(torch.sin(d), torch.cos(d))) * w).sum() / (w.sum() + 1e-8)


def objective(x, tm, w, fake, kind):
    """Raise the loss on the true label: benign toward 'fake', re-rendered toward 'benign'."""
    s = 0.0
    if kind in ("white_box_record", "white_box_joint"):
        e = template_mismatch(x, tm, w); s = s + (e if not fake else -e) * 100.0
    if kind in ("white_box_image", "white_box_joint"):
        g = image_only_logit(image_detector, x); s = s + (g if not fake else -g)
    if kind == "surrogate":
        g = image_only_logit(proxy, x); s = s + (g if not fake else -g)
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


def load_image(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.tensor(a, device=dev).permute(2, 0, 1)[None]


@torch.no_grad()
def scores(x, tm, w):
    return float(template_mismatch(x, tm, w)), float(torch.sigmoid(image_only_logit(image_detector, x)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = read_queries()
    ff = man[(man.dataset == "FF++") & (man.split == "test")]
    pos = ff[ff.protocol == "swap"].head(N)
    neg = ff[ff.protocol == "benign"].set_index("enrollment_id")
    train = man[(man.dataset == "FF++") & (man.split == "train") & (man.protocol == "benign")].head(200)

    cal_h, cal_d = [], []
    for q in train.itertuples():
        tm, w = record_tensors(q.template_artifact); h, d = scores(load_image(q.released_path), tm, w)
        cal_h.append(h); cal_d.append(d)

    rows = []
    for k, q in enumerate(pos.itertuples(), 1):
        if q.enrollment_id not in neg.index: continue
        tm, w = record_tensors(q.template_artifact)
        pair = [("benign", load_image(neg.loc[q.enrollment_id].released_path), False),
                ("re-rendered", load_image(q.query_path), True)]
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
    df["alarm"] = np.maximum(robust_z(df.rel_err, cal_h), np.minimum(robust_z(df.S_df, cal_d), 4))
    df.to_csv(OUT / "per_query.csv", index=False)
    out = []
    for atk, g in df.groupby("attack"):
        y = (g.label == "re-rendered").astype(int)
        out.append(dict(attack=atk, n_pairs=len(g) // 2,
                        record_only=roc_auc(y, g.rel_err), image_only=roc_auc(y, g.S_df),
                        alarm=roc_auc(y, g.alarm)))
    s = pd.DataFrame(out).set_index("attack").reindex(
        ["none", "white_box_record", "white_box_image", "white_box_joint", "score_only_nes", "surrogate"])
    s.to_csv(OUT / "summary.csv"); print(s.round(3).to_string())
    print("=== ADAPTIVE DONE ===")


if __name__ == "__main__":
    main()
