#!/usr/bin/env python
"""Appendix version of Fig. 2 (right): more assets, three corpora.

Each row is one enrolled asset: the released image, the same image after a benign channel (JPEG q85),
and its SimSwap re-rendering, with the relation H measured against that asset's own record and the
corpus cutoff (the 1% quantile of H on 100 train benign queries of that corpus). Drawn at the text
width so the paper includes it without scaling.

Env: DT_CANON_ROOT, DT_OUTPUTS. Writes $DT_OUTPUTS/figures/fig_channel_samples.pdf.
"""
import io, os
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from PIL import Image

mpl.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
R = os.environ["DT_CANON_ROOT"]; OUT = Path(os.environ["DT_OUTPUTS"]) / "figures"; OUT.mkdir(parents=True, exist_ok=True)
INK, GOOD, BAD = "#17202B", "#0C5E59", "#B63E30"
ROWS = [("FF++", 7), ("FF++", 21), ("CelebDF", 5), ("DFDC", 9)]     # corpus, index among test pairs


def phase(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    lum = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return np.angle(np.fft.fft2(lum))


def relation(img, tp):
    """Deployed reading: amplitude-weighted cosine against the stored template on the sub-band."""
    with np.load(tp) as z:
        tm = np.fft.ifftshift(z["template_phase"].astype(np.float64))
        sup = np.fft.ifftshift(z["support"].astype(bool))
        amp = np.fft.ifftshift(z["host_amplitude"].astype(np.float64))
    fy = np.fft.fftfreq(tm.shape[0])[:, None]; fx = np.fft.fftfreq(tm.shape[1])[None, :]
    w = np.where(sup & (np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5) <= 0.15), amp, 0.0)
    r = np.cos((phase(img) - tm + np.pi) % (2 * np.pi) - np.pi)
    return float((r * w).sum() / (w.sum() + 1e-8))


def jpeg(img, q=85):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=q); b.seek(0); return Image.open(b)


man = pd.read_csv(f"{R}/manifests/queries.csv"); man = man[man.status == "pass"]
tau = {}
for ds in {d for d, _ in ROWS}:
    tr = man[(man.dataset == ds) & (man.split == "train") & (man.protocol == "benign")].head(100)
    tau[ds] = float(np.quantile([relation(Image.open(q.released_path), q.template_artifact) for q in tr.itertuples()], 0.01))
print("cutoffs:", {k: round(v, 4) for k, v in tau.items()})

fig, axes = plt.subplots(len(ROWS), 3, figsize=(5.5, 1.95 * len(ROWS)))
titles = ["released image", "benign channel: JPEG", "face re-rendering"]
for r, (ds, idx) in enumerate(ROWS):
    te = man[(man.dataset == ds) & (man.split == "test")]
    swap_q = te[te.protocol == "swap"].iloc[idx]
    ben = te[(te.protocol == "benign") & (te.enrollment_id == swap_q.enrollment_id)].iloc[0]
    rel = Image.open(ben.released_path).convert("RGB")
    panels = [(rel, relation(rel, ben.template_artifact)),
              (jpeg(rel), relation(jpeg(rel), ben.template_artifact)),
              (Image.open(swap_q.query_path).convert("RGB"), relation(Image.open(swap_q.query_path), swap_q.template_artifact))]
    for c, (img, h) in enumerate(panels):
        ax = axes[r, c]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ok = h >= tau[ds]
        if r == 0: ax.set_title(titles[c], fontsize=8.5, pad=3, color=INK)
        ax.set_xlabel(f"$H$ = {h:.3f}, {'passes' if ok else 'fails'} {tau[ds]:.3f}", fontsize=7,
                      color=GOOD if ok else BAD, labelpad=2)
        for sp in ax.spines.values():
            sp.set_linewidth(0.9); sp.set_edgecolor(GOOD if ok else BAD)
    axes[r, 0].set_ylabel({"FF++": "FF++", "CelebDF": "Celeb-DF", "DFDC": "DFDC"}[ds], fontsize=8, color=INK)
fig.tight_layout(w_pad=0.6, h_pad=0.7, pad=0.2)
fig.savefig(OUT / "fig_channel_samples.pdf", pad_inches=0.01)
fig.savefig(OUT / "fig_channel_samples.png", dpi=200, pad_inches=0.01)
print("wrote fig_channel_samples")
