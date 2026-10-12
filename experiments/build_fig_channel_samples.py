#!/usr/bin/env python
"""Appendix version of Fig. 2 (right): more assets, three corpora.

Each row is one enrolled asset: the released image, the same image after a benign channel (JPEG q85),
and its SimSwap re-rendering, with the relation H measured against that asset's own record and the
corpus cutoff (the 1% quantile of H on 100 train benign queries of that corpus). Drawn at the text
width so the paper includes it without scaling.

Paths from config.yaml (doctor_trigger.config): queries_csv, outputs_root.
Writes <outputs_root>/figures/fig_channel_samples.{pdf,png}.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root holds doctor_trigger
from doctor_trigger import config, manifests
from doctor_trigger.calibration import lower_tail_threshold
from doctor_trigger.channels import jpeg
from doctor_trigger.readings import SumForm, spectral_agreement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase, subband_b

mpl.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
OUT = Path(config.path("outputs_root")) / "figures"; OUT.mkdir(parents=True, exist_ok=True)
INK, GOOD, BAD = "#17202B", "#0C5E59", "#B63E30"
ROWS = [("FF++", 7), ("FF++", 21), ("CelebDF", 5), ("DFDC", 9)]     # corpus, index among test pairs


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def relation(image, record_path):
    """Deployed reading C_B: amplitude-weighted cosine against the stored template on K_eff and
    sub-band B (Eq. 5), summed over the uncentered grid with stabilizer 1e-8, as the paper figure."""
    record = load_record(record_path)
    template = np.fft.ifftshift(record.template_phase)
    bins = np.fft.ifftshift(record.active_bins) & subband_b(record.shape, centered=False)
    return spectral_agreement(luminance_phase(image, centered=False), template,
                              np.fft.ifftshift(record.amplitude_weights), bins,
                              stabilizer=1e-8, form=SumForm.WEIGHT_MAP)


man = manifests.read_queries()
tau = {}
for ds in {d for d, _ in ROWS}:
    tr = manifests.select(man, dataset=ds, split="train", protocol="benign").head(100)
    tau[ds] = float(lower_tail_threshold([relation(rgb(q.released_path), q.template_artifact) for q in tr.itertuples()]))
print("cutoffs:", {k: round(v, 4) for k, v in tau.items()})

fig, axes = plt.subplots(len(ROWS), 3, figsize=(5.5, 1.95 * len(ROWS)))
titles = ["released image", "benign channel: JPEG", "face re-rendering"]
for r, (ds, idx) in enumerate(ROWS):
    te = manifests.select(man, dataset=ds, split="test")
    swap_q = te[te.protocol == "swap"].iloc[idx]
    ben = te[(te.protocol == "benign") & (te.enrollment_id == swap_q.enrollment_id)].iloc[0]
    rel, swapped = rgb(ben.released_path), rgb(swap_q.query_path)
    rel_jpeg = jpeg(rel, 85)
    panels = [(rel, relation(rel, ben.template_artifact)),
              (rel_jpeg, relation(rel_jpeg, ben.template_artifact)),
              (swapped, relation(swapped, swap_q.template_artifact))]
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
