"""What the archived record is, and what it is worth (canonical NPZ templates).

(a) the filed template phase on its ring
(b) phase-only reconstruction from the template alone -- it carries the face's
    own structure, which is why the record must stay private
(c) cross-asset matrix: query i against record j. The diagonal is the enrolled
    pairing; everything off it collapses.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root holds doctor_trigger
from doctor_trigger import config, manifests
from doctor_trigger.readings import SumForm, spectral_agreement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase

OUT = Path(config.path("outputs_root")) / "figures"; OUT.mkdir(parents=True, exist_ok=True)
N, EPS = 24, 1e-12
c = manifests.read_cases(dataset="FF++", split="test", protocol="benign").head(N)

records, queries = [], []
for r in c.itertuples():
    records.append(load_record(r.doctor_trigger_template_path))
    queries.append(luminance_phase(np.asarray(Image.open(r.input_path).convert("RGB")), centered=True))

# H of query i against record j, whole-grid weight map with stabilizer 1e-12 (as the paper figure)
M = np.full((N, N), np.nan)
for i, q in enumerate(queries):
    for j, rec in enumerate(records):
        if q.shape != rec.shape: continue
        M[i, j] = spectral_agreement(q, rec.original_phase, rec.amplitude_weights, rec.active_bins,
                                     stabilizer=EPS, form=SumForm.WEIGHT_MAP)
own = np.diag(M); off = M[~np.eye(N, dtype=bool)]; off = off[np.isfinite(off)]

tp0, sup0 = records[0].template_phase, records[0].active_bins
TS, LS = 12.5, 10.5
fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.9))
plt.subplots_adjust(left=0.02, right=0.965, top=0.86, bottom=0.10, wspace=0.20)

disp = np.where(sup0, tp0, np.nan)
im0 = ax[0].imshow(disp, cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
ax[0].set_title("(a) filed template $\\mathcal{T}_I$\non the mid-band ring", fontsize=TS)
ax[0].set_xticks([]); ax[0].set_yticks([])
cb = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03, ticks=[-np.pi, 0, np.pi])
cb.ax.set_yticklabels(["$-\\pi$", "$0$", "$\\pi$"], fontsize=LS); cb.outline.set_visible(False)

spec = np.zeros(tp0.shape, complex); spec[sup0] = np.exp(1j * tp0[sup0])
recon = np.real(np.fft.ifft2(np.fft.ifftshift(spec)))
rlim = np.percentile(np.abs(recon - recon.mean()), 99.3)
ax[1].imshow(np.clip(recon - recon.mean(), -rlim, rlim), cmap="Greys_r", interpolation="nearest")
ax[1].set_title("(b) phase-only reconstruction\nfrom $\\mathcal{T}_I$ alone", fontsize=TS)
ax[1].set_xticks([]); ax[1].set_yticks([])
ax[1].text(0.5, -0.075, "carries the enrolled face's own structure", transform=ax[1].transAxes,
           ha="center", fontsize=LS, color="#B63E30")

im2 = ax[2].imshow(M, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
ax[2].set_title("(c) query $i$ against record $j$\nrelation $H$", fontsize=TS)
ax[2].set_xlabel("record index $j$", fontsize=LS); ax[2].set_ylabel("query index $i$", fontsize=LS)
ax[2].tick_params(labelsize=9)
cb2 = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03, ticks=[0, 0.5, 1])
cb2.ax.tick_params(labelsize=LS); cb2.outline.set_visible(False)

for e in ("pdf", "png"):
    fig.savefig(OUT / f"fig_template.{e}", dpi=200, bbox_inches="tight", facecolor="white")
print(f"own {own.mean():.4f}+-{own.std():.4f} (min {np.nanmin(own):.4f}) | "
      f"cross {off.mean():.4f}+-{off.std():.4f} (max {off.max():.4f}) n_cross {len(off)}")
