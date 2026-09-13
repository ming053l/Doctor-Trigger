"""What the archived record is, and what it is worth (canonical NPZ templates).

(a) the filed template phase on its ring
(b) phase-only reconstruction from the template alone -- it carries the face's
    own structure, which is why the record must stay private
(c) cross-asset matrix: query i against record j. The diagonal is the enrolled
    pairing; everything off it collapses.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
from PIL import Image

R = os.environ["DT_CANON_ROOT"]
OUT = os.environ["DT_OUTPUTS"] + "/figures"; os.makedirs(OUT, exist_ok=True)
N, EPS = 24, 1e-12
c = pd.read_csv(f"{R}/common/cases.csv")
c = c[(c.split=="test") & (c.dataset=="FF++") & (c.protocol=="benign")].head(N)

def phase_of(p):
    rgb = np.asarray(Image.open(p).convert("RGB"), np.float64)
    lum = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    return np.fft.fftshift(np.angle(np.fft.fft2(lum)))

T, Q = [], []
for r in c.itertuples():
    with np.load(r.doctor_trigger_template_path) as rec:
        T.append((rec["template_phase"].astype(np.float64), rec["host_phase"].astype(np.float64),
                  rec["support"].astype(bool), rec["host_amplitude"].astype(np.float64)))
    Q.append(phase_of(r.input_path))

M = np.full((N, N), np.nan)
for i, q in enumerate(Q):
    for j, (tp, hp, sup, w) in enumerate(T):
        if q.shape != hp.shape: continue
        r = np.cos((q - hp + np.pi) % (2*np.pi) - np.pi)
        ww = np.where(sup, w, 0.0)
        M[i, j] = float(np.sum(r*ww) / (np.sum(ww) + EPS))
own = np.diag(M); off = M[~np.eye(N, dtype=bool)]; off = off[np.isfinite(off)]

tp0, hp0, sup0, w0 = T[0]
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
    fig.savefig(f"{OUT}/fig_template.{e}", dpi=200, bbox_inches="tight", facecolor="white")
print(f"own {own.mean():.4f}+-{own.std():.4f} (min {np.nanmin(own):.4f}) | "
      f"cross {off.mean():.4f}+-{off.std():.4f} (max {off.max():.4f}) n_cross {len(off)}")
