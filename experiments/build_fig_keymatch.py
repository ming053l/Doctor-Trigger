"""Appendix Fig 10: keyed displacement and query residual on one FF++ test asset.

What the key writes and the verifier reads: (a) the key's sign pattern, (b) the displacement,
(c) the stored template, (d) the query residual against the original phase, and the sign agreement
of that residual with the correct key (e) and with a wrong key (f), central region of the shifted
spectrum, bins outside the active support in gray.

Env: DT_CANON_ROOT, DT_OUTPUTS. Writes $DT_OUTPUTS/figures/fig_keymatch.pdf.
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
mpl.rcParams.update({"font.family": "Liberation Serif",
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
R = os.environ["DT_CANON_ROOT"]
OUT = Path(os.environ["DT_OUTPUTS"]) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, R)
from scripts.extract_features import wrong_keys, wrong_delta

cases = pd.read_csv(f"{R}/common/cases.csv")
q = cases[(cases.dataset == "FF++") & (cases.split == "test") & (cases.protocol == "benign")].iloc[7]
with np.load(q.doctor_trigger_template_path) as rec:
    tmpl_s = rec["template_phase"].astype(np.float64)
    host_s = rec["host_phase"].astype(np.float64)
    delta_s = rec["delta"].astype(np.float64)
    sup_s = rec["support"].astype(bool)
    amp_s = rec["host_amplitude"].astype(np.float64)
    code_s = rec["code"].astype(np.float64)
    key = int(rec["key"][0]); alpha = float(rec["alpha"][0]); beta = float(rec["beta"][0])
deff = abs(np.angle((1-beta) + beta*np.exp(1j*alpha*np.pi)))

def phase_of(pil):
    a = np.asarray(pil, dtype=np.float64)
    lum = 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
    return np.fft.fftshift(np.angle(np.fft.fft2(lum)))

ph_q = phase_of(Image.open(q.input_path).convert("RGB"))
resid = (ph_q - host_s + np.pi) % (2*np.pi) - np.pi
wd = wrong_delta(wrong_keys(key, 1)[0], tmpl_s.shape, alpha, beta, sup_s)

def G(delta):
    w = np.where(sup_s, amp_s, 0.0)
    return float(np.sum(w*delta*np.sin(resid)) / (np.sum(w*delta*delta) + 1e-12))
g_ok, g_wrong = G(delta_s), G(wd)

H0, W0 = tmpl_s.shape
cy, cx = H0//2, W0//2
Z = 110
sl = np.s_[cy-Z:cy+Z, cx-Z:cx+Z]
mask = sup_s[sl]

def masked(a):
    out = np.where(mask, a[sl], np.nan)
    return out

fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.3))
lim = deff * 1.25

im0 = axes[0,0].imshow(masked(code_s), cmap="coolwarm", vmin=-1, vmax=1)
axes[0,0].set_title("(a) key $s$: sign pattern", fontsize=13)

im1 = axes[0,1].imshow(masked(delta_s), cmap="coolwarm", vmin=-lim, vmax=lim)
axes[0,1].set_title(rf"(b) $\delta$: pattern $\times\,{deff:.3f}$ rad $(\alpha,\beta)$", fontsize=13)

im2 = axes[0,2].imshow(masked(tmpl_s), cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[0,2].set_title(r"(c) template $=$ original phase $+\ \delta$", fontsize=13)

im3 = axes[1,0].imshow(masked(resid), cmap="coolwarm", vmin=-lim, vmax=lim)
axes[1,0].set_title(r"(d) residual $r=\phi_J-\widehat\phi_I$", fontsize=13)

agree_ok = np.where(mask, (np.sign(resid[sl]) == np.sign(delta_s[sl])).astype(float), np.nan)
agree_wr = np.where(mask, (np.sign(resid[sl]) == np.sign(wd[sl])).astype(float), np.nan)
im4 = axes[1,1].imshow(agree_ok, cmap="RdYlGn", vmin=0, vmax=1)
axes[1,1].set_title(f"(e) sign agreement, correct key\n{np.nanmean(agree_ok)*100:.0f}%  ($G_s={g_ok:.2f}$)", fontsize=13)
im5 = axes[1,2].imshow(agree_wr, cmap="RdYlGn", vmin=0, vmax=1)
axes[1,2].set_title(f"(f) sign agreement, wrong key\n{np.nanmean(agree_wr)*100:.0f}%  ($G={g_wrong:.2f}$)", fontsize=13)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("0.92")
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.subplots_adjust(wspace=0.12, hspace=0.18, left=0.01, right=0.9, top=0.93, bottom=0.02)
cax = fig.add_axes([0.915, 0.2, 0.016, 0.6])
cb1 = fig.colorbar(im1, cax=cax)
cb1.set_label("radians", fontsize=11); cb1.ax.tick_params(labelsize=9)
fig.savefig(OUT / "fig_keymatch.pdf", dpi=300)
print(f"written; agree ok={np.nanmean(agree_ok)*100:.1f}% wrong={np.nanmean(agree_wr)*100:.1f}% G_ok={g_ok:.3f} G_wr={g_wrong:.3f}")
