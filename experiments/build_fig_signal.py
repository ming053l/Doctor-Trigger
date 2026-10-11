#!/usr/bin/env python3
"""Fig 6: complementary evidence for face consistency and release-signal retention.

Empirical anatomy of the directly computed measurements, main protocol.

Panels from the pipeline feature CSV (explicit NPZ templates,
PNG-read-back benign queries; test split):
(a-c) original-reference error 1-H (log) against key projection G_s on FF++, Celeb-DF, DFDC
(d,e) the wrong-key test read by the cosine control and by the signed projection (FF++).
Every annotation is computed from the data; the distributions overlap in 1-H,
and the figure shows that rather than a synthetic split line.

Env: DT_CANON_ROOT, DT_OUTPUTS; optional DT_FEATURES_CSV (default: the pipeline feature CSV).
Writes $DT_OUTPUTS/figures/fig_signal.{pdf,png}.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family": "Liberation Serif",
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
INK = "#17202B"; MUTED = "#5E6D82"; HAIRLINE = "#C8CED7"
GOLD = "#A87400"; BLUE = "#2B6CB0"; RED = "#B63E30"; TEAL = "#0C5E59"

import os
R = os.environ["DT_CANON_ROOT"]
F = os.environ.get("DT_FEATURES_CSV", f"{R}/features/calibrated_features.csv")
OUT = Path(os.environ["DT_OUTPUTS"]) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
cols = ["dataset", "split", "protocol", "y_fake", "H", "G_s", "G_wrong_max", "G_margin", "G_z"]
d = pd.read_csv(F, usecols=cols)
d = d[d.split == "test"]
ffpp = d[d.dataset == "FF++"]; cdf = d[d.dataset == "CelebDF"]; dfdc = d[d.dataset == "DFDC"]


# ---------------- wrong-key before/after, computed from the archived NPZ ----------------
EPS = 1e-12
def wrap(v): return (v + np.pi) % (2*np.pi) - np.pi
def code_of(shape, key):
    r = np.random.default_rng(key).standard_normal(shape)
    rows = np.mod(-np.arange(shape[0]), shape[0]); cols = np.mod(-np.arange(shape[1]), shape[1])
    c = np.sign(r - r[np.ix_(rows, cols)])
    c[np.ix_(np.arange(shape[0]) == rows, np.arange(shape[1]) == cols)] = 0.0
    return c
def wrong_key_of(key):
    rng = np.random.default_rng((key ^ 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
    while True:
        cand = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        if cand != key: return cand
def mmean(v, sup, w):
    m = sup & np.isfinite(v); t = float(np.sum(w[m]))
    return float(np.sum(v[m] * w[m]) / (t + EPS))

from PIL import Image
cases = pd.read_csv(f"{R}/common/cases.csv")
cb = cases[(cases.split == "test") & (cases.dataset == "FF++") & (cases.protocol == "benign")].head(250)
def qphase(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), np.float64)
    lum = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    return np.fft.fftshift(np.angle(np.fft.fft2(lum)))
cosC, cosW, gC, gW = [], [], [], []
for r in cb.itertuples():
    with np.load(r.doctor_trigger_template_path) as rec:
        tp = rec["template_phase"].astype(np.float64); hp = rec["host_phase"].astype(np.float64)
        dl = rec["delta"].astype(np.float64); sup = rec["support"].astype(bool)
        w = rec["host_amplitude"].astype(np.float64)
        key = int(rec["key"][0]); alpha = float(rec["alpha"][0]); beta = float(rec["beta"][0])
    ph = qphase(r.input_path)
    if ph.shape != hp.shape: continue
    wk = wrong_key_of(key)
    pre = alpha * np.pi * np.fft.fftshift(code_of(hp.shape, wk)) * sup
    dw = np.angle((1.0 - beta) + beta * np.exp(1j * pre)) * sup
    cosC.append(mmean(np.cos(wrap(ph - hp)), sup, w))
    cosW.append(cosC[-1])  # Paired projection-key test holds the original-phase reference fixed.
    res = wrap(ph - hp)
    tres = wrap(tp - hp)
    # release projection: the released copy's own projection, stored in the record at enrollment
    ref = float(np.sum(w[sup]*dl[sup]*np.sin(res[sup])) / (np.sum(w[sup]*dl[sup]**2) + EPS))
    ref = max(abs(ref), 1e-6)
    gC.append(float(np.sum(w[sup]*dl[sup]*np.sin(res[sup])) / (np.sum(w[sup]*dl[sup]**2) + EPS)) / ref)
    gW.append(float(np.sum(w[sup]*dw[sup]*np.sin(res[sup])) / (np.sum(w[sup]*dw[sup]**2) + EPS)) / ref)
cosC, cosW, gC, gW = map(np.array, (cosC, cosW, gC, gW))
from sklearn.metrics import roc_auc_score
auc_before = roc_auc_score([1]*len(cosC)+[0]*len(cosW), np.concatenate([cosC, cosW]))
auc_after = roc_auc_score([1]*len(gC)+[0]*len(gW), np.concatenate([gC, gW]))

TS, LS = 21.0, 20.0
FLOOR = 2e-4
fig = plt.figure(figsize=(12.6, 2.75))
# two groups: the two quantities on three corpora, then the two ways of reading the vertical axis
gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.25, 1, 0.12, 1], wspace=0.30,
                      left=0.055, right=0.995, top=0.73, bottom=0.27)
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]),
        fig.add_subplot(gs[4]), fig.add_subplot(gs[6])]

def plane(ax, df, title, sub_n=1100, ylab=True, xlab=True):
    ben = df[df.protocol == "benign"]; wk = df[df.protocol == "wrong_key"]
    fk = df[df.protocol == "swap"]
    fs = fk.iloc[rng.choice(len(fk), size=min(sub_n, len(fk)), replace=False)]
    bs = ben.iloc[rng.choice(len(ben), size=min(sub_n, len(ben)), replace=False)]
    ws = wk.iloc[rng.choice(len(wk), size=min(sub_n, len(wk)), replace=False)]
    for g, col, lab, marker in [(fs, RED, "Re-rendered", "x"), (bs, GOLD, "Benign, correct key", "o"),
                        (ws, BLUE, "Benign, wrong key", "^")]:
        ax.scatter(np.maximum(1 - g.H.values, FLOOR), g.G_s, s=7, color=col, marker=marker,
                   alpha=0.30, linewidths=0.5 if marker == "x" else 0, zorder=3, label=lab)
    ax.set_xscale("log"); ax.set_xlim(1.5e-4, 3.2); ax.set_ylim(-1.55, 1.75)
    if xlab: ax.set_xlabel("Original-reference error $1-H$ (log scale)", fontsize=LS)
    ax.set_ylabel("Key projection\n$G_s^{(\\mathrm{release})}$" if ylab else "", fontsize=LS, labelpad=1)
    if not ylab: ax.set_yticklabels([])
    ax.set_title(title, fontsize=TS); ax.tick_params(labelsize=17.0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    return ben, wk, fk

ben, wk, fk = plane(axes[0], ffpp, "(a) FF++", xlab=False)
plane(axes[1], cdf, "(b) Celeb-DF", sub_n=209, ylab=False, xlab=False)
plane(axes[2], dfdc, "(c) DFDC", sub_n=636, ylab=False, xlab=False)
# the three share one x axis, so label it once, centred under the group
_p0 = axes[0].get_position(); _p2 = axes[2].get_position()
fig.text((_p0.x0 + _p2.x1) / 2.0, 0.035, "Original-reference error $1-H$ (log scale)",
         ha="center", va="bottom", fontsize=LS)
handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.10),
                 fontsize=19.0, ncol=3, markerscale=4.0, frameon=False,
                 handletextpad=0.35, columnspacing=2.0)
for lh in leg.legend_handles:
    lh.set_alpha(1.0)
    lh.set_linewidth(1.6)

# (b)/(c) wrong-key verification, before and after the signed projection
def strip(ax, va, vb, title, ylab):
    ja = np.random.default_rng(1).normal(0, 0.05, len(va))
    jb = np.random.default_rng(2).normal(0, 0.05, len(vb))
    ax.scatter(ja, va, s=6, color=GOLD, alpha=0.4, linewidths=0)
    ax.scatter(1 + jb, vb, s=7, marker="^", color=BLUE, alpha=0.4, linewidths=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["correct\nkey", "wrong\nkey"], fontsize=18.0)
    ax.set_xlim(-0.5, 1.5); ax.set_title(title, fontsize=TS)
    ax.set_ylabel(ylab, fontsize=LS, labelpad=1); ax.tick_params(labelsize=17.0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

ax = axes[3]
strip(ax, cosC, cosW, "(d) Cosine", "Cosine control\n$H$")
lo = min(cosC.min(), cosW.min()); hi = max(cosC.max(), cosW.max()); pad = (hi-lo)*0.25 + 1e-4
ax.set_ylim(lo - pad, hi + pad)

ax = axes[4]
strip(ax, gC, gW, "(e) Projection", "Key projection\n$G_s^{(\\mathrm{release})}$")
ax.set_ylim(-0.6, 1.5)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_signal.{ext}", dpi=260, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print(f"saved; before AUC {auc_before:.3f}, after AUC {auc_after:.4f}, cos gap {cosC.mean()-cosW.mean():.5f}")
