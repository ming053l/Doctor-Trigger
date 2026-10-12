#!/usr/bin/env python3
"""Fig 6: complementary evidence for face consistency and release-signal retention.

Empirical anatomy of the directly computed measurements, main protocol.

Panels from the pipeline feature CSV (explicit NPZ templates,
PNG-read-back benign queries; test split):
(a-c) original-reference error 1-H (log) against key projection G_s on FF++, Celeb-DF, DFDC
(d,e) the wrong-key test read by the cosine control and by the signed projection (FF++).
Every annotation is computed from the data; the distributions overlap in 1-H,
and the figure shows that rather than a synthetic split line.

Paths from config.yaml (doctor_trigger.config): features_csv, cases_csv, outputs_root; the
environment variable DT_FEATURES_CSV, when set, replaces features_csv.
Writes <outputs_root>/figures/fig_signal.{pdf,png}.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root holds doctor_trigger
from doctor_trigger import config, manifests
from doctor_trigger.key import wrong_keys
from doctor_trigger.metrics import roc_auc
from doctor_trigger.readings import (SumForm, phase_drift, release_normalized, spectral_agreement,
                                     wrong_key_displacement)
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase

mpl.rcParams.update({"font.family": "Liberation Serif",
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
INK = "#17202B"; MUTED = "#5E6D82"; HAIRLINE = "#C8CED7"
GOLD = "#A87400"; BLUE = "#2B6CB0"; RED = "#B63E30"; TEAL = "#0C5E59"

F = os.environ.get("DT_FEATURES_CSV") or config.path("features_csv")
OUT = Path(config.path("outputs_root")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
cols = ["dataset", "split", "protocol", "y_fake", "H", "G_s", "G_wrong_max", "G_margin", "G_z"]
d = pd.read_csv(F, usecols=cols)
d = d[d.split == "test"]
ffpp = d[d.dataset == "FF++"]; cdf = d[d.dataset == "CelebDF"]; dfdc = d[d.dataset == "DFDC"]


# ---------------- wrong-key before/after, computed from the archived NPZ ----------------
EPS = 1e-12


def projection(drift, delta, weights, bins):
    """G_s (Eq. 6) on the gathered bins, with the denominator written sum w (delta^2) as this figure
    always was; key_projection's sum (w delta) delta differs from it in the last bits."""
    w, dl = weights[bins], delta[bins]
    return float(np.sum(w * dl * np.sin(drift[bins])) / (np.sum(w * dl ** 2) + EPS))


cb = manifests.read_cases(dataset="FF++", split="test", protocol="benign").head(250)
cosC, cosW, gC, gW = [], [], [], []
for r in cb.itertuples():
    record = load_record(r.doctor_trigger_template_path)
    ph = luminance_phase(np.asarray(Image.open(r.input_path).convert("RGB")), centered=True)
    if ph.shape != record.shape: continue
    hp, sup, w = record.original_phase, record.active_bins, record.amplitude_weights
    dw = wrong_key_displacement(record, wrong_keys(record.key, 1)[0])
    cosC.append(spectral_agreement(ph, hp, w, sup, stabilizer=EPS))
    cosW.append(cosC[-1])  # Paired projection-key test holds the original-phase reference fixed.
    res = phase_drift(ph, hp)
    # normalizer: the query's own correct-key projection, so the correct-key strip sits at +-1
    ref = projection(res, record.displacement, w, sup)
    gC.append(release_normalized(projection(res, record.displacement, w, sup), ref))
    gW.append(release_normalized(projection(res, dw, w, sup), ref))
cosC, cosW, gC, gW = map(np.array, (cosC, cosW, gC, gW))
auc_before = roc_auc([1]*len(cosC)+[0]*len(cosW), np.concatenate([cosC, cosW]))
auc_after = roc_auc([1]*len(gC)+[0]*len(gW), np.concatenate([gC, gW]))

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
