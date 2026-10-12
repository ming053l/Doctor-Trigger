#!/usr/bin/env python3
"""Fig 6: complementary evidence for face consistency and release-signal retention.

(a-c) original-reference error 1 - H (log) against the release-normalized key projection
      G_s^(release), test split of FF++, Celeb-DF and DFDC, from the pipeline feature CSV;
(d, e) a wrong projection key read by the original-reference agreement H (unchanged, since H does
      not use the key) and by G_s^(release) (disrupted), recomputed from the records of 250 FF++
      test releases. Nothing is fitted: the scatter shows where the classes overlap in 1 - H.

Paths from config.yaml (doctor_trigger.config): features_csv, cases_csv, outputs_root; the
environment variable DT_FEATURES_CSV, when set, replaces features_csv.
Writes <outputs_root>/figures/fig_signal.{pdf,png}.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from doctor_trigger import config, manifests
from doctor_trigger.key import wrong_keys
from doctor_trigger.metrics import roc_auc
from doctor_trigger.readings import key_projection, phase_drift, release_normalized, spectral_agreement, wrong_key_displacement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase

mpl.rcParams.update({"font.family": "Liberation Serif",
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
GOLD = "#A87400"; BLUE = "#2B6CB0"; RED = "#B63E30"

FEATURES_CSV = os.environ.get("DT_FEATURES_CSV") or config.path("features_csv")
OUT = Path(config.path("outputs_root")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
cols = ["dataset", "split", "protocol", "y_fake", "H", "G_s", "G_wrong_max", "G_margin", "G_z"]
features = pd.read_csv(FEATURES_CSV, usecols=cols)
features = features[features.split == "test"]
ffpp = features[features.dataset == "FF++"]
celebdf = features[features.dataset == "CelebDF"]
dfdc = features[features.dataset == "DFDC"]


# ---------------- panels (d, e): correct versus wrong projection key, from the records ----------------
cases = manifests.read_cases(dataset="FF++", split="test", protocol="benign").head(250)
h_correct, h_wrong, g_correct, g_wrong = [], [], [], []
for case in cases.itertuples():
    record = load_record(case.doctor_trigger_template_path)
    query_phase = luminance_phase(np.asarray(Image.open(case.input_path).convert("RGB")), centered=True)
    if query_phase.shape != record.shape: continue
    original_phase, active, weights = record.original_phase, record.active_bins, record.amplitude_weights
    wrong_delta = wrong_key_displacement(record, wrong_keys(record.key, 1)[0])
    h_correct.append(spectral_agreement(query_phase, original_phase, weights, active))
    h_wrong.append(h_correct[-1])  # H compares with the original phase only, so a wrong key leaves it unchanged
    drift = phase_drift(query_phase, original_phase)
    # release normalization by the query's own correct-key projection, so the correct-key strip sits at 1
    release_g = key_projection(drift, record.displacement, weights, active)
    g_correct.append(release_normalized(key_projection(drift, record.displacement, weights, active), release_g))
    g_wrong.append(release_normalized(key_projection(drift, wrong_delta, weights, active), release_g))
h_correct, h_wrong, g_correct, g_wrong = map(np.array, (h_correct, h_wrong, g_correct, g_wrong))
auc_h = roc_auc([1]*len(h_correct)+[0]*len(h_wrong), np.concatenate([h_correct, h_wrong]))
auc_g = roc_auc([1]*len(g_correct)+[0]*len(g_wrong), np.concatenate([g_correct, g_wrong]))

TS, LS = 21.0, 20.0
FLOOR = 2e-4
fig = plt.figure(figsize=(12.6, 2.75))
# two groups: the two quantities on three corpora, then the two ways of reading the vertical axis
gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.25, 1, 0.12, 1], wspace=0.30,
                      left=0.055, right=0.995, top=0.73, bottom=0.27)
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]),
        fig.add_subplot(gs[4]), fig.add_subplot(gs[6])]

def plane(ax, df, title, sub_n=1100, ylab=True, xlab=True):
    """1 - H against G_s^(release) for up to sub_n random swap, benign and wrong-key rows."""
    benign = df[df.protocol == "benign"]; wrong_key = df[df.protocol == "wrong_key"]
    swap = df[df.protocol == "swap"]
    fs = swap.iloc[rng.choice(len(swap), size=min(sub_n, len(swap)), replace=False)]
    bs = benign.iloc[rng.choice(len(benign), size=min(sub_n, len(benign)), replace=False)]
    ws = wrong_key.iloc[rng.choice(len(wrong_key), size=min(sub_n, len(wrong_key)), replace=False)]
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

plane(axes[0], ffpp, "(a) FF++", xlab=False)
plane(axes[1], celebdf, "(b) Celeb-DF", sub_n=209, ylab=False, xlab=False)
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

# (d, e) the same 250 releases read with the correct and with a wrong projection key
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
strip(ax, h_correct, h_wrong, "(d) Cosine", "Cosine control\n$H$")
lo = min(h_correct.min(), h_wrong.min()); hi = max(h_correct.max(), h_wrong.max()); pad = (hi-lo)*0.25 + 1e-4
ax.set_ylim(lo - pad, hi + pad)

ax = axes[4]
strip(ax, g_correct, g_wrong, "(e) Projection", "Key projection\n$G_s^{(\\mathrm{release})}$")
ax.set_ylim(-0.6, 1.5)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_signal.{ext}", dpi=260, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print(f"saved; before AUC {auc_h:.3f}, after AUC {auc_g:.4f}, cos gap {h_correct.mean()-h_wrong.mean():.5f}")
