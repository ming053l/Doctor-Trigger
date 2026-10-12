#!/usr/bin/env python3
"""Fig 7, drawn from the Table 4 evidence package with no recomputation.

Reads still_queries.csv and still_calibration.csv written by table4_artifact.py, so every point,
cutoff and rate is a per-image score behind a Table 4 row: relation error 1 - H_sub against the
release-normalized key projection.

Paths from config.yaml (doctor_trigger.config): outputs_root. Reads <outputs_root>/table4/,
writes <outputs_root>/figures/fig_roundtrip.{pdf,png}.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root holds doctor_trigger
from doctor_trigger import config
from doctor_trigger.calibration import lower_tail_threshold

mpl.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
GOLD = "#A87400"; RED = "#B63E30"; TEAL = "#0C5E59"; MUTED = "#5E6D82"
ART = Path(config.path("outputs_root")) / "table4"
OUT = Path(config.path("outputs_root")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FLOOR = 2e-4
q = pd.read_csv(ART / "still_queries.csv"); cal = pd.read_csv(ART / "still_calibration.csv")
PANELS = [("lossless", "(a) Lossless"), ("jpeg85", "(b) JPEG $q85$"),
          ("repost", "(c) Social repost"), ("transcode", "(d) H.264 intra")]
sub = q[q.channel.isin([c for c, _ in PANELS])]
lo, hi = np.percentile(sub.G_release_sub, [0.5, 99.5])
YLIM = (min(-0.55, lo - 0.05), max(1.35, hi + 0.05))
print(f"G_release_sub range over panels: p0.5={lo:.3f} p99.5={hi:.3f} max={sub.G_release_sub.max():.3f} -> ylim {YLIM}")

TS, LS = 11.5, 11.0
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.55), sharey=True)
plt.subplots_adjust(left=0.080, right=0.99, top=0.72, bottom=0.29, wspace=0.20)
for i, (ax, (ch, name)) in enumerate(zip(axes.flat, PANELS)):
    g = q[(q.channel == ch) & (q.protocol == "benign")]; f = q[(q.channel == ch) & (q.protocol == "swap")]
    tau = float(lower_tail_threshold(cal[(cal.kind == "cal") & (cal.channel == ch)].H_sub))
    ax.scatter(np.maximum(1 - g.H_sub, FLOOR), g.G_release_sub, s=11, marker="o", color=GOLD, alpha=0.60, linewidths=0, zorder=3)
    ax.scatter(np.maximum(1 - f.H_sub, FLOOR), f.G_release_sub, s=13, marker="x", color=RED, alpha=0.55, linewidths=0.65, zorder=3)
    ax.axvline(max(1 - tau, FLOOR), color=TEAL, lw=1.4)
    ax.set_xscale("log"); ax.set_xlim(1.5e-4, 3.2); ax.set_ylim(*YLIM)
    ax.set_title(name, fontsize=TS)
    ax.set_ylabel("Key projection\n$G_s^{(\\mathrm{release})}$" if i == 0 else "", fontsize=LS, labelpad=1)
    ax.tick_params(labelsize=8.5, length=2.5, pad=2)
    ax.minorticks_off()
    pres = float((g.H_sub >= tau).mean()); flag = float((f.H_sub < tau).mean())
    pct = lambda x: f"{100 * x:.1f}".rstrip("0").rstrip(".")     # same rounding as Table 4
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    print(f"{ch:10s} n={len(g)} tau_H={tau:.4f} flagged={flag:.3f} preserved={pres:.3f}")
fig.supxlabel(r"Original-reference error $1-H_B$ (log scale)", fontsize=LS, y=0.025)
handles = [
    Line2D([], [], color=GOLD, marker="o", linestyle="none", markersize=4, label="Benign"),
    Line2D([], [], color=RED, marker="x", linestyle="none", markersize=4, label="Face swap (SimSwap)"),
    Line2D([], [], color=TEAL, lw=1.4, label=r"Channel cutoff $1-\tau_H$"),
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.12), ncol=3,
           frameon=False, fontsize=11.5, handlelength=1.4, handletextpad=0.45, columnspacing=1.0)
for e in ("pdf", "png"):
    fig.savefig(OUT / f"fig_roundtrip.{e}", dpi=220, bbox_inches="tight", facecolor="white")
print("saved fig_roundtrip from", ART)
