#!/usr/bin/env python3
"""Appendix Fig 12: hyperparameter sensitivity beyond release strength: FRR target,
calibration-set size, and the record-model key-loss weight.

Draws the JSON written by the hyperparam_sensitivity entry.
Paths from config.yaml (doctor_trigger.config): outputs_root. The environment variable
DT_HYPER_JSON, when set, names the sweep JSON (default
<outputs_root>/recordscalar_hyper/hyper_sensitivity.json).
Writes <outputs_root>/figures/fig_hyper.{pdf,png}.
"""
from pathlib import Path
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "Liberation Serif",
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
INK = "#17202B"; MUTED = "#5E6D82"; GOLD = "#A87400"; BLUE = "#2B6CB0"; TEAL = "#0C5E59"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root holds doctor_trigger
from doctor_trigger import config

OUTPUTS = Path(config.path("outputs_root"))
OUT = OUTPUTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)
with open(os.environ.get("DT_HYPER_JSON", OUTPUTS / "recordscalar_hyper" / "hyper_sensitivity.json")) as handle:
    d = json.load(handle)
TS, LS, TK = 15.0, 13.0, 11.5
fig, axes = plt.subplots(1, 3, figsize=(8.9, 2.55))
plt.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.24, wspace=0.42)

ax = axes[0]
t = [r["target"] for r in d["frr_target"]]; rf = [r["realized_fpr"] for r in d["frr_target"]]
tp = [r["tpr"] for r in d["frr_target"]]
ax.plot([1e-3, 5e-2], [1e-3, 5e-2], ls=(0, (3, 3)), color=MUTED, lw=1.0, label="target = realized")
ax.plot(t, rf, "o-", color=TEAL, lw=1.6, ms=5, label="realized FPR")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("FRR target $t$", fontsize=LS); ax.set_ylabel("realized test FPR", fontsize=LS)
ax.set_title(f"(a) threshold target\n(TPR ${min(tp):.3f}$ at every $t$)", fontsize=TS)
ax.legend(fontsize=10.5, frameon=False, loc="upper left", handletextpad=0.4)
ax.tick_params(labelsize=TK)

ax = axes[1]
n = [r["n"] for r in d["cal_size"]]
ax.fill_between(n, [r["q10"] for r in d["cal_size"]], [r["q90"] for r in d["cal_size"]],
                color=TEAL, alpha=0.15, lw=0, label="10–90%")
ax.fill_between(n, [r["q25"] for r in d["cal_size"]], [r["q75"] for r in d["cal_size"]],
                color=TEAL, alpha=0.35, lw=0, label="25–75%")
ax.plot(n, [r["med"] for r in d["cal_size"]], "o-", color=TEAL, lw=1.6, ms=5, label="median")
ax.axhline(0.01, color=MUTED, ls=(0, (3, 3)), lw=1.0)
ax.text(390, 0.0112, "target $1\\%$", fontsize=10.5, color=MUTED, ha="right")
ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([str(v) for v in n])
ax.set_xlabel("calibration samples $n$", fontsize=LS); ax.set_ylabel("realized test FPR", fontsize=LS)
ax.set_title("(b) calibration-set size", fontsize=TS)
ax.legend(fontsize=10.5, frameon=False, loc="upper right", handletextpad=0.4)
ax.tick_params(labelsize=TK)

ax = axes[2]
import collections
agg = collections.defaultdict(list)
for r in d["lambda_key"]: agg[(r["lam"], r["eval"])].append((r["rel_auc"], r["key_auc"]))
lams = sorted({k[0] for k in agg})
x = [0.03 if l == 0 else l for l in lams]
for ev, ls_, mark in [("FF++", "-", "o"), ("CelebDF", "--", "s")]:
    ka = [np.mean([v[1] for v in agg[(l, ev)]]) for l in lams]
    ax.plot(x, ka, ls_, marker=mark, color=GOLD, lw=1.6, ms=5, label=f"$S_{{\\rm key}}$ {ev.replace('CelebDF', 'Celeb-DF')}")
ra = [np.mean([v[0] for v in agg[(l, "CelebDF")]]) for l in lams]
ax.plot(x, ra, ":", marker="^", color=BLUE, lw=1.6, ms=5, label="$S_{\\rm rel}$ Celeb-DF")
ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(["0", "0.1", "0.3", "1", "3"])
ax.set_ylim(0.52, 1.03)
ax.axvline(0.3, color=MUTED, ls=(0, (2, 2)), lw=0.9)
ax.set_xlabel("key-loss weight $\\lambda_{\\rm key}$", fontsize=LS); ax.set_ylabel("AUC", fontsize=LS)
ax.set_title("(c) record-model loss weight", fontsize=TS)
ax.legend(fontsize=10.0, frameon=False, loc="center right", handletextpad=0.4)
ax.tick_params(labelsize=TK)
for a in axes:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
for e in ("pdf", "png"):
    fig.savefig(OUT / f"fig_hyper.{e}", dpi=240, bbox_inches="tight", facecolor="white")
print("saved fig_hyper")
