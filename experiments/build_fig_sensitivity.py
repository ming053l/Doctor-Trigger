"""Appendix Figs 13 and 14 and the ring rows of Table 15: enrollment strength, ring location and
wrong-key count.

Fig 13 (2 x 4): strength alpha (top, x = |delta_eff|) and ring [kappa_min, kappa_max] (bottom)
against PSNR, the key-projection gap between releases and originals (tau_G minus the largest G_s
of an original), relation TPR after JPEG q90 at 1% benign rejection, and correct-vs-wrong-key AUC.
Fig 14: key AUC, the rate at which the correct key beats all m wrong keys, and scoring cost, for
m = 1..32. Fig 13 skips settings with fewer than 50 rows.

Paths from config.yaml (doctor_trigger.config): outputs_root. Reads the sweep CSVs in
<outputs_root>/sensitivity (sensitivity_release.py), writes <outputs_root>/figures/fig_sensitivity.pdf
and fig_sensitivity_m.pdf, and prints the Table 15 rows.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from doctor_trigger import config
from doctor_trigger.calibration import lower_tail_threshold
from doctor_trigger.key import displacement

mpl.rcParams.update({"font.family": "Liberation Serif",
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
SWEEP_DIR = config.path("outputs_root") + "/sensitivity"
OUT = config.path("outputs_root") + "/figures"
Path(OUT).mkdir(parents=True, exist_ok=True)
BETA = 0.15
ALPHAS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]
BANDS = ["0.05-0.45", "0.10-0.45", "0.15-0.45", "0.20-0.45",
         "0.10-0.30", "0.10-0.35", "0.10-0.40", "0.10-0.50"]
DATASETS = ["FF++", "DFDC", "CelebDF"]
WRONG_KEY_MS = 6.3   # scoring time of one wrong key (ms), not re-measured here; Fig 14 (right) plots m x this
COLORS = {"FF++": "#2166ac", "DFDC": "#b2182b", "CelebDF": "#1a7837"}
MARKS = {"FF++": "o", "DFDC": "s", "CelebDF": "^"}

def delta_eff(alpha):
    """|delta_eff|: the displacement of one active bin with h_s = +1 at (alpha, BETA) (Eq. 3)."""
    return abs(float(displacement(np.float64(1.0), alpha, BETA, True)))

def tag(spec):
    """File-name form of a sweep setting, e.g. "alpha:0.05" -> "alpha_0p05"."""
    return spec.replace(":", "_").replace(".", "p").replace("-", "_")

def metrics(d):
    """Fig 13 columns for one setting; JPEG-q90 readings are the *_j90 columns, Gw0..Gw7 the
    projections of eight wrong keys."""
    tau_g = lower_tail_threshold(d.G_ben_j90)
    margin = float(tau_g - d.G_orig.max())
    tau_h = np.quantile(1 - d.H_ben_j90, 0.99)
    tpr = float(((1 - d.H_swap_j90) >= tau_h).mean())
    pos = d.G_ben_j90.values
    neg = np.concatenate([d[f"Gw{j}"].values for j in range(8)])
    y = np.r_[np.ones_like(pos), np.zeros_like(neg)]
    key_auc = roc_auc_score(y, np.r_[pos, neg])
    return {"psnr": d.psnr.mean(), "release_gap": margin, "tpr": tpr, "key": key_auc}

def load(ds, spec):
    file = Path(f"{SWEEP_DIR}/{ds}_{tag(spec)}.csv")
    return pd.read_csv(file) if file.exists() else None

fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.72))   # drawn at the text width: no scaling on include
COLTITLES = ["PSNR (dB) $\\uparrow$", "post-hoc distance $\\uparrow$",
             "JPEG-q90 TPR@1% FPR $\\uparrow$", "key AUC under JPEG-q90 $\\uparrow$"]
KEYS = ["psnr", "release_gap", "tpr", "key"]

# row 1: alpha sweep, x = effective displacement
for ds in DATASETS:
    xs, ms = [], []
    for a in ALPHAS:
        d = load(ds, f"alpha:{a}")
        if d is None or len(d) < 50: continue
        xs.append(delta_eff(a)); ms.append(metrics(d))
    for c, k in enumerate(KEYS):
        axes[0, c].plot(xs, [m[k] for m in ms], marker=MARKS[ds], ms=4.5,
                        lw=1.6, color=COLORS[ds], label=ds if c == 0 else None)
for c in range(4):
    ax = axes[0, c]
    ax.set_xscale("log")
    ax.axvline(delta_eff(0.015), color="0.45", ls="--", lw=0.9)
    ax.axvline(delta_eff(0.05), color="0.45", ls=":", lw=1.1)
    ax.set_title(COLTITLES[c], fontsize=7.5)
    ax.set_xlabel(r"$|\delta_{\rm eff}|$ (rad)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.5)
axes[0, 0].annotate("DFDC\ndeployed", xy=(delta_eff(0.015), 0.02), xycoords=("data", "axes fraction"),
                    fontsize=5.5, color="0.35", ha="right")
axes[0, 0].annotate("FF++/CDF\ndeployed", xy=(delta_eff(0.05), 0.02), xycoords=("data", "axes fraction"),
                    fontsize=5.5, color="0.35", ha="left")

# row 2: ring sweep, categorical x
xt = np.arange(len(BANDS))
for ds in DATASETS:
    ms = []
    for b in BANDS:
        d = load(ds, f"band:{b}")
        ms.append(metrics(d) if d is not None and len(d) >= 50 else None)
    for c, k in enumerate(KEYS):
        ys = [m[k] if m else np.nan for m in ms]
        axes[1, c].plot(xt, ys, marker=MARKS[ds], ms=4.5, lw=1.6, color=COLORS[ds])
for c in range(4):
    ax = axes[1, c]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"[{b.replace('-', ',')}]".replace("0.", ".") for b in BANDS],
                       rotation=45, fontsize=5.5, ha="right")
    ax.axvline(1, color="0.45", ls=":", lw=1.1)
    ax.set_xlabel(r"ring $[\kappa_{\min},\kappa_{\max}]$", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(alpha=0.25, lw=0.5)
axes[1, 0].annotate("deployed", xy=(1, 0.03), xycoords=("data", "axes fraction"),
                    fontsize=5.5, color="0.35", ha="left")
for r in (0, 1):
    axes[r, 1].axhline(0, color="0.3", lw=0.8)
    axes[r, 2].set_ylim(-0.04, 1.04)
    axes[r, 3].set_ylim(0.48, 1.02)
fig.legend(loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.015))
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(f"{OUT}/fig_sensitivity.pdf")
plt.close(fig)

# ---- Fig 14: wrong-key count m, at the deployed strength ----
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75))   # text width, no scaling on include
MS = [1, 2, 4, 8, 16, 32]
for ds in DATASETS:
    spec = "alpha:0.015" if ds == "DFDC" else "alpha:0.05"
    d = load(ds, spec)
    if d is None: continue
    aucs, margins = [], []
    for m in MS:
        neg = np.concatenate([d[f"Gw{j}"].values for j in range(m)])
        pos = d.G_ben.values
        y = np.r_[np.ones_like(pos), np.zeros_like(neg)]
        aucs.append(roc_auc_score(y, np.r_[pos, neg]))
        wmax = np.max(np.stack([d[f"Gw{j}"].values for j in range(m)]), axis=0)
        margins.append(float((d.G_ben.values > wmax).mean()))
    axes[0].plot(MS, aucs, marker=MARKS[ds], color=COLORS[ds], lw=1.6, ms=4.5, label=ds)
    axes[1].plot(MS, margins, marker=MARKS[ds], color=COLORS[ds], lw=1.6, ms=4.5)
axes[2].plot(MS, [WRONG_KEY_MS * m for m in MS], marker="o", color="0.25", lw=1.6, ms=4.5)
for ax, t in zip(axes, ["key AUC $\\uparrow$", "correct$>$all wrong keys (rate)", "wrong-key scoring (ms)"]):
    ax.set_xscale("log", base=2)
    ax.set_xticks(MS); ax.set_xticklabels(MS)
    ax.set_xlabel("$m$ wrong keys", fontsize=7.5)
    ax.set_title(t, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.5)
    ax.axvline(8, color="0.45", ls=":", lw=1.0)
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_sensitivity_m.pdf")
plt.close(fig)

# ---- Table 15 ring rows (FF++) ----
rows = []
for b in BANDS:
    d = load("FF++", f"band:{b}")
    if d is None: continue
    m = metrics(d)
    clean_auc = roc_auc_score(np.r_[np.zeros(len(d)), np.ones(len(d))], np.r_[1 - d.H_ben, 1 - d.H_swap])
    j90_auc = roc_auc_score(np.r_[np.zeros(len(d)), np.ones(len(d))], np.r_[1 - d.H_ben_j90, 1 - d.H_swap_j90])
    blab = b.replace("-", ",\\,")
    rows.append(f"$[{blab}]$ & {m['psnr']:.1f} & {clean_auc:.3f} & {j90_auc:.3f} & {m['release_gap']:.2f} & {m['key']:.3f} \\\\")
print("\n".join(rows))
print("figures written")
