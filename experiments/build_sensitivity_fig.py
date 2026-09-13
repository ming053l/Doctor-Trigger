"""Build the 2x4 sensitivity figure, the m-sweep appendix figure, and the ring table."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest
import glob
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

mpl.rcParams.update({"font.family": "Liberation Serif",
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
SENS = _dt("DT_OUTPUTS", "/sensitivity")
OUT = _dt("DT_OUTPUTS", "/figures")
import os as _os; _os.makedirs(OUT, exist_ok=True)
BETA = 0.15
ALPHAS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]
BANDS = ["0.05-0.45", "0.10-0.45", "0.15-0.45", "0.20-0.45",
         "0.10-0.30", "0.10-0.35", "0.10-0.40", "0.10-0.50"]
DSETS = ["FF++", "DFDC", "CelebDF"]
COLORS = {"FF++": "#2166ac", "DFDC": "#b2182b", "CelebDF": "#1a7837"}
MARKS = {"FF++": "o", "DFDC": "s", "CelebDF": "^"}

def deff(alpha):
    return abs(np.angle((1 - BETA) + BETA * np.exp(1j * alpha * np.pi)))

def tag(spec):
    return spec.replace(":", "_").replace(".", "p").replace("-", "_")

def metrics(d):
    tau_g = np.quantile(d.G_ben_j90, 0.01)
    margin = float(tau_g - d.G_orig.max())
    tau_h = np.quantile(1 - d.H_ben_j90, 0.99)
    tpr = float(((1 - d.H_swap_j90) >= tau_h).mean())
    pos = d.G_ben_j90.values
    neg = np.concatenate([d[f"Gw{j}"].values for j in range(8)])
    y = np.r_[np.ones_like(pos), np.zeros_like(neg)]
    key_auc = roc_auc_score(y, np.r_[pos, neg])
    return {"psnr": d.psnr.mean(), "asr": margin, "tpr": tpr, "key": key_auc}

def load(ds, spec):
    f = f"{SENS}/{ds}_{tag(spec)}.csv"
    return pd.read_csv(f) if glob.glob(f) else None

fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.72))   # drawn at the text width: no scaling on include
COLTITLES = ["PSNR (dB) $\\uparrow$", "post-hoc distance $\\uparrow$",
             "JPEG-q90 TPR@1% FPR $\\uparrow$", "key AUC under JPEG-q90 $\\uparrow$"]
KEYS = ["psnr", "asr", "tpr", "key"]

# row 1: alpha sweep, x = effective displacement
for ds in DSETS:
    xs, ms = [], []
    for a in ALPHAS:
        d = load(ds, f"alpha:{a}")
        if d is None or len(d) < 50: continue
        xs.append(deff(a)); ms.append(metrics(d))
    for c, k in enumerate(KEYS):
        axes[0, c].plot(xs, [m[k] for m in ms], marker=MARKS[ds], ms=4.5,
                        lw=1.6, color=COLORS[ds], label=ds if c == 0 else None)
for c in range(4):
    ax = axes[0, c]
    ax.set_xscale("log")
    ax.axvline(deff(0.015), color="0.45", ls="--", lw=0.9)
    ax.axvline(deff(0.05), color="0.45", ls=":", lw=1.1)
    ax.set_title(COLTITLES[c], fontsize=7.5)
    ax.set_xlabel(r"$|\delta_{\rm eff}|$ (rad)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.5)
axes[0, 0].annotate("DFDC\ndeployed", xy=(deff(0.015), 0.02), xycoords=("data", "axes fraction"),
                    fontsize=5.5, color="0.35", ha="right")
axes[0, 0].annotate("FF++/CDF\ndeployed", xy=(deff(0.05), 0.02), xycoords=("data", "axes fraction"),
                    fontsize=5.5, color="0.35", ha="left")

# row 2: ring sweep, categorical x
xt = np.arange(len(BANDS))
for ds in DSETS:
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
    ax.set_xticklabels([f"[{b.split('-')[0][1:]},{b.split('-')[1][1:]}]".replace("0.", ".")
                        if False else f"[{b.replace('-', ',')}]".replace("0.", ".") for b in BANDS],
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

# ---- m-sweep appendix figure ----
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75))   # text width, no scaling on include
MS = [1, 2, 4, 8, 16, 32]
for ds in DSETS:
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
axes[2].plot(MS, [6.3 * m for m in MS], marker="o", color="0.25", lw=1.6, ms=4.5)
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

# ---- ring table (FF++) ----
rows = []
for b in BANDS:
    d = load("FF++", f"band:{b}")
    if d is None: continue
    m = metrics(d)
    clean_auc = roc_auc_score(np.r_[np.zeros(len(d)), np.ones(len(d))], np.r_[1 - d.H_ben, 1 - d.H_swap])
    j90_auc = roc_auc_score(np.r_[np.zeros(len(d)), np.ones(len(d))], np.r_[1 - d.H_ben_j90, 1 - d.H_swap_j90])
    blab = b.replace("-", ",\\,")
    rows.append(f"$[{blab}]$ & {m['psnr']:.1f} & {clean_auc:.3f} & {j90_auc:.3f} & {m['asr']:.2f} & {m['key']:.3f} \\\\")
print("\n".join(rows))
print("figures written")
