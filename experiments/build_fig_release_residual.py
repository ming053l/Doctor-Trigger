"""Appendix Fig 8: release residuals on four DFDC test subjects.

Top row the released image, bottom row |released - original| averaged over RGB, each panel scaled
to its own 99.5th percentile for visibility, with the panel's maximum in 1/255 units.

Paths from config.yaml (doctor_trigger.config): cases_csv, features_csv, outputs_root.
Writes <outputs_root>/figures/fig_supp_residual.pdf.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from doctor_trigger import config, manifests

mpl.rcParams.update({"font.family": "Liberation Serif",
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
OUT = Path(config.path("outputs_root")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# The cases are joined with the feature rows (a benign query has a benign and a wrong-key row), so
# each benign query appears twice; the fixed row picks below index that joined table.
feature_rows = pd.read_csv(config.path("features_csv"), usecols=["query_id"])
test = manifests.read_cases(split="test").merge(feature_rows, on="query_id")


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))

benign = test[(test.dataset == "DFDC") & (test.protocol == "benign")].sort_values("query_id")
picks = benign.iloc[[3, 11, 19, 27]]

fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.1))
for c, (_, case) in enumerate(picks.iterrows()):
    original = rgb(case.enrollment_input_path).astype(np.float64)
    released = rgb(case.doctor_trigger_released_path).astype(np.float64)
    res = np.abs(released - original).mean(axis=2)
    vmax = max(np.percentile(res, 99.5), 1e-3)
    axes[0, c].imshow(released.astype(np.uint8))
    im = axes[1, c].imshow(res, cmap="magma", vmin=0, vmax=vmax)
    axes[1, c].set_xlabel(f"max {res.max():.1f}/255", fontsize=9.5)
    for r in (0, 1):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        for sp in axes[r, c].spines.values():
            sp.set_visible(False)
axes[0, 0].set_ylabel("released", fontsize=12.5)
axes[1, 0].set_ylabel(r"$|$released $-$ original$|$ (panel-scaled)", fontsize=10.5)
fig.subplots_adjust(wspace=0.03, hspace=0.1, left=0.05, right=0.995, top=0.995, bottom=0.055)
fig.savefig(OUT / "fig_supp_residual.pdf", dpi=300)
plt.close(fig)
print("written", OUT / "fig_supp_residual.pdf")
