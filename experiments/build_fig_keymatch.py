"""Appendix Fig 10: keyed displacement and query residual on one FF++ test asset.

What the key writes and the verifier reads: (a) the key's sign pattern, (b) the displacement,
(c) the stored template, (d) the query residual against the original phase, and the sign agreement
of that residual with the correct key (e) and with a wrong key (f), central region of the shifted
spectrum, bins outside the active support in gray.

Paths from config.yaml (doctor_trigger.config): cases_csv, outputs_root.
Writes <outputs_root>/figures/fig_keymatch.pdf.
"""
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
mpl.rcParams.update({"font.family": "Liberation Serif",
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
from doctor_trigger import config, manifests
from doctor_trigger.key import displacement, wrong_keys
from doctor_trigger.readings import key_projection, phase_drift, wrong_key_displacement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase

OUT = Path(config.path("outputs_root")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)

case = manifests.read_cases(dataset="FF++", split="test", protocol="benign").iloc[7]
record = load_record(case.doctor_trigger_template_path)
template, delta = record.template_phase, record.displacement
active, weights = record.active_bins, record.amplitude_weights
# |delta_eff|: the displacement of one active bin with h_s = +1 (Eq. 3)
delta_eff = abs(float(displacement(np.float64(1.0), record.alpha, record.beta, True)))

query_phase = luminance_phase(np.asarray(Image.open(case.input_path).convert("RGB")), centered=True)
drift = phase_drift(query_phase, record.original_phase)
wrong_delta = wrong_key_displacement(record, wrong_keys(record.key, 1)[0])


def projection(displacement_map):
    """G_s (Eq. 6) on the active bins."""
    return key_projection(drift, displacement_map, weights, active)
g_correct, g_wrong = projection(delta), projection(wrong_delta)

# central 220 x 220 bins of the shifted spectrum
height, width = template.shape
cy, cx = height//2, width//2
Z = 110
sl = np.s_[cy-Z:cy+Z, cx-Z:cx+Z]
mask = active[sl]

def masked(a):
    return np.where(mask, a[sl], np.nan)

fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.3))
lim = delta_eff * 1.25

im0 = axes[0,0].imshow(masked(record.sign_pattern), cmap="coolwarm", vmin=-1, vmax=1)
axes[0,0].set_title("(a) key $s$: sign pattern", fontsize=13)

im1 = axes[0,1].imshow(masked(delta), cmap="coolwarm", vmin=-lim, vmax=lim)
axes[0,1].set_title(rf"(b) $\delta$: pattern $\times\,{delta_eff:.3f}$ rad $(\alpha,\beta)$", fontsize=13)

im2 = axes[0,2].imshow(masked(template), cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[0,2].set_title(r"(c) template $=$ original phase $+\ \delta$", fontsize=13)

im3 = axes[1,0].imshow(masked(drift), cmap="coolwarm", vmin=-lim, vmax=lim)
axes[1,0].set_title(r"(d) residual $r=\phi_J-\widehat\phi_I$", fontsize=13)

agree_correct = np.where(mask, (np.sign(drift[sl]) == np.sign(delta[sl])).astype(float), np.nan)
agree_wrong = np.where(mask, (np.sign(drift[sl]) == np.sign(wrong_delta[sl])).astype(float), np.nan)
im4 = axes[1,1].imshow(agree_correct, cmap="RdYlGn", vmin=0, vmax=1)
axes[1,1].set_title(f"(e) sign agreement, correct key\n{np.nanmean(agree_correct)*100:.0f}%  ($G_s={g_correct:.2f}$)", fontsize=13)
im5 = axes[1,2].imshow(agree_wrong, cmap="RdYlGn", vmin=0, vmax=1)
axes[1,2].set_title(f"(f) sign agreement, wrong key\n{np.nanmean(agree_wrong)*100:.0f}%  ($G={g_wrong:.2f}$)", fontsize=13)

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
print(f"written; agree ok={np.nanmean(agree_correct)*100:.1f}% wrong={np.nanmean(agree_wrong)*100:.1f}% "
      f"G_ok={g_correct:.3f} G_wr={g_wrong:.3f}")
