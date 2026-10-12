"""Appendix Fig 10: keyed displacement and query residual on one FF++ test asset.

What the key writes and the verifier reads: (a) the key's sign pattern, (b) the displacement,
(c) the stored template, (d) the query residual against the original phase, and the sign agreement
of that residual with the correct key (e) and with a wrong key (f), central region of the shifted
spectrum, bins outside the active support in gray.

Paths from config.yaml (doctor_trigger.config): cases_csv, outputs_root.
Writes <outputs_root>/figures/fig_keymatch.pdf.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
mpl.rcParams.update({"font.family": "Liberation Serif",
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root holds doctor_trigger
from doctor_trigger import config, manifests
from doctor_trigger.key import displacement, wrong_keys
from doctor_trigger.readings import SumForm, key_projection, phase_drift, wrong_key_displacement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase

OUT = Path(config.path("outputs_root")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)

q = manifests.read_cases(dataset="FF++", split="test", protocol="benign").iloc[7]
record = load_record(q.doctor_trigger_template_path)
tmpl_s, host_s, delta_s = record.template_phase, record.original_phase, record.displacement
sup_s, amp_s, code_s = record.active_bins, record.amplitude_weights, record.sign_pattern
# delta_eff: the displacement of one active bin with h_s = +1 (Eq. 3)
deff = abs(float(displacement(np.float64(1.0), record.alpha, record.beta, True)))

ph_q = luminance_phase(np.asarray(Image.open(q.input_path).convert("RGB")), centered=True)
resid = phase_drift(ph_q, host_s)
wd = wrong_key_displacement(record, wrong_keys(record.key, 1)[0])


def G(delta):
    """G_s (Eq. 6) on the active bins, whole-grid weight map, eps_0 = 1e-12."""
    return key_projection(resid, delta, amp_s, sup_s, form=SumForm.WEIGHT_MAP)
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
