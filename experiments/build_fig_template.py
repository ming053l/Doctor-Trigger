"""Appendix Fig 11: enrolled template and asset specificity (24 FF++ test assets).

(a) the template T_I on the active bins of the mid-band ring;
(b) unit-amplitude (phase-only) inversion of T_I alone, which shows facial structure and is why the
    record must stay private;
(c) original-reference agreement H of query i against record j: the enrolled pairs lie on the
    diagonal, mismatched pairs fall off it.

Paths from config.yaml (doctor_trigger.config): cases_csv, outputs_root.
Writes <outputs_root>/figures/fig_template.{pdf,png} and prints the matched / mismatched H.
"""
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
from PIL import Image

from doctor_trigger import config, manifests
from doctor_trigger.readings import SumForm, spectral_agreement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase

OUT = Path(config.path("outputs_root")) / "figures"; OUT.mkdir(parents=True, exist_ok=True)
N, EPS = 24, 1e-12
cases = manifests.read_cases(dataset="FF++", split="test", protocol="benign").head(N)

records, query_phases = [], []
for case in cases.itertuples():
    records.append(load_record(case.doctor_trigger_template_path))
    query_phases.append(luminance_phase(np.asarray(Image.open(case.input_path).convert("RGB")), centered=True))

# H of query i against record j (Eq. 5 with the original phase as reference), whole-grid weight
# map with stabilizer 1e-12; a size mismatch leaves the cell empty
agreement = np.full((N, N), np.nan)
for i, query_phase in enumerate(query_phases):
    for j, record in enumerate(records):
        if query_phase.shape != record.shape: continue
        agreement[i, j] = spectral_agreement(query_phase, record.original_phase, record.amplitude_weights,
                                             record.active_bins, stabilizer=EPS, form=SumForm.WEIGHT_MAP)
matched = np.diag(agreement)
mismatched = agreement[~np.eye(N, dtype=bool)]; mismatched = mismatched[np.isfinite(mismatched)]

template, active = records[0].template_phase, records[0].active_bins
TS, LS = 12.5, 10.5
fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.9))
plt.subplots_adjust(left=0.02, right=0.965, top=0.86, bottom=0.10, wspace=0.20)

im0 = ax[0].imshow(np.where(active, template, np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
ax[0].set_title("(a) filed template $\\mathcal{T}_I$\non the mid-band ring", fontsize=TS)
ax[0].set_xticks([]); ax[0].set_yticks([])
cb = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03, ticks=[-np.pi, 0, np.pi])
cb.ax.set_yticklabels(["$-\\pi$", "$0$", "$\\pi$"], fontsize=LS); cb.outline.set_visible(False)

unit_spectrum = np.zeros(template.shape, complex); unit_spectrum[active] = np.exp(1j * template[active])
recon = np.real(np.fft.ifft2(np.fft.ifftshift(unit_spectrum)))
rlim = np.percentile(np.abs(recon - recon.mean()), 99.3)
ax[1].imshow(np.clip(recon - recon.mean(), -rlim, rlim), cmap="Greys_r", interpolation="nearest")
ax[1].set_title("(b) phase-only reconstruction\nfrom $\\mathcal{T}_I$ alone", fontsize=TS)
ax[1].set_xticks([]); ax[1].set_yticks([])
ax[1].text(0.5, -0.075, "carries the enrolled face's own structure", transform=ax[1].transAxes,
           ha="center", fontsize=LS, color="#B63E30")

im2 = ax[2].imshow(agreement, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
ax[2].set_title("(c) query $i$ against record $j$\nrelation $H$", fontsize=TS)
ax[2].set_xlabel("record index $j$", fontsize=LS); ax[2].set_ylabel("query index $i$", fontsize=LS)
ax[2].tick_params(labelsize=9)
cb2 = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03, ticks=[0, 0.5, 1])
cb2.ax.tick_params(labelsize=LS); cb2.outline.set_visible(False)

for e in ("pdf", "png"):
    fig.savefig(OUT / f"fig_template.{e}", dpi=200, bbox_inches="tight", facecolor="white")
print(f"own {matched.mean():.4f}+-{matched.std():.4f} (min {np.nanmin(matched):.4f}) | "
      f"cross {mismatched.mean():.4f}+-{mismatched.std():.4f} (max {mismatched.max():.4f}) n_cross {len(mismatched)}")
