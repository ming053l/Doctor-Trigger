#!/usr/bin/env python
"""How small can the per-face record get? (Sec. 4.2 storage paragraph, appendix Table 16 records)

Variants, each verified on 150 FF++ test pairs (benign release and its swap) for the swap AUC of
1 - C, the benign mean of C, and the correct-versus-wrong-key AUC of G_s:
  full   : the deployed NPZ as shipped (baseline bytes)
  derive : drop every field that can be recomputed (displacement, sign pattern, original phase);
           keep the template phase and amplitude on the active bins of the ring
           kappa in [0.10, 0.45] as float16, plus a bitmask of those bins
  q84    : derive, with the phase quantized to 8 bits and the log-amplitude to 4 bits
  sub    : q84 restricted to the ring's part in sub-band B (kappa <= 0.15)
Bytes are zlib-compressed serialized payloads. The displacement is regenerated from the record's
key; G_s is compared with 4 wrong keys per record.

Paths: config.yaml (cases_csv). Prints the table; writes nothing.
"""
import io
import os
import zlib

import numpy as np
from PIL import Image

from doctor_trigger.key import wrong_keys
from doctor_trigger.manifests import read_cases
from doctor_trigger.metrics import roc_auc
from doctor_trigger.readings import key_projection, phase_drift, spectral_agreement, wrong_key_displacement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import kappa_grid, luminance_phase

PAIR_COUNT = 150
WRONG_KEY_COUNT = 4
RING_KAPPA_MIN, RING_KAPPA_MAX = 0.10, 0.45
SUBBAND_KAPPA_MAX = 0.15
VARIANTS = ["derive", "q84", "sub"]


def unshifted_phase(path):
    """Unshifted luminance phase of an image file."""
    return luminance_phase(np.asarray(Image.open(path).convert("RGB"), dtype=np.float64), centered=False)


def payload_bytes(arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return len(zlib.compress(buffer.getvalue(), 9))


def encode(record, kappa, variant):
    """The variant's payload arrays and the boolean mask of the bins it keeps."""
    ring = (kappa >= RING_KAPPA_MIN) & (kappa <= (SUBBAND_KAPPA_MAX if variant == "sub" else RING_KAPPA_MAX))
    kept = record["active_bins"] & ring
    template = record["template_phase"][kept]
    amplitude = record["amplitude_weights"][kept]
    mask = np.packbits(kept[ring])
    if variant == "derive":
        return {"t": template.astype(np.float16), "a": amplitude.astype(np.float16), "m": mask}, kept
    template_q = np.round((template + np.pi) / (2*np.pi) * 255).astype(np.uint8)
    log_amplitude = np.log10(np.maximum(amplitude, 1e-6))
    lo, hi = log_amplitude.min(), log_amplitude.max()
    amplitude_q = np.round((log_amplitude - lo) / max(hi - lo, 1e-9) * 15).astype(np.uint8)
    return {"t": template_q, "a": amplitude_q, "m": mask, "r": np.array([lo, hi], np.float32)}, kept


def decode(payload, variant):
    """(template phase, amplitude) on the kept bins, as float64."""
    if variant == "derive":
        return payload["t"].astype(np.float64), payload["a"].astype(np.float64)
    template = payload["t"].astype(np.float64) / 255 * 2*np.pi - np.pi
    lo, hi = payload["r"]
    amplitude = 10 ** (payload["a"].astype(np.float64) / 15 * (hi - lo) + lo)
    return template, amplitude


test = read_cases(dataset="FF++", split="test")
benign = test[test.protocol == "benign"].drop_duplicates("enrollment_id")
swaps = test[test.protocol == "swap"].drop_duplicates("enrollment_id")[["enrollment_id", "input_path"]]
pairs = benign.merge(swaps, on="enrollment_id", suffixes=("", "_swap")).head(PAIR_COUNT)

sizes = {v: [] for v in VARIANTS + ["full"]}
readings = {v: {"C_benign": [], "C_swap": [], "G": [], "G_wrong": []} for v in VARIANTS}
for _, pair in pairs.iterrows():
    sizes["full"].append(os.path.getsize(pair.doctor_trigger_template_path))
    stored = load_record(pair.doctor_trigger_template_path)
    record = {"template_phase": np.fft.ifftshift(stored.template_phase),
              "active_bins": np.fft.ifftshift(stored.active_bins),
              "amplitude_weights": np.fft.ifftshift(stored.amplitude_weights)}
    kappa = kappa_grid(stored.shape, centered=False)
    benign_phase = unshifted_phase(pair.input_path)
    swap_phase = unshifted_phase(pair.input_path_swap)
    displacement = np.fft.ifftshift(wrong_key_displacement(stored, stored.key))   # regenerated from the key
    wrong_displacements = [np.fft.ifftshift(wrong_key_displacement(stored, k))
                           for k in wrong_keys(stored.key, WRONG_KEY_COUNT)]
    for variant in VARIANTS:
        payload, kept = encode(record, kappa, variant)
        sizes[variant].append(payload_bytes(payload))
        template, amplitude = decode(payload, variant)
        every_bin = np.ones(template.shape, dtype=bool)
        original_phase = template - displacement[kept]
        for name, phase in [("C_benign", benign_phase), ("C_swap", swap_phase)]:
            readings[variant][name].append(spectral_agreement(phase[kept], template, amplitude, every_bin))
        drift = phase_drift(benign_phase[kept], original_phase)
        readings[variant]["G"].append(key_projection(drift, displacement[kept], amplitude, every_bin))
        for wrong in wrong_displacements:
            readings[variant]["G_wrong"].append(key_projection(drift, wrong[kept], amplitude, every_bin))

print(f"{'variant':8s} {'KiB':>7s} {'det AUC':>8s} {'benH':>7s} {'keyAUC':>7s}")
print(f"{'full':8s} {np.mean(sizes['full'])/1024:7.1f}    (shipped NPZ)")
for variant in VARIANTS:
    labels = np.r_[np.zeros(PAIR_COUNT), np.ones(PAIR_COUNT)]
    swap_auc = roc_auc(labels, np.r_[1 - np.array(readings[variant]["C_benign"]),
                                     1 - np.array(readings[variant]["C_swap"])])
    correct, wrong = np.array(readings[variant]["G"]), np.array(readings[variant]["G_wrong"])
    key_auc = roc_auc(np.r_[np.ones_like(correct), np.zeros_like(wrong)], np.r_[correct, wrong])
    print(f"{variant:8s} {np.mean(sizes[variant])/1024:7.1f} {swap_auc:8.3f} "
          f"{np.mean(readings[variant]['C_benign']):7.3f} {key_auc:7.3f}")
