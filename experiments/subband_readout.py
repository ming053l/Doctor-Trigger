"""Channel-adaptive sub-band readout: can a declared lossy channel be partially
recovered by reading only the part of the ring it preserves?

Blur and codec are low-pass: they attenuate the upper end of the ring. The
record already stores amplitudes and support, so restricting the readout to
kappa <= kappa_cut is a per-channel declaration, training-free. AUC per cut on
the explicit reading (C_B against the template T_I), FF++ test."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from doctor_trigger.channels import apply_channel  # noqa: E402
from doctor_trigger.manifests import read_cases  # noqa: E402
from doctor_trigger.metrics import roc_auc  # noqa: E402
from doctor_trigger.readings import SumForm, amplitude_weighted_mean  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from doctor_trigger.spectrum import kappa_grid, luminance_phase, wrap_phase  # noqa: E402

N = 150
STABILIZER = 1e-8

te = read_cases(dataset="FF++", split="test")
ben = te[te.protocol == "benign"].drop_duplicates("enrollment_id")[
    ["enrollment_id", "doctor_trigger_template_path", "doctor_trigger_released_path"]]
swaps = te[te.protocol == "swap"].drop_duplicates("enrollment_id")[["enrollment_id", "input_path"]]
pool = ben.merge(swaps, on="enrollment_id").head(N)

CUTS = [0.15, 0.20, 0.25, 0.30, 0.45]
CHANNELS = ["blur_k7", "x264_crf35", "jpeg_q50"]
scores = {(c, k): {"b": [], "f": []} for c in CHANNELS for k in CUTS}
for _, q in pool.iterrows():
    record = load_record(q.doctor_trigger_template_path)
    # uncentered layout, as the readout always summed in it
    template = np.fft.ifftshift(record.template_phase)
    active = np.fft.ifftshift(record.active_bins)
    weights = np.fft.ifftshift(record.amplitude_weights)
    kappa = kappa_grid(template.shape, centered=False)
    rel = Image.open(q.doctor_trigger_released_path).convert("RGB")
    fake = Image.open(q.input_path).convert("RGB")
    for ci, ch in enumerate(CHANNELS):
        pb = luminance_phase(np.asarray(apply_channel(ch, rel, seed=11 + ci)), centered=False)
        pf = luminance_phase(np.asarray(apply_channel(ch, fake, seed=17 + ci)), centered=False)
        cos_benign = np.cos(wrap_phase(pb - template))
        cos_fake = np.cos(wrap_phase(pf - template))
        for k in CUTS:
            bins = active & (kappa <= k)
            for values, side in ((cos_benign, "b"), (cos_fake, "f")):
                scores[(ch, k)][side].append(amplitude_weighted_mean(
                    values, weights, bins, stabilizer=STABILIZER, form=SumForm.WEIGHT_MAP))

print(f"{'channel':11s}" + "".join(f"  cut<={k:.2f}" for k in CUTS) + "   (AUC, benign-H mean)")
for ch in CHANNELS:
    line = f"{ch:11s}"
    for k in CUTS:
        s_ = scores[(ch, k)]
        y = np.r_[np.zeros(len(s_["b"])), np.ones(len(s_["f"]))]
        auc = roc_auc(y, np.r_[1-np.array(s_["b"]), 1-np.array(s_["f"])])
        line += f"  {auc:.3f}/{np.mean(s_['b']):.2f}"
    print(line)
