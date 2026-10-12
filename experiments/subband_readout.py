"""Five-cut sub-band diagnostic (App. G.2): which part of the ring survives a lossy channel?

Blur and video coding are low-pass: they attenuate the upper end of the ring. The record already
stores the amplitudes and K_eff, so restricting the reading to kappa <= cut needs no training.
For 150 FF++ test releases and their SimSwap queries, each sent through blur_k7, x264_crf35 and
jpeg_q50, prints the AUC of 1 - C_B (agreement with the template T_I on K_eff and kappa <= cut)
and the benign mean C_B, for cuts 0.15, 0.20, 0.25, 0.30, 0.45. The paper fixes cut 0.15
(sub-band B).
"""
import numpy as np
from PIL import Image

from doctor_trigger.channels import apply_channel
from doctor_trigger.manifests import read_cases
from doctor_trigger.metrics import roc_auc
from doctor_trigger.readings import SumForm, amplitude_weighted_mean
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import kappa_grid, luminance_phase, wrap_phase

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
    # uncentered layout: the weight-map sums below run in this order
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
