"""Benign channels on one dataset (Table 10; full-ring row of Table 1): 1 - H and S_df per channel.

Every test query of the dataset is sent through each benign channel (seed (k + 17) * 1009 for the
k-th query) and scored by the full-ring original-reference agreement H against its record
(luminance phase, centered layout, amplitude weights, K_eff; resized to the record's grid when the
channel changed the size) and by the image-only detector S_df (the dataset's augmented detector,
or $IMAGE_DETECTOR_CKPT). A query whose channel or reading fails is left out. Prints the AUC of
1 - H, of S_df and of their fusion max(S_df, 1 - H) per channel family (mean over severities).
Usage: benign_channels.py [FF++|DFDC|CelebDF]
"""
import sys

import numpy as np
from PIL import Image

from doctor_trigger.channels import apply_channel
from doctor_trigger.manifests import read_cases
from doctor_trigger.metrics import roc_auc
from doctor_trigger.readings import SizeMismatch, SumForm, query_phase, spectral_agreement
from doctor_trigger.record import load_record
from experiments.shared.image_only_detector import deployed_checkpoint_path, image_only_score, load_image_only_detector

DEVICE = "cuda"
STABILIZER = 1e-12
DATASET = sys.argv[1] if len(sys.argv) > 1 else "FF++"
CHANNELS = ["clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "blur_k7", "x264_crf30", "x264_crf35", "x264_crf40", "scale_0.5"]
FAMILIES = {"JPEG": ["jpeg_q90", "jpeg_q70", "jpeg_q50"], "blur": ["blur_k7"],
            "codec": ["x264_crf30", "x264_crf35", "x264_crf40"], "pixel": ["scale_0.5"]}

cases = read_cases(split="test", dataset=DATASET)
image_detector = load_image_only_detector(deployed_checkpoint_path(DATASET), DEVICE)


def original_reference_agreement(image, record_path):
    """Full-ring H of a query against its record (weight-map summation, stabilizer 1e-12)."""
    record = load_record(record_path)
    phase = query_phase(image, record.shape, SizeMismatch.RESIZE, centered=True)
    return spectral_agreement(phase, record.original_phase, record.amplitude_weights, record.active_bins,
                              stabilizer=STABILIZER, form=SumForm.WEIGHT_MAP)


results = {}
for channel in CHANNELS:
    labels, mismatch, detector = [], [], []
    for k, query in enumerate(cases.itertuples()):
        try:
            image = apply_channel(channel, Image.open(query.input_path).convert("RGB"), seed=(k + 17) * 1009)
            h = original_reference_agreement(image, query.doctor_trigger_template_path)
            s_df = image_only_score(image_detector, image, DEVICE)
        except Exception:
            continue
        labels.append(query.y_fake); mismatch.append(1 - h); detector.append(s_df)
    results[channel] = (roc_auc(labels, mismatch), roc_auc(labels, detector),
                        roc_auc(labels, np.maximum(detector, mismatch)), len(labels))
print(f"{'family':8s} {'H(1-)':>8s} {'image-only detector':>8s} {'Fusion':>8s}   n")
for family, members in FAMILIES.items():
    a = [results[c][0] for c in members]; b = [results[c][1] for c in members]; c_ = [results[c][2] for c in members]
    print(f"{family:8s} {np.mean(a):8.4f} {np.mean(b):8.4f} {np.mean(c_):8.4f}  {results[members[0]][3]}")
print(f"{'clean':8s} {results['clean'][0]:8.4f} {results['clean'][1]:8.4f} {results['clean'][2]:8.4f}")
