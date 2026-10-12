"""Benign channels under the canonical templates: distort the canonical query
bytes, score the closed-form original-reference agreement H against the record
(luminance phase, centered layout, original-amplitude weights, full ring K_eff)
and the augmented image-only detector. Usage: benign_canonical.py [FF++|DFDC|CelebDF]"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from doctor_trigger.channels import apply_channel  # noqa: E402
from doctor_trigger.manifests import read_cases  # noqa: E402
from doctor_trigger.metrics import roc_auc  # noqa: E402
from doctor_trigger.readings import SizeMismatch, SumForm, query_phase, spectral_agreement  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from experiments.shared.image_only_detector import (deployed_checkpoint_path, image_only_score,  # noqa: E402
                                                    load_image_only_detector)

dev = "cuda"; EPS = 1e-12
DS = sys.argv[1] if len(sys.argv) > 1 else "FF++"
cases = read_cases(split="test", dataset=DS)


def H_of(pil, tp):
    record = load_record(tp)
    phase = query_phase(pil, record.shape, SizeMismatch.RESIZE, centered=True)
    return spectral_agreement(phase, record.original_phase, record.amplitude_weights, record.active_bins,
                              stabilizer=EPS, form=SumForm.WEIGHT_MAP)


image_detector = load_image_only_detector(deployed_checkpoint_path(DS), dev)


def prob(pil):
    return image_only_score(image_detector, pil, dev)


PERTS = ["clean","jpeg_q90","jpeg_q70","jpeg_q50","blur_k7","x264_crf30","x264_crf35","x264_crf40","scale_0.5"]
FAM = {"JPEG":["jpeg_q90","jpeg_q70","jpeg_q50"],"blur":["blur_k7"],
       "codec":["x264_crf30","x264_crf35","x264_crf40"],"pixel":["scale_0.5"]}
res = {}
for p in PERTS:
    ys, hs, ds_ = [], [], []
    for k, q in enumerate(cases.itertuples()):
        try:
            img = apply_channel(p, Image.open(q.input_path).convert("RGB"), seed=(k+17)*1009)
            h = H_of(img, q.doctor_trigger_template_path); dp = prob(img)
        except Exception: continue
        ys.append(q.y_fake); hs.append(1-h); ds_.append(dp)
    res[p] = (roc_auc(ys, hs), roc_auc(ys, ds_),
              roc_auc(ys, np.maximum(ds_, hs)), len(ys))
print(f"{'family':8s} {'H(1-)':>8s} {'image-only detector':>8s} {'Fusion':>8s}   n")
for fam, ps in FAM.items():
    a=[res[p][0] for p in ps]; b=[res[p][1] for p in ps]; c=[res[p][2] for p in ps]
    print(f"{fam:8s} {np.mean(a):8.4f} {np.mean(b):8.4f} {np.mean(c):8.4f}  {res[ps[0]][3]}")
print(f"{'clean':8s} {res['clean'][0]:8.4f} {res['clean'][1]:8.4f} {res['clean'][2]:8.4f}")
