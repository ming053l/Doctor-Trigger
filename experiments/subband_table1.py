"""The C_B row of Table 1 (and the direct 1 - C_B row of Table 2): AUC of 1 - C_B per cell.

C_B is the agreement with the template T_I on K_eff and sub-band B (kappa <= 0.15). CPU only.
Usage: subband_table1.py <part>
  benign    the four benign families on the full FF++ test split, each the mean AUC over its
            severities, with the channel seeds (k + 17) * 1009 of the full-ring row
  datasets  clean SimSwap detection on each dataset's test split
  engines   the other face manipulations: released image vs its cached engine swap
            (<engine_cache>/<engine>/<enrollment_id>.png, written by other_manipulations.py)
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from doctor_trigger import config
from doctor_trigger.channels import apply_channel
from doctor_trigger.manifests import read_cases
from doctor_trigger.metrics import roc_auc
from doctor_trigger.readings import SizeMismatch, SumForm, query_phase, spectral_agreement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import subband_b

STABILIZER = 1e-8
PART = sys.argv[1]

@lru_cache(maxsize=None)
def subband_b_uncentered(shape):
    return subband_b(shape, centered=False)


def subband_spectral_agreement(pil, record_path):
    """C_B of a query against its record, summed over the uncentered grid (a weight map)."""
    record = load_record(record_path)
    template = np.fft.ifftshift(record.template_phase)
    active = np.fft.ifftshift(record.active_bins)
    weights = np.fft.ifftshift(record.amplitude_weights)
    phase = query_phase(pil, template.shape, SizeMismatch.RESIZE, centered=False)
    return spectral_agreement(phase, template, weights, active & subband_b_uncentered(template.shape),
                              stabilizer=STABILIZER, form=SumForm.WEIGHT_MAP)


if PART == "benign":
    te = read_cases(dataset="FF++", split="test")
    PERTS = ["jpeg_q90", "jpeg_q70", "jpeg_q50", "blur_k7", "x264_crf30", "x264_crf35", "x264_crf40", "scale_0.5"]
    FAM = {"JPEG": ["jpeg_q90", "jpeg_q70", "jpeg_q50"], "blur": ["blur_k7"],
           "codec": ["x264_crf30", "x264_crf35", "x264_crf40"], "pixel": ["scale_0.5"]}
    res = {p: {"s": [], "y": []} for p in PERTS}
    for k, (_, q) in enumerate(te.iterrows()):
        img = Image.open(q.input_path).convert("RGB")
        for p in PERTS:
            res[p]["s"].append(1 - subband_spectral_agreement(apply_channel(p, img, seed=(k+17)*1009),
                                                              q.doctor_trigger_template_path))
            res[p]["y"].append(q.y_fake)
    for fam, ps in FAM.items():
        aucs = [roc_auc(res[p]["y"], res[p]["s"]) for p in ps]
        print(f"SUBBAND {fam}: {np.mean(aucs):.3f}  (per-sev: {' '.join(f'{a:.3f}' for a in aucs)})")

elif PART == "datasets":
    te = read_cases(split="test")
    for ds, g in te.groupby("dataset"):
        ss = [1 - subband_spectral_agreement(Image.open(q.input_path).convert("RGB"), q.doctor_trigger_template_path)
              for _, q in g.iterrows()]
        print(f"SUBBAND {ds}: {roc_auc(g.y_fake.values, ss):.3f}")

elif PART == "engines":
    CACHE = config.path("engine_cache")
    te = read_cases(dataset="FF++", split="test")
    ben = te[te.protocol == "benign"].drop_duplicates("enrollment_id")
    for eng in ["stargan", "uniface", "inswapper", "sdinpaint"]:
        ys, ss = [], []
        for _, q in ben.iterrows():
            cf = Path(f"{CACHE}/{eng}/{q.enrollment_id}.png")
            if not cf.exists():
                continue
            ss.append(1 - subband_spectral_agreement(Image.open(q.doctor_trigger_released_path).convert("RGB"),
                                                     q.doctor_trigger_template_path)); ys.append(0)
            ss.append(1 - subband_spectral_agreement(Image.open(cf).convert("RGB"),
                                                     q.doctor_trigger_template_path)); ys.append(1)
        print(f"SUBBAND {eng}: {roc_auc(ys, ss):.3f}  n={len(ys)//2}")
