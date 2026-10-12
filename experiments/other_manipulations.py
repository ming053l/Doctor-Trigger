"""Other face manipulations (Table 1 right columns, Table 12 engine rows): H and S_df per engine.

ENGINE (stargan, uniface, inswapper or sdinpaint; experiments/shared/swap_engines.py) re-renders
each FF++ test release with the enrollment's recorded swap source. The release (benign) and the
re-rendering (attack) are scored by the full-ring original-reference agreement H against the
record and by the FF++ image-only detector S_df ($IMAGE_DETECTOR_CKPT overrides it). Swaps are
cached in <engine_cache>/ENGINE/<enrollment_id>.png and reused when present; delete the cache to
regenerate. An enrollment whose swap or reading fails is counted as dropped.
Prints the AUC of 1 - H, S_df and max(S_df, 1 - H), the pair count and the mean benign/attack H.
Usage: other_manipulations.py ENGINE
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from doctor_trigger import config
from doctor_trigger.manifests import read_cases
from doctor_trigger.metrics import roc_auc
from doctor_trigger.readings import SizeMismatch, query_phase, spectral_agreement
from doctor_trigger.record import load_record
from experiments.shared.image_only_detector import deployed_checkpoint_path, image_only_score, load_image_only_detector
from experiments.shared.swap_engines import ENGINES

ENGINE = sys.argv[1]
DEVICE = "cuda"
cases = read_cases(split="test", dataset="FF++")
image_detector = load_image_only_detector(deployed_checkpoint_path("FF++"), DEVICE)


def original_reference_agreement(image, record_path):
    """Full-ring H of a query against its record."""
    record = load_record(record_path)
    phase = query_phase(image, record.shape, SizeMismatch.RESIZE, centered=True)
    return spectral_agreement(phase, record.original_phase, record.amplitude_weights, record.active_bins)


engine = ENGINES[ENGINE]()
cache = Path(config.path("engine_cache")) / ENGINE
cache.mkdir(parents=True, exist_ok=True)
labels, mismatch, detector, dropped = [], [], [], 0
for enrollment_id, group in cases.groupby("enrollment_id"):
    released_path = group.doctor_trigger_released_path.iloc[0]
    record_path = group.doctor_trigger_template_path.iloc[0]
    source_path = group[group.protocol == "swap"].attack_source_path.iloc[0]
    cached = cache / f"{enrollment_id}.png"
    try:
        released = Image.open(released_path).convert("RGB")
        h_benign = original_reference_agreement(released, record_path)
        if cached.exists():
            swapped = Image.open(cached).convert("RGB")
        else:
            swapped = engine(released, Image.open(source_path).convert("RGB"))
            swapped.save(cached)
        h_attack = original_reference_agreement(swapped, record_path)
    except Exception:
        dropped += 1
        continue
    labels += [0, 1]; mismatch += [1 - h_benign, 1 - h_attack]
    detector += [image_only_score(image_detector, released, DEVICE), image_only_score(image_detector, swapped, DEVICE)]
fusion = [max(a, b) for a, b in zip(detector, mismatch)]
print(f"{ENGINE}: AUC(1-H)={roc_auc(labels, mismatch):.4f} AUC(image-only detector)={roc_auc(labels, detector):.4f} "
      f"AUC(Fus)={roc_auc(labels, fusion):.4f} n={len(labels)//2} dropped={dropped} "
      f"benignH={1-np.mean([s for s, y in zip(mismatch, labels) if y == 0]):.4f} "
      f"attackH={1-np.mean([s for s, y in zip(mismatch, labels) if y == 1]):.4f}")
