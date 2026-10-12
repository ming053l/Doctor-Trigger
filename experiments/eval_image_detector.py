"""Canonical-trained image-only detector: full matrix on canonical test + fusion with the relation.

Reads checkpoints.image_detector_aug (the deployed augmentation recipe) and writes
<outputs_root>/image_detector_aug.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from doctor_trigger import config  # noqa: E402
from doctor_trigger.manifests import DATASET_TAGS, read_cases  # noqa: E402
from doctor_trigger.metrics import roc_auc  # noqa: E402
from experiments.shared.image_only_detector import (checkpoint_path, image_only_score,  # noqa: E402
                                                    load_image_only_detector)

OUT_CSV = config.path("outputs_root") + "/image_detector_aug.csv"
dev = "cuda"
cases = read_cases(split="test")
HEADS = {tag: load_image_only_detector(checkpoint_path(tag, augmented=True), dev)
         for tag in ["ffpp", "dfdc", "celebdf"]}
rows = []
for _, q in cases.iterrows():
    pil = Image.open(q.input_path).convert("RGB")
    r = {"query_id": q.query_id, "dataset": q.dataset, "y_fake": q.y_fake}
    for tag, m in HEADS.items(): r[f"image_detector_{tag}"] = image_only_score(m, pil, dev)
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(OUT_CSV, index=False)
print("canonical-trained image-only detector, test AUC:")
for ds, g in d.groupby("dataset"):
    line = f"  {ds:8s}"
    for tag in HEADS: line += f"  {tag}={roc_auc(g.y_fake, g[f'image_detector_{tag}']):.4f}"
    print(line)
f = pd.read_csv(config.path("canonical_root") + "/features/calibrated_features.csv", usecols=["query_id", "H"])
m = d.merge(f, on="query_id")
print("fusion max(in-domain image-only detector, 1-H):")
for ds, g in m.groupby("dataset"):
    head = f"image_detector_{DATASET_TAGS[ds]}"
    fus = np.maximum(g[head], 1 - g.H)
    print(f"  {ds:8s} H={roc_auc(g.y_fake, -g.H):.4f} image_detector={roc_auc(g.y_fake, g[head]):.4f} fusion={roc_auc(g.y_fake, fus):.4f}")
