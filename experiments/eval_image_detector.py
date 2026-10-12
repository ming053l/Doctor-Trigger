"""Image-only detector S_df on every test query (Table 2 S_df row), and its fusion with 1 - H.

Scores each test query of cases.csv with the augmented detector of each training dataset
(checkpoints.image_detector_aug/{ffpp,dfdc,celebdf}), prints the within- and cross-dataset AUC
matrix, then the in-domain fusion max(S_df, 1 - H) with H from the pipeline features (features_csv).
Writes <outputs_root>/image_detector_aug.csv (query_id, dataset, y_fake, image_detector_<tag>).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from doctor_trigger import config
from doctor_trigger.manifests import DATASET_TAGS, read_cases
from doctor_trigger.metrics import roc_auc
from experiments.shared.image_only_detector import checkpoint_path, image_only_score, load_image_only_detector

DEVICE = "cuda"
out_csv = Path(config.path("outputs_root")) / "image_detector_aug.csv"
cases = read_cases(split="test")
detectors = {tag: load_image_only_detector(checkpoint_path(tag, augmented=True), DEVICE)
             for tag in ["ffpp", "dfdc", "celebdf"]}
rows = []
for _, query in cases.iterrows():
    image = Image.open(query.input_path).convert("RGB")
    row = {"query_id": query.query_id, "dataset": query.dataset, "y_fake": query.y_fake}
    for tag, detector in detectors.items():
        row[f"image_detector_{tag}"] = image_only_score(detector, image, DEVICE)
    rows.append(row)
scores = pd.DataFrame(rows)
out_csv.parent.mkdir(parents=True, exist_ok=True)
scores.to_csv(out_csv, index=False)
print("canonical-trained image-only detector, test AUC:")
for dataset, group in scores.groupby("dataset"):
    line = f"  {dataset:8s}"
    for tag in detectors:
        line += f"  {tag}={roc_auc(group.y_fake, group[f'image_detector_{tag}']):.4f}"
    print(line)
features = pd.read_csv(config.path("features_csv"), usecols=["query_id", "H"])
merged = scores.merge(features, on="query_id")
print("fusion max(in-domain image-only detector, 1-H):")
for dataset, group in merged.groupby("dataset"):
    in_domain = f"image_detector_{DATASET_TAGS[dataset]}"
    fusion = np.maximum(group[in_domain], 1 - group.H)
    print(f"  {dataset:8s} H={roc_auc(group.y_fake, -group.H):.4f} image_detector={roc_auc(group.y_fake, group[in_domain]):.4f} "
          f"fusion={roc_auc(group.y_fake, fusion):.4f}")
