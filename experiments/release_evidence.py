#!/usr/bin/env python
"""Release-version evidence: only an encoded release carries the keyed displacement (appendix Table 7).

For 400 sampled benign test queries, score two images against the SAME record: the released copy
(the benign query) and its unreleased original (the enrollment input). H cannot separate the two
(the displacement is deliberately weak); the key projection G_s can. A reference-only record has no
analog: any image can have a record filed for it after the fact, so the existence of a record
alone certifies nothing about enrollment. Images whose size differs from the record are skipped.

Paths: config.yaml (cases_csv, outputs_root). Writes $DT_OUTPUTS/release_evidence.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from doctor_trigger import config
from doctor_trigger.manifests import read_cases
from doctor_trigger.metrics import roc_auc
from doctor_trigger.readings import SizeMismatch, query_phase, record_features
from doctor_trigger.record import load_record

benign = read_cases(split="test", protocol="benign").sample(400, random_state=0)
rows = []
for _, case in benign.iterrows():
    record = load_record(case.doctor_trigger_template_path)
    for kind, image_path in [("released", case.input_path), ("original", case.enrollment_input_path)]:
        try:
            phase = query_phase(Image.open(image_path), record.shape, SizeMismatch.RAISE)
        except ValueError:
            continue
        features = record_features(phase, record)
        rows.append({"dataset": case.dataset, "kind": kind, "H": features["H"], "G_s": features["G_s"],
                     "G_margin": features.get("G_margin", np.nan)})
readings = pd.DataFrame(rows)
readings.to_csv(Path(config.path("outputs_root")) / "release_evidence.csv", index=False)
released = (readings.kind == "released").astype(int)
print(f"n={len(readings)} ({(released==1).sum()} released / {(released==0).sum()} untriggered originals)")
for col in ["H", "G_s"]:
    print(f"released-vs-original AUC {col:4s}: {roc_auc(released, readings[col]):.4f}  "
          f"released mean {readings[released==1][col].mean():.3f}  original mean {readings[released==0][col].mean():.3f}")
