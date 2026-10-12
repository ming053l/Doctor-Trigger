#!/usr/bin/env python
"""Per-release attribution: can each verifier tell WHICH release a copy came from? (appendix Table 7)

For test source groups holding at least two enrollments of the same person, a benign copy of
release i is scored against its own record i and against a sibling record j of the same person.
Identity features cannot separate the two by construction; the per-asset phase reference H and the
key projection G_s can. AUC is own-vs-sibling. Stops after about 1,600 rows.

Paths: config.yaml (cases_csv, outputs_root). Writes $DT_OUTPUTS/attribution_matrix.csv.
"""
from pathlib import Path

import pandas as pd
from PIL import Image

from doctor_trigger import config
from doctor_trigger.manifests import read_cases
from doctor_trigger.metrics import roc_auc
from doctor_trigger.readings import SizeMismatch, query_phase, record_features
from doctor_trigger.record import load_record
from experiments.shared.face_identity import face_analysis, identity_embedding, identity_similarity

MAX_ROWS = 1600

face = face_analysis()


def embedding(path):
    return identity_embedding(face, Image.open(path))


benign = read_cases(split="test", protocol="benign")
rows = []
for _, group in benign.groupby("source_group_id"):
    if len(group) < 2:
        continue
    group = group.sample(2, random_state=0)
    first, second = group.index.tolist()
    for own_index, sibling_index in ((first, second), (second, first)):
        query = group.loc[own_index]
        for pair, record_case in [("own", query), ("sibling", group.loc[sibling_index])]:
            record = load_record(record_case.doctor_trigger_template_path)
            try:
                phase = query_phase(Image.open(query.input_path), record.shape, SizeMismatch.RAISE)
            except ValueError:
                continue
            features = record_features(phase, record)
            enrolled, queried = embedding(record_case.enrollment_input_path), embedding(query.input_path)
            rows.append({"dataset": query.dataset, "pair": pair, "H": features["H"], "G_s": features["G_s"],
                         "arc": identity_similarity(enrolled, queried)})
    if len(rows) > MAX_ROWS:
        break
readings = pd.DataFrame(rows)
readings.to_csv(Path(config.path("outputs_root")) / "attribution_matrix.csv", index=False)
own = (readings.pair == "own").astype(int)
print(f"n={len(readings)} pairs ({(own==1).sum()} own / {(own==0).sum()} sibling, same person)")
for col in ["H", "G_s", "arc"]:
    available = readings.dropna(subset=[col]); available_own = (available.pair == "own").astype(int)
    print(f"attribution AUC {col:4s}: {roc_auc(available_own, available[col]):.4f}   "
          f"own mean {available[available_own==1][col].mean():.3f}  "
          f"sibling mean {available[available_own==0][col].mean():.3f}")
