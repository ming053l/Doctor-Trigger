#!/usr/bin/env python
"""Equally-privileged reference baselines on the main test split (Table 3, appendix Table 7 swap columns).

How much of the swap-detection AUC comes from having a private per-asset reference at all, rather
than from the keyed displacement? Every test query is scored against:
  A) phase references: the agreement with the stored original phase φ̂_I (no key information;
     CSV column ``H_host``) and with the stored template phase T_I (``H_tmpl``)
  B) pixel reference: SSIM between the query and the stored release
  C) identity reference: ArcFace cosine similarity between the query and the enrolled face
All of them see the same private state and the same queries as the record path. A query whose
size differs from the record is resized (bicubic) to the record's grid.

Paths: config.yaml (cases_csv, outputs_root). Writes $DT_OUTPUTS/reference_baselines.csv and prints
the swap AUC per dataset.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from doctor_trigger import config
from doctor_trigger.manifests import read_cases
from doctor_trigger.metrics import roc_auc
from doctor_trigger.readings import SizeMismatch, SumForm, query_phase, spectral_agreement
from doctor_trigger.record import load_record
from experiments.shared.face_identity import face_analysis, identity_embedding, identity_similarity

AGREEMENT_STABILIZER = 1e-8


def agreement_with(phase, record, reference_phase):
    """Spectral agreement of a query phase with one stored reference phase (weight-map form)."""
    return spectral_agreement(phase, reference_phase, record.amplitude_weights, record.active_bins,
                              stabilizer=AGREEMENT_STABILIZER, form=SumForm.WEIGHT_MAP)


face = face_analysis()

cases = read_cases(split="test")
rows, enrolled_embeddings = [], {}
for _, case in cases.iterrows():
    query = Image.open(case.input_path).convert("RGB")
    released = Image.open(case.doctor_trigger_released_path).convert("RGB")
    row = {"query_id": case.query_id, "dataset": case.dataset, "y_fake": case.y_fake}
    record = load_record(case.doctor_trigger_template_path)
    phase = query_phase(query, record.shape, SizeMismatch.RESIZE)
    row["H_host"] = agreement_with(phase, record, record.original_phase)
    row["H_tmpl"] = agreement_with(phase, record, record.template_phase)
    query_gray = np.asarray(query.resize(released.size).convert("L"))
    released_gray = np.asarray(released.convert("L"))
    row["ssim"] = ssim(query_gray, released_gray)
    if case.enrollment_id not in enrolled_embeddings:
        enrolled_embeddings[case.enrollment_id] = identity_embedding(
            face, Image.open(case.enrollment_input_path).convert("RGB"))
    row["arc"] = identity_similarity(enrolled_embeddings[case.enrollment_id], identity_embedding(face, query))
    rows.append(row)
scores = pd.DataFrame(rows)
scores.to_csv(Path(config.path("outputs_root")) / "reference_baselines.csv", index=False)
print(f"{'dataset':9s} {'H(tmpl)':>8s} {'H(host)':>8s} {'SSIM':>7s} {'ArcFace':>8s} {'arc-nan':>7s}")
for dataset, group in scores.groupby("dataset"):
    with_face = group.dropna(subset=["arc"])
    print(f"{dataset:9s} {roc_auc(group.y_fake, -group.H_tmpl):8.4f} {roc_auc(group.y_fake, -group.H_host):8.4f} "
          f"{roc_auc(group.y_fake, -group.ssim):7.4f} {roc_auc(with_face.y_fake, -with_face.arc):8.4f} "
          f"{group.arc.isna().sum():7d}")
