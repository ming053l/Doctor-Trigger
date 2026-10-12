"""Equally-privileged reference baselines on the canonical test split.

Control: how much of the detection AUC comes from having a
private per-asset reference at all, versus from the keyed phase trigger?
  A) no-trigger phase reference: H computed against host_phase (the original
     face's own phase), ignoring the keyed displacement entirely
  B) pixel reference: SSIM between the query and the stored release
  C) identity reference: ArcFace embedding distance between the query and the
     enrolled face
All three see the same private state and the same queries as the record path.
"""
import sys
from pathlib import Path

import numpy as np, pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402
from doctor_trigger.manifests import read_cases  # noqa: E402
from doctor_trigger.metrics import roc_auc  # noqa: E402
from doctor_trigger.readings import SizeMismatch, SumForm, query_phase, spectral_agreement  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from experiments.shared.face_identity import face_analysis, identity_embedding, identity_similarity  # noqa: E402

EPS = 1e-8


def H_against(phase, record, reference):
    """Spectral agreement of a query phase with one record reference, weight-map form, +1e-8."""
    return spectral_agreement(phase, reference, record.amplitude_weights, record.active_bins,
                              stabilizer=EPS, form=SumForm.WEIGHT_MAP)


fa = face_analysis()

cases = read_cases(split="test")
rows, emb_cache = [], {}
for _, q in cases.iterrows():
    quer = Image.open(q.input_path).convert("RGB")
    rel = Image.open(q.doctor_trigger_released_path).convert("RGB")
    r = {"query_id": q.query_id, "dataset": q.dataset, "y_fake": q.y_fake}
    record = load_record(q.doctor_trigger_template_path)
    phase = query_phase(quer, record.shape, SizeMismatch.RESIZE)   # bicubic resize to the record's grid
    r["H_host"] = H_against(phase, record, record.original_phase)
    r["H_tmpl"] = H_against(phase, record, record.template_phase)
    qa = np.asarray(quer.resize(rel.size).convert("L")); ra = np.asarray(rel.convert("L"))
    r["ssim"] = ssim(qa, ra)
    if q.enrollment_id not in emb_cache:
        emb_cache[q.enrollment_id] = identity_embedding(fa, Image.open(q.enrollment_input_path).convert("RGB"))
    er, eq = emb_cache[q.enrollment_id], identity_embedding(fa, quer)
    r["arc"] = identity_similarity(er, eq)
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(Path(config.path("outputs_root")) / "reference_baselines.csv", index=False)
print(f"{'dataset':9s} {'H(tmpl)':>8s} {'H(host)':>8s} {'SSIM':>7s} {'ArcFace':>8s} {'arc-nan':>7s}")
for ds, g in d.groupby("dataset"):
    ga = g.dropna(subset=["arc"])
    print(f"{ds:9s} {roc_auc(g.y_fake, -g.H_tmpl):8.4f} {roc_auc(g.y_fake, -g.H_host):8.4f} "
          f"{roc_auc(g.y_fake, -g.ssim):7.4f} {roc_auc(ga.y_fake, -ga.arc):8.4f} {g.arc.isna().sum():7d}")
