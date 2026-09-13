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
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py (see config.example.yaml)"
    return v + rest

import sys, numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as ssim
EPS = 1e-8
R = _dt("DT_CANON_ROOT")

def phase_of(pil):
    rgb = np.asarray(pil, dtype=np.float64)
    lum = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    return np.fft.fftshift(np.angle(np.fft.fft2(lum)))

def H_against(pil, tp, key):
    with np.load(tp) as rec:
        ref = rec[key].astype(np.float64); sup = rec["support"].astype(bool)
        w = rec["host_amplitude"].astype(np.float64)
    ph = phase_of(pil)
    if ph.shape != ref.shape:
        pil = pil.resize((ref.shape[1], ref.shape[0])); ph = phase_of(pil)
    r = np.cos((ph - ref + np.pi) % (2*np.pi) - np.pi)
    ww = np.where(sup, w, 0.0); return float(np.sum(r*ww) / (np.sum(ww) + EPS))

import insightface
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)

def embed(pil):
    faces = fa.get(np.asarray(pil)[:, :, ::-1])
    return faces[0].normed_embedding if faces else None

cases = pd.read_csv(f"{R}/common/cases.csv")
cases = cases[cases.split == "test"]
rows, emb_cache = [], {}
for _, q in cases.iterrows():
    quer = Image.open(q.input_path).convert("RGB")
    rel = Image.open(q.doctor_trigger_released_path).convert("RGB")
    r = {"query_id": q.query_id, "dataset": q.dataset, "y_fake": q.y_fake}
    r["H_host"] = H_against(quer, q.doctor_trigger_template_path, "host_phase")
    r["H_tmpl"] = H_against(quer, q.doctor_trigger_template_path, "template_phase")
    qa = np.asarray(quer.resize(rel.size).convert("L")); ra = np.asarray(rel.convert("L"))
    r["ssim"] = ssim(qa, ra)
    if q.enrollment_id not in emb_cache:
        emb_cache[q.enrollment_id] = embed(Image.open(q.enrollment_input_path).convert("RGB"))
    er, eq = emb_cache[q.enrollment_id], embed(quer)
    r["arc"] = float(np.dot(er, eq)) if er is not None and eq is not None else np.nan
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(_dt("DT_OUTPUTS", "/reference_baselines.csv"), index=False)
print(f"{'dataset':9s} {'H(tmpl)':>8s} {'H(host)':>8s} {'SSIM':>7s} {'ArcFace':>8s} {'arc-nan':>7s}")
for ds, g in d.groupby("dataset"):
    ga = g.dropna(subset=["arc"])
    print(f"{ds:9s} {roc_auc_score(g.y_fake, -g.H_tmpl):8.4f} {roc_auc_score(g.y_fake, -g.H_host):8.4f} "
          f"{roc_auc_score(g.y_fake, -g.ssim):7.4f} {roc_auc_score(ga.y_fake, -ga.arc):8.4f} {g.arc.isna().sum():7d}")
