"""Per-release attribution: can each verifier tell WHICH release a copy came from?

For test source groups holding at least two enrollments of the same person, a
benign copy of release i is scored against its own record i and against a
sibling record j of the same person. Identity features cannot separate the two
by construction; the per-asset phase reference and the keyed projection can.
AUC is own-vs-sibling."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py"
    return v + rest

import sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score
R = _dt("DT_CANON_ROOT")
sys.path.insert(0, R)
from scripts.extract_features import feature_row

from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
def embed(p):
    faces = fa.get(np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1])
    return faces[0].normed_embedding if faces else None

cases = pd.read_csv(f"{R}/common/cases.csv")
ben = cases[(cases.split == "test") & (cases.protocol == "benign")]
rng = np.random.RandomState(0)
rows = []
for gid, g in ben.groupby("source_group_id"):
    if len(g) < 2: continue
    g = g.sample(min(len(g), 2), random_state=0)
    ids = g.index.tolist()
    for k, i in enumerate(ids):
        qi = g.loc[i]; j = g.loc[ids[1 - k]]
        for tag, rec in [("own", qi), ("sibling", j)]:
            try:
                feats, _ = feature_row(Path(qi.input_path), Path(rec.doctor_trigger_template_path))
            except ValueError:
                continue
            e_r = embed(rec.enrollment_input_path); e_q = embed(qi.input_path)
            rows.append({"dataset": qi.dataset, "pair": tag, "H": feats["H"], "G_s": feats["G_s"],
                         "arc": float(np.dot(e_r, e_q)) if e_r is not None and e_q is not None else np.nan})
    if len(rows) > 1600: break
d = pd.DataFrame(rows)
d.to_csv(_dt("DT_OUTPUTS", "/attribution_matrix.csv"), index=False)
y = (d.pair == "own").astype(int)
print(f"n={len(d)} pairs ({(y==1).sum()} own / {(y==0).sum()} sibling, same person)")
for col in ["H", "G_s", "arc"]:
    dd = d.dropna(subset=[col]); yy = (dd.pair == "own").astype(int)
    print(f"attribution AUC {col:4s}: {roc_auc_score(yy, dd[col]):.4f}   own mean {dd[yy==1][col].mean():.3f}  sibling mean {dd[yy==0][col].mean():.3f}")
