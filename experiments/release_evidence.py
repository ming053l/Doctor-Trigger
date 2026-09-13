"""Release-event evidence: only an enrolled release carries the keyed nudge.

For each sampled test enrollment, score three images against the SAME record:
the released copy (benign query), the untriggered original (enrollment input),
and a sibling person's released copy. H cannot separate the first two (the
nudge is deliberately weak); the signed projection G_s can. A reference-only
record has no analog: any image can have a record filed for it after the fact,
so record existence alone certifies nothing about enrollment."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
R = _dt("DT_CANON_ROOT")
sys.path.insert(0, R)
from scripts.extract_features import feature_row

cases = pd.read_csv(f"{R}/common/cases.csv")
ben = cases[(cases.split == "test") & (cases.protocol == "benign")].sample(400, random_state=0)
rows = []
for _, q in ben.iterrows():
    for tag, img in [("released", q.input_path), ("original", q.enrollment_input_path)]:
        try:
            f, _ = feature_row(Path(img), Path(q.doctor_trigger_template_path))
        except ValueError:
            continue
        rows.append({"dataset": q.dataset, "kind": tag, "H": f["H"], "G_s": f["G_s"], "G_margin": f.get("G_margin", np.nan)})
d = pd.DataFrame(rows)
d.to_csv(_dt("DT_OUTPUTS", "/release_evidence.csv"), index=False)
y = (d.kind == "released").astype(int)
print(f"n={len(d)} ({(y==1).sum()} released / {(y==0).sum()} untriggered originals)")
for col in ["H", "G_s"]:
    print(f"released-vs-original AUC {col:4s}: {roc_auc_score(y, d[col]):.4f}  released mean {d[y==1][col].mean():.3f}  original mean {d[y==0][col].mean():.3f}")
