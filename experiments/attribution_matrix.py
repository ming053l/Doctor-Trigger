"""Per-release attribution: can each verifier tell WHICH release a copy came from?

For test source groups holding at least two enrollments of the same person, a
benign copy of release i is scored against its own record i and against a
sibling record j of the same person. Identity features cannot separate the two
by construction; the per-asset phase reference and the keyed projection can.
AUC is own-vs-sibling."""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402
from doctor_trigger.manifests import read_cases  # noqa: E402
from doctor_trigger.metrics import roc_auc  # noqa: E402
from doctor_trigger.readings import SizeMismatch, query_phase, record_features  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from experiments.shared.face_identity import face_analysis, identity_embedding, identity_similarity  # noqa: E402

fa = face_analysis()


def embed(p):
    return identity_embedding(fa, Image.open(p))


ben = read_cases(split="test", protocol="benign")
rows = []
for gid, g in ben.groupby("source_group_id"):
    if len(g) < 2: continue
    g = g.sample(min(len(g), 2), random_state=0)
    ids = g.index.tolist()
    for k, i in enumerate(ids):
        qi = g.loc[i]; j = g.loc[ids[1 - k]]
        for tag, rec in [("own", qi), ("sibling", j)]:
            record = load_record(rec.doctor_trigger_template_path)
            try:
                phase = query_phase(Image.open(qi.input_path), record.shape, SizeMismatch.RAISE)
            except ValueError:
                continue
            feats = record_features(phase, record)
            e_r = embed(rec.enrollment_input_path); e_q = embed(qi.input_path)
            rows.append({"dataset": qi.dataset, "pair": tag, "H": feats["H"], "G_s": feats["G_s"],
                         "arc": identity_similarity(e_r, e_q)})
    if len(rows) > 1600: break
d = pd.DataFrame(rows)
d.to_csv(Path(config.path("outputs_root")) / "attribution_matrix.csv", index=False)
y = (d.pair == "own").astype(int)
print(f"n={len(d)} pairs ({(y==1).sum()} own / {(y==0).sum()} sibling, same person)")
for col in ["H", "G_s", "arc"]:
    dd = d.dropna(subset=[col]); yy = (dd.pair == "own").astype(int)
    print(f"attribution AUC {col:4s}: {roc_auc(yy, dd[col]):.4f}   own mean {dd[yy==1][col].mean():.3f}  sibling mean {dd[yy==0][col].mean():.3f}")
