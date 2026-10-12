"""Release-event evidence: only an enrolled release carries the keyed nudge.

For each sampled test enrollment, score three images against the SAME record:
the released copy (benign query), the untriggered original (enrollment input),
and a sibling person's released copy. H cannot separate the first two (the
nudge is deliberately weak); the signed projection G_s can. A reference-only
record has no analog: any image can have a record filed for it after the fact,
so record existence alone certifies nothing about enrollment."""
import sys
from pathlib import Path

import numpy as np, pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402
from doctor_trigger.manifests import read_cases  # noqa: E402
from doctor_trigger.metrics import roc_auc  # noqa: E402
from doctor_trigger.readings import SizeMismatch, query_phase, record_features  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402

ben = read_cases(split="test", protocol="benign").sample(400, random_state=0)
rows = []
for _, q in ben.iterrows():
    record = load_record(q.doctor_trigger_template_path)
    for tag, img in [("released", q.input_path), ("original", q.enrollment_input_path)]:
        try:
            phase = query_phase(Image.open(img), record.shape, SizeMismatch.RAISE)
        except ValueError:
            continue
        f = record_features(phase, record)
        rows.append({"dataset": q.dataset, "kind": tag, "H": f["H"], "G_s": f["G_s"], "G_margin": f.get("G_margin", np.nan)})
d = pd.DataFrame(rows)
d.to_csv(Path(config.path("outputs_root")) / "release_evidence.csv", index=False)
y = (d.kind == "released").astype(int)
print(f"n={len(d)} ({(y==1).sum()} released / {(y==0).sum()} untriggered originals)")
for col in ["H", "G_s"]:
    print(f"released-vs-original AUC {col:4s}: {roc_auc(y, d[col]):.4f}  released mean {d[y==1][col].mean():.3f}  original mean {d[y==0][col].mean():.3f}")
