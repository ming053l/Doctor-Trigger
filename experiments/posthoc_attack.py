"""Post-hoc enrollment attack, measured at deployed thresholds.

Threat model: verification claims a returning copy descends from an enrolled
release. The attacker holds content that never passed through release (here:
the untriggered original of each enrolled asset, the strongest case since it
matches the record's structure perfectly) and presents it for verification.
A signed database timestamp cannot help: the record itself is genuine; the
query is the impostor. Each verifier's threshold is set on genuine released
copies at 1% false rejection; the attack succeeds if the impostor passes.
"""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest
import sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim
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
ben = cases[(cases.split == "test") & (cases.protocol == "benign")].sample(400, random_state=0)
rows = []
for _, q in ben.iterrows():
    rel_img = Image.open(q.doctor_trigger_released_path).convert("RGB")
    e_rel = embed(q.doctor_trigger_released_path)
    for kind, img in [("released", q.input_path), ("original", q.enrollment_input_path)]:
        try:
            f, _ = feature_row(Path(img), Path(q.doctor_trigger_template_path))
        except ValueError:
            continue
        qa = np.asarray(Image.open(img).convert("RGB").resize(rel_img.size).convert("L"))
        ra = np.asarray(rel_img.convert("L"))
        e_q = embed(img)
        rows.append({"kind": kind, "H": f["H"], "G_s": f["G_s"], "ssim": ssim(qa, ra),
                     "arc": float(np.dot(e_rel, e_q)) if e_rel is not None and e_q is not None else np.nan})
d = pd.DataFrame(rows)
d.to_csv(_dt("DT_OUTPUTS", "/posthoc_attack.csv"), index=False)
gen = d[d.kind == "released"]; att = d[d.kind == "original"]
print(f"n: {len(gen)} genuine released copies, {len(att)} never-released originals")
print(f"{'verifier':26s} {'tau@1%FRR':>10s} {'genuine pass':>13s} {'attack pass':>12s}")
for col, name in [("H", "no-trigger/H reference"), ("ssim", "released-image SSIM"),
                  ("arc", "ArcFace identity ref"), ("G_s", "keyed projection G_s")]:
    g = gen.dropna(subset=[col])[col]; a = att.dropna(subset=[col])[col]
    tau = np.quantile(g, 0.01)
    print(f"{name:26s} {tau:10.3f} {(g >= tau).mean():12.1%} {(a >= tau).mean():11.1%}")
