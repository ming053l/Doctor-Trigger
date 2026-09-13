"""Bootstrap 95% CIs for the headline AUCs, canonical test split (1,000 resamples)."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest
import numpy as np, pandas as pd, glob
from sklearn.metrics import roc_auc_score
R = _dt("DT_CANON_ROOT")
rng = np.random.RandomState(0)

def ci(y, s, n=1000):
    y, s = np.asarray(y), np.asarray(s)
    point = roc_auc_score(y, s)
    aucs = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(set(y[idx])) < 2: continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return point, lo, hi

feats = pd.read_csv(f"{R}/features/calibrated_features.csv", usecols=["query_id", "H"])
cases = pd.read_csv(f"{R}/common/cases.csv", usecols=["query_id", "dataset", "split", "y_fake"])
m = cases[cases.split == "test"].merge(feats, on="query_id")
image_detector = pd.read_csv(_dt("DT_OUTPUTS", "/image_detector_aug.csv"))
m = m.merge(image_detector[["query_id", "image_detector_ffpp", "image_detector_dfdc", "image_detector_celebdf"]], on="query_id")
IDM = {"FF++": "image_detector_ffpp", "DFDC": "image_detector_dfdc", "CelebDF": "image_detector_celebdf"}
print(f"{'dataset':9s} {'signal':22s} {'AUC':>7s}  95% CI")
for ds, g in m.groupby("dataset"):
    for name, score in [("relation 1-H", 1 - g.H),
                        ("image path (in-domain)", g[IDM[ds]]),
                        ("fusion max", np.maximum(g[IDM[ds]], 1 - g.H))]:
        pt, lo, hi = ci(g.y_fake, score)
        print(f"{ds:9s} {name:22s} {pt:7.4f}  [{lo:.4f}, {hi:.4f}]")
# record status S_rel per seed (FF++-trained, transfer)
for ds, tag in [("FF++", "ffpp"), ("DFDC", "dfdc"), ("CelebDF", "celebdf")]:
    pts, los, his = [], [], []
    for seed in [0, 1, 2]:
        f = glob.glob(_dtos.environ.get("DT_RECORD_MODEL_SCORES_DIR", f"{R}/results/record_model/scores")
                      + f"/*full_record_model_seed{seed}_{tag}_test_scores.csv")
        if not f: continue
        d = pd.read_csv(f[0])
        pt, lo, hi = ci(d.y_fake, d.S_fake)
        pts.append(pt); los.append(lo); his.append(hi)
    print(f"{ds:9s} {'record status S':22s} {np.mean(pts):7.4f}  [{np.mean(los):.4f}, {np.mean(his):.4f}] (3 seeds)")
