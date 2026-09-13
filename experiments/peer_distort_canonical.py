import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
"""Peer robustness to benign distortions on the canonical protocol (FF++ test).

Embeds the peer's mark into the canonical enrollment bytes, makes the recorded
SimSwap re-rendering once, then scores both under each benign-channel family.
Usage: peer_distort_canonical.py <method>
"""
import sys, os
import numpy as np, pandas as pd, torch
from PIL import Image
sys.path.insert(0, _dt("DT_PEERS_ROOT"))
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from scripts.benign_channels import transform_image

METHOD = sys.argv[1]
sys.argv = ["x", METHOD, "ffpp"]
import peer_robust
from engines_api import ENGINES
from sklearn.metrics import roc_auc_score

R = _dt("DT_CANON_ROOT")
cases = pd.read_csv(f"{R}/common/cases.csv")
cases = cases[(cases.split == "test") & (cases.dataset == "FF++")]
peer_robust.DS = "ffpp"
embed, score = peer_robust.load()
swap = ENGINES["simswap"]()
PERTS = ["clean","jpeg_q90","jpeg_q70","jpeg_q50","blur_k7","x264_crf30","x264_crf35","x264_crf40","scale_0.5"]
FAM = {"JPEG":["jpeg_q90","jpeg_q70","jpeg_q50"],"blur":["blur_k7"],
       "codec":["x264_crf30","x264_crf35","x264_crf40"],"pixel":["scale_0.5"]}

pairs, drop = [], 0
for eid, g in cases.groupby("enrollment_id"):
    try:
        wm = embed(Image.open(g.enrollment_input_path.iloc[0]).convert("RGB"))
        src = g[g.protocol == "swap"].attack_source_path.iloc[0]
        sw = swap(wm, Image.open(src).convert("RGB")).resize(wm.size)
        pairs.append((wm, sw))
    except Exception:
        drop += 1
res = {}
for p in PERTS:
    ys, ss = [], []
    for k, (b, sw) in enumerate(pairs):
        try:
            ss.append(score(transform_image(b, p, seed=(k+17)*1009))); ys.append(0)
            ss.append(score(transform_image(sw, p, seed=(k+17)*1009))); ys.append(1)
        except Exception:
            continue
    ss = np.nan_to_num(np.array(ss, float), posinf=1e12, neginf=-1e12)
    res[p] = roc_auc_score(ys, ss)
print(f"== {METHOD} canonical-distort n={len(pairs)} dropped={drop}")
for fam, ps in FAM.items():
    v = [res[x] for x in ps]
    print(f"   {fam:6s} min={min(v):.4f} mean={np.mean(v):.4f} max={max(v):.4f}")
print(f"   clean  {res['clean']:.4f}")
