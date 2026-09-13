"""Video transcode, clip-level verdict (Table 5 x264 row).

Frame-level reading under x264 ranks well but the 1%-FRR operating point sits
below the swap body, so almost nothing is flagged. A transcode channel returns
a clip, not a still: every enrolled frame of the video passes through the same
codec, and the deployment decides per clip. Clip score = mean sub-band H over
the video's enrolled frames; tau at 1% clip-level FRR on train-benign clips.

Usage: x264_clip_verdict.py <split>:<shard>/<nshards>   (split in {train,test})
Writes per-clip scores to $DT_OUT/x264_clip_<split>_<shard>.csv;
run collect after all shards: x264_clip_verdict.py collect
"""
import os as _dtos, sys
def _dt(env, rest=""):
    v = _dtos.environ.get(env); assert v, f"{env} unset"; return v + rest
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from scripts.benign_channels import transform_image
R = _dt("DT_CANON_ROOT"); OUT = Path(_dtos.environ.get("DT_OUT", "."))
EPS = 1e-12; CUT = 0.15
cases = pd.read_csv(f"{R}/common/cases.csv")

def lum_of(img):
    a = np.asarray(img.convert("RGB"), np.float64)
    return 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
def wrap(v): return (v + np.pi) % (2*np.pi) - np.pi
def H_sub(img, tp):
    with np.load(tp) as rec:
        tpz = rec["template_phase"].astype(np.float64)
        sup = rec["support"].astype(bool); w = rec["host_amplitude"].astype(np.float64)
    ph = np.fft.fftshift(np.angle(np.fft.fft2(lum_of(img))))
    fy = np.fft.fftshift(np.fft.fftfreq(tpz.shape[0])).reshape(-1, 1)
    fx = np.fft.fftshift(np.fft.fftfreq(tpz.shape[1])).reshape(1, -1)
    rad = np.sqrt(fy*fy + fx*fx) / np.sqrt(0.5)
    ww = np.where(sup & (rad <= CUT), w, 0.0)
    return float(np.sum(np.cos(wrap(ph - tpz))*ww) / (np.sum(ww) + EPS))

if sys.argv[1] == "collect":
    tr = pd.concat([pd.read_csv(f) for f in OUT.glob("x264_clip_train_*.csv")])
    tebn = pd.concat([pd.read_csv(f) for f in OUT.glob("x264_clip_test_*.csv")])
    tau = float(np.quantile(tr[tr.protocol == "benign"].groupby("video").H.mean(), 0.01))
    gb = tebn[tebn.protocol == "benign"].groupby("video").H.mean()
    fb = tebn[tebn.protocol == "swap"].groupby("video").H.mean()
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score([0]*len(gb)+[1]*len(fb), np.concatenate([-gb.values, -fb.values]))
    print(f"clips: {len(gb)} genuine / {len(fb)} swapped | tau(clip,1%FRR,train n={tr[tr.protocol=='benign'].video.nunique()}): {tau:.4f}")
    print(f"benign clip H {gb.mean():.3f}+-{gb.std():.3f} | swap clip H {fb.mean():.3f}+-{fb.std():.3f}")
    print(f"preserved {np.mean(gb >= tau):.1%} | flagged {np.mean(fb < tau):.1%} | clip AUC {auc:.3f}")
    frame_g = tebn[tebn.protocol == "benign"].H; frame_f = tebn[tebn.protocol == "swap"].H
    ftau = float(np.quantile(tr[tr.protocol == "benign"].H, 0.01))
    print(f"[frame-level ref] preserved {np.mean(frame_g >= ftau):.1%} | flagged {np.mean(frame_f < ftau):.1%}")
    sys.exit(0)

split, shard = sys.argv[1].split(":")
k, n = map(int, shard.split("/"))
d = cases[(cases.split == split) & (cases.dataset == "FF++") & (cases.protocol.isin(["benign", "swap"]))]
vids = sorted(d.source_group_id.unique())
if split == "train": vids = vids[:100]
mine = [v for i, v in enumerate(vids) if i % n == k]
rows = []
for v in mine:
    for _, q in d[d.source_group_id == v].iterrows():
        t = transform_image(Image.open(q.input_path).convert("RGB"), "x264_crf30", seed=hash(q.query_id) % 65521)
        rows.append(dict(video=v, protocol=q.protocol, H=H_sub(t, q.doctor_trigger_template_path)))
pd.DataFrame(rows).to_csv(OUT / f"x264_clip_{split}_{k}.csv", index=False)
print(f"shard {k}/{n} {split}: {len(mine)} videos, {len(rows)} frames", flush=True)
