#!/usr/bin/env python
"""Copy-ring images against the image-only detector and the fusion alarm (FF++).

Rebuilds the transplant of copy_ring_canonical.py (manipulated magnitude, released phase on the record's
active bins, per RGB channel, rounded to 8 bits) and scores released copy, manipulated query and
transplant with: the relation error 1 - H over the declared sub-band, the FF++-trained image-only
detector S_df, and the alarm max(z_rel, min(z_df, 4)) with robust z-scores from 200 FF++ train benign
releases. Each flag sits at 1% FPR on those train releases. Saves four side-by-side examples.

Env: DT_CANON_ROOT, DT_OUTPUTS, DT_MODEL_SNAPSHOT, DT_IMAGE_DETECTOR_AUG; optional DT_COPY_N (default 200).
Writes $DT_OUTPUTS/copy_ring/{detector_per_query.csv,detector_summary.csv,examples.png}.
"""
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score


def _dt(env, rest=""):
    v = os.environ.get(env)
    assert v, f"{env} unset"
    return v + rest


R = _dt("DT_CANON_ROOT")
OUT = Path(_dt("DT_OUTPUTS")) / "copy_ring"
N = int(os.environ.get("DT_COPY_N", "200"))
DS = os.environ.get("DT_COPY_DATASET", "FF++")      # target corpus; the detector stays FF++-trained
TAG = "" if DS == "FF++" else "_" + DS.lower().replace("+", "p")
CUT = 0.15

sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from src import models  # noqa: E402

cfg = yaml.safe_load(open(_dt("DT_MODEL_SNAPSHOT", "/config_ffpp_std_dfd_resnet50.yaml")))
image_detector = models.build_model(cfg).cuda()
image_detector.load_state_dict(torch.load(_dt("DT_IMAGE_DETECTOR_AUG", "/ffpp/best_val_auc.pt"), map_location="cuda", weights_only=False))
image_detector.eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])


@torch.no_grad()
def prob(a):
    o = image_detector(tf(Image.fromarray(a.astype(np.uint8))).unsqueeze(0).cuda())
    lg = o["logits"] if isinstance(o, dict) else o
    return float(torch.softmax(lg, 1)[0, 1]) if lg.ndim > 1 and lg.shape[1] == 2 else float(torch.sigmoid(lg.reshape(-1))[0])


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def lum_phase(a):
    y = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return np.fft.fftshift(np.angle(np.fft.fft2(y)))


def transplant(query, donor, mask_shifted):
    m = np.fft.ifftshift(mask_shifted)
    out = np.empty_like(query)
    for c in range(3):
        q = np.fft.fft2(query[:, :, c])
        d = np.fft.fft2(donor[:, :, c])
        q[m] = np.abs(q[m]) * np.exp(1j * np.angle(d[m]))
        out[:, :, c] = np.fft.ifft2(q).real
    return np.clip(np.round(out), 0, 255)


def load_rec(tp):
    with np.load(tp) as rec:
        return {k: rec[k] for k in ("support", "template_phase", "host_amplitude")}


def rel_err(a, rec):
    tm = np.fft.ifftshift(rec["template_phase"].astype(np.float64))
    sup = np.fft.ifftshift(rec["support"].astype(bool))
    amp = np.fft.ifftshift(rec["host_amplitude"].astype(np.float64))
    fy = np.fft.fftfreq(tm.shape[0]).reshape(-1, 1)
    fx = np.fft.fftfreq(tm.shape[1]).reshape(1, -1)
    w = np.where(sup & (np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5) <= CUT), amp, 0.0)
    r = np.cos((np.fft.ifftshift(lum_phase(a)) - tm + np.pi) % (2 * np.pi) - np.pi)
    return 1.0 - float(np.sum(r * w) / (np.sum(w) + 1e-8))


def rz(x, cal):
    cal = np.asarray(cal, dtype=np.float64)
    med = np.median(cal)
    mad = np.median(np.abs(cal - med)) + 1e-6
    return (np.asarray(x, dtype=np.float64) - med) / mad


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(f"{R}/manifests/queries.csv")
    man = man[man.status == "pass"]
    test = man[(man.dataset == DS) & (man.split == "test") & (man.protocol == "swap")].head(N)
    train = man[(man.dataset == DS) & (man.split == "train") & (man.protocol == "benign")].head(200)

    cal_d, cal_h = [], []
    for q in train.itertuples():
        a = rgb(q.released_path)
        cal_d.append(prob(a)); cal_h.append(rel_err(a, load_rec(q.template_artifact)))
    cal_d, cal_h = np.array(cal_d), np.array(cal_h)
    cal_f = np.maximum(rz(cal_h, cal_h), np.minimum(rz(cal_d, cal_d), 4))
    tau = dict(rel=np.quantile(cal_h, 0.99), df=np.quantile(cal_d, 0.99), alarm=np.quantile(cal_f, 0.99))

    rows, examples = [], []
    for k, q in enumerate(test.itertuples()):
        rec = load_rec(q.template_artifact)
        rel, mq = rgb(q.released_path), rgb(q.query_path)
        cp = transplant(mq, rel, rec["support"].astype(bool))
        if k < 4:   # crops differ in size across corpora: show each at 256 px high
            row = np.concatenate([rel, mq, cp], axis=1).astype(np.uint8)
            h, w = row.shape[:2]
            examples.append(np.asarray(Image.fromarray(row).resize((round(w * 256 / h / 3) * 3, 256))))
        for name, a in (("released", rel), ("manipulated", mq), ("copy_ring", cp)):
            rows.append(dict(query_id=q.query_id, condition=name, rel_err=rel_err(a, rec), S_df=prob(a)))
    df = pd.DataFrame(rows)
    df["alarm"] = np.maximum(rz(df.rel_err, cal_h), np.minimum(rz(df.S_df, cal_d), 4))
    df["flag_rel"] = df.rel_err > tau["rel"]
    df["flag_df"] = df.S_df > tau["df"]
    df["flag_alarm"] = df.alarm > tau["alarm"]
    df["flag_or"] = df.flag_rel | df.flag_df
    df.to_csv(OUT / f"detector_per_query{TAG}.csv", index=False)

    summ = df.groupby("condition").agg(n=("query_id", "size"), S_df_med=("S_df", "median"),
                                       flag_rel=("flag_rel", "mean"), flag_df=("flag_df", "mean"),
                                       flag_or=("flag_or", "mean"),
                                       flag_alarm=("flag_alarm", "mean")).reindex(["released", "manipulated", "copy_ring"])
    ben = df[df.condition == "released"]
    for cond in ("manipulated", "copy_ring"):
        g = df[df.condition == cond]
        y = np.r_[np.zeros(len(ben)), np.ones(len(g))]
        for s in ("rel_err", "S_df", "alarm"):
            summ.loc[cond, f"AUC_{s}"] = roc_auc_score(y, np.r_[ben[s], g[s]])
    summ.to_csv(OUT / f"detector_summary{TAG}.csv")
    Image.fromarray(np.concatenate(examples, axis=0).astype(np.uint8)).save(OUT / f"examples{TAG}.png")
    print(f"dataset {DS}, n={len(test)}")
    print("thresholds (1% train-benign FPR):", {k: round(float(v), 4) for k, v in tau.items()})
    with pd.option_context("display.width", 200):
        print(summ.round(4).to_string())
    print("=== COPY RING DETECTOR DONE ===")


if __name__ == "__main__":
    main()
