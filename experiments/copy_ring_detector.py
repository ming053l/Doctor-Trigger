#!/usr/bin/env python
"""Copy-ring images against the image-only detector and the fusion alarm (FF++).

Rebuilds the transplant of copy_ring_canonical.py (manipulated magnitude, released phase on the record's
active bins, per RGB channel, rounded to 8 bits) and scores released copy, manipulated query and
transplant with: the relation error 1 - H over the declared sub-band, the FF++-trained image-only
detector S_df, and the alarm max(z_rel, min(z_df, 4)) with robust z-scores from 200 FF++ train benign
releases. Each flag sits at 1% FPR on those train releases. Saves four side-by-side examples.

Paths: config.yaml (queries_csv, outputs_root, model_snapshot, checkpoints.image_detector_aug);
optional env DT_COPY_N (default 200).
Writes $DT_OUTPUTS/copy_ring/{detector_per_query.csv,detector_summary.csv,examples.png}.
"""
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402
from doctor_trigger.manifests import read_queries, select  # noqa: E402
from doctor_trigger.metrics import robust_z, roc_auc  # noqa: E402
from doctor_trigger.readings import SumForm, spectral_agreement  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from doctor_trigger.spectrum import luminance_phase, subband_b  # noqa: E402
from experiments.shared.image_only_detector import checkpoint_path, image_only_score, load_image_only_detector  # noqa: E402

OUT = Path(config.path("outputs_root")) / "copy_ring"
N = int(os.environ.get("DT_COPY_N", "200"))
DS = os.environ.get("DT_COPY_DATASET", "FF++")      # target corpus; the detector stays FF++-trained
TAG = "" if DS == "FF++" else "_" + DS.lower().replace("+", "p")
SUBBAND_STABILIZER = 1e-8

# the FF++-trained deployed detector (IMAGE_DETECTOR_CKPT is not consulted here)
image_detector = load_image_only_detector(checkpoint_path("FF++", augmented=True))


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def transplant(query, donor, mask_shifted):
    m = np.fft.ifftshift(mask_shifted)
    out = np.empty_like(query)
    for c in range(3):
        q = np.fft.fft2(query[:, :, c])
        d = np.fft.fft2(donor[:, :, c])
        q[m] = np.abs(q[m]) * np.exp(1j * np.angle(d[m]))
        out[:, :, c] = np.fft.ifft2(q).real
    return np.clip(np.round(out), 0, 255)


def rel_err(a, record):
    """1 - C_B against the stored template, summed in the unshifted layout."""
    agreement = spectral_agreement(
        luminance_phase(a, centered=False),
        np.fft.ifftshift(record.template_phase),
        np.fft.ifftshift(record.amplitude_weights),
        np.fft.ifftshift(record.active_bins) & subband_b(record.shape, centered=False),
        stabilizer=SUBBAND_STABILIZER,
        form=SumForm.WEIGHT_MAP,
    )
    return 1.0 - agreement


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = read_queries()
    test = select(man, dataset=DS, split="test", protocol="swap").head(N)
    train = select(man, dataset=DS, split="train", protocol="benign").head(200)

    cal_d, cal_h = [], []
    for q in train.itertuples():
        a = rgb(q.released_path)
        cal_d.append(image_only_score(image_detector, a)); cal_h.append(rel_err(a, load_record(q.template_artifact)))
    cal_d, cal_h = np.array(cal_d), np.array(cal_h)
    cal_f = np.maximum(robust_z(cal_h, cal_h), np.minimum(robust_z(cal_d, cal_d), 4))
    tau = dict(rel=np.quantile(cal_h, 0.99), df=np.quantile(cal_d, 0.99), alarm=np.quantile(cal_f, 0.99))

    rows, examples = [], []
    for k, q in enumerate(test.itertuples()):
        rec = load_record(q.template_artifact)
        rel, mq = rgb(q.released_path), rgb(q.query_path)
        cp = transplant(mq, rel, rec.active_bins)
        if k < 4:   # crops differ in size across corpora: show each at 256 px high
            row = np.concatenate([rel, mq, cp], axis=1).astype(np.uint8)
            h, w = row.shape[:2]
            examples.append(np.asarray(Image.fromarray(row).resize((round(w * 256 / h / 3) * 3, 256))))
        for name, a in (("released", rel), ("manipulated", mq), ("copy_ring", cp)):
            rows.append(dict(query_id=q.query_id, condition=name, rel_err=rel_err(a, rec),
                             S_df=image_only_score(image_detector, a)))
    df = pd.DataFrame(rows)
    df["alarm"] = np.maximum(robust_z(df.rel_err, cal_h), np.minimum(robust_z(df.S_df, cal_d), 4))
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
            summ.loc[cond, f"AUC_{s}"] = roc_auc(y, np.r_[ben[s], g[s]])
    summ.to_csv(OUT / f"detector_summary{TAG}.csv")
    Image.fromarray(np.concatenate(examples, axis=0).astype(np.uint8)).save(OUT / f"examples{TAG}.png")
    print(f"dataset {DS}, n={len(test)}")
    print("thresholds (1% train-benign FPR):", {k: round(float(v), 4) for k, v in tau.items()})
    with pd.option_context("display.width", 200):
        print(summ.round(4).to_string())
    print("=== COPY RING DETECTOR DONE ===")


if __name__ == "__main__":
    main()
