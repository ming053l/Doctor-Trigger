#!/usr/bin/env python
"""Phase-ring transplantation against the image-only detector and the fusion alarm (appendix Table 18).

Rebuilds the transplant of ring_transplant.py (manipulated magnitude, released phase on the record's
active bins, per RGB channel, rounded to 8 bits) and scores released copy, manipulated query and
transplant with: the relation error 1 - C_B over sub-band B, the FF++-trained image-only detector
S_df, and the alarm max(z_rel, min(z_df, 4)) with robust z-scores from 200 train benign releases of
the target corpus. Each flag sits at 1% FPR on those train releases. Saves four side-by-side examples.

Paths: config.yaml (queries_csv, outputs_root, model_snapshot, checkpoints.image_detector_aug);
optional env DT_COPY_N (number of test queries, default 200) and DT_COPY_DATASET (target corpus,
default FF++; other corpora get a file suffix such as ``_dfdc``).
Writes $DT_OUTPUTS/copy_ring/{detector_per_query,detector_summary}<suffix>.csv and examples<suffix>.png.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from doctor_trigger import config
from doctor_trigger.manifests import read_queries, select
from doctor_trigger.metrics import robust_z, roc_auc
from doctor_trigger.readings import SumForm, spectral_agreement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase, subband_b
from experiments.shared.image_only_detector import checkpoint_path, image_only_score, load_image_only_detector

OUT = Path(config.path("outputs_root")) / "copy_ring"
TEST_COUNT = int(os.environ.get("DT_COPY_N", "200"))
CALIBRATION_COUNT = 200
DATASET = os.environ.get("DT_COPY_DATASET", "FF++")      # target corpus; the detector stays FF++-trained
SUFFIX = "" if DATASET == "FF++" else "_" + DATASET.lower().replace("+", "p")
SUBBAND_STABILIZER = 1e-8
EXAMPLE_COUNT = 4

# the FF++-trained deployed detector (IMAGE_DETECTOR_CKPT is not consulted here)
image_detector = load_image_only_detector(checkpoint_path("FF++", augmented=True))


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def transplant(query, donor, mask_centered):
    """``query`` with the donor's phase on ``mask_centered`` (centered layout), per RGB channel, 8-bit."""
    mask = np.fft.ifftshift(mask_centered)
    out = np.empty_like(query)
    for channel in range(3):
        spectrum = np.fft.fft2(query[:, :, channel])
        donor_spectrum = np.fft.fft2(donor[:, :, channel])
        spectrum[mask] = np.abs(spectrum[mask]) * np.exp(1j * np.angle(donor_spectrum[mask]))
        out[:, :, channel] = np.fft.ifft2(spectrum).real
    return np.clip(np.round(out), 0, 255)


def relation_error(image, record):
    """1 - C_B against the stored template, summed in the unshifted layout."""
    agreement = spectral_agreement(
        luminance_phase(image, centered=False),
        np.fft.ifftshift(record.template_phase),
        np.fft.ifftshift(record.amplitude_weights),
        np.fft.ifftshift(record.active_bins) & subband_b(record.shape, centered=False),
        stabilizer=SUBBAND_STABILIZER,
        form=SumForm.WEIGHT_MAP,
    )
    return 1.0 - agreement


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queries = read_queries()
    test = select(queries, dataset=DATASET, split="test", protocol="swap").head(TEST_COUNT)
    calibration = select(queries, dataset=DATASET, split="train", protocol="benign").head(CALIBRATION_COUNT)

    calibration_df, calibration_rel = [], []
    for query in calibration.itertuples():
        released = rgb(query.released_path)
        calibration_df.append(image_only_score(image_detector, released))
        calibration_rel.append(relation_error(released, load_record(query.template_artifact)))
    calibration_df, calibration_rel = np.array(calibration_df), np.array(calibration_rel)
    calibration_alarm = np.maximum(robust_z(calibration_rel, calibration_rel),
                                   np.minimum(robust_z(calibration_df, calibration_df), 4))
    tau = dict(rel=np.quantile(calibration_rel, 0.99), df=np.quantile(calibration_df, 0.99),
               alarm=np.quantile(calibration_alarm, 0.99))

    rows, examples = [], []
    for k, query in enumerate(test.itertuples()):
        record = load_record(query.template_artifact)
        released, manipulated = rgb(query.released_path), rgb(query.query_path)
        transplanted = transplant(manipulated, released, record.active_bins)
        if k < EXAMPLE_COUNT:   # crops differ in size across corpora: show each at 256 px high
            strip = np.concatenate([released, manipulated, transplanted], axis=1).astype(np.uint8)
            height, width = strip.shape[:2]
            examples.append(np.asarray(Image.fromarray(strip).resize((round(width * 256 / height / 3) * 3, 256))))
        for condition, image in (("released", released), ("manipulated", manipulated), ("copy_ring", transplanted)):
            rows.append(dict(query_id=query.query_id, condition=condition, rel_err=relation_error(image, record),
                             S_df=image_only_score(image_detector, image)))
    per_query = pd.DataFrame(rows)
    per_query["alarm"] = np.maximum(robust_z(per_query.rel_err, calibration_rel),
                                    np.minimum(robust_z(per_query.S_df, calibration_df), 4))
    per_query["flag_rel"] = per_query.rel_err > tau["rel"]
    per_query["flag_df"] = per_query.S_df > tau["df"]
    per_query["flag_alarm"] = per_query.alarm > tau["alarm"]
    per_query["flag_or"] = per_query.flag_rel | per_query.flag_df
    per_query.to_csv(OUT / f"detector_per_query{SUFFIX}.csv", index=False)

    summary = per_query.groupby("condition").agg(
        n=("query_id", "size"), S_df_med=("S_df", "median"), flag_rel=("flag_rel", "mean"),
        flag_df=("flag_df", "mean"), flag_or=("flag_or", "mean"),
        flag_alarm=("flag_alarm", "mean")).reindex(["released", "manipulated", "copy_ring"])
    benign = per_query[per_query.condition == "released"]
    for condition in ("manipulated", "copy_ring"):
        attacked = per_query[per_query.condition == condition]
        labels = np.r_[np.zeros(len(benign)), np.ones(len(attacked))]
        for score in ("rel_err", "S_df", "alarm"):
            summary.loc[condition, f"AUC_{score}"] = roc_auc(labels, np.r_[benign[score], attacked[score]])
    summary.to_csv(OUT / f"detector_summary{SUFFIX}.csv")
    Image.fromarray(np.concatenate(examples, axis=0).astype(np.uint8)).save(OUT / f"examples{SUFFIX}.png")
    print(f"dataset {DATASET}, n={len(test)}")
    print("thresholds (1% train-benign FPR):", {k: round(float(v), 4) for k, v in tau.items()})
    with pd.option_context("display.width", 200):
        print(summary.round(4).to_string())
    print("=== COPY RING DETECTOR DONE ===")


if __name__ == "__main__":
    main()
