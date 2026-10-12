#!/usr/bin/env python
"""Phase-ring transplantation under the main protocol (appendix Table 18, template and record-model rows).

The attack keeps the manipulated image's Fourier magnitude and replaces its phase on the ring by the
released image's phase, per RGB channel. The face region is the whole aligned crop, and the ring is
either the record's active bins (an attacker who knows them) or the full radial band those bins
span. The result is rounded to 8 bits before scoring.

Each FF++ test manipulated query is scored as released copy, as manipulated, and after each
transplant: the closed-form agreement H over the active bins (original phase, as in the features),
the sub-band agreement C_B (template phase, kappa <= 0.15, as in Table 1; CSV column ``H_sub``),
G_s on the release scale, and the record model's S_rel/S_key (FF++-trained, record-scalar
features, three seeds). Thresholds sit at 1% FPR on FF++ train benign queries.

Paths: config.yaml (queries_csv, outputs_root, record_model_runs); optional env DT_COPY_N (number of
test queries, default 200). Writes $DT_OUTPUTS/copy_ring/{per_query.csv,summary.csv}.
"""
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from doctor_trigger import config
from doctor_trigger.calibration import lower_tail_threshold
from doctor_trigger.manifests import read_queries, select
from doctor_trigger.readings import normalized_pixel_radius, record_features, spectral_agreement, subband_bins
from doctor_trigger.record import load_record
from doctor_trigger.record_model import load_checkpoint, record_model_scores
from doctor_trigger.spectrum import luminance_phase

OUTPUTS = Path(config.path("outputs_root"))
OUT = OUTPUTS / "copy_ring"
TEST_COUNT = int(os.environ.get("DT_COPY_N", "200"))
CALIBRATION_COUNT = 200

CHECKPOINT = (config.path("record_model_runs")
              + "/recordscalar_ffpp/checkpoints/full_record_model_seed{}_recordscalar_ffpp_to_all.pt")
TRAIN_FEATURES = OUTPUTS / "features_record_scalar/recordscalar_features_ffpp_train.csv"
C_FAMILY = ["C"] + [f"C_b{i}" for i in range(8)]
SCORE_ORDER = ("S_rel", "S_key", "S_fake")
"""Order of the record-model score columns in per_query.csv."""
CONDITIONS = ["released", "manipulated", "copy_active_bins", "copy_ring_band"]


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


def subband_agreement(phase_centered, record):
    """C_B against the stored template (CSV column ``H_sub``)."""
    return spectral_agreement(phase_centered, record.template_phase, record.amplitude_weights, subband_bins(record))


def test_rows(query):
    """One row per condition for a manipulated test query."""
    record = load_record(query["template_artifact"])
    active = record.active_bins
    released, manipulated = rgb(query["released_path"]), rgb(query["query_path"])
    radius = normalized_pixel_radius(active.shape)
    band = (radius >= radius[active].min()) & (radius <= radius[active].max())
    images = {"released": released, "manipulated": manipulated,
              "copy_active_bins": transplant(manipulated, released, active),
              "copy_ring_band": transplant(manipulated, released, band)}
    released_phase = luminance_phase(released, centered=True)
    rows = []
    for condition, image in images.items():
        phase = luminance_phase(image, centered=True)
        features = record_features(phase, record, reference_phase=released_phase)
        template_reference = record_features(phase, record)        # C family against the stored template
        for c in C_FAMILY:
            features[c] = template_reference[c]
        mse = np.mean((image - manipulated) ** 2)
        rows.append(dict(query_id=query["query_id"], condition=condition, H_sub=subband_agreement(phase, record),
                         psnr_vs_manipulated=float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else np.inf,
                         ring_fraction=float(band.mean()), **features))
    return rows


def calibration_subband_agreement(query):
    return subband_agreement(luminance_phase(rgb(query["released_path"]), centered=True),
                             load_record(query["template_artifact"]))


def predict(features):
    """S_rel, S_key, S_fake averaged over the three seeds.

    NaN features are read as 0 here (``missing="zero"``), whereas the trainer and the other
    experiments fill them with the training mean. The two differ only on rows with a NaN feature;
    this script keeps 0 so its published outputs stay reproducible.
    """
    outputs = [record_model_scores(load_checkpoint(CHECKPOINT.format(seed)), features, missing="zero")
               for seed in range(3)]
    return {name: np.mean([o[name] for o in outputs], axis=0) for name in SCORE_ORDER}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queries = read_queries()
    test = select(queries, dataset="FF++", split="test", protocol="swap").head(TEST_COUNT).to_dict("records")
    calibration = select(queries, dataset="FF++", split="train", protocol="benign").head(CALIBRATION_COUNT)
    with Pool(16) as pool:
        rows = [r for query_rows in pool.map(test_rows, test) for r in query_rows]
        calibration_subband = np.array(pool.map(calibration_subband_agreement, calibration.to_dict("records")))
    per_query = pd.DataFrame(rows)
    for name, values in predict(per_query).items():
        per_query[name] = values

    train = pd.read_csv(TRAIN_FEATURES)
    train_benign = train[train.protocol == "benign"].reset_index(drop=True)
    train_scores = predict(train_benign)
    tau = dict(H=float(lower_tail_threshold(train_benign.H)),
               H_sub=float(lower_tail_threshold(calibration_subband)),
               S_rel=float(lower_tail_threshold(train_scores["S_rel"])),
               S_key=float(lower_tail_threshold(train_scores["S_key"])))
    print("thresholds at 1% train-benign FPR:", {k: round(v, 4) for k, v in tau.items()}, flush=True)

    per_query["acc_H"] = per_query.H >= tau["H"]
    per_query["acc_H_sub"] = per_query.H_sub >= tau["H_sub"]
    per_query["acc_S_rel"] = per_query.S_rel >= tau["S_rel"]
    per_query["acc_S_key"] = per_query.S_key >= tau["S_key"]
    per_query["acc_joint"] = per_query.acc_S_rel & per_query.acc_S_key
    per_query.to_csv(OUT / "per_query.csv", index=False)
    summary = per_query.groupby("condition").agg(
        n=("query_id", "size"), H_med=("H", "median"), H_sub_med=("H_sub", "median"), G_s_med=("G_s", "median"),
        S_rel_mean=("S_rel", "mean"), S_key_mean=("S_key", "mean"),
        psnr_vs_manip_med=("psnr_vs_manipulated", "median"),
        accept_H=("acc_H", "mean"), accept_H_sub=("acc_H_sub", "mean"), key_pass=("acc_S_key", "mean"),
        rel_pass=("acc_S_rel", "mean"), accept_joint=("acc_joint", "mean")).reindex(CONDITIONS)
    summary.to_csv(OUT / "summary.csv")
    with pd.option_context("display.width", 200):
        print(summary.round(4).to_string(), flush=True)
    print("=== COPY RING DONE ===", flush=True)


if __name__ == "__main__":
    main()
