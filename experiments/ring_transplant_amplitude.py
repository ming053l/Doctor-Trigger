#!/usr/bin/env python
"""Does a ring-amplitude check catch phase-ring transplantation, and what does a full copy cost?

The phase-only transplant (ring_transplant.py) keeps the manipulated image's Fourier magnitude on the
ring. The record already stores the original amplitude there (the weights w_k), so an amplitude
check can compare the query's ring magnitude with it: the correlation of log magnitudes over the
record's active bins and over their part in sub-band B (kappa <= 0.15). A full transplant copies
magnitude and phase, which passes that check by construction but writes the released face's ring
content back into the manipulated image; its cost is reported as PSNR against the manipulated image
and against the release.

Per corpus (FF++, CelebDF, DFDC): 200 test manipulated queries, thresholds at 1% FPR on 200 train
releases. Paths: config.yaml (queries_csv, outputs_root); optional env DT_COPY_N (number of test
queries per corpus, default 200). Writes $DT_OUTPUTS/copy_ring/amplitude_{per_query,summary}.csv.
"""
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from doctor_trigger import config
from doctor_trigger.manifests import read_queries, select
from doctor_trigger.readings import spectral_agreement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_spectrum, subband_b

OUT = Path(config.path("outputs_root")) / "copy_ring"
TEST_COUNT = int(os.environ.get("DT_COPY_N", "200"))
CALIBRATION_COUNT = 200
DATASETS = ("FF++", "CelebDF", "DFDC")
CONDITIONS = ["released", "manipulated", "copy_phase", "copy_phase_and_magnitude"]


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def transplant(query, donor, mask, magnitude):
    """``query`` with the donor's spectrum on ``mask`` (unshifted layout), per RGB channel, 8-bit.

    ``magnitude=False`` copies the phase only; ``True`` copies magnitude and phase.
    """
    out = np.empty_like(query)
    for channel in range(3):
        spectrum = np.fft.fft2(query[:, :, channel])
        donor_spectrum = np.fft.fft2(donor[:, :, channel])
        spectrum[mask] = (donor_spectrum[mask] if magnitude
                          else np.abs(spectrum[mask]) * np.exp(1j * np.angle(donor_spectrum[mask])))
        out[:, :, channel] = np.fft.ifft2(spectrum).real
    return np.clip(np.round(out), 0, 255)


def readings(image, record):
    """Relation error 1 - C_B and the two amplitude errors (1 - log-magnitude correlation)."""
    spectrum = luminance_spectrum(image)
    agreement = spectral_agreement(np.angle(spectrum), record["template"], record["amplitude"],
                                   record["subband"])
    out = {"rel_err": 1.0 - agreement}
    log_amplitude = np.log1p(np.abs(spectrum))
    for name, mask in (("amp_err_ring", record["active"]), ("amp_err_sub", record["subband"])):
        out[name] = 1.0 - float(np.corrcoef(log_amplitude[mask], record["log_amplitude"][mask])[0, 1])
    return out


def load_unshifted_record(path):
    """Record arrays in the unshifted layout, with K_eff and its part in sub-band B."""
    record = load_record(path)
    active = np.fft.ifftshift(record.active_bins)
    amplitude = np.fft.ifftshift(record.amplitude_weights)
    template = np.fft.ifftshift(record.template_phase)
    subband = active & subband_b(record.shape, centered=False)
    return dict(active=active, subband=subband, amplitude=amplitude, log_amplitude=np.log1p(amplitude),
                template=template)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else np.inf


def test_rows(query):
    record = load_unshifted_record(query["template_artifact"])
    released, manipulated = rgb(query["released_path"]), rgb(query["query_path"])
    images = {"released": released, "manipulated": manipulated,
              "copy_phase": transplant(manipulated, released, record["active"], False),
              "copy_phase_and_magnitude": transplant(manipulated, released, record["active"], True)}
    return [dict(dataset=query["dataset"], query_id=query["query_id"], condition=condition,
                 psnr_vs_manipulated=psnr(image, manipulated), psnr_vs_released=psnr(image, released),
                 **readings(image, record)) for condition, image in images.items()]


def calibration_row(query):
    return dict(dataset=query["dataset"],
                **readings(rgb(query["released_path"]), load_unshifted_record(query["template_artifact"])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queries = read_queries()
    tests, calibration = [], []
    for dataset in DATASETS:
        tests += select(queries, dataset=dataset, split="test", protocol="swap").head(TEST_COUNT).to_dict("records")
        calibration += select(queries, dataset=dataset, split="train",
                              protocol="benign").head(CALIBRATION_COUNT).to_dict("records")
    with Pool(24) as pool:
        per_query = pd.DataFrame([r for rows in pool.map(test_rows, tests) for r in rows])
        calibration = pd.DataFrame(pool.map(calibration_row, calibration))
    scores = ["rel_err", "amp_err_ring", "amp_err_sub"]
    tau = calibration.groupby("dataset")[scores].quantile(0.99)
    print("thresholds (1% train-benign FPR):\n" + tau.round(5).to_string(), flush=True)
    for score in scores:
        per_query[f"flag_{score}"] = per_query[score] > per_query.dataset.map(tau[score])
    per_query["flag_rel_or_amp_ring"] = per_query.flag_rel_err | per_query.flag_amp_err_ring
    per_query["flag_rel_or_amp_sub"] = per_query.flag_rel_err | per_query.flag_amp_err_sub
    per_query.to_csv(OUT / "amplitude_per_query.csv", index=False)
    flags = [c for c in per_query.columns if c.startswith("flag_")]
    summary = per_query.groupby(["dataset", "condition"]).agg(
        n=("query_id", "size"), psnr_vs_manipulated=("psnr_vs_manipulated", "median"),
        psnr_vs_released=("psnr_vs_released", "median"), **{f: (f, "mean") for f in flags})
    summary = summary.reindex(pd.MultiIndex.from_product([list(DATASETS), CONDITIONS]))
    summary.to_csv(OUT / "amplitude_summary.csv")
    with pd.option_context("display.width", 250):
        print(summary.round(3).to_string(), flush=True)
    print("=== COPY RING AMPLITUDE DONE ===", flush=True)


if __name__ == "__main__":
    main()
