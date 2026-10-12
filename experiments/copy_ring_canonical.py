#!/usr/bin/env python
"""Copy-ring transplant under the main protocol.

Attack from the earlier ablation: keep the
manipulated image's Fourier magnitude and replace its phase on the ring by the released image's phase,
per RGB channel. Here the face region is the whole aligned crop, and the ring is either the record's
active bins (an attacker who knows them) or the full radial band those bins span. The result is rounded
to 8 bits before scoring.

Each FF++ test manipulated query is scored as released copy, as manipulated, and after each transplant:
closed-form H over the ring (host phase, as in the features) and over the declared sub-band (template
phase, radius <= 0.15, as in Table 1), G_s on the release scale, and the record model's S_rel/S_key
(FF++-trained, record-scalar features, three seeds). Thresholds sit at 1% FPR on FF++ train benign.

Paths: config.yaml (queries_csv, outputs_root, record_model_runs); optional env DT_COPY_N (default 200).
Writes $DT_OUTPUTS/copy_ring/{per_query.csv,summary.csv}.
"""
import os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402
from doctor_trigger.calibration import lower_tail_threshold  # noqa: E402
from doctor_trigger.manifests import read_queries, select  # noqa: E402
from doctor_trigger.readings import SumForm, normalized_pixel_radius, record_features, spectral_agreement  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from doctor_trigger.record_model import load_checkpoint, record_model_scores  # noqa: E402
from doctor_trigger.spectrum import luminance_phase, subband_b  # noqa: E402

OUTS = Path(config.path("outputs_root"))
OUT = OUTS / "copy_ring"
N = int(os.environ.get("DT_COPY_N", "200"))

CKPT = config.path("record_model_runs") + "/recordscalar_ffpp/checkpoints/full_record_model_seed{}_recordscalar_ffpp_to_all.pt"
TRAIN_FEAT = OUTS / "features_record_scalar/recordscalar_features_ffpp_train.csv"
C_FAMILY = ["C"] + [f"C_b{i}" for i in range(8)]
SCORE_ORDER = ("S_rel", "S_key", "S_fake")
"""Order of the record-model score columns in per_query.csv."""
SUBBAND_STABILIZER = 1e-8


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


def h_sub(ph_shifted, record):
    """C_B against the stored template (named H_sub in the CSV), summed in the unshifted layout."""
    return spectral_agreement(
        np.fft.ifftshift(ph_shifted),
        np.fft.ifftshift(record.template_phase),
        np.fft.ifftshift(record.amplitude_weights),
        np.fft.ifftshift(record.active_bins) & subband_b(record.shape, centered=False),
        stabilizer=SUBBAND_STABILIZER,
        form=SumForm.WEIGHT_MAP,
    )


def work(q):
    record = load_record(q["template_artifact"])
    sup = record.active_bins
    rel, man = rgb(q["released_path"]), rgb(q["query_path"])
    rad = normalized_pixel_radius(sup.shape)
    band = (rad >= rad[sup].min()) & (rad <= rad[sup].max())
    imgs = {"released": rel, "manipulated": man,
            "copy_active_bins": transplant(man, rel, sup), "copy_ring_band": transplant(man, rel, band)}
    rel_phase = luminance_phase(rel, centered=True)
    rows = []
    for name, a in imgs.items():
        ph = luminance_phase(a, centered=True)
        feats = record_features(ph, record, reference_phase=rel_phase)
        tmpl_ref = record_features(ph, record)        # C family against the stored template
        for c in C_FAMILY:
            feats[c] = tmpl_ref[c]
        mse = np.mean((a - man) ** 2)
        rows.append(dict(query_id=q["query_id"], condition=name, H_sub=h_sub(ph, record),
                         psnr_vs_manipulated=float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else np.inf,
                         ring_fraction=float(band.mean()), **feats))
    return rows


def cal_hsub(q):
    return h_sub(luminance_phase(rgb(q["released_path"]), centered=True), load_record(q["template_artifact"]))


def predict(df):
    """S_rel, S_key, S_fake averaged over the three seeds; NaN features read as 0 (as this script always has)."""
    outs = [record_model_scores(load_checkpoint(CKPT.format(s)), df, missing="zero") for s in range(3)]
    return {k: np.mean([o[k] for o in outs], axis=0) for k in SCORE_ORDER}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = read_queries()
    test = select(man, dataset="FF++", split="test", protocol="swap").head(N).to_dict("records")
    train = select(man, dataset="FF++", split="train", protocol="benign").head(200).to_dict("records")
    with Pool(16) as p:
        rows = [r for rs in p.map(work, test) for r in rs]
        hsub_cal = np.array(p.map(cal_hsub, train))
    df = pd.DataFrame(rows)
    sc = predict(df)
    for k, v in sc.items():
        df[k] = v

    tr = pd.read_csv(TRAIN_FEAT)
    trb = tr[tr.protocol == "benign"].reset_index(drop=True)
    trs = predict(trb)
    tau = dict(H=float(lower_tail_threshold(trb.H)), H_sub=float(lower_tail_threshold(hsub_cal)),
               S_rel=float(lower_tail_threshold(trs["S_rel"])), S_key=float(lower_tail_threshold(trs["S_key"])))
    print("thresholds at 1% train-benign FPR:", {k: round(v, 4) for k, v in tau.items()}, flush=True)

    df["acc_H"] = df.H >= tau["H"]
    df["acc_H_sub"] = df.H_sub >= tau["H_sub"]
    df["acc_S_rel"] = df.S_rel >= tau["S_rel"]
    df["acc_S_key"] = df.S_key >= tau["S_key"]
    df["acc_joint"] = df.acc_S_rel & df.acc_S_key
    df.to_csv(OUT / "per_query.csv", index=False)
    order = ["released", "manipulated", "copy_active_bins", "copy_ring_band"]
    summ = df.groupby("condition").agg(
        n=("query_id", "size"), H_med=("H", "median"), H_sub_med=("H_sub", "median"), G_s_med=("G_s", "median"),
        S_rel_mean=("S_rel", "mean"), S_key_mean=("S_key", "mean"), psnr_vs_manip_med=("psnr_vs_manipulated", "median"),
        accept_H=("acc_H", "mean"), accept_H_sub=("acc_H_sub", "mean"), key_pass=("acc_S_key", "mean"),
        rel_pass=("acc_S_rel", "mean"), accept_joint=("acc_joint", "mean")).reindex(order)
    summ.to_csv(OUT / "summary.csv")
    with pd.option_context("display.width", 200):
        print(summ.round(4).to_string(), flush=True)
    print("=== COPY RING DONE ===", flush=True)


if __name__ == "__main__":
    main()
