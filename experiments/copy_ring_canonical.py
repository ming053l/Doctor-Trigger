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

Env: DT_CANON_ROOT, DT_OUTPUTS, optional DT_COPY_N (default 200).
Writes $DT_OUTPUTS/copy_ring/{per_query.csv,summary.csv}.
"""
import os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

R = os.environ["DT_CANON_ROOT"]
OUTS = Path(os.environ["DT_OUTPUTS"])
OUT = OUTS / "copy_ring"
N = int(os.environ.get("DT_COPY_N", "200"))
sys.path.insert(0, R)
from scripts.extract_features import feature_row, radial_values  # noqa: E402  (pipeline code, unchanged)

CKPT = os.environ["DT_RECORD_MODEL_RUNS"] + "/recordscalar_ffpp/checkpoints/full_record_model_seed{}_recordscalar_ffpp_to_all.pt"
TRAIN_FEAT = OUTS / "features_record_scalar/recordscalar_features_ffpp_train.csv"
C_FAMILY = ["C"] + [f"C_b{i}" for i in range(8)]
CUT = 0.15


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


def h_sub(ph_shifted, rec):
    tm = np.fft.ifftshift(rec["template_phase"].astype(np.float64))
    sup = np.fft.ifftshift(rec["support"].astype(bool))
    amp = np.fft.ifftshift(rec["host_amplitude"].astype(np.float64))
    fy = np.fft.fftfreq(tm.shape[0]).reshape(-1, 1)
    fx = np.fft.fftfreq(tm.shape[1]).reshape(1, -1)
    w = np.where(sup & (np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5) <= CUT), amp, 0.0)
    r = np.cos((np.fft.ifftshift(ph_shifted) - tm + np.pi) % (2 * np.pi) - np.pi)
    return float(np.sum(r * w) / (np.sum(w) + 1e-8))


def work(q):
    rec = load_rec(q["template_artifact"])
    sup = rec["support"].astype(bool)
    rel, man = rgb(q["released_path"]), rgb(q["query_path"])
    rad = radial_values(sup.shape)
    band = (rad >= rad[sup].min()) & (rad <= rad[sup].max())
    imgs = {"released": rel, "manipulated": man,
            "copy_active_bins": transplant(man, rel, sup), "copy_ring_band": transplant(man, rel, band)}
    rel_phase = lum_phase(rel)
    tp, qp, rp = Path(q["template_artifact"]), Path(q["query_path"]), Path(q["released_path"])
    rows = []
    for name, a in imgs.items():
        ph = lum_phase(a)
        feats, _ = feature_row(qp, tp, reference_path=rp, phase_override=ph, reference_phase_override=rel_phase)
        tmpl_ref, _ = feature_row(qp, tp, phase_override=ph)        # C family against the stored template
        for c in C_FAMILY:
            feats[c] = tmpl_ref[c]
        mse = np.mean((a - man) ** 2)
        rows.append(dict(query_id=q["query_id"], condition=name, H_sub=h_sub(ph, rec),
                         psnr_vs_manipulated=float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else np.inf,
                         ring_fraction=float(band.mean()), **feats))
    return rows


def cal_hsub(q):
    return h_sub(lum_phase(rgb(q["released_path"])), load_rec(q["template_artifact"]))


def predict(df):
    import torch
    sys.path.insert(0, f"{R}/record_model")
    from src.record_model import RecordModelConfig, RecordModelMLP
    outs = []
    for s in range(3):
        ck = torch.load(CKPT.format(s), map_location="cpu", weights_only=False)
        m = RecordModelMLP(RecordModelConfig(variant=ck["variant"], input_dim=len(ck["feature_columns"]), hidden_dim=64))
        m.load_state_dict(ck["state_dict"]); m.eval()
        x = df[ck["feature_columns"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        x = (np.nan_to_num(x, nan=0.0) - ck["mean"]) / ck["std"]
        with torch.no_grad():
            o = m(torch.tensor(x.astype(np.float32)))
        outs.append({k: torch.sigmoid(o[k]).numpy() for k in ("S_rel", "S_key", "S_fake")})
    return {k: np.mean([o[k] for o in outs], axis=0) for k in ("S_rel", "S_key", "S_fake")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(f"{R}/manifests/queries.csv")
    man = man[man.status == "pass"]
    test = man[(man.dataset == "FF++") & (man.split == "test") & (man.protocol == "swap")].head(N).to_dict("records")
    train = man[(man.dataset == "FF++") & (man.split == "train") & (man.protocol == "benign")].head(200).to_dict("records")
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
    tau = dict(H=float(np.quantile(trb.H, 0.01)), H_sub=float(np.quantile(hsub_cal, 0.01)),
               S_rel=float(np.quantile(trs["S_rel"], 0.01)), S_key=float(np.quantile(trs["S_key"], 0.01)))
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
