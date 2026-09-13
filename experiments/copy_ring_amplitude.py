#!/usr/bin/env python
"""Does a ring-amplitude check catch the copy-ring transplant, and what does a full copy cost?

The phase-only transplant (copy_ring_canonical.py) keeps the manipulated image's Fourier magnitude on the
ring. The record already stores the host amplitude there, so an amplitude check can compare the query's
ring magnitude with it: the correlation of log magnitudes over the record's active bins, over the full ring
and over the declared sub-band (radius <= 0.15). A full transplant copies magnitude and phase, which passes
that check by construction but writes the released face's ring content back into the manipulated image;
its cost is reported as PSNR against the manipulated image and against the release.

Per corpus (FF++, CelebDF, DFDC): 200 test manipulated queries, thresholds at 1% FPR on 200 train releases.
Env: DT_CANON_ROOT, DT_OUTPUTS, optional DT_COPY_N. Writes $DT_OUTPUTS/copy_ring/amplitude_{per_query,summary}.csv.
"""
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

R = os.environ["DT_CANON_ROOT"]
OUT = Path(os.environ["DT_OUTPUTS"]) / "copy_ring"
N = int(os.environ.get("DT_COPY_N", "200"))
CUT = 0.15


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def lum_fft(a):
    return np.fft.fft2(0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2])


def transplant(query, donor, m, magnitude):
    out = np.empty_like(query)
    for c in range(3):
        q = np.fft.fft2(query[:, :, c])
        d = np.fft.fft2(donor[:, :, c])
        q[m] = d[m] if magnitude else np.abs(q[m]) * np.exp(1j * np.angle(d[m]))
        out[:, :, c] = np.fft.ifft2(q).real
    return np.clip(np.round(out), 0, 255)


def scores(a, rec):
    f = lum_fft(a)
    r = np.cos((np.angle(f) - rec["tm"] + np.pi) % (2 * np.pi) - np.pi)
    w = np.where(rec["sub"], rec["amp"], 0.0)
    out = {"rel_err": 1.0 - float(np.sum(r * w) / (np.sum(w) + 1e-8))}
    la = np.log1p(np.abs(f))
    for name, m in (("amp_err_ring", rec["sup"]), ("amp_err_sub", rec["sub"])):
        out[name] = 1.0 - float(np.corrcoef(la[m], rec["lamp"][m])[0, 1])
    return out


def load_rec(tp):
    with np.load(tp) as z:
        sup = np.fft.ifftshift(z["support"].astype(bool))
        amp = np.fft.ifftshift(z["host_amplitude"].astype(np.float64))
        tm = np.fft.ifftshift(z["template_phase"].astype(np.float64))
    fy = np.fft.fftfreq(sup.shape[0]).reshape(-1, 1)
    fx = np.fft.fftfreq(sup.shape[1]).reshape(1, -1)
    sub = sup & (np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5) <= CUT)
    return dict(sup=sup, sub=sub, amp=amp, lamp=np.log1p(amp), tm=tm)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else np.inf


def work_test(q):
    rec = load_rec(q["template_artifact"])
    rel, man = rgb(q["released_path"]), rgb(q["query_path"])
    imgs = {"released": rel, "manipulated": man,
            "copy_phase": transplant(man, rel, rec["sup"], False),
            "copy_phase_and_magnitude": transplant(man, rel, rec["sup"], True)}
    return [dict(dataset=q["dataset"], query_id=q["query_id"], condition=k, psnr_vs_manipulated=psnr(a, man),
                 psnr_vs_released=psnr(a, rel), **scores(a, rec)) for k, a in imgs.items()]


def work_cal(q):
    return dict(dataset=q["dataset"], **scores(rgb(q["released_path"]), load_rec(q["template_artifact"])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(f"{R}/manifests/queries.csv")
    man = man[man.status == "pass"]
    tests, cals = [], []
    for ds in ("FF++", "CelebDF", "DFDC"):
        tests += man[(man.dataset == ds) & (man.split == "test") & (man.protocol == "swap")].head(N).to_dict("records")
        cals += man[(man.dataset == ds) & (man.split == "train") & (man.protocol == "benign")].head(200).to_dict("records")
    with Pool(24) as p:
        df = pd.DataFrame([r for rs in p.map(work_test, tests) for r in rs])
        cal = pd.DataFrame(p.map(work_cal, cals))
    S = ["rel_err", "amp_err_ring", "amp_err_sub"]
    tau = cal.groupby("dataset")[S].quantile(0.99)
    print("thresholds (1% train-benign FPR):\n" + tau.round(5).to_string(), flush=True)
    for s in S:
        df[f"flag_{s}"] = df[s] > df.dataset.map(tau[s])
    df["flag_rel_or_amp_ring"] = df.flag_rel_err | df.flag_amp_err_ring
    df["flag_rel_or_amp_sub"] = df.flag_rel_err | df.flag_amp_err_sub
    df.to_csv(OUT / "amplitude_per_query.csv", index=False)
    flags = [c for c in df.columns if c.startswith("flag_")]
    summ = df.groupby(["dataset", "condition"]).agg(
        n=("query_id", "size"), psnr_vs_manipulated=("psnr_vs_manipulated", "median"),
        psnr_vs_released=("psnr_vs_released", "median"), **{f: (f, "mean") for f in flags})
    order = ["released", "manipulated", "copy_phase", "copy_phase_and_magnitude"]
    summ = summ.reindex(pd.MultiIndex.from_product([["FF++", "CelebDF", "DFDC"], order]))
    summ.to_csv(OUT / "amplitude_summary.csv")
    with pd.option_context("display.width", 250):
        print(summ.round(3).to_string(), flush=True)
    print("=== COPY RING AMPLITUDE DONE ===", flush=True)


if __name__ == "__main__":
    main()
