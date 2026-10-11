"""Unreleased-original control with training calibration (Table 8; pass columns of Table 3).

For 400 held-out test enrollments, score three images against the same record: the lossless
release, its JPEG-q90 copy, and the unreleased original. SSIM and ArcFace use the released image
as their private reference; H (original phase), C (template phase) and G_s read the stored record,
with G_s divided by the template's own projection. Each reading's cutoff is the 1% quantile of the
same reading on benign training releases of that dataset, sent through the calibration channel
(lossless or JPEG q90). Calibration and test source videos are disjoint. Faces ArcFace cannot
detect are counted as unavailable and never accepted.

The paper's two panels use different calibration sizes:
  lossless panel     python original_control.py 100      (100 training releases per dataset)
  JPEG-q90 panel     python original_control.py 1580     (all 1,144 / 1,580 / 1,203 releases)
Table 3's pass columns are the JPEG-q90 panel's original and JPEG-release acceptance.

Env: DT_CANON_ROOT, DT_OUTPUTS. Writes $DT_OUTPUTS/original_control_cal<N>/
{scores.csv, rates.csv, thresholds.csv, summary.csv}.
"""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

R = _dt("DT_CANON_ROOT")
sys.path.insert(0, R)
from scripts.extract_features import projection, wrap_phase

CAL_PER_DATASET = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TEST_COUNT = 400
OUT = Path(_dt("DT_OUTPUTS")) / f"original_control_cal{CAL_PER_DATASET}"


def jpeg(image):
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=90)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def phase(image):
    a = np.asarray(image, dtype=np.float64)
    return np.fft.fftshift(np.angle(np.fft.fft2(0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from insightface.app import FaceAnalysis
    face = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)

    def embedding(image):
        faces = face.get(np.asarray(image)[:, :, ::-1])
        return faces[0].normed_embedding if faces else None

    cases = pd.read_csv(f"{R}/common/cases.csv")
    benign = cases[cases.protocol == "benign"].drop_duplicates("enrollment_id")
    train = benign[benign.split == "train"].groupby("dataset", group_keys=False).head(CAL_PER_DATASET)
    test = benign[benign.split == "test"].sample(TEST_COUNT, random_state=0)
    assert not set(train.source_group_id) & set(test.source_group_id)
    selected = pd.concat([train.assign(cohort="calibration"), test.assign(cohort="test")])
    rows = []
    for i, q in enumerate(selected.itertuples()):
        released = Image.open(q.doctor_trigger_released_path).convert("RGB")
        original = Image.open(q.enrollment_input_path).convert("RGB")
        with np.load(q.doctor_trigger_template_path) as z:
            hp = z["host_phase"].astype(np.float64)
            tp = z["template_phase"].astype(np.float64)
            delta = z["delta"].astype(np.float64)
            sup = z["support"].astype(bool)
            w = z["host_amplitude"].astype(np.float64)
        template_scale = max(abs(projection(tp, hp, delta, sup, w)), 1e-6)
        er = embedding(released)
        rgray = np.asarray(released.convert("L"))
        for kind, image in [("release_clean", released), ("release_jpeg90", jpeg(released)), ("original", original)]:
            ph = phase(image)
            eq = er if kind == "release_clean" else embedding(image)
            rows.append(dict(
                enrollment_id=q.enrollment_id, dataset=q.dataset, source_group_id=q.source_group_id,
                cohort=q.cohort, kind=kind,
                H=float(np.sum(w[sup] * np.cos(wrap_phase(ph - hp))[sup]) / (w[sup].sum() + 1e-12)),
                C=float(np.sum(w[sup] * np.cos(wrap_phase(ph - tp))[sup]) / (w[sup].sum() + 1e-12)),
                G_s=projection(ph, hp, delta, sup, w) / template_scale,
                ssim=float(structural_similarity(np.asarray(image.convert("L")), rgray, data_range=255)),
                arc=float(np.dot(er, eq)) if er is not None and eq is not None else np.nan))
        if (i + 1) % 250 == 0:
            print(f"processed {i + 1}/{len(selected)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scores.csv", index=False)

    summary, thresholds = [], []
    for ds, group in df.groupby("dataset"):
        for channel in ["release_clean", "release_jpeg90"]:
            cal = group[(group.cohort == "calibration") & (group.kind == channel)]
            for metric in ["H", "C", "G_s", "ssim", "arc"]:
                finite = cal[metric].dropna()
                if len(finite) == 0:
                    raise ValueError(f"no calibration readings: {ds}/{metric}")
                tau = float(finite.quantile(0.01))
                thresholds.append(dict(dataset=ds, channel=channel, metric=metric, tau=tau,
                                       n=len(cal), available=len(finite)))
                for kind, t in group[group.cohort == "test"].groupby("kind"):
                    summary.append(dict(dataset=ds, channel=channel, metric=metric, kind=kind, n=len(t),
                                        available=int(t[metric].notna().sum()),
                                        accepted=int((t[metric] >= tau).sum()), tau=tau))
    rates = pd.DataFrame(summary)
    rates.to_csv(OUT / "rates.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "thresholds.csv", index=False)
    pooled = rates.groupby(["channel", "metric", "kind"])[["n", "available", "accepted"]].sum().reset_index()
    pooled["accepted_pct"] = 100 * pooled.accepted / pooled.n
    pooled.to_csv(OUT / "summary.csv", index=False)
    table = pooled.pivot_table(index="metric", columns=["channel", "kind"], values="accepted_pct")
    print(f"calibration: up to {CAL_PER_DATASET} benign training releases per dataset; test: {TEST_COUNT} enrollments")
    print(table.reindex(["H", "C", "ssim", "arc", "G_s"]).round(2).to_string())


if __name__ == "__main__":
    main()
