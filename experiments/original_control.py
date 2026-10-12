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

Paths: config.yaml (cases_csv, outputs_root). Writes $DT_OUTPUTS/original_control_cal<N>/
{scores.csv, rates.csv, thresholds.csv, summary.csv}.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402
from doctor_trigger.calibration import lower_tail_threshold  # noqa: E402
from doctor_trigger.channels import jpeg  # noqa: E402
from doctor_trigger.manifests import read_cases  # noqa: E402
from doctor_trigger.readings import key_projection, phase_drift, release_normalized, spectral_agreement  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from doctor_trigger.spectrum import luminance_phase  # noqa: E402
from experiments.shared.face_identity import face_analysis, identity_embedding, identity_similarity  # noqa: E402

CAL_PER_DATASET = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TEST_COUNT = 400
OUT = Path(config.path("outputs_root")) / f"original_control_cal{CAL_PER_DATASET}"
AGREEMENT_STABILIZER = 1e-12


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    face = face_analysis(allowed_modules=("detection", "recognition"))

    benign = read_cases(protocol="benign").drop_duplicates("enrollment_id")
    train = benign[benign.split == "train"].groupby("dataset", group_keys=False).head(CAL_PER_DATASET)
    test = benign[benign.split == "test"].sample(TEST_COUNT, random_state=0)
    assert not set(train.source_group_id) & set(test.source_group_id)
    selected = pd.concat([train.assign(cohort="calibration"), test.assign(cohort="test")])
    rows = []
    for i, q in enumerate(selected.itertuples()):
        released = Image.open(q.doctor_trigger_released_path).convert("RGB")
        original = Image.open(q.enrollment_input_path).convert("RGB")
        record = load_record(q.doctor_trigger_template_path)
        hp, tp, delta = record.original_phase, record.template_phase, record.displacement
        sup, w = record.active_bins, record.amplitude_weights
        template_projection = key_projection(phase_drift(tp, hp), delta, w, sup)
        er = identity_embedding(face, released)
        rgray = np.asarray(released.convert("L"))
        release_jpeg90 = Image.fromarray(jpeg(np.asarray(released), 90))
        for kind, image in [("release_clean", released), ("release_jpeg90", release_jpeg90), ("original", original)]:
            ph = luminance_phase(np.asarray(image, dtype=np.float64), centered=True)
            eq = er if kind == "release_clean" else identity_embedding(face, image)
            rows.append(dict(
                enrollment_id=q.enrollment_id, dataset=q.dataset, source_group_id=q.source_group_id,
                cohort=q.cohort, kind=kind,
                H=spectral_agreement(ph, hp, w, sup, stabilizer=AGREEMENT_STABILIZER),
                C=spectral_agreement(ph, tp, w, sup, stabilizer=AGREEMENT_STABILIZER),
                G_s=release_normalized(key_projection(phase_drift(ph, hp), delta, w, sup), template_projection),
                ssim=float(structural_similarity(np.asarray(image.convert("L")), rgray, data_range=255)),
                arc=identity_similarity(er, eq)))
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
                tau = float(lower_tail_threshold(finite.to_numpy()))
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
