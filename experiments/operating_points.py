#!/usr/bin/env python
"""Empirical operating points of the direct relation test (appendix Table 12).

The deployed reading, template mismatch 1 - C_B (sub-band B, against the stored template), scored
on benign test queries as the negatives and re-rendered queries as the positives, per re-rendering
engine and per corpus: AUC, and the TPR at a benign false-alarm rate of 1e-2 and 1e-3.

SimSwap rows use the swap queries of each corpus from the query manifest. The other engines' rows
use the swapped FF++ images cached by the engines experiment under engine_cache/<engine>/, each
scored against its asset's own record. Images whose size differs from the record are left out.

Paths: config.yaml (queries_csv, outputs_root, engine_cache). Writes $DT_OUTPUTS/operating_points.csv.
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from doctor_trigger import config
from doctor_trigger.manifests import read_queries, select
from doctor_trigger.metrics import roc_auc, tpr_at_fpr
from doctor_trigger.readings import SumForm, spectral_agreement
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase, subband_b

OUT = Path(config.path("outputs_root"))
ENGINE_CACHE = Path(config.path("engine_cache"))
SUBBAND_STABILIZER = 1e-8


def relation_error(image_and_record):
    """1 - C_B of an image against a record, summed in the unshifted layout; NaN on a size mismatch."""
    image_path, record_path = image_and_record
    record = load_record(record_path)
    template = np.fft.ifftshift(record.template_phase)
    phase = luminance_phase(np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64), centered=False)
    if phase.shape != template.shape:
        return np.nan
    bins = np.fft.ifftshift(record.active_bins) & subband_b(record.shape, centered=False)
    agreement = spectral_agreement(phase, template, np.fft.ifftshift(record.amplitude_weights), bins,
                                   stabilizer=SUBBAND_STABILIZER, form=SumForm.WEIGHT_MAP_VALUE_LAYOUT)
    return 1.0 - agreement


def main():
    queries = read_queries()
    comparisons = []          # (engine, dataset, benign scores, re-rendered scores)
    with Pool(24) as pool:
        benign = {}
        for dataset in ("FF++", "CelebDF", "DFDC"):
            benign_queries = select(queries, dataset=dataset, split="test", protocol="benign")
            benign[dataset] = (benign_queries, np.array(pool.map(
                relation_error, [(q.released_path, q.template_artifact) for q in benign_queries.itertuples()])))
            swaps = select(queries, dataset=dataset, split="test", protocol="swap")
            positives = np.array(pool.map(relation_error,
                                          [(q.query_path, q.template_artifact) for q in swaps.itertuples()]))
            comparisons.append(("SimSwap", dataset, benign[dataset][1], positives))
            print(f"SimSwap {dataset}: n={len(positives)}", flush=True)
        ffpp_benign, ffpp_benign_scores = benign["FF++"]
        for engine in sorted(d.name for d in ENGINE_CACHE.iterdir() if d.is_dir()):
            pairs = [(ENGINE_CACHE / engine / f"{q.enrollment_id}.png", q.template_artifact)
                     for q in ffpp_benign.itertuples() if (ENGINE_CACHE / engine / f"{q.enrollment_id}.png").exists()]
            positives = np.array(pool.map(relation_error, pairs))
            comparisons.append((engine, "FF++", ffpp_benign_scores, positives))
            print(f"{engine} FF++: n={len(positives)}", flush=True)

    rows = []
    for engine, dataset, negatives, positives in comparisons:
        negatives, positives = negatives[np.isfinite(negatives)], positives[np.isfinite(positives)]
        labels = np.r_[np.zeros(len(negatives)), np.ones(len(positives))]
        scores = np.r_[negatives, positives]
        rows.append(dict(re_rendering=engine, dataset=dataset, n_pos=len(positives), n_neg=len(negatives),
                         auc=roc_auc(labels, scores), tpr_1e2=tpr_at_fpr(labels, scores, 1e-2),
                         tpr_1e3=tpr_at_fpr(labels, scores, 1e-3)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "operating_points.csv", index=False)
    print(table.round(4).to_string(index=False)); print("=== OPERATING POINTS DONE ===")


if __name__ == "__main__":
    main()
