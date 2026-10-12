#!/usr/bin/env python
"""Record-model features Φ(J, R) with the stored template T_I as the reference.

The pipeline's feature CSV (calibrated_features.csv) uses the released PNG as the reference, so C,
C_b*, and the normalization of G_s, G_wrong_*, G_margin, G_b* and G_margin_b* read released pixels
that the deployed record does not contain, and a benign query (which is that PNG) scores
C = G_s = 1 by construction. This script repeats the same extraction with the stored template phase
as the reference, keeping every other field and the wrong-key row unchanged. Before the full run it
extracts a subset with the released reference and requires an exact match with the pipeline's
feature CSV, so the only difference in the output is the reference.

Paths: config.yaml (canonical_root, queries_csv, outputs_root). Optional env DT_FEAT_WORKERS (worker
processes, default 48) and DT_FEAT_CHECK (queries in the equivalence check, default 60).
Output: $DT_OUTPUTS/features_record_ref/recordref_features.csv and per-split files named like the
pipeline's (..._{ffpp,dfdc,celebdf}_{train,val,test}.csv).
"""
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from doctor_trigger import config
from doctor_trigger.key import wrong_key_query_key
from doctor_trigger.manifests import read_queries
from doctor_trigger.readings import SizeMismatch, query_phase, record_features
from doctor_trigger.record import load_record

PIPELINE_ROOT = config.path("canonical_root")
OUT = Path(config.path("outputs_root")) / "features_record_ref"

WRONG_KEY_COUNT, BAND_COUNT = 8, 8
WORKERS = int(os.environ.get("DT_FEAT_WORKERS", "48"))
CHECK_COUNT = int(os.environ.get("DT_FEAT_CHECK", "60"))


def feature_rows(query, use_released_reference):
    """Feature rows of one query: its own key, plus a wrong-key row for a benign query."""
    query_path = Path(query["query_path"]).resolve()
    record_path = Path(query["template_artifact"]).resolve()
    released_path = Path(query["released_path"]).resolve()
    record = load_record(record_path)
    if use_released_reference:
        # the pipeline's arrangement, used only for the equivalence check
        reference = query_phase(Image.open(released_path), record.shape, SizeMismatch.RAISE)
        phase = (reference if query_path == released_path
                 else query_phase(Image.open(query_path), record.shape, SizeMismatch.RAISE))
    else:
        reference = None                      # record_features then reads the stored template phase
        phase = query_phase(Image.open(query_path), record.shape, SizeMismatch.RAISE)
    key = record.key
    features = record_features(phase, record, reference_phase=reference, wrong_key_count=WRONG_KEY_COUNT,
                               band_count=BAND_COUNT)
    base = dict(sample_id=query["query_id"], query_id=query["query_id"], enrollment_id=query["enrollment_id"],
                source_group_id=query["source_group_id"], dataset=query["dataset"], split=query["split"],
                protocol=query["protocol"], source_path=str(query_path), template_id=str(record_path),
                key_id=key, original_key_id=key, host_recovery_key_id=key, code_mode="signed_hermitian_pm_pi",
                y_fake=query["y_fake"], y_rel=query["y_rel"], y_keycorr=query["y_keycorr"])
    rows = [{**base, **features}]
    if query["protocol"] == "benign":
        wrong_key = wrong_key_query_key(key)
        wrong_features = record_features(phase, record, reference_phase=reference, projection_key=wrong_key,
                                         wrong_key_count=WRONG_KEY_COUNT, band_count=BAND_COUNT)
        rows.append({**base, "sample_id": f"{query['query_id']}_wrong_key", "protocol": "wrong_key",
                     "key_id": wrong_key, "y_fake": 0, "y_rel": 1, "y_keycorr": 0, **wrong_features})
    return rows


def released_reference_rows(query): return feature_rows(query, True)
def template_reference_rows(query): return feature_rows(query, False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queries = read_queries().to_dict("records")
    print(f"queries passing: {len(queries)}", flush=True)

    # 1) equivalence: the released reference must reproduce the pipeline's feature CSV exactly
    pipeline = pd.read_csv(f"{PIPELINE_ROOT}/features/calibrated_features.csv").set_index("sample_id")
    with Pool(min(WORKERS, 16)) as pool:
        check = pd.DataFrame([r for rows in pool.map(released_reference_rows, queries[:CHECK_COUNT])
                              for r in rows]).set_index("sample_id")
    numeric = [c for c in check.columns if pd.api.types.is_numeric_dtype(check[c]) and c in pipeline.columns]
    diff = (check[numeric] - pipeline.loc[check.index, numeric]).abs().max().max()
    print(f"equivalence check on {len(check)} rows, {len(numeric)} numeric columns: max |diff| = {diff:.3e}",
          flush=True)
    if not np.isfinite(diff) or diff > 1e-9:
        raise SystemExit("re-implementation does not reproduce the pipeline's features; aborting")

    # 2) full run with the template reference
    start = time.time(); rows = []
    with Pool(WORKERS) as pool:
        for i, query_rows in enumerate(pool.imap(template_reference_rows, queries, chunksize=8), 1):
            rows.extend(query_rows)
            if i % 1000 == 0 or i == len(queries):
                print(f"processed={i}/{len(queries)} rows={len(rows)} {time.time() - start:.0f}s", flush=True)
    features = pd.DataFrame(rows)
    main_csv = OUT / "recordref_features.csv"
    features.to_csv(main_csv, index=False)
    for (dataset, split), part in features.groupby(["dataset", "split"]):
        part.to_csv(OUT / f"{main_csv.stem}_{str(dataset).lower().replace('+', 'p')}_{split}.csv", index=False)
    print(f"wrote {main_csv} rows={len(features)} (pipeline rows={len(pipeline)})", flush=True)
    benign = features[(features.protocol == "benign") & (features.split == "test")]
    print(f"test benign: C==1 {np.mean(np.isclose(benign.C, 1)):.1%} (was 100%), "
          f"G_s==1 {np.mean(np.isclose(benign.G_s, 1)):.1%}, G_s median {benign.G_s.median():.3f}", flush=True)
    print("=== FEATURES RECORD-REF DONE ===", flush=True)


if __name__ == "__main__":
    main()
