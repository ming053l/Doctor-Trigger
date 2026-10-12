#!/usr/bin/env python
"""Record-model features with a record-only reference.

The pipeline's rows_for_query passes the released PNG as reference_path, so C, C_b*, and the
normalization of G_s, G_wrong_*, G_margin, G_b* and G_margin_b* read released pixels that the
deployed record does not contain, and a benign query (which is that PNG) scores C = G_s = 1 by
construction. This script reruns the same extraction with the stored template phase as the reference
(reference_path=None), keeping every other field and the wrong-key row unchanged. Before the full run
it restores the released reference on a subset and requires an exact match with the pipeline's feature CSV, so
the only difference in the output is the reference.

Paths: config.yaml (canonical_root, queries_csv, outputs_root). Output: $DT_OUTPUTS/features_record_ref/recordref_features.csv
and per-split files named like the pipeline's (..._{ffpp,dfdc,celebdf}_{train,val,test}.csv).
"""
import os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402
from doctor_trigger.key import wrong_key_query_key  # noqa: E402
from doctor_trigger.manifests import read_queries  # noqa: E402
from doctor_trigger.readings import SizeMismatch, query_phase, record_features  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402

R = config.path("canonical_root")
OUT = Path(config.path("outputs_root")) / "features_record_ref"

N_WRONG, N_BANDS = 8, 8
WORKERS = int(os.environ.get("DT_FEAT_WORKERS", "48"))
CHECK_N = int(os.environ.get("DT_FEAT_CHECK", "60"))


def rows_for_query(q, use_released):
    path = Path(q["query_path"]).resolve()
    template = Path(q["template_artifact"]).resolve()
    released = Path(q["released_path"]).resolve()
    record = load_record(template)
    if use_released:                                   # the pipeline's arrangement, for the equivalence check
        reference = query_phase(Image.open(released), record.shape, SizeMismatch.RAISE)
        phase = reference if path == released else query_phase(Image.open(path), record.shape, SizeMismatch.RAISE)
    else:                                              # the fix: reference = stored template phase
        reference = None
        phase = query_phase(Image.open(path), record.shape, SizeMismatch.RAISE)
    k0 = record.key
    feats = record_features(phase, record, reference_phase=reference, wrong_key_count=N_WRONG, band_count=N_BANDS)
    base = dict(sample_id=q["query_id"], query_id=q["query_id"], enrollment_id=q["enrollment_id"],
                source_group_id=q["source_group_id"], dataset=q["dataset"], split=q["split"],
                protocol=q["protocol"], source_path=str(path), template_id=str(template), key_id=k0,
                original_key_id=k0, host_recovery_key_id=k0, code_mode="signed_hermitian_pm_pi",
                y_fake=q["y_fake"], y_rel=q["y_rel"], y_keycorr=q["y_keycorr"])
    rows = [{**base, **feats}]
    if q["protocol"] == "benign":
        wk = wrong_key_query_key(k0)
        wf = record_features(phase, record, reference_phase=reference, projection_key=wk,
                             wrong_key_count=N_WRONG, band_count=N_BANDS)
        rows.append({**base, "sample_id": f"{q['query_id']}_wrong_key", "protocol": "wrong_key", "key_id": wk,
                     "y_fake": 0, "y_rel": 1, "y_keycorr": 0, **wf})
    return rows


def work_released(q): return rows_for_query(q, True)
def work_record(q): return rows_for_query(q, False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = read_queries().to_dict("records")
    print(f"queries passing: {len(recs)}", flush=True)

    # 1) equivalence: restoring the released reference must reproduce the pipeline's feature CSV exactly
    canon = pd.read_csv(f"{R}/features/calibrated_features.csv").set_index("sample_id")
    with Pool(min(WORKERS, 16)) as p:
        chk = pd.DataFrame([r for rs in p.map(work_released, recs[:CHECK_N]) for r in rs]).set_index("sample_id")
    num = [c for c in chk.columns if pd.api.types.is_numeric_dtype(chk[c]) and c in canon.columns]
    diff = (chk[num] - canon.loc[chk.index, num]).abs().max().max()
    print(f"equivalence check on {len(chk)} rows, {len(num)} numeric columns: max |diff| = {diff:.3e}", flush=True)
    if not np.isfinite(diff) or diff > 1e-9:
        raise SystemExit("re-implementation does not reproduce the pipeline's features; aborting")

    # 2) full run with the record-only reference
    t0 = time.time(); rows = []
    with Pool(WORKERS) as p:
        for i, rs in enumerate(p.imap(work_record, recs, chunksize=8), 1):
            rows.extend(rs)
            if i % 1000 == 0 or i == len(recs):
                print(f"processed={i}/{len(recs)} rows={len(rows)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    main_csv = OUT / "recordref_features.csv"
    df.to_csv(main_csv, index=False)
    for (ds, split), part in df.groupby(["dataset", "split"]):
        part.to_csv(OUT / f"{main_csv.stem}_{str(ds).lower().replace('+', 'p')}_{split}.csv", index=False)
    print(f"wrote {main_csv} rows={len(df)} (pipeline rows={len(canon)})", flush=True)
    b = df[(df.protocol == "benign") & (df.split == "test")]
    print(f"test benign: C==1 {np.mean(np.isclose(b.C, 1)):.1%} (was 100%), G_s==1 {np.mean(np.isclose(b.G_s, 1)):.1%}, "
          f"G_s median {b.G_s.median():.3f}", flush=True)
    print("=== FEATURES RECORD-REF DONE ===", flush=True)


if __name__ == "__main__":
    main()
