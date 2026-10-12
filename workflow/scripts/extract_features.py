#!/usr/bin/env python3
"""Stage 5: record features x = Phi(J, R) of every query, the record model's input.

One row per query, plus one wrong-key row per benign query (the same image read with the
wrong-key query key, labels y_rel = 1, y_key = 0). Writes the full table and one table per
(dataset, split) next to it (``<stem>_<dataset>_<split>.csv``). The feature columns are
``doctor_trigger.readings.FEATURE_COLUMNS``; ``y_keycorr`` is the paper's y_key.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from doctor_trigger.key import wrong_key_query_key
from doctor_trigger.readings import record_features
from doctor_trigger.record import load_record
from doctor_trigger.spectrum import luminance_phase
from workflow.pipeline_paths import WORKFLOW_ROOT, require_input_path, require_output_path


def image_phase(path: Path) -> np.ndarray:
    """Centered luminance phase of an image file (record layout)."""
    return luminance_phase(np.asarray(Image.open(path).convert("RGB"), dtype=np.float64), centered=True)


def rows_for_query(query: Any, wrong_keys: int, feature_bands: int) -> list[dict[str, Any]]:
    """Feature rows of one query-manifest row (two rows for a benign query)."""
    path = require_input_path(query.query_path)
    record_path = require_input_path(query.template_artifact)
    release_path = require_input_path(query.released_path)
    record = load_record(record_path)
    release_phase = image_phase(release_path)
    phase = release_phase if path == release_path else image_phase(path)
    original_key = record.key
    features = record_features(phase, record, release_phase, wrong_key_count=wrong_keys, band_count=feature_bands)
    rows = [
        {
            "sample_id": query.query_id,
            "query_id": query.query_id,
            "enrollment_id": query.enrollment_id,
            "source_group_id": query.source_group_id,
            "dataset": query.dataset,
            "split": query.split,
            "protocol": query.protocol,
            "source_path": str(path),
            "template_id": str(record_path),
            "key_id": original_key,
            "original_key_id": original_key,
            "host_recovery_key_id": original_key,
            "code_mode": "signed_hermitian_pm_pi",
            "y_fake": query.y_fake,
            "y_rel": query.y_rel,
            "y_keycorr": query.y_keycorr,
            **features,
        }
    ]
    if query.protocol == "benign":
        wrong_key = wrong_key_query_key(original_key)
        wrong_features = record_features(
            phase, record, release_phase, projection_key=wrong_key, wrong_key_count=wrong_keys, band_count=feature_bands
        )
        rows.append(
            {
                "sample_id": f"{query.query_id}_wrong_key",
                "query_id": query.query_id,
                "enrollment_id": query.enrollment_id,
                "source_group_id": query.source_group_id,
                "dataset": query.dataset,
                "split": query.split,
                "protocol": "wrong_key",
                "source_path": str(path),
                "template_id": str(record_path),
                "key_id": wrong_key,
                "original_key_id": original_key,
                "host_recovery_key_id": original_key,
                "code_mode": "signed_hermitian_pm_pi",
                "y_fake": 0,
                "y_rel": 1,
                "y_keycorr": 0,
                **wrong_features,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=WORKFLOW_ROOT / "features/features.csv")
    parser.add_argument("--wrong-keys", type=int, default=8, help="m, wrong keys per query for the G statistics")
    parser.add_argument("--feature-bands", type=int, default=8, help="radial bands of the per-band features")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    queries = pd.read_csv(require_input_path(args.queries))
    queries = queries[queries["status"] == "pass"].copy()
    output = require_output_path(args.output)
    rows: list[dict[str, Any]] = []
    query_rows = list(queries.itertuples(index=False))
    worker_count = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        iterator = executor.map(
            lambda query: rows_for_query(query, args.wrong_keys, args.feature_bands),
            query_rows,
        )
        for count, produced in enumerate(iterator, start=1):
            rows.extend(produced)
            if count % 100 == 0:
                print(f"processed={count}/{len(queries)} output_rows={len(rows)} workers={worker_count}", flush=True)

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    for (dataset, split), part in result.groupby(["dataset", "split"]):
        split_path = output.with_name(f"{output.stem}_{str(dataset).lower().replace('+', 'p')}_{split}{output.suffix}")
        part.to_csv(split_path, index=False)
    print(f"Wrote {output} rows={len(result)} columns={len(result.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
