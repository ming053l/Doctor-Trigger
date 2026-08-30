#!/usr/bin/env python3
"""Build collision-free benign and SimSwap swap queries; failures are quarantined."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repo"))

from src.simswap_engine import SimSwapConfig, SimSwapEngine, SimSwapRequest
from workflow.paths import require_input_path, require_output_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_id(enrollment_id: str, fake_sample_id: str) -> str:
    payload = f"query-v1\x1f{enrollment_id}\x1f{fake_sample_id}\x1fsimswap".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def choose_fake_candidates(enrollments: pd.DataFrame, alignment: pd.DataFrame, pairs_per_enrollment: int) -> pd.DataFrame:
    fakes = alignment[(alignment["class"] != "Real") & (alignment["status"] == "pass")].copy()
    fakes = fakes.sort_values(["dataset", "split", "class", "video_id", "sample_id"])
    rows = []
    for enrollment in enrollments.itertuples(index=False):
        pool = fakes[(fakes["dataset"] == enrollment.dataset) & (fakes["split"] == enrollment.split)]
        if pool.empty:
            continue
        classes = sorted(pool["class"].unique())
        selected = []
        class_digest = hashlib.sha256(f"class-cycle-v1\x1f{enrollment.enrollment_id}".encode()).digest()
        class_start = int.from_bytes(class_digest[:8], "big") % len(classes)
        for offset in range(max(1, pairs_per_enrollment)):
            # A one-pair run must not collapse FF++ to the alphabetically
            # first manipulation.  Stable per-enrollment rotation gives a
            # deterministic, approximately balanced manipulation mixture.
            fake_class = classes[(class_start + offset) % len(classes)]
            class_pool = pool[pool["class"] == fake_class].reset_index(drop=True)
            digest = hashlib.sha256(f"{enrollment.enrollment_id}\x1f{fake_class}\x1f{offset}".encode()).digest()
            index = int.from_bytes(digest[:8], "big") % len(class_pool)
            selected.append(class_pool.iloc[index])
        for fake in selected:
            rows.append(
                {
                    **enrollment._asdict(),
                    "fake_sample_id": fake["sample_id"],
                    "fake_class": fake["class"],
                    "fake_aligned_path": fake["aligned_path"],
                    "fake_source_group_id": fake["source_group_id"],
                }
            )
    return pd.DataFrame(rows)


def finalize_results(
    pending: list[dict[str, Any]],
    errors: list[str | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record, error in zip(pending, errors):
        query_path = Path(record["query_path"])
        if error or not query_path.is_file():
            rejected.append({**record, "status": "quarantine", "reason": error or "missing_output"})
            continue
        try:
            with Image.open(query_path) as image:
                image.verify()
        except Exception as exc:  # pylint: disable=broad-except
            rejected.append({**record, "status": "quarantine", "reason": f"decode_failed:{exc}"})
            continue
        final_record = {
            key: value
            for key, value in record.items()
            if key not in {"receipt_path", "fake_input_sha256", "released_input_sha256", "reason"}
        }
        final_record.update({"query_sha256": sha256_file(query_path), "status": "pass"})
        receipt = {
            "query_id": record["query_id"],
            "fake_input_sha256": record["fake_input_sha256"],
            "released_input_sha256": record["released_input_sha256"],
            "query_sha256": final_record["query_sha256"],
            "record": final_record,
        }
        Path(record["receipt_path"]).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        accepted.append(final_record)
    return accepted, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enrollments", type=Path, required=True)
    parser.add_argument("--alignment-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/queries")
    parser.add_argument("--output-manifest", type=Path, default=ROOT / "manifests/queries.csv")
    parser.add_argument("--quarantine", type=Path, default=ROOT / "manifests/query_quarantine.csv")
    parser.add_argument("--simswap-dir", type=Path, default=Path(os.environ.get("DT_SIMSWAP_ROOT", "SimSwap")))
    parser.add_argument("--det-thresh", type=float, default=0.3)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pairs-per-enrollment", type=int, default=1)
    parser.add_argument("--fallback-candidates", type=int, default=3)
    parser.add_argument("--fallback-det-thresh", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse an existing query only when its sidecar receipt and all input/output hashes match.",
    )
    args = parser.parse_args()

    enrollments_path = require_input_path(args.enrollments)
    alignment_path = require_input_path(args.alignment_manifest)
    simswap_dir = require_input_path(args.simswap_dir)
    output_root = require_output_path(args.output_root)
    output_manifest = require_output_path(args.output_manifest)
    quarantine_path = require_output_path(args.quarantine)
    enrollments = pd.read_csv(enrollments_path)
    enrollments = enrollments[enrollments["status"] == "pass"].copy()
    if args.limit > 0:
        enrollments = enrollments.groupby(["dataset", "split"], group_keys=False).head(args.limit)
    alignment = pd.read_csv(alignment_path)
    selected = choose_fake_candidates(enrollments, alignment, args.pairs_per_enrollment)
    if selected.empty:
        raise RuntimeError("No same-dataset/split fake candidates available")

    records: list[dict[str, Any]] = []
    requests: list[SimSwapRequest] = []
    pending: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        identifier = pair_id(str(row.enrollment_id), str(row.fake_sample_id))
        destination = output_root / str(row.dataset) / str(row.split) / f"{identifier}.png"
        receipt_path = destination.with_suffix(".json")
        if destination.exists():
            if not args.resume_existing:
                raise RuntimeError(f"Refusing pre-existing query output without --resume-existing: {destination}")
            if not receipt_path.is_file():
                raise RuntimeError(f"Existing query has no resume receipt: {destination}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected = {
                "query_id": identifier,
                "fake_input_sha256": sha256_file(require_input_path(row.fake_aligned_path)),
                "released_input_sha256": sha256_file(require_input_path(row.released_path)),
                "query_sha256": sha256_file(destination),
            }
            for key, value in expected.items():
                if receipt.get(key) != value:
                    raise RuntimeError(f"Resume receipt mismatch ({key}): {destination}")
            with Image.open(destination) as image:
                image.verify()
            records.append(dict(receipt["record"]))
            continue
        request = SimSwapRequest(
            source_path=require_input_path(row.fake_aligned_path),
            target_path=require_input_path(row.released_path),
            output_path=destination,
        )
        requests.append(request)
        pending.append(
            {
                "query_id": identifier,
                "sample_id": row.sample_id,
                "enrollment_id": row.enrollment_id,
                "dataset": row.dataset,
                "split": row.split,
                "source_group_id": row.source_group_id,
                "fake_sample_id": row.fake_sample_id,
                "fake_source_group_id": row.fake_source_group_id,
                "fake_class": row.fake_class,
                "fake_aligned_path": row.fake_aligned_path,
                "template_artifact": row.template_artifact,
                "released_path": row.released_path,
                "query_path": str(destination),
                "protocol": "swap",
                "backend": "simswap",
                "y_fake": 1,
                "y_rel": 0,
                "y_keycorr": 0,
                "method_version": "doctor-trigger-workflow-v1",
                "simswap_det_thresh": args.det_thresh,
                "receipt_path": str(receipt_path),
                "fake_input_sha256": sha256_file(require_input_path(row.fake_aligned_path)),
                "released_input_sha256": sha256_file(require_input_path(row.released_path)),
            }
        )

    errors: list[str | None] = []
    engine: SimSwapEngine | None = None
    if requests:
        engine = SimSwapEngine(
            SimSwapConfig(
                simswap_dir=simswap_dir,
                device="cuda:0",
                crop_size=args.crop_size,
                det_thresh=args.det_thresh,
            )
        )
        errors = engine.swap_batch(requests, batch_size=args.batch_size)
    accepted, rejected = finalize_results(pending, errors)
    records.extend(accepted)

    # A detector can occasionally reject a particular already-aligned fake.
    # Retry with deterministic alternatives from the same dataset/split/class;
    # never substitute the target, full frame, class, or split.
    fakes = alignment[(alignment["class"] != "Real") & (alignment["status"] == "pass")].copy()
    fakes = fakes.sort_values(["dataset", "split", "class", "video_id", "sample_id"])
    for attempt in range(1, max(0, int(args.fallback_candidates)) + 1):
        if not rejected:
            break
        if engine is None:
            raise RuntimeError("Fallback requested without an initialized SimSwap engine")
        retry_pending: list[dict[str, Any]] = []
        retry_requests: list[SimSwapRequest] = []
        for failed in rejected:
            pool = fakes[
                (fakes["dataset"] == failed["dataset"])
                & (fakes["split"] == failed["split"])
                & (fakes["class"] == failed["fake_class"])
            ].reset_index(drop=True)
            if pool.empty:
                continue
            matches = pool.index[pool["sample_id"] == failed["fake_sample_id"]].tolist()
            start = matches[0] if matches else 0
            fake = pool.iloc[(start + attempt) % len(pool)]
            identifier = pair_id(str(failed["enrollment_id"]), str(fake["sample_id"]))
            destination = output_root / str(failed["dataset"]) / str(failed["split"]) / f"{identifier}.png"
            receipt_path = destination.with_suffix(".json")
            retry_record = {
                key: value
                for key, value in failed.items()
                if key not in {"status", "reason", "receipt_path", "fake_input_sha256", "released_input_sha256"}
            }
            retry_record.update(
                {
                    "query_id": identifier,
                    "fake_sample_id": fake["sample_id"],
                    "fake_source_group_id": fake["source_group_id"],
                    "fake_aligned_path": fake["aligned_path"],
                    "query_path": str(destination),
                    "fallback_attempt": attempt,
                    "receipt_path": str(receipt_path),
                    "fake_input_sha256": sha256_file(require_input_path(fake["aligned_path"])),
                    "released_input_sha256": sha256_file(require_input_path(failed["released_path"])),
                }
            )
            if destination.exists() and args.resume_existing and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    receipt.get("query_id") == identifier
                    and receipt.get("fake_input_sha256") == retry_record["fake_input_sha256"]
                    and receipt.get("released_input_sha256") == retry_record["released_input_sha256"]
                    and receipt.get("query_sha256") == sha256_file(destination)
                ):
                    records.append(dict(receipt["record"]))
                    continue
            retry_pending.append(retry_record)
            retry_requests.append(
                SimSwapRequest(
                    source_path=require_input_path(fake["aligned_path"]),
                    target_path=require_input_path(failed["released_path"]),
                    output_path=destination,
                )
            )
        retry_errors = engine.swap_batch(retry_requests, batch_size=args.batch_size)
        retry_accepted, rejected = finalize_results(retry_pending, retry_errors)
        records.extend(retry_accepted)

    # If the released target (not the fake source) is the low-confidence
    # detection, retry the original pair at an explicitly recorded lower
    # SimSwap detector threshold.  The aligned/released bytes are unchanged.
    if rejected and float(args.fallback_det_thresh) < float(args.det_thresh):
        low_engine = SimSwapEngine(
            SimSwapConfig(
                simswap_dir=simswap_dir,
                device="cuda:0",
                crop_size=args.crop_size,
                det_thresh=float(args.fallback_det_thresh),
            )
        )
        low_pending: list[dict[str, Any]] = []
        low_requests: list[SimSwapRequest] = []
        unresolved: list[dict[str, Any]] = []
        for failed in rejected:
            originals = selected[
                (selected["enrollment_id"] == failed["enrollment_id"])
                & (selected["fake_class"] == failed["fake_class"])
            ]
            if originals.empty:
                unresolved.append(failed)
                continue
            original = originals.iloc[0]
            identifier = pair_id(str(failed["enrollment_id"]), str(original["fake_sample_id"]))
            destination = output_root / str(failed["dataset"]) / str(failed["split"]) / f"{identifier}.png"
            receipt_path = destination.with_suffix(".json")
            low_record = {
                key: value
                for key, value in failed.items()
                if key not in {"status", "reason", "receipt_path", "fake_input_sha256", "released_input_sha256"}
            }
            low_record.update(
                {
                    "query_id": identifier,
                    "fake_sample_id": original["fake_sample_id"],
                    "fake_source_group_id": original["fake_source_group_id"],
                    "fake_aligned_path": original["fake_aligned_path"],
                    "query_path": str(destination),
                    "fallback_attempt": "target_det_thresh",
                    "simswap_det_thresh": float(args.fallback_det_thresh),
                    "receipt_path": str(receipt_path),
                    "fake_input_sha256": sha256_file(require_input_path(original["fake_aligned_path"])),
                    "released_input_sha256": sha256_file(require_input_path(failed["released_path"])),
                }
            )
            low_pending.append(low_record)
            low_requests.append(
                SimSwapRequest(
                    source_path=require_input_path(original["fake_aligned_path"]),
                    target_path=require_input_path(failed["released_path"]),
                    output_path=destination,
                )
            )
        low_errors = low_engine.swap_batch(low_requests, batch_size=args.batch_size)
        low_accepted, low_rejected = finalize_results(low_pending, low_errors)
        records.extend(low_accepted)
        rejected = unresolved + low_rejected

    # Benign queries use the released image and the same enrollment record.
    for row in enrollments.itertuples(index=False):
        identifier = hashlib.sha256(f"benign-v1\x1f{row.enrollment_id}".encode()).hexdigest()[:24]
        released_path = require_input_path(row.released_path)
        records.append(
            {
                "query_id": identifier,
                "sample_id": row.sample_id,
                "enrollment_id": row.enrollment_id,
                "dataset": row.dataset,
                "split": row.split,
                "source_group_id": row.source_group_id,
                "fake_sample_id": "",
                "fake_source_group_id": "",
                "fake_class": "Real",
                "template_artifact": row.template_artifact,
                "released_path": str(released_path),
                "query_path": str(released_path),
                "protocol": "benign",
                "backend": "identity",
                "y_fake": 0,
                "y_rel": 1,
                "y_keycorr": 1,
                "method_version": "doctor-trigger-workflow-v1",
                "query_sha256": sha256_file(released_path),
                "status": "pass",
            }
        )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_manifest, index=False)
    pd.DataFrame(rejected).to_csv(quarantine_path, index=False)
    print(f"Wrote {output_manifest} rows={len(records)}")
    print(f"Wrote {quarantine_path} rows={len(rejected)}")
    if rejected:
        raise RuntimeError(f"SimSwap produced {len(rejected)} quarantined rows; canonical run is fail-closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
