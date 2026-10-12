#!/usr/bin/env python3
"""Stage 4: build the queries J of every enrollment.

A benign query is the released image itself. A swap query is SimSwap applied to the release with
an aligned fake face of the same dataset and split as identity source (FF++ manipulation types
rotate across enrollments). Query ids are hashes of (enrollment, source face), so paths never
collide. A failed swap is retried with other source faces of the same class, then once at the
lower detector threshold; any swap that still fails is quarantined and fails the stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from doctor_trigger.manifests import sha256_file
from workflow.pipeline_paths import WORKFLOW_ROOT, require_input_path, require_output_path, simswap_root
from workflow.repo.src.simswap_engine import SimSwapConfig, SimSwapEngine, SimSwapRequest


def pair_id(enrollment_id: str, fake_sample_id: str) -> str:
    """Swap query id: first 24 hex digits of SHA-256 over (enrollment, source face, "simswap")."""
    payload = f"query-v1\x1f{enrollment_id}\x1f{fake_sample_id}\x1fsimswap".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def choose_fake_candidates(enrollments: pd.DataFrame, alignment: pd.DataFrame, pairs_per_enrollment: int) -> pd.DataFrame:
    """One identity-source fake face per (enrollment, pair), chosen by hash within dataset and split."""
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
            # Start the class cycle at a hashed offset per enrollment, so a one-pair run mixes
            # the FF++ manipulation types instead of always taking the first one.
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
    """Split swap attempts into accepted rows (decodable output, receipt written) and rejected rows."""
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
    parser.add_argument("--output-root", type=Path, default=WORKFLOW_ROOT / "artifacts/queries")
    parser.add_argument("--output-manifest", type=Path, default=WORKFLOW_ROOT / "manifests/queries.csv")
    parser.add_argument("--quarantine", type=Path, default=WORKFLOW_ROOT / "manifests/query_quarantine.csv")
    parser.add_argument("--simswap-dir", type=Path, default=None, help="SimSwap checkout (default: simswap_root in config.yaml)")
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
    simswap_dir = require_input_path(args.simswap_dir or simswap_root())
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

    # SimSwap's detector can reject a particular aligned fake. Retry with the next source faces of
    # the same dataset, split and class; the target, class and split are never substituted.
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

    # When the release (the target) is the low-confidence detection, retry the first pair at the
    # lower detector threshold, recorded in simswap_det_thresh. No input image changes.
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
