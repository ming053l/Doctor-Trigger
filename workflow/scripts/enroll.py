#!/usr/bin/env python3
"""Stage 3: enroll every aligned real face.

For each face this writes the released image I' (PNG), its private record R (NPZ: template T_I,
original phase, displacement, amplitude weights, active bins, sign pattern h, key, alpha, beta, ring,
amplitude floor) and a JSON sidecar. The fidelity gate (PSNR >= 35 dB, SSIM >= 0.98) is applied to
the encoded release and again to the PNG read back from disk; faces that fail go to the quarantine
manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from doctor_trigger.enrollment import EnrollmentSettings, alpha_ladder, enroll, passes_fidelity_gate, psnr, ssim
from doctor_trigger.key import Ring
from doctor_trigger.manifests import sha256_file
from doctor_trigger.record import UnitId, enrollment_id, image_unit_path, save_record
from workflow.pipeline_paths import WORKFLOW_ROOT, require_input_path, require_output_path


def settings_of(args: argparse.Namespace) -> EnrollmentSettings:
    """Enrollment settings from the command line."""
    return EnrollmentSettings(
        master_seed=args.master_seed,
        beta=args.beta,
        ring=Ring(args.kappa_min, args.kappa_max),
        amplitude_floor_ratio=args.amplitude_floor_ratio,
        minimum_psnr=args.minimum_psnr,
        minimum_ssim=args.minimum_ssim,
    )


def enroll_row(
    row: Any,
    args: argparse.Namespace,
    released_root: Path,
    record_root: Path,
) -> tuple[str, dict[str, Any]]:
    """Enroll one alignment-manifest row: ("pass", enrollment row) or ("quarantine", reason row)."""
    aligned_path = Path(str(row.aligned_path))
    try:
        aligned_path = require_input_path(aligned_path)
        unit = UnitId(str(row.sample_id))
        released_path = image_unit_path(released_root, str(row.dataset), str(row.split), unit, ".png")
        record_path = image_unit_path(record_root, str(row.dataset), str(row.split), unit, ".npz")
        metadata_path = record_path.with_suffix(".json")
        if args.skip_existing and released_path.is_file() and record_path.is_file() and metadata_path.is_file():
            return "pass", json.loads(metadata_path.read_text(encoding="utf-8"))

        original = np.asarray(Image.open(aligned_path).convert("RGB"), dtype=np.uint8)
        settings = settings_of(args)
        # Raises ValueError("quality_gate_failed:[...]") when no alpha on the ladder passes.
        result = enroll(original, unit, alpha_ladder(str(row.dataset)), settings)
        released_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.released).save(released_path)
        stored = np.asarray(Image.open(released_path).convert("RGB"), dtype=np.uint8)
        stored_psnr = psnr(original, stored)
        stored_ssim = ssim(original, stored)
        if not passes_fidelity_gate(stored_psnr, stored_ssim, settings):
            raise ValueError(f"persisted_quality_gate_failed:psnr={stored_psnr}:ssim={stored_ssim}")
        save_record(record_path, result.record)
        enrollment = {
            "enrollment_id": enrollment_id(unit.asset_id),
            "sample_id": row.sample_id,
            "dataset": row.dataset,
            "split": row.split,
            "video_id": row.video_id,
            "source_group_id": row.source_group_id,
            "face_slot_id": "face_0",
            "source_path": row.source_path,
            "aligned_path": str(aligned_path),
            "released_path": str(released_path),
            "template_artifact": str(record_path),
            "key_reference": f"{record_path}#key",
            "alignment_metadata": str(Path(row.aligned_path).with_suffix(".json")),
            "alpha": result.alpha,
            "beta": args.beta,
            "kappa_min": args.kappa_min,
            "kappa_max": args.kappa_max,
            "amplitude_floor_ratio": args.amplitude_floor_ratio,
            "persisted_psnr": stored_psnr,
            "persisted_ssim": stored_ssim,
            "method_version": "doctor-trigger-workflow-v1",
            "calibration_version": "pending-v1",
            "released_sha256": sha256_file(released_path),
            "template_sha256": sha256_file(record_path),
            "status": "pass",
        }
        metadata_path.write_text(json.dumps(enrollment, indent=2) + "\n", encoding="utf-8")
        return "pass", enrollment
    except Exception as exc:  # pylint: disable=broad-except
        return "quarantine", {
            "sample_id": row.sample_id,
            "dataset": row.dataset,
            "split": row.split,
            "aligned_path": str(aligned_path),
            "status": "quarantine",
            "reason": f"{type(exc).__name__}:{exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-manifest", type=Path, required=True)
    parser.add_argument("--released-root", type=Path, default=WORKFLOW_ROOT / "artifacts/released")
    parser.add_argument("--record-root", type=Path, default=WORKFLOW_ROOT / "private/templates",
                        help="where the private records R are written (<dataset>/<split>/<enrollment_id>.npz)")
    parser.add_argument("--output-manifest", type=Path, default=WORKFLOW_ROOT / "manifests/enrollments.csv")
    parser.add_argument("--quarantine", type=Path, default=WORKFLOW_ROOT / "manifests/enrollment_quarantine.csv")
    parser.add_argument("--master-seed", type=int, default=17)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--kappa-min", type=float, default=0.10)
    parser.add_argument("--kappa-max", type=float, default=0.45)
    parser.add_argument("--amplitude-floor-ratio", type=float, default=1e-6)
    parser.add_argument("--minimum-psnr", type=float, default=35.0)
    parser.add_argument("--minimum-ssim", type=float, default=0.98)
    parser.add_argument("--limit", type=int, default=0, help="at most this many faces per (dataset, split) (0 = all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse a face whose release, record and sidecar already exist")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    alignment_manifest = require_input_path(args.alignment_manifest)
    released_root = require_output_path(args.released_root)
    record_root = require_output_path(args.record_root)
    output_manifest = require_output_path(args.output_manifest)
    quarantine_path = require_output_path(args.quarantine)
    frame = pd.read_csv(alignment_manifest)
    frame = frame[(frame["class"] == "Real") & (frame["status"] == "pass")].copy()
    frame = frame.sort_values(["dataset", "split", "video_id", "sample_id"])
    if args.limit > 0:
        frame = frame.groupby(["dataset", "split"], group_keys=False).head(args.limit)

    enrollments: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    input_rows = list(frame.itertuples(index=False))
    worker_count = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        iterator = executor.map(
            lambda row: enroll_row(row, args, released_root, record_root),
            input_rows,
        )
        for count, (status, row) in enumerate(iterator, start=1):
            (enrollments if status == "pass" else rejected).append(row)
            if count % 100 == 0:
                print(
                    f"enrolled={len(enrollments)} quarantine={len(rejected)} processed={count}/{len(frame)} workers={worker_count}",
                    flush=True,
                )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(enrollments).to_csv(output_manifest, index=False)
    pd.DataFrame(rejected).to_csv(quarantine_path, index=False)
    print(f"Wrote {output_manifest} rows={len(enrollments)}")
    print(f"Wrote {quarantine_path} rows={len(rejected)}")
    if not enrollments:
        raise RuntimeError("No enrollment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
