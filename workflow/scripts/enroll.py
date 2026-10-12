#!/usr/bin/env python3
"""Create released images and explicit private phase-template enrollment records."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT.parent))  # the doctor_trigger package

from doctor_trigger.enrollment import EnrollmentSettings, alpha_ladder, enroll, passes_fidelity_gate, psnr, ssim
from doctor_trigger.key import Ring
from doctor_trigger.manifests import sha256_file
from doctor_trigger.record import UnitId, enrollment_id, image_unit_path, save_record
from workflow.paths import require_input_path, require_output_path


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


def process_enrollment(
    row: Any,
    args: argparse.Namespace,
    released_root: Path,
    template_root: Path,
) -> tuple[str, dict[str, Any]]:
    aligned_path = Path(str(row.aligned_path))
    try:
        aligned_path = require_input_path(aligned_path)
        unit = UnitId(str(row.sample_id))
        released_path = image_unit_path(released_root, str(row.dataset), str(row.split), unit, ".png")
        template_path = image_unit_path(template_root, str(row.dataset), str(row.split), unit, ".npz")
        metadata_path = template_path.with_suffix(".json")
        if args.skip_existing and released_path.is_file() and template_path.is_file() and metadata_path.is_file():
            return "pass", json.loads(metadata_path.read_text(encoding="utf-8"))

        original = np.asarray(Image.open(aligned_path).convert("RGB"), dtype=np.uint8)
        settings = settings_of(args)
        # Raises ValueError("quality_gate_failed:[...]") when no alpha on the ladder passes.
        result = enroll(original, unit, alpha_ladder(str(row.dataset)), settings)
        released_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.released).save(released_path)
        persisted = np.asarray(Image.open(released_path).convert("RGB"), dtype=np.uint8)
        persisted_psnr = psnr(original, persisted)
        persisted_ssim = ssim(original, persisted)
        if not passes_fidelity_gate(persisted_psnr, persisted_ssim, settings):
            raise ValueError(f"persisted_quality_gate_failed:psnr={persisted_psnr}:ssim={persisted_ssim}")
        save_record(template_path, result.record)
        record = {
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
            "template_artifact": str(template_path),
            "key_reference": f"{template_path}#key",
            "alignment_metadata": str(Path(row.aligned_path).with_suffix(".json")),
            "alpha": result.alpha,
            "beta": args.beta,
            "kappa_min": args.kappa_min,
            "kappa_max": args.kappa_max,
            "amplitude_floor_ratio": args.amplitude_floor_ratio,
            "persisted_psnr": persisted_psnr,
            "persisted_ssim": persisted_ssim,
            "method_version": "doctor-trigger-workflow-v1",
            "calibration_version": "pending-v1",
            "released_sha256": sha256_file(released_path),
            "template_sha256": sha256_file(template_path),
            "status": "pass",
        }
        metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return "pass", record
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
    parser.add_argument("--released-root", type=Path, default=ROOT / "artifacts/released")
    parser.add_argument("--template-root", type=Path, default=ROOT / "private/templates")
    parser.add_argument("--output-manifest", type=Path, default=ROOT / "manifests/enrollments.csv")
    parser.add_argument("--quarantine", type=Path, default=ROOT / "manifests/enrollment_quarantine.csv")
    parser.add_argument("--master-seed", type=int, default=17)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--kappa-min", type=float, default=0.10)
    parser.add_argument("--kappa-max", type=float, default=0.45)
    parser.add_argument("--amplitude-floor-ratio", type=float, default=1e-6)
    parser.add_argument("--minimum-psnr", type=float, default=35.0)
    parser.add_argument("--minimum-ssim", type=float, default=0.98)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    alignment_manifest = require_input_path(args.alignment_manifest)
    released_root = require_output_path(args.released_root)
    template_root = require_output_path(args.template_root)
    output_manifest = require_output_path(args.output_manifest)
    quarantine_path = require_output_path(args.quarantine)
    frame = pd.read_csv(alignment_manifest)
    frame = frame[(frame["class"] == "Real") & (frame["status"] == "pass")].copy()
    frame = frame.sort_values(["dataset", "split", "video_id", "sample_id"])
    if args.limit > 0:
        frame = frame.groupby(["dataset", "split"], group_keys=False).head(args.limit)

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    input_rows = list(frame.itertuples(index=False))
    worker_count = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        iterator = executor.map(
            lambda row: process_enrollment(row, args, released_root, template_root),
            input_rows,
        )
        for count, (status, record) in enumerate(iterator, start=1):
            (records if status == "pass" else rejected).append(record)
            if count % 100 == 0:
                print(
                    f"enrolled={len(records)} quarantine={len(rejected)} processed={count}/{len(frame)} workers={worker_count}",
                    flush=True,
                )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_manifest, index=False)
    pd.DataFrame(rejected).to_csv(quarantine_path, index=False)
    print(f"Wrote {output_manifest} rows={len(records)}")
    print(f"Wrote {quarantine_path} rows={len(rejected)}")
    if not records:
        raise RuntimeError("No enrollment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
