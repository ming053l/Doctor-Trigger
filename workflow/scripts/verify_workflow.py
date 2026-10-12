#!/usr/bin/env python3
"""Read-only acceptance check of a finished pipeline run (writes nothing).

Re-hashes every file listed in FINAL_RECEIPT.json and checks the expected row counts of the
paper's run, the record-model metrics, checkpoints and score files, and the comparison
interface. ``--full`` also re-runs the integrity gate's checks on every stage.
"""

from __future__ import annotations

import argparse
import json
import math
import stat
from pathlib import Path

import pandas as pd

from doctor_trigger.manifests import sha256_file
from workflow.pipeline_paths import WORKFLOW_ROOT


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also re-run the integrity checks, re-hashing every release, record and query")
    args = parser.parse_args()
    failures: list[str] = []

    receipt_path = WORKFLOW_ROOT / "FINAL_RECEIPT.json"
    integrity_path = WORKFLOW_ROOT / "reports/calibrated_integrity.json"
    require(receipt_path.is_file(), f"missing {receipt_path}", failures)
    require(integrity_path.is_file(), f"missing {integrity_path}", failures)
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
        return 2

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    require(receipt.get("status") == "PASS", "FINAL_RECEIPT status is not PASS", failures)
    require(integrity.get("status") == "PASS", "integrity status is not PASS", failures)
    require(Path(receipt.get("workflow_root", "")).resolve() == WORKFLOW_ROOT.resolve(), "receipt workflow_root does not match this copy", failures)

    verified_hashes = 0
    for section in ("manifests", "code", "interfaces"):
        for name, record in receipt.get(section, {}).items():
            path = Path(record["path"])
            require(path.is_file(), f"missing hashed {section} file {name}: {path}", failures)
            if path.is_file():
                actual = sha256_file(path)
                require(actual == record["sha256"], f"hash mismatch for {section} file {name}: {path}", failures)
                verified_hashes += 1

    expected_counts = {
        "base": (WORKFLOW_ROOT / "manifests/base_samples.csv", 24527),
        "alignment": (WORKFLOW_ROOT / "manifests/alignment.csv", 24209),
        "alignment_quarantine": (WORKFLOW_ROOT / "manifests/alignment_quarantine.csv", 318),
        "enrollments": (WORKFLOW_ROOT / "manifests/enrollments.csv", 7366),
        "queries": (WORKFLOW_ROOT / "manifests/queries.csv", 14732),
        "features": (WORKFLOW_ROOT / "features/calibrated_features.csv", 22098),
        "common_cases": (WORKFLOW_ROOT / "common/cases.csv", 14732),
        "metrics": (
            WORKFLOW_ROOT / "results/record_model/metrics/record_model_release_calibrated_ffpp_to_all_metrics.csv",
            108,
        ),
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, (path, expected) in expected_counts.items():
        frame = pd.read_csv(path)
        frames[name] = frame
        require(len(frame) == expected, f"{name} rows={len(frame)}, expected={expected}", failures)
        require(receipt.get("counts", {}).get(name) == expected, f"receipt count mismatch for {name}", failures)

    base = frames["base"]
    require(set(base["dataset"]) == {"FF++", "DFDC", "CelebDF"}, "base dataset set is incomplete", failures)
    missing_sources = sum(not Path(value).is_file() for value in base["frame_path"])
    require(missing_sources == 0, f"base source files missing={missing_sources}", failures)

    metrics = frames["metrics"]
    metric_key = ["eval_name", "variant", "seed", "score"]
    require(not metrics.duplicated(metric_key).any(), "duplicate final metric key", failures)
    require(set(metrics["eval_name"]) == {"ffpp_test", "dfdc_test", "celebdf_test"}, "metric eval set is incomplete", failures)
    require(set(metrics["seed"]) == {0, 1, 2}, "metric seed set is incomplete", failures)
    numeric = metrics.select_dtypes("number")
    require(not numeric.isna().any().any(), "final metrics contain NaN", failures)
    require(all(math.isfinite(float(value)) for value in numeric.to_numpy().ravel()), "final metrics contain non-finite values", failures)
    require(len(list((WORKFLOW_ROOT / "results/record_model/checkpoints").glob("*.pt"))) == 12, "expected 12 final checkpoints", failures)
    require(len(list((WORKFLOW_ROOT / "results/record_model/scores").glob("*.csv"))) == 36, "expected 36 final score files", failures)

    cases = frames["common_cases"]
    required_case_columns = {
        "query_id", "enrollment_input_path", "enrollment_input_sha256", "attack_source_path",
        "input_path", "input_sha256", "protocol", "y_fake", "y_rel", "y_keycorr",
    }
    require(required_case_columns <= set(cases.columns), "common case v2 columns are incomplete", failures)
    require((WORKFLOW_ROOT / "common/peer_score_contract.json").is_file(), "missing peer score contract", failures)
    require((WORKFLOW_ROOT / "common/workflow_contract.json").is_file(), "missing workflow stage contract", failures)

    mode = WORKFLOW_ROOT.stat().st_mode
    require(bool(mode & stat.S_IROTH) and bool(mode & stat.S_IXOTH), "workflow root is not world-readable/traversable", failures)

    full_checks: dict[str, object] = {}
    if args.full and not failures:
        from workflow.scripts.integrity_gate import check_alignment, check_base, check_enrollment, check_features, check_queries

        full_checks = {
            "base": check_base(WORKFLOW_ROOT / "manifests/base_samples.csv"),
            "alignment": check_alignment(
                WORKFLOW_ROOT / "manifests/alignment.csv",
                WORKFLOW_ROOT / "manifests/alignment_quarantine.csv",
            ),
            "enrollment": check_enrollment(
                WORKFLOW_ROOT / "manifests/enrollments.csv",
                WORKFLOW_ROOT / "manifests/enrollment_quarantine.csv",
            ),
            "queries": check_queries(
                WORKFLOW_ROOT / "manifests/queries.csv",
                WORKFLOW_ROOT / "manifests/query_quarantine.csv",
            ),
            "features": check_features(WORKFLOW_ROOT / "features/calibrated_features.csv"),
        }

    report = {
        "status": "PASS" if not failures else "FAIL",
        "workflow_root": str(WORKFLOW_ROOT),
        "mode": "full" if args.full else "quick",
        "verified_receipt_hashes": verified_hashes,
        "counts": {name: len(frame) for name, frame in frames.items()},
        "full_checks": full_checks,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
