#!/usr/bin/env python3
"""Create the final reproducibility/deployment receipt from completed workflow outputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT.parent))  # the doctor_trigger package

from doctor_trigger.manifests import sha256_file
from workflow.paths import require_input_path, require_output_path


def csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(require_input_path(path))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def hash_record(path: Path) -> dict[str, Any]:
    resolved = require_input_path(path)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--alignment-quarantine", type=Path, required=True)
    parser.add_argument("--enrollments", type=Path, required=True)
    parser.add_argument("--enrollment-quarantine", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-quarantine", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--common-cases", type=Path, required=True)
    parser.add_argument("--peer-contract", type=Path)
    parser.add_argument("--workflow-contract", type=Path)
    parser.add_argument("--output-json", type=Path, default=ROOT / "FINAL_RECEIPT.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "FINAL_RECEIPT.md")
    args = parser.parse_args()

    tables = {
        "base": csv_or_empty(args.base),
        "alignment": csv_or_empty(args.alignment),
        "alignment_quarantine": csv_or_empty(args.alignment_quarantine),
        "enrollments": csv_or_empty(args.enrollments),
        "enrollment_quarantine": csv_or_empty(args.enrollment_quarantine),
        "queries": csv_or_empty(args.queries),
        "query_quarantine": csv_or_empty(args.query_quarantine),
        "features": csv_or_empty(args.features),
        "metrics": csv_or_empty(args.metrics),
        "common_cases": csv_or_empty(args.common_cases),
    }
    integrity = json.loads(require_input_path(args.integrity).read_text(encoding="utf-8"))
    if integrity.get("status") != "PASS":
        raise RuntimeError("Refusing final receipt without a PASS integrity report")
    if not tables["query_quarantine"].empty:
        raise RuntimeError("Refusing final receipt with quarantined SimSwap queries")

    manifests = {
        name: hash_record(path)
        for name, path in {
            "base": args.base,
            "alignment": args.alignment,
            "alignment_quarantine": args.alignment_quarantine,
            "enrollments": args.enrollments,
            "enrollment_quarantine": args.enrollment_quarantine,
            "queries": args.queries,
            "query_quarantine": args.query_quarantine,
            "features": args.features,
            "integrity": args.integrity,
            "metrics": args.metrics,
            "common_cases": args.common_cases,
        }.items()
    }
    interface_paths = {
        "peer_score_contract": args.peer_contract,
        "workflow_contract": args.workflow_contract,
    }
    interfaces = {
        name: hash_record(path)
        for name, path in interface_paths.items()
        if path is not None
    }
    code_paths = sorted((ROOT / "scripts").glob("*.py")) + [
        ROOT / "workflow/paths.py",
        ROOT / "record_model/src/record_model.py",
        ROOT / "record_model/scripts/train_eval_record_model_matrix.py",
        ROOT / "configs/record_model.yaml",
        ROOT / "run_workflow.sh",
        ROOT / "requirements-workflow.txt",
        ROOT / "ENVIRONMENT.md",
        ROOT / "WORKFLOW.md",
    ]
    code = {str(path.relative_to(ROOT)): hash_record(path) for path in code_paths}

    alignment_quarantine = tables["alignment_quarantine"]
    metrics = tables["metrics"]
    metric_summary = []
    if not metrics.empty:
        columns = [column for column in ["auc", "eer", "tpr@1e-2", "tpr@1e-3", "tpr@1e-4"] if column in metrics]
        summary = metrics.groupby(["eval_name", "variant", "score"])[columns].mean().reset_index()
        metric_summary = summary.to_dict(orient="records")

    receipt = {
        "receipt_version": "doctor-trigger-calibrated-1.0",
        "status": "PASS",
        "workflow_root": str(ROOT),
        "write_boundary": str(ROOT),
        "source_inputs_read_only": {
            "FF++": "site-local FF++ frame store (see config)",
            "DFDC": "site-local DFDC frame store (see config)",
            "CelebDF": "site-local Celeb-DF frame store (see config)",
            "SimSwap": "site-local SimSwap weights (see config)",
        },
        "counts": {
            name: int(len(table)) for name, table in tables.items()
        },
        "alignment_quarantine_reasons": (
            {str(k): int(v) for k, v in alignment_quarantine["reason"].value_counts().items()}
            if not alignment_quarantine.empty and "reason" in alignment_quarantine else {}
        ),
        "query_protocol_counts": {
            str(k): int(v) for k, v in tables["queries"]["protocol"].value_counts().items()
        },
        "swap_fake_class_counts": {
            str(k): int(v)
            for k, v in tables["queries"].query("protocol == 'swap'")["fake_class"].value_counts().items()
        },
        "feature_protocol_counts": {
            str(k): int(v) for k, v in tables["features"]["protocol"].value_counts().items()
        },
        "integrity": integrity,
        "metric_summary_mean_over_seeds": metric_summary,
        "manifests": manifests,
        "interfaces": interfaces,
        "code": code,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "onnxruntime-gpu": package_version("onnxruntime-gpu"),
            "insightface": package_version("insightface"),
            "opencv": package_version("opencv-python"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
        },
        "interpretation": "Smoke results are wiring evidence only; final calibrated metrics are in the hashed metrics table.",
    }
    output_json = require_output_path(args.output_json)
    output_md = require_output_path(args.output_md)
    output_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Doctor Trigger calibrated receipt",
        "",
        "Status: **PASS**",
        "",
        f"Write boundary: `{ROOT}`",
        "",
        "The SSD dataset/model trees were read-only inputs. All generated files are under the write boundary.",
        "",
        "| Artifact table | Rows | SHA-256 |",
        "|---|---:|---|",
    ]
    for name, record in manifests.items():
        lines.append(f"| {name} | {len(tables[name]) if name in tables else '-'} | `{record['sha256']}` |")
    lines.extend(
        [
            "",
            f"Alignment accepted: {len(tables['alignment'])}; quarantined: {len(alignment_quarantine)}.",
            f"Enrollments: {len(tables['enrollments'])}; queries: {len(tables['queries'])}; features: {len(tables['features'])}.",
            "",
            "See `reports/calibrated_integrity.json` for gate details and the metrics CSV for per-seed results.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_json} and {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
