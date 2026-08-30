#!/usr/bin/env python3
"""Fail-closed integrity checks for the canonical Doctor Trigger workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "record_model"))

from src.record_model import feature_columns_for_variant
from workflow.paths import require_input_path, require_output_path


TEMPLATE_FIELDS = {
    "template_phase",
    "host_phase",
    "delta",
    "host_amplitude",
    "support",
    "code",
    "key",
    "alpha",
    "beta",
    "bands",
    "amplitude_floor_ratio",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(require_input_path(path))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def generated_path(path: object) -> Path:
    resolved = Path(str(path)).expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise AssertionError(f"generated path escapes workflow root: {resolved}") from exc
    return resolved


def split_overlap(frame: pd.DataFrame, group_column: str = "source_group_id") -> int:
    if frame.empty:
        return 0
    return int((frame.groupby(["dataset", group_column])["split"].nunique() > 1).sum())


def assert_columns(frame: pd.DataFrame, columns: set[str], stage: str) -> None:
    missing = columns - set(frame.columns)
    assert not missing, f"{stage} missing columns: {sorted(missing)}"


def check_base(path: Path) -> dict[str, Any]:
    frame = read_csv(path)
    assert_columns(
        frame,
        {"sample_id", "dataset", "split", "class", "source_group_id", "frame_path", "source_exists"},
        "base",
    )
    assert not frame.empty, "base manifest is empty"
    assert not frame["sample_id"].duplicated().any(), "duplicate base sample_id"
    assert frame["source_exists"].astype(bool).all(), "base contains missing sources"
    assert split_overlap(frame) == 0, "source group crosses canonical splits in base manifest"
    missing_files = sum(not Path(value).is_file() for value in frame["frame_path"])
    assert missing_files == 0, f"base source files missing: {missing_files}"
    return {
        "rows": len(frame),
        "datasets": sorted(frame["dataset"].unique().tolist()),
        "source_group_split_overlap": 0,
        "counts": {
            "|".join(map(str, key)): int(value)
            for key, value in frame.groupby(["dataset", "split", "class"]).size().items()
        },
    }


def check_alignment(path: Path, quarantine_path: Path | None) -> dict[str, Any]:
    frame = read_csv(path)
    assert_columns(
        frame,
        {
            "sample_id", "dataset", "split", "class", "source_group_id", "aligned_path",
            "scrfd_bbox", "yunet_bbox", "haar_bbox", "agreeing_detectors", "padding", "resized", "status",
        },
        "alignment",
    )
    assert not frame.empty, "alignment manifest is empty"
    assert not frame["sample_id"].duplicated().any(), "duplicate aligned sample_id"
    assert (frame["status"] == "pass").all(), "non-pass row in alignment manifest"
    assert (pd.to_numeric(frame["agreeing_detectors"]) >= 2).all(), "alignment has fewer than two agreeing detectors"
    assert (pd.to_numeric(frame["padding"]) == 0).all(), "alignment used padding"
    resized = frame["resized"].astype(str).str.lower().isin({"true", "1"})
    assert not resized.any(), "alignment resized an image"
    assert frame["scrfd_bbox"].astype(str).str.len().gt(0).all(), "missing SCRFD detection record"
    for value in frame["aligned_path"]:
        assert generated_path(value).is_file(), f"missing aligned output: {value}"
    assert split_overlap(frame) == 0, "source group crosses splits after alignment"
    quarantine = read_csv(quarantine_path) if quarantine_path else pd.DataFrame()
    return {
        "rows": len(frame),
        "quarantine_rows": len(quarantine),
        "source_group_split_overlap": 0,
        "detector_agreement": {str(int(k)): int(v) for k, v in frame["agreeing_detectors"].value_counts().items()},
    }


def check_enrollment(path: Path, quarantine_path: Path | None) -> dict[str, Any]:
    frame = read_csv(path)
    assert_columns(
        frame,
        {
            "enrollment_id", "sample_id", "dataset", "split", "source_group_id", "released_path",
            "template_artifact", "released_sha256", "template_sha256", "persisted_psnr", "persisted_ssim", "status",
        },
        "enrollment",
    )
    assert not frame.empty, "enrollment manifest is empty"
    assert not frame["enrollment_id"].duplicated().any(), "duplicate enrollment_id"
    assert not frame["template_artifact"].duplicated().any(), "template artifact reused by multiple enrollments"
    assert (frame["status"] == "pass").all(), "non-pass row in enrollment manifest"
    assert (pd.to_numeric(frame["persisted_psnr"]) >= 35.0).all(), "released image fails PSNR gate"
    assert (pd.to_numeric(frame["persisted_ssim"]) >= 0.98).all(), "released image fails SSIM gate"
    assert split_overlap(frame) == 0, "enrollment group crosses splits"
    for row in frame.itertuples(index=False):
        released = generated_path(row.released_path)
        template = generated_path(row.template_artifact)
        assert released.is_file(), f"missing released image: {released}"
        assert template.is_file(), f"missing template: {template}"
        assert sha256_file(released) == row.released_sha256, f"released hash mismatch: {released}"
        assert sha256_file(template) == row.template_sha256, f"template hash mismatch: {template}"
        with np.load(template) as data:
            assert TEMPLATE_FIELDS <= set(data.files), f"incomplete formal template: {template}"
            shape = data["template_phase"].shape
            assert len(shape) == 2 and min(shape) > 0, f"invalid template dimensions: {template}"
            for key in ("host_phase", "delta", "host_amplitude", "support", "code"):
                assert data[key].shape == shape, f"template field shape mismatch {key}: {template}"
            assert np.isfinite(data["template_phase"]).all(), f"non-finite template: {template}"
    quarantine = read_csv(quarantine_path) if quarantine_path else pd.DataFrame()
    return {
        "rows": len(frame),
        "quarantine_rows": len(quarantine),
        "source_group_split_overlap": 0,
        "minimum_psnr": float(pd.to_numeric(frame["persisted_psnr"]).min()),
        "minimum_ssim": float(pd.to_numeric(frame["persisted_ssim"]).min()),
    }


def check_queries(path: Path, quarantine_path: Path | None) -> dict[str, Any]:
    frame = read_csv(path)
    assert_columns(
        frame,
        {
            "query_id", "enrollment_id", "dataset", "split", "source_group_id", "query_path",
            "template_artifact", "protocol", "query_sha256", "status",
        },
        "queries",
    )
    assert not frame.empty, "query manifest is empty"
    assert not frame["query_id"].duplicated().any(), "duplicate query_id"
    assert (frame["status"] == "pass").all(), "non-pass row in query manifest"
    assert {"benign", "swap"} <= set(frame["protocol"]), "query protocols incomplete"
    swap = frame[frame["protocol"] == "swap"]
    assert not swap.empty, "no swap queries"
    assert not swap["query_path"].duplicated().any(), "swap output path collision"
    assert split_overlap(frame) == 0, "query enrollment group crosses splits"
    if "fake_source_group_id" in swap.columns:
        assert split_overlap(swap, "fake_source_group_id") == 0, "query fake source group crosses splits"
    for row in frame.itertuples(index=False):
        query = generated_path(row.query_path)
        template = generated_path(row.template_artifact)
        assert query.is_file(), f"missing query image: {query}"
        assert template.is_file(), f"missing query template: {template}"
        assert sha256_file(query) == row.query_sha256, f"query hash mismatch: {query}"
    quarantine = read_csv(quarantine_path) if quarantine_path else pd.DataFrame()
    assert quarantine.empty, f"canonical query run has {len(quarantine)} quarantined rows"
    return {
        "rows": len(frame),
        "ttf_rows": len(swap),
        "benign_rows": int((frame["protocol"] == "benign").sum()),
        "quarantine_rows": 0,
        "source_group_split_overlap": 0,
    }


def check_features(path: Path) -> dict[str, Any]:
    frame = read_csv(path)
    assert_columns(
        frame,
        {"sample_id", "dataset", "split", "protocol", "source_group_id", "y_fake", "y_rel", "y_keycorr"},
        "features",
    )
    assert not frame.empty, "features are empty"
    assert not frame["sample_id"].duplicated().any(), "duplicate feature sample_id"
    assert {"benign", "swap", "wrong_key"} <= set(frame["protocol"]), "feature protocols incomplete"
    assert split_overlap(frame) == 0, "feature source group crosses splits"
    selected = feature_columns_for_variant("full_record_model", list(frame.columns))
    assert len(selected) == 75, f"expected 75 calibrated full-record-model features, found {len(selected)}"
    values = frame[selected].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    assert np.isfinite(values).all(), "non-finite selected feature value"
    return {
        "rows": len(frame),
        "selected_full_model_features": len(selected),
        "protocol_counts": {str(k): int(v) for k, v in frame["protocol"].value_counts().items()},
        "source_group_split_overlap": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--alignment-quarantine", type=Path)
    parser.add_argument("--enrollments", type=Path)
    parser.add_argument("--enrollment-quarantine", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--query-quarantine", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    checks: dict[str, dict[str, Any]] = {"base": check_base(args.base)}
    if args.alignment:
        checks["alignment"] = check_alignment(args.alignment, args.alignment_quarantine)
    if args.enrollments:
        checks["enrollment"] = check_enrollment(args.enrollments, args.enrollment_quarantine)
    if args.queries:
        checks["queries"] = check_queries(args.queries, args.query_quarantine)
    if args.features:
        checks["features"] = check_features(args.features)

    report = {
        "status": "PASS",
        "workflow_root": str(ROOT),
        "write_boundary": str(ROOT),
        "checks": checks,
    }
    output_json = require_output_path(args.output_json)
    output_md = require_output_path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Workflow Integrity Gate",
        "",
        "Status: **PASS**",
        "",
        f"All generated artifacts are confined to `{ROOT}`.",
        "",
        "| Stage | Rows | Quarantine | Group split overlap |",
        "|---|---:|---:|---:|",
    ]
    for stage, result in checks.items():
        lines.append(
            f"| {stage} | {result.get('rows', 0)} | {result.get('quarantine_rows', 0)} | "
            f"{result.get('source_group_split_overlap', 0)} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: wrote {output_json} and {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
