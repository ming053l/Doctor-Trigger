#!/usr/bin/env python3
"""Export the method-neutral comparison interface: common/cases.csv and the two contracts.

cases.csv has one row per query with the exact bytes every method sees (query, enrollment input,
swap identity source, Doctor Trigger release and record, all with SHA-256) and the targets.
peer_score_contract.json says how a peer method returns scores; workflow_contract.json says
which inputs each comparison mode uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from doctor_trigger.manifests import sha256_file
from workflow.pipeline_paths import WORKFLOW_ROOT, require_input_path, require_output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--enrollments", type=Path, required=True)
    parser.add_argument("--alignment-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=WORKFLOW_ROOT / "common")
    args = parser.parse_args()

    queries = pd.read_csv(require_input_path(args.queries))
    queries = queries[queries["status"] == "pass"].copy()
    required = {
        "query_id", "enrollment_id", "dataset", "split", "source_group_id", "query_path",
        "protocol", "y_fake", "y_rel", "y_keycorr", "query_sha256",
    }
    missing = required - set(queries.columns)
    if missing:
        raise RuntimeError(f"Query manifest missing common-interface fields: {sorted(missing)}")
    if queries["query_id"].duplicated().any():
        raise RuntimeError("query_id collision in common interface")

    enrollments = pd.read_csv(require_input_path(args.enrollments))
    enrollments = enrollments[enrollments["status"] == "pass"].copy()
    enrollment_required = {
        "enrollment_id", "aligned_path", "released_path", "template_artifact",
        "released_sha256", "template_sha256",
    }
    missing = enrollment_required - set(enrollments.columns)
    if missing:
        raise RuntimeError(f"Enrollment manifest missing common-interface fields: {sorted(missing)}")
    if enrollments["enrollment_id"].duplicated().any():
        raise RuntimeError("enrollment_id collision in common interface")

    alignment = pd.read_csv(require_input_path(args.alignment_manifest))
    alignment = alignment[alignment["status"] == "pass"].copy()
    if alignment["sample_id"].duplicated().any():
        raise RuntimeError("alignment sample_id collision in common interface")
    aligned_by_sample = alignment.set_index("sample_id")["aligned_path"].to_dict()

    enrollment_columns = enrollments[
        ["enrollment_id", "aligned_path", "released_path", "template_artifact", "released_sha256", "template_sha256"]
    ].rename(
        columns={
            "aligned_path": "enrollment_input_path",
            "released_path": "doctor_trigger_released_path",
            "template_artifact": "doctor_trigger_template_path",
            "released_sha256": "doctor_trigger_released_sha256",
            "template_sha256": "doctor_trigger_template_sha256",
        }
    )
    enrollment_columns["enrollment_input_sha256"] = enrollment_columns["enrollment_input_path"].map(sha256_file)
    queries = queries.merge(enrollment_columns, on="enrollment_id", how="left", validate="many_to_one")
    if queries["enrollment_input_path"].isna().any():
        raise RuntimeError("query has no matching enrollment input")

    queries["attack_source_path"] = queries["fake_sample_id"].map(aligned_by_sample).fillna("")
    attack_hash_cache: dict[str, str] = {}
    def attack_hash(path: object) -> str:
        value = str(path)
        if not value:
            return ""
        if value not in attack_hash_cache:
            attack_hash_cache[value] = sha256_file(value)
        return attack_hash_cache[value]
    queries["attack_source_sha256"] = queries["attack_source_path"].map(attack_hash)
    swap = queries["protocol"] == "swap"
    if queries.loc[swap, "attack_source_path"].eq("").any():
        raise RuntimeError("swap query has no canonical attack source")
    # A swap row made at the default SimSwap detector threshold may not store it; write 0.3 (the
    # default) so every swap row states its attack parameters. Rows retried at 0.1 store 0.1.
    queries["simswap_det_thresh"] = pd.to_numeric(queries["simswap_det_thresh"], errors="coerce")
    queries.loc[swap, "simswap_det_thresh"] = queries.loc[swap, "simswap_det_thresh"].fillna(0.3)
    queries.loc[~swap, "simswap_det_thresh"] = float("nan")
    queries["fallback_attempt"] = queries["fallback_attempt"].fillna("").astype(str)
    queries.loc[swap & queries["fallback_attempt"].eq(""), "fallback_attempt"] = "none"
    queries.loc[~swap, "fallback_attempt"] = "not_applicable"

    output_dir = require_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = queries[
        [
            "query_id", "enrollment_id", "dataset", "split", "source_group_id", "query_path",
            "query_sha256", "enrollment_input_path", "enrollment_input_sha256",
            "attack_source_path", "attack_source_sha256", "doctor_trigger_released_path",
            "doctor_trigger_released_sha256", "doctor_trigger_template_path",
            "doctor_trigger_template_sha256", "protocol", "backend", "simswap_det_thresh",
            "fallback_attempt",
            "y_fake", "y_rel", "y_keycorr",
        ]
    ].rename(columns={"query_path": "input_path", "query_sha256": "input_sha256"})
    cases.insert(0, "case_schema_version", "doctor-trigger-common-case-v2")
    cases.to_csv(output_dir / "cases.csv", index=False)

    score_contract = {
        "schema_version": "doctor-trigger-peer-score-v2",
        "join_key": "query_id",
        "required_columns": {
            "query_id": "string; must appear exactly once and match cases.csv",
            "method_id": "string; immutable method/config identifier",
            "score_fake": "float; larger means more likely fake",
            "status": "one of pass, quarantine",
        },
        "optional_columns": {
            "score_relation": "float; larger means enrolled relation is preserved",
            "score_key": "float; larger means enrollment key matches",
            "error": "string; required when status=quarantine",
        },
        "fairness_rules": [
            "Declare detector_only or end_to_end as the comparison mode.",
            "Detector-only mode must score input_path bytes without modification.",
            "End-to-end mode must enroll enrollment_input_path and apply the recorded attack using attack_source_path.",
            "Do not change labels, canonical splits, enrollment bytes, attack-source bytes, or case IDs.",
            "Fit/calibrate only on split=train; use val only for threshold selection.",
            "Report test results per dataset and protocol, including quarantines.",
            "Never silently replace a failed input with a different crop or full frame.",
        ],
    }
    (output_dir / "peer_score_contract.json").write_text(
        json.dumps(score_contract, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        columns=["query_id", "method_id", "score_fake", "score_relation", "score_key", "status", "error"]
    ).to_csv(output_dir / "peer_scores_template.csv", index=False)
    workflow_contract = {
        "schema_version": "doctor-trigger-stage-contract-v1",
        "comparison_modes": {
            "detector_only": {
                "input": "cases.csv:input_path",
                "rule": "Every method scores the exact same finalized query bytes.",
            },
            "end_to_end": {
                "enroll_input": "cases.csv:enrollment_input_path",
                "attack_source": "cases.csv:attack_source_path (swap only)",
                "attack_parameters": ["cases.csv:backend", "cases.csv:simswap_det_thresh", "cases.csv:fallback_attempt"],
                "doctor_trigger_reference": [
                    "cases.csv:doctor_trigger_released_path",
                    "cases.csv:doctor_trigger_template_path",
                    "cases.csv:input_path",
                ],
                "rule": "Each proactive method creates its own released enrollment and applies the same recorded attack protocol/source.",
            },
        },
        "stages": [
            {"name": "canonical_alignment", "input": "source frame", "output": "enrollment_input_path"},
            {"name": "method_enrollment", "input": "enrollment_input_path", "output": "method release plus private state/template"},
            {"name": "query_generation", "input": "method release plus attack_source_path", "output": "method query"},
            {"name": "method_scoring", "input": "method query plus private state if required", "output": "peer score contract row"},
        ],
        "join_key": "query_id",
    }
    (output_dir / "workflow_contract.json").write_text(
        json.dumps(workflow_contract, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'cases.csv'} rows={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
