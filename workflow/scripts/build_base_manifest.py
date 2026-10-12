#!/usr/bin/env python3
"""Stage 1: merge the three source manifests into one base manifest of real and fake frames.

Each row gets a sample id and a source-group (video) id. A dataset whose source splits put one
video in more than one split is re-split by video group (70/15/15, hashed); otherwise the source
protocol's split is kept. The source manifests are named in config.yaml (``source_manifests``).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from workflow.pipeline_paths import WORKFLOW_ROOT, require_input_path, require_output_path, source_manifests


def hashed_id(*parts: object) -> str:
    """First 24 hex digits of SHA-256 over the parts joined by \x1f."""
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def normalize_class(value: object) -> str:
    text = str(value).strip()
    return "Real" if text.lower() in {"real", "real_youtube", "real_actors"} else text


def load_dataset(dataset: str, path: Path) -> pd.DataFrame:
    source = pd.read_csv(require_input_path(path))
    required = {"split", "class", "video_id", "frame_path"}
    missing = required - set(source.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    frame = source[list(required)].copy()
    frame["dataset"] = dataset
    frame["class"] = frame["class"].map(normalize_class)
    frame["frame_path"] = frame["frame_path"].astype(str)
    frame = frame.rename(columns={"split": "original_split"})
    frame["source_group_id"] = frame["video_id"].map(lambda value: hashed_id(dataset, value))
    frame["logical_frame"] = frame["frame_path"].map(lambda value: Path(value).name)
    frame["sample_id"] = frame.apply(
        lambda row: hashed_id(dataset, row["class"], row["video_id"], row["logical_frame"]),
        axis=1,
    )
    frame["source_exists"] = frame["frame_path"].map(lambda value: Path(value).is_file())
    split_counts = frame.groupby("source_group_id")["original_split"].nunique()
    crossing = set(split_counts[split_counts > 1].index)
    if crossing:
        def assign_split(group_id: str) -> str:
            value = int(hashlib.sha256(f"canonical-split-v1\x1f{dataset}\x1f{group_id}".encode()).hexdigest()[:16], 16)
            unit = value / float(16**16)
            if unit < 0.70:
                return "train"
            if unit < 0.85:
                return "val"
            return "test"

        frame["split"] = frame["source_group_id"].map(assign_split)
        frame["split_source"] = "deterministic_group_resplit_v1"
    else:
        frame["split"] = frame["original_split"]
        frame["split_source"] = "source_protocol"
    # The Celeb-DF frame tree lists some frames under more than one split; keep one row per
    # sample once the whole video group has its split.
    frame = frame.sort_values(["dataset", "class", "video_id", "logical_frame", "original_split", "frame_path"])
    frame = frame.drop_duplicates(subset=["sample_id"], keep="first")
    return frame[
        [
            "sample_id",
            "dataset",
            "split",
            "original_split",
            "split_source",
            "class",
            "video_id",
            "source_group_id",
            "frame_path",
            "source_exists",
        ]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKFLOW_ROOT / "manifests/base_samples.csv")
    parser.add_argument("--limit-per-dataset", type=int, default=0,
                        help="keep at most this many frames per (split, class) of each dataset (0 = all)")
    args = parser.parse_args()
    output = require_output_path(args.output)

    parts = []
    for dataset, path in source_manifests().items():
        frame = load_dataset(dataset, path)
        if args.limit_per_dataset > 0:
            frame = (
                frame.sort_values(["split", "class", "video_id", "frame_path"])
                .groupby(["split", "class"], group_keys=False)
                .head(args.limit_per_dataset)
            )
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    if not combined["source_exists"].all():
        missing = combined[~combined["source_exists"]]
        raise RuntimeError(f"Source manifest contains {len(missing)} missing files")
    if combined["sample_id"].duplicated().any():
        raise RuntimeError("sample_id collision in base manifest")

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    counts = combined.groupby(["dataset", "split", "class"]).size()
    print(f"Wrote {output} rows={len(combined)}")
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
