#!/usr/bin/env python3
"""Stage 0: build a source manifest CSV from a directory of extracted frames.

The canonical workflow consumes one manifest per dataset with columns
    split,class,video_id,frame_path
Point this script at a frame tree laid out as
    <frames_root>/<split>/<class>/<video_id>/<frame>.png
(the standard layout produced by ffmpeg extraction of FF++ / Celeb-DF / DFDC),
and it writes the manifest the pipeline expects.  Any per-dataset frame
sampling policy (frames per video, sampling stride) is applied upstream when
you extract the frames; this script only records what exists.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_root", type=Path)
    parser.add_argument("out_csv", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    rows = []
    for split in args.splits:
        split_dir = args.frames_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for video_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                for frame in sorted(video_dir.iterdir()):
                    if frame.suffix.lower() in IMG_EXTS:
                        rows.append((split, class_dir.name, video_dir.name, str(frame.resolve())))

    if not rows:
        raise SystemExit(f"no frames found under {args.frames_root}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "video_id", "frame_path"])
        writer.writerows(rows)
    print(f"{args.out_csv}: {len(rows)} frames, "
          f"{len({r[2] for r in rows})} videos, splits={sorted({r[0] for r in rows})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
