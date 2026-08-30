#!/usr/bin/env python3
"""Create no-padding/no-resize face crops and a three-detector audit manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow.paths import require_input_path, require_output_path


def box_iou(first: tuple[int, int, int, int] | None, second: tuple[int, int, int, int] | None) -> float:
    if first is None or second is None:
        return 0.0
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection)
    return float(intersection / union)


def clip_box(box: np.ndarray | list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
    x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
    x2, y2 = max(x1 + 1, min(width, x2)), max(y1 + 1, min(height, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def ratio_crop(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    ratio_w: float,
    ratio_h: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    face_w = max(1.0, x2 - x1)
    face_h = max(1.0, y2 - y1)
    target_w = min(float(width), max(face_w, face_w / ratio_w))
    target_h = min(float(height), max(face_h, face_h / ratio_h))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1, ny1 = cx - target_w / 2.0, cy - target_h / 2.0
    nx2, ny2 = nx1 + target_w, ny1 + target_h
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0.0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0.0
    if nx2 > width:
        nx1 -= nx2 - width
        nx2 = float(width)
    if ny2 > height:
        ny1 -= ny2 - height
        ny2 = float(height)
    return (
        max(0, int(round(nx1))),
        max(0, int(round(ny1))),
        min(width, int(round(nx2))),
        min(height, int(round(ny2))),
    )


class Detectors:
    def __init__(self, scrfd_model: Path, yunet_model: Path, ctx_id: int, det_thresh: float) -> None:
        import onnxruntime as ort
        from insightface.model_zoo import get_model

        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in ort.get_available_providers()
        ]
        self.scrfd = get_model(str(scrfd_model), providers=providers)
        self.scrfd.prepare(ctx_id=ctx_id, input_size=(640, 640), det_thresh=det_thresh)
        self.yunet_path = str(yunet_model)
        self.yunet = cv2.FaceDetectorYN_create(self.yunet_path, "", (320, 320), 0.6, 0.3, 5000)
        haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(str(haar_path))

    @staticmethod
    def largest(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
        if not boxes:
            return None
        return max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))

    def detect_scrfd(self, bgr: np.ndarray) -> tuple[tuple[int, int, int, int] | None, list[list[float]]]:
        height, width = bgr.shape[:2]
        bboxes, keypoints = self.scrfd.detect(bgr, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return None, []
        index = int(np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
        box = clip_box(bboxes[index, :4], width, height)
        points = keypoints[index].tolist() if keypoints is not None else []
        return box, points

    def detect_yunet(self, bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        height, width = bgr.shape[:2]
        self.yunet.setInputSize((width, height))
        _, faces = self.yunet.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        candidates = []
        for face in faces:
            x, y, w, h = face[:4]
            box = clip_box([x, y, x + w, y + h], width, height)
            if box:
                candidates.append(box)
        return self.largest(candidates)

    def detect_haar(self, bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        height, width = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        candidates = []
        for x, y, w, h in found:
            box = clip_box([x, y, x + w, y + h], width, height)
            if box:
                candidates.append(box)
        return self.largest(candidates)


def serialize_box(box: tuple[int, int, int, int] | None) -> str:
    return json.dumps(list(box)) if box else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/aligned")
    parser.add_argument("--output-manifest", type=Path, default=ROOT / "manifests/alignment.csv")
    parser.add_argument("--quarantine", type=Path, default=ROOT / "manifests/alignment_quarantine.csv")
    parser.add_argument("--scrfd-model", type=Path, default=ROOT / "assets/detectors/scrfd_10g_bnkps.onnx")
    parser.add_argument("--yunet-model", type=Path, default=ROOT / "assets/detectors/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--ctx-id", type=int, default=0)
    parser.add_argument("--det-thresh", type=float, default=0.3)
    parser.add_argument("--ratio-w", type=float, default=0.224)
    parser.add_argument("--ratio-h", type=float, default=0.334)
    parser.add_argument("--validator-iou", type=float, default=0.2)
    parser.add_argument("--minimum-agreeing-detectors", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    manifest = require_input_path(args.manifest)
    output_root = require_output_path(args.output_root)
    output_manifest = require_output_path(args.output_manifest)
    quarantine_path = require_output_path(args.quarantine)
    scrfd_model = require_input_path(args.scrfd_model)
    yunet_model = require_input_path(args.yunet_model)
    frame = pd.read_csv(manifest).sort_values(["dataset", "split", "class", "video_id", "frame_path"])
    frame = frame.rename(columns={"class": "class_name"})
    if args.limit > 0:
        frame = frame.groupby(["dataset", "split", "class_name"], group_keys=False).head(args.limit)

    detectors = Detectors(scrfd_model, yunet_model, args.ctx_id, args.det_thresh)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for count, row in enumerate(frame.itertuples(index=False), start=1):
        source = Path(str(row.frame_path))
        # Canonical crops are always lossless PNG so the alignment stage does
        # not introduce an unmeasured JPEG recompression variable.
        destination = output_root / str(row.dataset) / str(row.split) / str(row.class_name) / f"{row.sample_id}.png"
        meta_path = destination.with_suffix(".json")
        if args.skip_existing and destination.is_file() and meta_path.is_file():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            accepted.append(metadata)
            continue
        try:
            bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("decode_failed")
            height, width = bgr.shape[:2]
            primary, keypoints = detectors.detect_scrfd(bgr)
            yunet = detectors.detect_yunet(bgr)
            haar = detectors.detect_haar(bgr)
            if primary is None:
                raise ValueError("scrfd_no_face")
            yunet_iou = box_iou(primary, yunet)
            haar_iou = box_iou(primary, haar)
            agreeing = 1 + int(yunet_iou >= args.validator_iou) + int(haar_iou >= args.validator_iou)
            if agreeing < args.minimum_agreeing_detectors:
                raise ValueError(f"detector_disagreement:{agreeing}")
            crop = ratio_crop(primary, width, height, args.ratio_w, args.ratio_h)
            x1, y1, x2, y2 = crop
            if x2 <= x1 or y2 <= y1:
                raise ValueError("empty_crop")
            face_cx = (primary[0] + primary[2]) / 2.0
            face_cy = (primary[1] + primary[3]) / 2.0
            center_error_x = abs(face_cx - (x1 + x2) / 2.0) / max(1.0, x2 - x1)
            center_error_y = abs(face_cy - (y1 + y2) / 2.0) / max(1.0, y2 - y1)
            if max(center_error_x, center_error_y) > 0.35:
                raise ValueError("off_center")
            rgb = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(destination)
            metadata = {
                "sample_id": row.sample_id,
                "dataset": row.dataset,
                "split": row.split,
                "class": row.class_name,
                "video_id": row.video_id,
                "source_group_id": row.source_group_id,
                "source_path": str(source),
                "aligned_path": str(destination),
                "source_width": width,
                "source_height": height,
                "aligned_width": x2 - x1,
                "aligned_height": y2 - y1,
                "scrfd_bbox": serialize_box(primary),
                "scrfd_keypoints": json.dumps(keypoints),
                "yunet_bbox": serialize_box(yunet),
                "haar_bbox": serialize_box(haar),
                "yunet_iou": yunet_iou,
                "haar_iou": haar_iou,
                "agreeing_detectors": agreeing,
                "crop_bbox": serialize_box(crop),
                "ratio_w": args.ratio_w,
                "ratio_h": args.ratio_h,
                "padding": 0,
                "resized": False,
                "center_error_x": center_error_x,
                "center_error_y": center_error_y,
                "status": "pass",
            }
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            accepted.append(metadata)
        except Exception as exc:  # pylint: disable=broad-except
            rejected.append(
                {
                    "sample_id": row.sample_id,
                    "dataset": row.dataset,
                    "split": row.split,
                    "class": row.class_name,
                    "video_id": row.video_id,
                    "source_group_id": row.source_group_id,
                    "source_path": str(source),
                    "status": "quarantine",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
        if count % 100 == 0:
            print(f"aligned={len(accepted)} quarantine={len(rejected)} processed={count}/{len(frame)}", flush=True)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accepted).to_csv(output_manifest, index=False)
    pd.DataFrame(rejected).to_csv(quarantine_path, index=False)
    print(f"Wrote {output_manifest} rows={len(accepted)}")
    print(f"Wrote {quarantine_path} rows={len(rejected)}")
    if not accepted:
        raise RuntimeError("No aligned faces passed the integrity gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
