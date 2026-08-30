#!/usr/bin/env python3
"""Create released images and explicit private phase-template enrollment records."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow.paths import require_input_path, require_output_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(master_seed: int, sample_id: str) -> int:
    payload = f"doctor-trigger-v1\x1f{master_seed}\x1f{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def wrap_phase(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def signed_hermitian_code(shape: tuple[int, int], key: int) -> np.ndarray:
    rng = np.random.default_rng(key)
    raw = rng.standard_normal(shape)
    rows = np.mod(-np.arange(shape[0]), shape[0])
    cols = np.mod(-np.arange(shape[1]), shape[1])
    partner = raw[np.ix_(rows, cols)]
    code = np.sign(raw - partner)
    self_rows = np.arange(shape[0]) == rows
    self_cols = np.arange(shape[1]) == cols
    code[np.ix_(self_rows, self_cols)] = 0.0
    return code.astype(np.float64)


def frequency_radius(shape: tuple[int, int]) -> np.ndarray:
    fy = np.fft.fftfreq(shape[0]).reshape(-1, 1)
    fx = np.fft.fftfreq(shape[1]).reshape(1, -1)
    return np.sqrt(fy * fy + fx * fx) / math.sqrt(0.5**2 + 0.5**2)


def rgb_to_y(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2))
    return 100.0 if mse <= 1e-12 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def embed(
    image: np.ndarray,
    key: int,
    alpha: float,
    beta: float,
    bands: tuple[float, float],
    amplitude_floor_ratio: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    height, width = image.shape[:2]
    code = signed_hermitian_code((height, width), key)
    radius = frequency_radius((height, width))
    host_y = rgb_to_y(image.astype(np.float64))
    host_y_fft = np.fft.fft2(host_y)
    host_amplitude = np.abs(host_y_fft)
    amplitude_floor = float(np.max(host_amplitude) * amplitude_floor_ratio)
    support = (radius >= bands[0]) & (radius <= bands[1]) & (host_amplitude > amplitude_floor) & (code != 0)
    preblend_shift = alpha * np.pi * code * support

    modified_channels = []
    for channel in range(3):
        frequency = np.fft.fft2(image[:, :, channel].astype(np.float64))
        modified = np.abs(frequency) * np.exp(1j * (np.angle(frequency) + preblend_shift))
        modified_channels.append(np.fft.ifft2(modified).real)
    modified_rgb = np.stack(modified_channels, axis=2)
    mixed = (1.0 - beta) * image.astype(np.float64) + beta * modified_rgb
    persisted = np.clip(mixed, 0.0, 255.0).astype(np.uint8)

    effective_delta = np.angle((1.0 - beta) + beta * np.exp(1j * preblend_shift))
    template_phase = wrap_phase(np.angle(host_y_fft) + effective_delta)
    record = {
        "template_phase": np.fft.fftshift(template_phase).astype(np.float16),
        "host_phase": np.fft.fftshift(np.angle(host_y_fft)).astype(np.float16),
        "delta": np.fft.fftshift(effective_delta).astype(np.float16),
        "host_amplitude": np.fft.fftshift(host_amplitude / max(float(np.max(host_amplitude)), 1e-12)).astype(np.float16),
        "support": np.fft.fftshift(support).astype(np.uint8),
        "code": np.fft.fftshift(code).astype(np.int8),
    }
    return persisted, record


def alpha_candidates(dataset: str) -> list[float]:
    if dataset == "DFDC":
        return [0.015, 0.010, 0.005]
    return [0.050, 0.030, 0.015]


def process_enrollment(
    row: Any,
    args: argparse.Namespace,
    released_root: Path,
    template_root: Path,
) -> tuple[str, dict[str, Any]]:
    aligned_path = Path(str(row.aligned_path))
    try:
        aligned_path = require_input_path(aligned_path)
        enrollment_id = hashlib.sha256(f"enrollment-v1\x1f{row.sample_id}".encode("utf-8")).hexdigest()[:24]
        released_path = released_root / str(row.dataset) / str(row.split) / f"{enrollment_id}.png"
        template_path = template_root / str(row.dataset) / str(row.split) / f"{enrollment_id}.npz"
        metadata_path = template_path.with_suffix(".json")
        if args.skip_existing and released_path.is_file() and template_path.is_file() and metadata_path.is_file():
            return "pass", json.loads(metadata_path.read_text(encoding="utf-8"))

        image = np.asarray(Image.open(aligned_path).convert("RGB"), dtype=np.uint8)
        key = stable_key(args.master_seed, str(row.sample_id))
        chosen: tuple[float, np.ndarray, dict[str, np.ndarray], float, float] | None = None
        attempts = []
        for alpha in alpha_candidates(str(row.dataset)):
            released, template = embed(
                image,
                key,
                alpha,
                args.beta,
                (args.kappa_min, args.kappa_max),
                args.amplitude_floor_ratio,
            )
            candidate_psnr = psnr(image, released)
            candidate_ssim = float(structural_similarity(image, released, channel_axis=2, data_range=255))
            attempts.append({"alpha": alpha, "psnr": candidate_psnr, "ssim": candidate_ssim})
            if candidate_psnr >= args.minimum_psnr and candidate_ssim >= args.minimum_ssim:
                chosen = (alpha, released, template, candidate_psnr, candidate_ssim)
                break
        if chosen is None:
            raise ValueError(f"quality_gate_failed:{attempts}")
        alpha, released, template, persisted_psnr, persisted_ssim = chosen
        released_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(released).save(released_path)
        persisted = np.asarray(Image.open(released_path).convert("RGB"), dtype=np.uint8)
        persisted_psnr = psnr(image, persisted)
        persisted_ssim = float(structural_similarity(image, persisted, channel_axis=2, data_range=255))
        if persisted_psnr < args.minimum_psnr or persisted_ssim < args.minimum_ssim:
            raise ValueError(f"persisted_quality_gate_failed:psnr={persisted_psnr}:ssim={persisted_ssim}")
        np.savez_compressed(
            template_path,
            **template,
            key=np.asarray([key], dtype=np.uint64),
            alpha=np.asarray([alpha], dtype=np.float32),
            beta=np.asarray([args.beta], dtype=np.float32),
            bands=np.asarray([args.kappa_min, args.kappa_max], dtype=np.float32),
            amplitude_floor_ratio=np.asarray([args.amplitude_floor_ratio], dtype=np.float32),
        )
        record = {
            "enrollment_id": enrollment_id,
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
            "alpha": alpha,
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
