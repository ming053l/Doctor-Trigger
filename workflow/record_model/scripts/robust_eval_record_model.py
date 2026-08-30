#!/usr/bin/env python3
"""Robustness eval for record_model_release RecordModel from existing sample rows and checkpoints."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.record_model import RecordModelConfig, RecordModelMLP, safe_auc, safe_eer, tpr_at_fpr
from src.trigger_conditioned import feature_vector

ImageFile.LOAD_TRUNCATED_IMAGES = True

_CTX: Dict[str, object] = {}


def clip_bbox(bbox: Iterable[float], shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    vals = list(bbox)
    if len(vals) != 4:
        return 0, 0, w, h
    x1, y1, x2, y2 = [int(round(float(v))) for v in vals]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def rgb_to_y(arr: np.ndarray) -> np.ndarray:
    arr01 = np.clip(arr / 255.0, 0.0, 1.0).astype(np.float64)
    return 0.299 * arr01[:, :, 0] + 0.587 * arr01[:, :, 1] + 0.114 * arr01[:, :, 2]


def transform_image(img: Image.Image, perturbation: str, seed: int) -> Image.Image:
    img = img.convert("RGB")
    if perturbation == "clean":
        return img
    if perturbation.startswith("jpeg_q"):
        q = int(perturbation.replace("jpeg_q", ""))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if perturbation == "noise_sigma4":
        rng = np.random.default_rng(seed)
        arr = np.asarray(img, dtype=np.float32)
        arr = np.clip(arr + rng.normal(0.0, 4.0, size=arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if perturbation == "blur_k7":
        return img.filter(ImageFilter.GaussianBlur(radius=3.0))
    if perturbation == "scale_0.5":
        w, h = img.size
        small = img.resize((max(1, w // 2), max(1, h // 2)), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if perturbation.startswith("x264_crf") or perturbation.startswith("x265_crf"):
        crf = perturbation.split("crf", 1)[1]
        codec = "libx264" if perturbation.startswith("x264") else "libx265"
        orig_size = img.size
        pad_w = orig_size[0] + (orig_size[0] % 2)
        pad_h = orig_size[1] + (orig_size[1] % 2)
        if (pad_w, pad_h) != orig_size:
            padded = Image.new("RGB", (pad_w, pad_h))
            padded.paste(img, (0, 0))
            if pad_w > orig_size[0]:
                padded.paste(img.crop((orig_size[0] - 1, 0, orig_size[0], orig_size[1])), (orig_size[0], 0))
            if pad_h > orig_size[1]:
                padded.paste(padded.crop((0, orig_size[1] - 1, pad_w, orig_size[1])), (0, orig_size[1]))
            img_for_codec = padded
        else:
            img_for_codec = img
        with tempfile.TemporaryDirectory(prefix="record_model_release_codec_") as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "in.png"
            video_path = tmp_path / "encoded.mp4"
            out_path = tmp_path / "out.png"
            img_for_codec.save(in_path)
            cmd_encode = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-i",
                str(in_path),
                "-frames:v",
                "1",
                "-c:v",
                codec,
                "-threads",
                "1",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
            ]
            if codec == "libx265":
                cmd_encode.extend(["-x265-params", "pools=1:frame-threads=1:log-level=error"])
            cmd_encode.append(str(video_path))
            cmd_decode = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video_path), "-frames:v", "1", str(out_path)]
            subprocess.run(cmd_encode, check=True)
            subprocess.run(cmd_decode, check=True)
            return Image.open(out_path).convert("RGB").crop((0, 0, orig_size[0], orig_size[1]))
    raise ValueError(f"unknown perturbation: {perturbation}")


def phase_from_image(img: Image.Image, bbox: Iterable[float]) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    x1, y1, x2, y2 = clip_bbox(bbox, arr.shape)
    roi = arr[y1:y2, x1:x2]
    if roi.size == 0:
        roi = arr
    return np.angle(np.fft.fft2(rgb_to_y(roi)))


def resize_phase_like(phi: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    if phi.shape == ref_shape:
        return phi
    img = Image.fromarray(((phi + np.pi) / (2.0 * np.pi) * 65535.0).astype(np.uint16))
    img = img.resize((ref_shape[1], ref_shape[0]), Image.BILINEAR)
    return np.asarray(img).astype(np.float64) / 65535.0 * (2.0 * np.pi) - np.pi


def init_worker(ctx: Dict[str, object]) -> None:
    global _CTX
    _CTX = ctx


def compute_one(task: Tuple[int, Dict[str, object], str]) -> Dict[str, object]:
    idx, row, perturbation = task
    source_path = Path(str(row["source_path"]))
    template_path = Path(str(row["template_id"]))
    meta = json.loads(template_path.with_suffix(".json").read_text(encoding="utf-8"))
    bbox = meta.get("bbox", [0, 0, 10_000, 10_000])
    bands_raw = meta.get("bands", [float(row.get("kappa_min", 0.1)), float(row.get("kappa_max", 0.45))])
    bands = (float(bands_raw[0]), float(bands_raw[1]))
    template_img = Image.open(template_path).convert("RGB")
    query_img = transform_image(Image.open(source_path).convert("RGB"), perturbation, seed=(idx + 17) * 1009)
    template_phase = phase_from_image(template_img, bbox)
    query_phase = resize_phase_like(phase_from_image(query_img, bbox), template_phase.shape)
    feats = feature_vector(
        query_phase,
        template_phase,
        key=int(row["key_id"]),
        alpha=float(row["alpha"]),
        beta=float(row["beta"]),
        bands=bands,
        code_mode=str(row.get("code_mode", "zero_pi")),
        n_wrong=int(_CTX["n_wrong"]),
        n_bands=int(_CTX["n_bands"]),
        host_key=int(row.get("original_key_id", row["key_id"])),
    )
    return {
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "protocol": row["protocol"],
        "perturbation": perturbation,
        "source_path": str(source_path),
        "template_id": str(template_path),
        "key_id": int(row["key_id"]),
        "original_key_id": int(row.get("original_key_id", row["key_id"])),
        "alpha": float(row["alpha"]),
        "beta": float(row["beta"]),
        "kappa_min": bands[0],
        "kappa_max": bands[1],
        "code_mode": str(row.get("code_mode", "zero_pi")),
        "y_fake": int(row["y_fake"]),
        "y_rel": int(row["y_rel"]),
        "y_keycorr": int(row["y_keycorr"]),
        **feats,
    }


def predict(features: pd.DataFrame, ckpt_path: Path) -> pd.DataFrame:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cols = ckpt["feature_columns"]
    config = RecordModelConfig(variant=str(ckpt["variant"]), input_dim=len(cols), hidden_dim=64)
    model = RecordModelMLP(config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x_raw = features[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x = (np.nan_to_num(x_raw, nan=np.asarray(ckpt["mean"])) - np.asarray(ckpt["mean"])) / np.asarray(ckpt["std"])
    with torch.no_grad():
        out = model(torch.tensor(x.astype(np.float32)))
    scores = features[["sample_id", "dataset", "protocol", "perturbation", "y_fake", "y_rel", "y_keycorr"]].copy()
    for key in ["S_fake", "S_rel", "S_key"]:
        scores[key] = torch.sigmoid(out[key]).numpy()
    return scores


def metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    targets = {"S_fake": "y_fake", "S_rel": "y_rel", "S_key": "y_keycorr"}
    for (perturb, protocol), group in scores.groupby(["perturbation", "protocol"], dropna=False):
        for score, target in targets.items():
            y = group[target].to_numpy(dtype=int)
            s = group[score].to_numpy(dtype=float)
            rows.append(
                {
                    "perturbation": perturb,
                    "protocol": protocol,
                    "score": score,
                    "target": target,
                    "auc": safe_auc(y, s),
                    "eer": safe_eer(y, s),
                    "tpr@1e-2": tpr_at_fpr(y, s, 1e-2),
                    "tpr@1e-3": tpr_at_fpr(y, s, 1e-3),
                    "tpr@1e-4": tpr_at_fpr(y, s, 1e-4),
                    "n": int(len(group)),
                    "n_pos": int(np.sum(y == 1)),
                    "n_neg": int(np.sum(y == 0)),
                }
            )
    for perturb, group in scores.groupby("perturbation", dropna=False):
        for score, target in targets.items():
            y = group[target].to_numpy(dtype=int)
            s = group[score].to_numpy(dtype=float)
            rows.append(
                {
                    "perturbation": perturb,
                    "protocol": "ALL",
                    "score": score,
                    "target": target,
                    "auc": safe_auc(y, s),
                    "eer": safe_eer(y, s),
                    "tpr@1e-2": tpr_at_fpr(y, s, 1e-2),
                    "tpr@1e-3": tpr_at_fpr(y, s, 1e-3),
                    "tpr@1e-4": tpr_at_fpr(y, s, 1e-4),
                    "n": int(len(group)),
                    "n_pos": int(np.sum(y == 1)),
                    "n_neg": int(np.sum(y == 0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("results/record_model_release/features/record_model_release_matrix_ffpp_test_features.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/record_model_release/checkpoints/full_record_model_seed0_record_model_ffpp_train_cross_eval.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/record_model_release/robust"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-per-protocol", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--perturbations", nargs="+", default=["clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "noise_sigma4", "blur_k7", "scale_0.5"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.out_dir / "record_model_release_robust_features.csv"
    df = pd.read_csv(args.features)
    df = df[df["protocol"].isin(["benign", "swap"])].copy()
    if args.limit_per_protocol > 0:
        df = df.groupby("protocol", group_keys=False).head(args.limit_per_protocol).copy()
    tasks = [(idx, row.to_dict(), perturb) for perturb in args.perturbations for idx, row in df.iterrows()]
    expected = len(tasks)
    existing_rows: List[Dict[str, object]] = []
    done_keys = set()
    if features_path.exists():
        existing = pd.read_csv(features_path)
        if not existing.empty and {"sample_id", "perturbation"}.issubset(existing.columns):
            existing_rows = existing.to_dict("records")
            done_keys = set(zip(existing["sample_id"].astype(str), existing["perturbation"].astype(str)))
            print(f"[record_model_release_robust_eval] resume existing features: {len(existing_rows)}/{expected}", flush=True)
    remaining = [(idx, row, perturb) for idx, row, perturb in tasks if (str(row["sample_id"]), str(perturb)) not in done_keys]
    rows = existing_rows
    if remaining:
        ctx = {"n_wrong": 8, "n_bands": 8}
        if args.num_workers <= 1:
            init_worker(ctx)
            for local_done, task in enumerate(remaining, start=1):
                rows.append(compute_one(task))
                total_done = len(rows)
                if total_done % 100 == 0 or local_done == 1:
                    print(f"[record_model_release_robust_eval] {total_done}/{expected}", flush=True)
                if args.save_every > 0 and (total_done % args.save_every == 0):
                    pd.DataFrame(rows).to_csv(features_path, index=False)
            features = pd.DataFrame(rows)
            features.to_csv(features_path, index=False)
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker, initargs=(ctx,)) as pool:
                for local_done, row in enumerate(pool.map(compute_one, remaining, chunksize=8), start=1):
                    rows.append(row)
                    total_done = len(rows)
                    if total_done % 500 == 0 or local_done == 1:
                        print(f"[record_model_release_robust_eval] {total_done}/{expected}", flush=True)
                    if args.save_every > 0 and (total_done % args.save_every == 0):
                        pd.DataFrame(rows).to_csv(features_path, index=False)
            features = pd.DataFrame(rows)
            features.to_csv(features_path, index=False)
    else:
        print(f"[record_model_release_robust_eval] reusing complete existing features: {features_path}", flush=True)
        features = pd.DataFrame(rows)
    scores = predict(features, args.checkpoint)
    scores_path = args.out_dir / "record_model_release_robust_scores.csv"
    scores.to_csv(scores_path, index=False)
    metrics_df = metrics(scores)
    metrics_path = args.out_dir / "record_model_release_robust_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(json.dumps({"features": str(features_path.resolve()), "scores": str(scores_path.resolve()), "metrics": str(metrics_path.resolve()), "rows": len(features)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
