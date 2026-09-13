"""Benign channels applied to a query image: JPEG, noise, blur, rescale, and one-frame x264/x265.

Trimmed from the original robustness-evaluation script; only ``transform_image`` is kept.
The x264/x265 channels call ``ffmpeg``, which must be on PATH.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        with tempfile.TemporaryDirectory(prefix="channel_codec_") as tmp:
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
