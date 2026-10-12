"""Benign and delivery channels, the set 𝒞 of processing a deployment supports (paper App. A).

A channel P maps a released or query image to the image a verifier receives after delivery
(re-saving, resizing, blurring, a video codec, ...). Under the benign hypothesis H_c the query is
P(I') with P ∈ 𝒞; thresholds are calibrated per channel on channel-processed benign releases.

Every channel here works on one RGB image held as an ``(H, W, 3)`` uint8 array and takes a
``numpy.random.Generator`` for channels that draw noise:

    Channel = Callable[[np.ndarray, np.random.Generator], np.ndarray]

``CHANNELS`` maps the channel labels used in the experiments and the paper tables ("jpeg_q90",
"blur_k7", "x264_crf30", ...) to channels; ``apply_channel(name, image, seed)`` applies one by
name, ``compose(*names)`` chains several. The pixel operations are moved verbatim from the
earlier ``benign_channels.transform_image`` (and from the JPEG/resize copies in the quickstart and
the Table 4 script), so every output is bit-identical to those implementations, including
``transform_image(img, name, seed) == apply_channel(name, img, seed)``.

Labels follow the paper's notation table: "blur k7" is a channel label, the blur is a Gaussian of
radius 3 (App. A, "JPEG q90, blur k7"); "scale_0.5" halves each side with bicubic resampling and
restores the enrollment geometry.

Video channels. A video codec acts on a frame sequence, not one image. The video phase will add

    ClipChannel = Callable[[Iterable[np.ndarray]], Iterator[np.ndarray]]

and a second dict ``CLIP_CHANNELS`` (H.264 / H.265 through ffmpeg over whole clips) next to
``CHANNELS``. The one-frame x264/x265 channels below (``single_frame_codec``) encode each image as
a one-frame video; they are the starting point for that code and stay image channels here.
"""

from __future__ import annotations

import io
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFile, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True

Channel = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


def _to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


# ----------------------------------------------------------------------------- pixel operations
def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """Re-save as JPEG at ``quality`` (PIL defaults otherwise: 4:2:0 subsampling, no optimize)."""
    buf = io.BytesIO()
    _to_pil(image).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return _to_array(Image.open(buf))


def gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add i.i.d. Gaussian noise of standard deviation ``sigma`` (in 8-bit units), clip, truncate to uint8."""
    arr = np.asarray(image, dtype=np.float32)
    return np.clip(arr + rng.normal(0.0, sigma, size=arr.shape), 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, radius: float) -> np.ndarray:
    """PIL Gaussian blur of the given radius (the paper's "blur k7" uses radius 3)."""
    return _to_array(_to_pil(image).filter(ImageFilter.GaussianBlur(radius=radius)))


def resize(image: np.ndarray, factor: float) -> np.ndarray:
    """Bicubic resize of each side by ``factor`` (``int(side * factor)``); the geometry changes."""
    pil = _to_pil(image)
    return _to_array(pil.resize((int(pil.width * factor), int(pil.height * factor)), Image.BICUBIC))


def rescale_round_trip(image: np.ndarray, factor: float) -> np.ndarray:
    """Bicubic shrink of each side by ``factor``, then bicubic resize back to the original geometry.

    The reduced side is ``max(1, int(side * factor))``; for ``factor = 0.5`` this equals
    ``side // 2``, as in the earlier ``scale_0.5`` channel.
    """
    pil = _to_pil(image)
    w, h = pil.size
    small = pil.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BICUBIC)
    return _to_array(small.resize((w, h), Image.BICUBIC))


def single_frame_codec(image: np.ndarray, codec: str, crf: int | str) -> np.ndarray:
    """Encode the image as a one-frame video with ffmpeg and decode it back (yuv420p).

    ``codec`` is ``"libx264"`` or ``"libx265"``; ``crf`` is the constant-rate factor. Odd sides are
    padded by repeating the last column/row (4:2:0 needs even sizes) and cropped after decoding.
    ffmpeg must be on PATH. Single-threaded encode so the output does not depend on the machine.
    """
    if codec not in ("libx264", "libx265"):
        raise ValueError(f"unsupported codec {codec!r} (libx264 or libx265)")
    img = _to_pil(image)
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
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-loop", "1", "-i", str(in_path), "-frames:v", "1",
            "-c:v", codec, "-threads", "1", "-crf", str(crf), "-pix_fmt", "yuv420p",
        ]
        if codec == "libx265":
            cmd_encode.extend(["-x265-params", "pools=1:frame-threads=1:log-level=error"])
        cmd_encode.append(str(video_path))
        cmd_decode = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video_path),
                      "-frames:v", "1", str(out_path)]
        subprocess.run(cmd_encode, check=True)
        subprocess.run(cmd_decode, check=True)
        with Image.open(out_path) as decoded:
            return _to_array(decoded.convert("RGB").crop((0, 0, orig_size[0], orig_size[1])))


# ----------------------------------------------------------------------------- named channels
def identity(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The clean channel: the image as released."""
    return np.asarray(image, dtype=np.uint8)


def jpeg_channel(quality: int) -> Channel:
    return lambda image, rng: jpeg(image, quality)


def rescale_channel(factor: float) -> Channel:
    return lambda image, rng: rescale_round_trip(image, factor)


def codec_channel(codec: str, crf: int) -> Channel:
    return lambda image, rng: single_frame_codec(image, codec, crf)


CHANNELS: dict[str, Channel] = {
    "clean": identity,
    **{f"jpeg_q{q}": jpeg_channel(q) for q in (50, 70, 75, 80, 85, 90)},
    "noise_sigma4": lambda image, rng: gaussian_noise(image, 4.0, rng),
    "blur_k7": lambda image, rng: gaussian_blur(image, 3.0),
    "scale_0.5": rescale_channel(0.5),
    **{f"x264_crf{crf}": codec_channel("libx264", crf) for crf in (30, 35, 40)},
    **{f"x265_crf{crf}": codec_channel("libx265", crf) for crf in (30, 35, 40)},
}
"""Channel label -> channel. Labels are the on-disk names used in result tables."""

_PARAMETRIC = re.compile(r"^(?:jpeg_q(?P<quality>\d+)|(?P<codec>x264|x265)_crf(?P<crf>\d+))$")


def get_channel(name: str) -> Channel:
    """Look up a channel by label; ``jpeg_q<N>`` and ``x264_crf<N>``/``x265_crf<N>`` accept any N."""
    if name in CHANNELS:
        return CHANNELS[name]
    match = _PARAMETRIC.match(name)
    if match and match["quality"]:
        return jpeg_channel(int(match["quality"]))
    if match:
        return codec_channel("lib" + match["codec"], int(match["crf"]))
    raise ValueError(f"unknown channel: {name}")


def compose(*names: str) -> Channel:
    """Chain channels left to right (``compose("scale_0.5", "jpeg_q80")`` resizes, then re-saves).

    All channels draw from the one generator they are given, in order.
    """
    steps = [get_channel(name) for name in names]

    def composed(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for step in steps:
            image = step(image, rng)
        return image

    return composed


def apply_channel(name: str, image: np.ndarray | Image.Image, seed: int) -> np.ndarray | Image.Image:
    """Apply the channel ``name`` to one image with ``numpy.random.default_rng(seed)``.

    Accepts an RGB uint8 array or a PIL image and returns the same kind (PIL input is converted to
    RGB first, as the earlier ``transform_image`` did).
    """
    channel = get_channel(name)
    rng = np.random.default_rng(seed)
    if isinstance(image, Image.Image):
        return Image.fromarray(channel(_to_array(image), rng))
    return channel(np.asarray(image), rng)
