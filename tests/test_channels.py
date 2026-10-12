"""doctor_trigger.channels against the implementations it replaces, on real released/query images.

References: experiments/_snapshot/scripts/benign_channels.transform_image, the quickstart's
``jpeg``/``resized`` and the Table 4 repost/screenshot resize (copied below verbatim, because
importing table4_artifact.py runs the experiment). Every comparison is exact.
"""

from __future__ import annotations

import importlib.util
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from doctor_trigger import channels

REPO = Path(__file__).resolve().parents[1]
CASES = Path("/ssd2/ming/ttf_run/public_view/common/cases.csv")


def import_by_path(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old_channels = import_by_path("old_benign_channels", REPO / "experiments/_snapshot/scripts/benign_channels.py")


@pytest.fixture(scope="module")
def images() -> list[Image.Image]:
    if not CASES.exists():
        pytest.skip("public_view not available")
    cases = pd.read_csv(CASES)
    picked = pd.concat([cases[(cases.dataset == ds) & (cases.split == "test")].head(2)
                        for ds in ("FF++", "DFDC", "CelebDF")])
    out = []
    for row in picked.itertuples():
        out.append(Image.open(row.input_path).convert("RGB"))
        out.append(Image.open(row.doctor_trigger_released_path).convert("RGB"))
    out.append(out[0].crop((0, 0, out[0].width - 1, out[0].height - 3)))  # odd sides: codec padding path
    return out


OLD_NAMES = ["clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "noise_sigma4", "blur_k7", "scale_0.5"]
CODEC_NAMES = ["x264_crf30", "x264_crf35", "x264_crf40", "x265_crf30"]


@pytest.mark.parametrize("name", OLD_NAMES + CODEC_NAMES)
def test_transform_image_bit_identical(images, name):
    if name in CODEC_NAMES and shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    for k, img in enumerate(images):
        seed = (k + 17) * 1009
        old = np.asarray(old_channels.transform_image(img, name, seed=seed))
        new_pil = channels.apply_channel(name, img, seed)
        new_arr = channels.apply_channel(name, np.asarray(img), seed)
        assert isinstance(new_pil, Image.Image) and isinstance(new_arr, np.ndarray)
        assert np.array_equal(old, np.asarray(new_pil)), (name, k)
        assert np.array_equal(old, new_arr), (name, k)
        assert new_arr.dtype == np.uint8 and new_arr.shape == old.shape


def test_registry_and_parametric_names(images):
    for name in channels.CHANNELS:
        assert channels.get_channel(name) is channels.CHANNELS[name]
    img = images[0]
    for q in (60, 95):  # not in CHANNELS: parsed like transform_image did
        assert np.array_equal(np.asarray(old_channels.transform_image(img, f"jpeg_q{q}", 0)),
                              channels.apply_channel(f"jpeg_q{q}", np.asarray(img), 0))
    with pytest.raises(ValueError):
        channels.apply_channel("nope", np.asarray(img), 0)


def test_noise_seed_matters(images):
    a = channels.apply_channel("noise_sigma4", np.asarray(images[0]), 1)
    b = channels.apply_channel("noise_sigma4", np.asarray(images[0]), 2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------- quickstart copies
def test_quickstart_jpeg_and_resize(images):
    quickstart = import_by_path("old_quickstart", REPO / "examples/quickstart.py")
    for img in images:
        a = np.asarray(img)
        for q in (70, 75, 80, 85, 90):
            assert np.array_equal(quickstart.jpeg(a, q), channels.jpeg(a, q))
        for s in (0.6, 0.75, 0.9):
            assert np.array_equal(quickstart.resized(a, s), channels.rescale_round_trip(a, s))
        assert np.array_equal(quickstart.jpeg(quickstart.resized(a, 0.75), 80),
                              channels.jpeg(channels.rescale_round_trip(a, 0.75), 80))


def test_compose_order(images):
    a = np.asarray(images[1])
    composed = channels.compose("scale_0.5", "jpeg_q80")(a, np.random.default_rng(0))
    assert np.array_equal(composed, channels.jpeg(channels.rescale_round_trip(a, 0.5), 80))


# ---------------------------------------------------------------- Table 4 copies (verbatim)
def _t4_attach_jpeg(img, quality):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=quality or 85)
    b.seek(0)
    return Image.open(b)


def _t4_repost(img):
    x = img.resize((int(img.width * 0.75), int(img.height * 0.75)), Image.BICUBIC)
    return _t4_attach_jpeg(x, 70)


def _t4_screenshot_pixels(img):
    return img.resize((int(img.width * 1.25), int(img.height * 1.25)), Image.BICUBIC)


def test_table4_resize_channels(images):
    for img in images:
        a = np.asarray(img)
        assert np.array_equal(np.asarray(_t4_repost(img).convert("RGB")), channels.jpeg(channels.resize(a, 0.75), 70))
        assert np.array_equal(np.asarray(_t4_screenshot_pixels(img)), channels.resize(a, 1.25))
