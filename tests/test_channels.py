"""doctor_trigger.channels against the implementations it replaces, on real released/query images.

References (frozen outputs, see golden_values.py): the former benign_channels.transform_image
(tests/golden/benign_channels.json), the quickstart's former ``jpeg``/``resized``
(tests/golden/quickstart_copies.json)
and the Table 4 repost/screenshot resize (copied below verbatim, because
importing table4_artifact.py runs the experiment). Every comparison is exact.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from doctor_trigger import channels
from golden_values import digest, load

CASES = Path("/ssd2/ming/ttf_run/public_view/common/cases.csv")


OLD_CHANNELS = load("benign_channels")


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
    expected = OLD_CHANNELS["transform_image"][name]
    assert len(expected) == len(images)
    for k, (img, old) in enumerate(zip(images, expected)):
        seed = (k + 17) * 1009
        new_pil = channels.apply_channel(name, img, seed)
        new_arr = channels.apply_channel(name, np.asarray(img), seed)
        assert isinstance(new_pil, Image.Image) and isinstance(new_arr, np.ndarray)
        assert digest(np.asarray(new_pil)) == old, (name, k)
        assert digest(new_arr) == old, (name, k)
        assert new_arr.dtype == np.uint8 and new_arr.shape == np.asarray(img).shape


def test_registry_and_parametric_names(images):
    for name in channels.CHANNELS:
        assert channels.get_channel(name) is channels.CHANNELS[name]
    img = images[0]
    for q in (60, 95):  # not in CHANNELS: parsed like transform_image did
        assert digest(channels.apply_channel(f"jpeg_q{q}", np.asarray(img), 0)) == OLD_CHANNELS["parametric_jpeg"][str(q)]
    with pytest.raises(ValueError):
        channels.apply_channel("nope", np.asarray(img), 0)


def test_noise_seed_matters(images):
    a = channels.apply_channel("noise_sigma4", np.asarray(images[0]), 1)
    b = channels.apply_channel("noise_sigma4", np.asarray(images[0]), 2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------- quickstart copies
def test_quickstart_jpeg_and_resize(images):
    expected = load("quickstart_copies")["images"]
    assert len(expected) == len(images)
    for img, old in zip(images, expected):
        a = np.asarray(img)
        for q in (70, 75, 80, 85, 90):
            assert digest(channels.jpeg(a, q)) == old["jpeg"][str(q)]
        for s in (0.6, 0.75, 0.9):
            assert digest(channels.rescale_round_trip(a, s)) == old["resized"][str(s)]
        assert digest(channels.jpeg(channels.rescale_round_trip(a, 0.75), 80)) == old["jpeg80_resized075"]


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
