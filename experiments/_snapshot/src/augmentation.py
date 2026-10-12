"""Training-time distortion augmentation of the image-only detector (the deployed recipe, App. E.2).

``build_augmentation(cfg)`` turns the ``train.albumentations`` block of
config_ffpp_std_dfd_resnet50.yaml into an albumentations pipeline: JPEG recompression, Gaussian
blur, Gaussian noise and hue/saturation/value jitter, each applied with its own probability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import albumentations as A


def build_augmentation(cfg: Optional[Dict[str, Any]]) -> Optional[A.Compose]:
    """The albumentations pipeline described by ``cfg``, or None when it is absent or not enabled."""
    if not cfg or not cfg.get("enabled", False):
        return None
    transforms: List[Any] = []
    if cfg.get("jpeg"):
        jpeg_cfg = cfg["jpeg"]
        transforms.append(
            A.ImageCompression(
                p=jpeg_cfg.get("p", 0.5),
                quality_lower=jpeg_cfg.get("quality_min", 30),
                quality_upper=jpeg_cfg.get("quality_max", 50),
            )
        )
    if cfg.get("blur"):
        blur_cfg = cfg["blur"]
        transforms.append(
            A.GaussianBlur(
                p=blur_cfg.get("p", 0.3),
                blur_limit=blur_cfg.get("blur_limit", (3, 7)),
            )
        )
    if cfg.get("noise"):
        noise_cfg = cfg["noise"]
        transforms.append(
            A.GaussNoise(
                p=noise_cfg.get("p", 0.3),
                var_limit=noise_cfg.get("var_limit", (5.0, 30.0)),
            )
        )
    if cfg.get("color"):
        color_cfg = cfg["color"]
        transforms.append(
            A.HueSaturationValue(
                p=color_cfg.get("p", 0.4),
                hue_shift_limit=color_cfg.get("hue_limit", 8),
                sat_shift_limit=color_cfg.get("sat_limit", 12),
                val_shift_limit=color_cfg.get("val_limit", 12),
            )
        )
    if not transforms:
        return None
    return A.Compose(transforms)
