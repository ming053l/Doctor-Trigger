"""Training-time augmentation for the image-only detector (the deployed recipe).

Trimmed from the original dataset module; only ``build_albu_transform`` is kept."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import albumentations as A
except ImportError:  # pragma: no cover - optional
    A = None


def build_albu_transform(cfg: Optional[Dict[str, Any]]) -> Optional["A.Compose"]:
    if not cfg or not cfg.get("enabled", False):
        return None
    if A is None:
        raise RuntimeError("Albumentations is not installed but albumentations config is enabled.")
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
    if cfg.get("gamma"):
        gamma_cfg = cfg["gamma"]
        transforms.append(
            A.RandomGamma(
                p=gamma_cfg.get("p", 0.3),
                gamma_limit=gamma_cfg.get("gamma_limit", (80, 120)),
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
    if cfg.get("affine"):
        affine_cfg = cfg["affine"]
        transforms.append(
            A.Affine(
                p=affine_cfg.get("p", 0.2),
                scale=affine_cfg.get("scale", (0.95, 1.05)),
                translate_percent=affine_cfg.get("translate", (0.0, 0.02)),
                rotate=affine_cfg.get("rotate", (-2, 2)),
            )
        )
    if not transforms:
        return None
    return A.Compose(transforms)
