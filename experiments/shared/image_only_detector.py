"""The image-only detector g_ψ behind S_df = g_ψ(J₂₂₄) (paper §3.3, "image-only review").

A ResNet-50 classifier that sees only the query pixels, resized to 224 × 224, and never the
release record. Checkpoints are per training corpus, ``<root>/<tag>/best_val_auc.pt`` with tag
``ffpp``, ``dfdc`` or ``celebdf``, under config key ``checkpoints.image_detector_aug`` (the
deployed augmentation recipe) or ``checkpoints.image_detector`` (same architecture, trained
without augmentation; appendix E.2).

The architecture is ``experiments/_snapshot/src/models.py`` (config key ``model_snapshot``),
loaded here by file path so it does not collide with other ``src`` packages. The benign-channel,
engine, ring-transplant, diffusion, adaptive-attack and evaluation scripts all build, load and
score the detector through this module.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image

from doctor_trigger import config
from doctor_trigger.manifests import DATASET_TAGS

INPUT_SIZE = 224
INPUT_TRANSFORM = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor()])
"""PIL image -> (3, 224, 224) float tensor in [0, 1] (bilinear resize, no normalization)."""

CHECKPOINT_OVERRIDE_ENV = "IMAGE_DETECTOR_CKPT"
"""Set by ``reproduce ... head=<tag>`` / ``plain_head=<tag>`` to choose the detector checkpoint."""


@lru_cache(maxsize=None)
def _architecture() -> ModuleType:
    file = Path(config.path("model_snapshot")) / "src" / "models.py"
    spec = importlib.util.spec_from_file_location("doctor_trigger_image_only_detector_models", file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dataset_tag(dataset: str) -> str:
    """``"FF++"`` -> ``"ffpp"``; tags pass through unchanged."""
    return DATASET_TAGS.get(dataset, dataset)


def checkpoint_path(dataset: str, augmented: bool = True) -> str:
    """Checkpoint of the detector trained on ``dataset`` (label or tag)."""
    root = config.path("checkpoints.image_detector_aug" if augmented else "checkpoints.image_detector")
    return os.path.join(root, dataset_tag(dataset), "best_val_auc.pt")


def deployed_checkpoint_path(dataset: str) -> str:
    """``$IMAGE_DETECTOR_CKPT`` when set, else the augmented detector of ``dataset``."""
    return os.environ.get(CHECKPOINT_OVERRIDE_ENV) or checkpoint_path(dataset, augmented=True)


def build_image_only_detector(pretrained_backbone: bool = False) -> torch.nn.Module:
    """The untrained architecture from the snapshot model config (config key ``model_config``).

    ``pretrained_backbone=False`` skips the ImageNet weight download; it makes no difference once a
    checkpoint is loaded, because every parameter and buffer comes from the checkpoint.
    """
    with open(config.path("model_config"), encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)
    if not pretrained_backbone:
        model_config["model"]["pretrained"] = False
    return _architecture().build_model(model_config)


def load_image_only_detector(checkpoint: str | Path, device: str = "cuda") -> torch.nn.Module:
    """Build the detector, load ``checkpoint`` (a plain state dict) strictly, move it, eval mode."""
    model = build_image_only_detector().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    return model.eval()


def _logits(output: torch.Tensor | dict) -> torch.Tensor:
    return output["logits"] if isinstance(output, dict) else output


@torch.no_grad()
def image_only_score(model: torch.nn.Module, image: Image.Image | np.ndarray, device: str = "cuda") -> float:
    """S_df for one RGB image: the softmax probability of the "manipulated" class."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image).astype(np.uint8))
    logits = _logits(model(INPUT_TRANSFORM(image).unsqueeze(0).to(device)))
    if logits.ndim > 1 and logits.shape[1] == 2:
        return float(torch.softmax(logits, 1)[0, 1])
    return float(torch.sigmoid(logits.reshape(-1))[0])


def image_only_logit(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Differentiable detector logit (manipulated minus real) for a batch ``x`` in [0, 1], any size.

    Resizes with bilinear interpolation (``align_corners=False``) to 224 × 224; used by the
    gradient-based adaptive attacks.
    """
    logits = _logits(model(F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)))
    return logits[:, 1] - logits[:, 0] if logits.ndim > 1 and logits.shape[1] == 2 else logits.reshape(-1)
