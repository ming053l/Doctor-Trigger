"""Image-only detector g_psi behind S_df: a ResNet-50 classifier on the query pixels.

Built from the ``model`` section of config_ffpp_std_dfd_resnet50.yaml. Every detector checkpoint
used in the paper holds exactly ``backbone.*`` and ``trigger_head.*`` tensors, and the module
attributes below are named so those checkpoints load with ``strict=True``. ``trigger_head`` is an
auxiliary head of the original training code: it is loaded and counted in the parameter total
(Table 16) but its output is not used by any score.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AuxiliaryHead(nn.Module):
    """Global-average-pooled features -> one scalar; kept only so checkpoints load strictly."""

    def __init__(self, in_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])
        return self.net(x).squeeze(-1)


class ImageOnlyDetector(nn.Module):
    """ResNet backbone whose ``fc`` gives the two class logits (real, manipulated)."""

    def __init__(self, backbone: str = "resnet18", auxiliary_dim: int = 128, num_classes: int = 2,
                 pretrained: bool = False):
        super().__init__()
        net, in_features = self._build_backbone(str(backbone).lower(), pretrained)
        net.fc = nn.Linear(in_features, num_classes)
        self.trigger_head = AuxiliaryHead(in_features, auxiliary_dim)
        self.backbone = net

    @staticmethod
    def _build_backbone(backbone: str, pretrained: bool) -> Tuple[nn.Module, int]:
        if backbone not in {"resnet18", "resnet50"}:
            raise ValueError(f"Unsupported backbone: {backbone!r} (this module builds resnet18/resnet50 only)")
        weights = None
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT if backbone == "resnet50" else models.ResNet18_Weights.DEFAULT
        net = (models.resnet50 if backbone == "resnet50" else models.resnet18)(weights=weights)
        return net, int(net.fc.in_features)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = self.backbone
        features = b.layer4(b.layer3(b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x))))))))
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        return {"logits": b.fc(pooled.view(pooled.size(0), -1))}


def build_model(cfg: Dict[str, Any]) -> ImageOnlyDetector:
    """Detector from ``cfg["model"]``: ``backbone``, ``pretrained`` (ImageNet weights), ``trigger_head_dim``."""
    m = cfg["model"]
    return ImageOnlyDetector(backbone=m.get("backbone", "resnet18"), auxiliary_dim=m.get("trigger_head_dim", 128),
                             pretrained=bool(m.get("pretrained", False)))
