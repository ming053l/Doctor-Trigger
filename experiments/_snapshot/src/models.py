"""Image-only detector behind S_df: a ResNet backbone with a trigger head.

Trimmed from the original training code. Only the plain ResNet path is kept: every detector
checkpoint used in the paper holds exactly ``backbone.*`` and ``trigger_head.*`` tensors, and the
module attributes below are named so those checkpoints load with ``strict=True``. Cross-attention,
order heads, Swin, Xception and the LRGCN backbone were never used and are not included.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TriggerHead(nn.Module):
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
    def __init__(self, backbone: str = "resnet18", trigger_dim: int = 128, num_classes: int = 2,
                 pretrained: bool = False):
        super().__init__()
        backbone = str(backbone).lower()
        self.backbone_name = backbone
        self.backbone_kind, net, in_features = self._build_backbone(backbone, pretrained)
        self.use_cross_attn = False
        net.fc = nn.Linear(in_features, num_classes)
        self.trigger_head = TriggerHead(in_features, trigger_dim)
        self.backbone = net
        self.order_head = None

    @staticmethod
    def _build_backbone(backbone: str, pretrained: bool) -> Tuple[str, nn.Module, int]:
        if backbone not in {"resnet18", "resnet50"}:
            raise ValueError(f"Unsupported backbone: {backbone!r} (this module builds resnet18/resnet50 only)")
        weights = None
        if pretrained:
            try:
                weights = models.ResNet50_Weights.DEFAULT if backbone == "resnet50" else models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        fn = models.resnet50 if backbone == "resnet50" else models.resnet18
        try:
            net = fn(weights=weights)
        except TypeError:
            net = fn(pretrained=bool(pretrained))
        return "resnet", net, int(net.fc.in_features)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self._forward_features(x)
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        cls_vec = pooled.view(pooled.size(0), -1)
        logits = self.backbone.fc(cls_vec)
        trigger_score = self.trigger_head(features)
        return {"logits": logits, "trigger": trigger_score}

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        return b.layer4(b.layer3(b.layer2(b.layer1(x))))


def build_model(cfg: Dict[str, Any]) -> ImageOnlyDetector:
    m = cfg["model"]
    if m.get("use_cross_attn") or (m.get("order_head") or {}).get("enabled") or m.get("dfdml"):
        raise ValueError("this module builds only the plain ResNet detector "
                         "(no cross-attention, order head, or LRGCN backbone)")
    pretrained = m.get("pretrained", False)
    pretrained_flag = bool(pretrained) and str(pretrained).lower() not in {"false", "0", "none", "no"}
    return ImageOnlyDetector(backbone=m.get("backbone", "resnet18"), trigger_dim=m.get("trigger_head_dim", 128),
                    pretrained=pretrained_flag)
