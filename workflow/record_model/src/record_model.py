"""Minimal configurable Trigger-Conditioned Relation Head for record_model_release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn


HOST_COLUMNS = ["H", "C", "residual_mean", "residual_std", "cos_consistency_mean"]
TRIGGER_COLUMNS = ["G_s", "G_wrong_mean", "G_wrong_max", "G_margin", "G_z", "sin_projection_mean"]


def _band_columns(columns: Iterable[str]) -> List[str]:
    prefixes = ("H_b", "C_b", "G_b", "G_margin_b", "residual_mean_b", "residual_std_b", "sin_projection_mean_b", "cos_consistency_mean_b")
    return [col for col in columns if col.startswith(prefixes)]


def feature_columns_for_variant(variant: str, available_columns: Sequence[str]) -> List[str]:
    available = set(available_columns)
    cols: List[str] = []
    if variant in {"host_only", "host_plus_trigger", "full_record_model"}:
        cols.extend([col for col in HOST_COLUMNS if col in available])
    if variant in {"trigger_only", "host_plus_trigger", "full_record_model"}:
        cols.extend([col for col in TRIGGER_COLUMNS if col in available])
        cols.extend(_band_columns(available_columns))
    if variant == "full_record_model":
        cols.extend([col for col in ["detector_score", "alignment_quality", "D_face_context"] if col in available])
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    if not out:
        raise ValueError(f"No usable features for variant {variant}")
    return out


@dataclass
class RecordModelConfig:
    variant: str
    input_dim: int
    hidden_dim: int = 32
    lambda_rel: float = 1.0
    lambda_key: float = 0.3
    lambda_rank: float = 0.2
    lambda_bind: float = 0.1
    lambda_copy: float = 0.2


class RecordModelMLP(nn.Module):
    def __init__(self, config: RecordModelConfig):
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.rel_head = nn.Linear(hidden, 1)
        self.key_head = nn.Linear(hidden, 1)
        self.fake_head = nn.Linear(hidden, 1)
        self.rel_embed = nn.Linear(hidden, 8)
        self.key_embed = nn.Linear(hidden, 8)
        self.bind = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        z_rel = self.rel_embed(h)
        z_key = self.key_embed(h)
        return {
            "S_rel": self.rel_head(h).squeeze(-1),
            "S_key": self.key_head(h).squeeze(-1),
            "S_fake": self.fake_head(h).squeeze(-1),
            "z_rel": z_rel,
            "z_key": z_key,
            "z_df": h,
        }


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: RecordModelConfig,
    features: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss()
    loss = bce(outputs["S_fake"], targets["y_fake"])
    loss = loss + config.lambda_rel * bce(outputs["S_rel"], targets["y_rel"])
    loss = loss + config.lambda_key * bce(outputs["S_key"], targets["y_keycorr"])
    if features and "G_s" in features and "G_wrong_max" in features:
        rank = torch.relu(0.1 - (features["G_s"] - features["G_wrong_max"]))
        loss = loss + config.lambda_rank * rank.mean()
    z_rel = outputs["z_rel"]
    z_key = outputs["z_key"]
    mapped = z_key
    dist = torch.linalg.norm(z_rel - mapped, dim=1)
    rel_mask = targets["y_rel"] > 0.5
    copy_mask = torch.logical_and(targets["y_rel"] < 0.5, targets["y_keycorr"] > 0.5)
    if torch.any(rel_mask):
        loss = loss + config.lambda_bind * torch.mean(dist[rel_mask] ** 2)
    if torch.any(copy_mask):
        loss = loss + config.lambda_copy * torch.mean(torch.relu(0.5 - dist[copy_mask]) ** 2)
    return loss


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    keep = tpr[fpr <= float(target_fpr)]
    return float(np.max(keep)) if keep.size else 0.0
