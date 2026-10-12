"""The record model f_θ (paper §3.3, Eqs. 8–10; App. D.1).

The record model reads the feature vector x = Φ(J, R) of a query–record comparison, standardizes
it with training-set means μ and standard deviations σ (x̃ = (x − μ) ⊘ σ, near-zero σ replaced by
one, Eq. 9) and maps it with a small MLP to three sigmoid scores (S_rel, S_key, S_fake) = f_θ(x̃):
face consistency, correct-key release association, and re-rendering.

Feature columns are the CSV column names of the feature tables (unchanged on disk). Which columns
a model uses depends on its variant:

* ``host_only``          – agreement statistics only (``AGREEMENT_COLUMNS``)
* ``trigger_only``       – key statistics and every per-band column (``KEY_COLUMNS`` + bands)
* ``host_plus_trigger``  – both
* ``full_record_model``  – both, plus the context columns when present

The variant names are stored in checkpoints and file names and stay as they are. Column order
defines the checkpoints' input layout and must not change.

Checkpoints are ``torch.save`` dicts with keys ``state_dict``, ``variant``, ``seed``,
``feature_columns``, ``mean``, ``std``, ``train_features_csv``; the state-dict parameter names
(``encoder``, ``rel_head``, ``key_head``, ``fake_head``, ``rel_embed``, ``key_embed``, ``bind``)
are kept so existing checkpoints load unchanged. ``bind`` is unused by the forward pass but is part
of every saved state dict.

Moved from workflow/record_model/src/record_model.py (model, loss, columns) and
workflow/record_model/scripts/train_eval_record_model_matrix.py (standardize, fit, predict); the
arithmetic is unchanged, so scores are bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

AGREEMENT_COLUMNS = ["H", "C", "residual_mean", "residual_std", "cos_consistency_mean"]
"""Full-ring agreement and phase-residual statistics of f_0 (H, C and the drift r_k summaries)."""

KEY_COLUMNS = ["G_s", "G_wrong_mean", "G_wrong_max", "G_margin", "G_z", "sin_projection_mean"]
"""Full-ring key statistics of f_0: G_s, Ḡ⁻, G⁻_max, G_Δ (column ``G_margin``), G_z (Eq. 7)."""

BAND_PREFIXES = ("H_b", "C_b", "G_b", "G_margin_b", "residual_mean_b", "residual_std_b",
                 "sin_projection_mean_b", "cos_consistency_mean_b")
"""Prefixes of the per-band columns f_ℓ, ℓ = 1..8 (pixel-radius bands). ``G_margin_b{i}`` holds
the same value as ``G_b{i}``; both columns are kept because the checkpoints read both."""

CONTEXT_COLUMNS = ["detector_score", "alignment_quality", "D_face_context"]
"""Optional context columns used by ``full_record_model`` when the feature table has them."""

VARIANTS = ("host_only", "trigger_only", "host_plus_trigger", "full_record_model")

SCORE_NAMES = ("S_fake", "S_rel", "S_key")

MissingValues = Literal["column_mean", "zero"]


def band_columns(columns: Iterable[str]) -> list[str]:
    """Per-band feature columns among ``columns``, in their given order."""
    return [col for col in columns if col.startswith(BAND_PREFIXES)]


def feature_columns(variant: str, available_columns: Sequence[str]) -> list[str]:
    """Input columns of a record-model variant, given the columns a feature table has."""
    available = set(available_columns)
    cols: list[str] = []
    if variant in {"host_only", "host_plus_trigger", "full_record_model"}:
        cols.extend([col for col in AGREEMENT_COLUMNS if col in available])
    if variant in {"trigger_only", "host_plus_trigger", "full_record_model"}:
        cols.extend([col for col in KEY_COLUMNS if col in available])
        cols.extend(band_columns(available_columns))
    if variant == "full_record_model":
        cols.extend([col for col in CONTEXT_COLUMNS if col in available])
    out = list(dict.fromkeys(cols))
    if not out:
        raise ValueError(f"No usable features for variant {variant}")
    return out


@dataclass
class RecordModelConfig:
    """Architecture and loss weights. λ_rel, λ_key, λ_agr are the paper's Eq. (10) weights.

    ``lambda_rank`` (key-ranking hinge) and ``lambda_copy`` (copy-separation hinge) are terms of the
    implemented loss that Eq. (10) does not list; see ``multitask_loss``.
    """

    variant: str
    input_dim: int
    hidden_dim: int = 32
    lambda_rel: float = 1.0
    lambda_key: float = 0.3
    lambda_rank: float = 0.2
    lambda_agreement: float = 0.1
    lambda_copy: float = 0.2


class RecordModel(nn.Module):
    """f_θ: two hidden ReLU layers shared by three linear heads, plus the projections z_rel, z_key ∈ R⁸."""

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
        self.bind = nn.Linear(8, 8, bias=False)  # unused in forward; kept so checkpoints load strictly

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits of S_rel, S_key, S_fake, the projections z_rel, z_key and the shared features."""
        # The order of these calls fixes the order in which backward accumulates the gradient of h;
        # keep it, or retrained weights change in the last bits.
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
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    config: RecordModelConfig,
    features: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Training loss: Eq. (10) plus two hinge terms the implementation has always used.

    BCE(S_fake, y_fake) + λ_rel BCE(S_rel, y_rel) + λ_key BCE(S_key, y_key)
    + λ_rank · mean relu(0.1 − (G_s − G⁻_max))            (when ``features`` holds both columns)
    + λ_agr · mean_{y_rel=1} ‖z_rel − z_key‖²               (the agreement term of Eq. 10)
    + λ_copy · mean_{y_rel=0, y_key=1} relu(0.5 − ‖z_rel − z_key‖)²   (no such rows in the
      benign/swap/wrong-key training sets, so this term is inactive there)

    ``targets`` holds ``y_fake``, ``y_rel`` and ``y_key`` (CSV column ``y_keycorr``).
    """
    bce = nn.BCEWithLogitsLoss()
    loss = bce(outputs["S_fake"], targets["y_fake"])
    loss = loss + config.lambda_rel * bce(outputs["S_rel"], targets["y_rel"])
    loss = loss + config.lambda_key * bce(outputs["S_key"], targets["y_key"])
    if features and "G_s" in features and "G_wrong_max" in features:
        rank = torch.relu(0.1 - (features["G_s"] - features["G_wrong_max"]))
        loss = loss + config.lambda_rank * rank.mean()
    dist = torch.linalg.norm(outputs["z_rel"] - outputs["z_key"], dim=1)
    rel_mask = targets["y_rel"] > 0.5
    copy_mask = torch.logical_and(targets["y_rel"] < 0.5, targets["y_key"] > 0.5)
    if torch.any(rel_mask):
        loss = loss + config.lambda_agreement * torch.mean(dist[rel_mask] ** 2)
    if torch.any(copy_mask):
        loss = loss + config.lambda_copy * torch.mean(torch.relu(0.5 - dist[copy_mask]) ** 2)
    return loss


# ----------------------------------------------------------------------------- inputs
def feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """The feature columns as float64, non-numeric entries as NaN."""
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


def standardize(x: np.ndarray, std_floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-set standardization (Eq. 9): returns x̃, μ, σ as float32.

    μ and σ ignore NaN; σ below ``std_floor`` is replaced by one; NaN entries become μ before
    scaling. ``std_floor`` is 1e-6 for the record-model trainer (other scripts used 1e-8; pass it).
    """
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < std_floor] = 1.0
    x = (np.nan_to_num(x, nan=mean) - mean) / std
    return x.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardized_input(x_raw: np.ndarray, mean: np.ndarray, std: np.ndarray,
                       missing: MissingValues = "column_mean") -> np.ndarray:
    """x̃ = (x − μ) ⊘ σ with a stored μ, σ, as float32.

    ``missing`` sets what NaN features become before scaling: ``"column_mean"`` (μ, as in training
    and the trainer's evaluation) or ``"zero"`` (0, as the copy-ring experiment has always done).
    """
    fill = mean if missing == "column_mean" else 0.0
    return ((np.nan_to_num(x_raw, nan=fill) - mean) / std).astype(np.float32)


# ----------------------------------------------------------------------------- training
@dataclass
class FittedRecordModel:
    """A trained f_θ with the columns and standardization it expects."""

    model: RecordModel
    variant: str
    seed: int
    feature_columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    final_loss: float = float("nan")


def fit_record_model(train: pd.DataFrame, variant: str, seed: int, steps: int, hidden_dim: int,
                     lr: float, batch_size: int) -> FittedRecordModel:
    """Train one variant with AdamW (weight decay 1e-4) on a feature table with y_* targets.

    Seeds numpy's global RNG, torch and a ``default_rng(seed)`` mini-batch sampler exactly as the
    record-model trainer does, so a CPU run reproduces its weights.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cols = feature_columns(variant, list(train.columns))
    x, mean, std = standardize(feature_matrix(train, cols))
    x_t = torch.tensor(x)
    targets = {
        "y_fake": torch.tensor(train["y_fake"].to_numpy(dtype=np.float32)),
        "y_rel": torch.tensor(train["y_rel"].to_numpy(dtype=np.float32)),
        "y_key": torch.tensor(train["y_keycorr"].to_numpy(dtype=np.float32)),
    }
    key_features = {}
    for name in ["G_s", "G_wrong_max"]:
        if name in train.columns:
            key_features[name] = torch.tensor(
                pd.to_numeric(train[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    config = RecordModelConfig(variant=variant, input_dim=x.shape[1], hidden_dim=hidden_dim)
    model = RecordModel(config)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    final_loss = float("nan")
    for _ in range(int(steps)):
        if 0 < batch_size < len(train):
            batch = torch.tensor(rng.integers(0, len(train), size=batch_size), dtype=torch.long)
            x_batch = x_t[batch]
            target_batch = {name: value[batch] for name, value in targets.items()}
            feature_batch = {name: value[batch] for name, value in key_features.items()}
        else:
            x_batch, target_batch, feature_batch = x_t, targets, key_features
        opt.zero_grad(set_to_none=True)
        loss = multitask_loss(model(x_batch), target_batch, config, feature_batch)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu().item())
    return FittedRecordModel(model, variant, int(seed), cols, mean, std, final_loss)


def save_checkpoint(fitted: FittedRecordModel, file: str | Path, train_features_csv: str | Path) -> None:
    """Write the checkpoint dict the trainer has always written."""
    torch.save(
        {
            "state_dict": fitted.model.state_dict(),
            "variant": fitted.variant,
            "seed": int(fitted.seed),
            "feature_columns": fitted.feature_columns,
            "mean": fitted.mean,
            "std": fitted.std,
            "train_features_csv": str(train_features_csv),
        },
        file,
    )


def load_checkpoint(file: str | Path) -> FittedRecordModel:
    """Load a record-model checkpoint on CPU in eval mode (hidden width read from the weights)."""
    ck = torch.load(file, map_location="cpu", weights_only=False)
    state = ck["state_dict"]
    config = RecordModelConfig(variant=ck["variant"], input_dim=len(ck["feature_columns"]),
                               hidden_dim=state["encoder.0.weight"].shape[0])
    model = RecordModel(config)
    model.load_state_dict(state)
    model.eval()
    return FittedRecordModel(model, ck["variant"], int(ck["seed"]), list(ck["feature_columns"]),
                             ck["mean"], ck["std"])


# ----------------------------------------------------------------------------- prediction
def record_model_scores(fitted: FittedRecordModel, frame: pd.DataFrame,
                        missing: MissingValues = "column_mean") -> dict[str, np.ndarray]:
    """Sigmoid scores S_fake, S_rel, S_key (float32 arrays) for every row of a feature table."""
    x = standardized_input(feature_matrix(frame, fitted.feature_columns), fitted.mean, fitted.std, missing)
    with torch.no_grad():
        out = fitted.model(torch.tensor(x))
    return {name: torch.sigmoid(out[name]).numpy() for name in SCORE_NAMES}


def score_table(fitted: FittedRecordModel, frame: pd.DataFrame) -> pd.DataFrame:
    """The trainer's per-query score table: ids, labels, then S_fake, S_rel, S_key."""
    scores = frame[["sample_id", "dataset", "protocol", "y_fake", "y_rel", "y_keycorr"]].copy()
    for name, values in record_model_scores(fitted, frame).items():
        scores[name] = values
    return scores
