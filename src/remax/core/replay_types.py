"""Typed tensor boundaries for replay scoring and objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CanonicalReplayLoss:
    """Differentiable replay loss and detached model-score diagnostics."""

    loss: torch.Tensor
    normalized_entropy: torch.Tensor
    cross_entropy_excess: torch.Tensor
    score_gradients: torch.Tensor
    eligible_groups: int
    retained_modes: int
    actuator_groups: int
    actuator_modes: int


@dataclass(frozen=True)
class CanonicalReplayBatch:
    """Right-padded teacher-forcing batch over verified exemplars."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_masks: torch.Tensor
    group_sizes: tuple[int, ...]
    fresh_observation_counts: torch.Tensor
    mass_weights: torch.Tensor
    priority_modes: int


@dataclass(frozen=True)
class CanonicalReplaySplitLoss:
    """Independent verified-mass and known-mode-balance replay terms."""

    mass_loss: torch.Tensor
    balance_loss: torch.Tensor
    normalized_entropy: torch.Tensor
    mass_score_gradients: torch.Tensor
    balance_score_gradients: torch.Tensor
    actuator_groups: int
    actuator_modes: int
    balance_eligible_groups: int
    balance_retained_modes: int
