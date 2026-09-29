"""Auditable DAPO primitives for the direct comparative baseline.

The implementation follows Yu et al. (2025), Equation 8: standard GRPO
advantages, asymmetric PPO clipping, token-level loss reduction, dynamic
filtering of zero-variance reward groups, and soft overlong reward shaping.
This module keeps the numerical pieces pure so the frozen experiment can test
them independently of OAT's distributed runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class DAPOGroupDiagnostics:
    """Counts produced by DAPO's accuracy-group filter."""

    groups: int
    eligible_groups: int
    all_zero_groups: int
    all_one_groups: int


@dataclass(frozen=True)
class DAPOOverlongDiagnostics:
    """Summary of the soft overlong reward term."""

    rows: int
    shaped_rows: int
    truncated_rows: int
    penalty_min: float
    penalty_mean: float


def dapo_group_diagnostics(
    rewards: Sequence[float],
    *,
    num_samples: int,
) -> DAPOGroupDiagnostics:
    """Classify complete binary-reward groups for dynamic sampling.

    DAPO retains only groups whose verifier accuracy is strictly between zero
    and one. The campaign's rule-based task reward is binary, so accepting a
    non-binary value here would silently change the paper's estimand.
    """

    if isinstance(num_samples, bool) or int(num_samples) != num_samples:
        raise ValueError("num_samples must be an integer")
    num_samples = int(num_samples)
    if num_samples <= 1:
        raise ValueError("num_samples must exceed one")
    values = [float(value) for value in rewards]
    if not values or len(values) % num_samples:
        raise ValueError("DAPO requires non-empty complete prompt groups")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("DAPO rewards must be finite")
    if any(value not in {0.0, 1.0} for value in values):
        raise ValueError("DAPO dynamic sampling requires binary task rewards")

    all_zero = 0
    all_one = 0
    eligible = 0
    for start in range(0, len(values), num_samples):
        total = sum(values[start : start + num_samples])
        if total == 0.0:
            all_zero += 1
        elif total == float(num_samples):
            all_one += 1
        else:
            eligible += 1
    groups = len(values) // num_samples
    return DAPOGroupDiagnostics(
        groups=groups,
        eligible_groups=eligible,
        all_zero_groups=all_zero,
        all_one_groups=all_one,
    )


def dapo_resample_prompt_index(
    *,
    base_seed: int,
    learner_step: int,
    generation_batch: int,
    dataset_size: int,
) -> int:
    """Choose a deterministic fresh prompt for a rejected DAPO group.

    The index is a pure function of registered run identity and checkpointed
    learner step, so infrastructure retries cannot choose prompts differently.
    """

    for name, value, minimum in (
        ("learner_step", learner_step, 0),
        ("generation_batch", generation_batch, 1),
        ("dataset_size", dataset_size, 1),
    ):
        if isinstance(value, bool) or int(value) != value or int(value) < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    payload = (
        f"dapo-dynamic-v1|{int(base_seed)}|{int(learner_step)}|{int(generation_batch)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % int(
        dataset_size
    )


def dapo_soft_overlong_penalty(
    response_lengths: torch.Tensor,
    *,
    max_length: int,
    buffer_ratio: float,
    penalty_factor: float,
) -> tuple[torch.Tensor, DAPOOverlongDiagnostics]:
    """Return Equation 13's soft overlong penalty at the local token scale."""

    if response_lengths.ndim != 1 or response_lengths.numel() == 0:
        raise ValueError("response_lengths must be a non-empty vector")
    if not bool(torch.isfinite(response_lengths.float()).all()):
        raise ValueError("response_lengths must be finite")
    if bool((response_lengths < 0).any()):
        raise ValueError("response_lengths must be non-negative")
    if isinstance(max_length, bool) or int(max_length) != max_length:
        raise ValueError("max_length must be an integer")
    max_length = int(max_length)
    if max_length <= 1:
        raise ValueError("max_length must exceed one")
    buffer_ratio = float(buffer_ratio)
    penalty_factor = float(penalty_factor)
    if not math.isfinite(buffer_ratio) or not 0.0 < buffer_ratio < 1.0:
        raise ValueError("buffer_ratio must be finite and in (0, 1)")
    if not math.isfinite(penalty_factor) or penalty_factor < 0.0:
        raise ValueError("penalty_factor must be finite and non-negative")

    # The paper uses an expected maximum of 16,384 plus a 4,096-token cache:
    # the unpenalized boundary is therefore 80% of the generation ceiling.
    buffer_tokens = max(1, int(round(max_length * buffer_ratio)))
    unpenalized_max = max_length - buffer_tokens
    lengths = response_lengths.to(dtype=torch.float32)
    penalties = torch.zeros_like(lengths)
    soft = lengths > float(unpenalized_max)
    penalties = torch.where(
        soft,
        (float(unpenalized_max) - lengths) / float(buffer_tokens),
        penalties,
    )
    penalties = penalties.clamp(min=-1.0, max=0.0) * penalty_factor
    truncated = lengths >= float(max_length)
    diagnostics = DAPOOverlongDiagnostics(
        rows=int(lengths.numel()),
        shaped_rows=int(soft.sum().item()),
        truncated_rows=int(truncated.sum().item()),
        penalty_min=float(penalties.min().item()),
        penalty_mean=float(penalties.mean().item()),
    )
    return penalties, diagnostics


def dapo_token_level_policy_loss(
    token_losses: torch.Tensor,
    response_masks: torch.Tensor,
    loss_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce policy losses over active tokens, not sequence means.

    Returns the scalar loss and its active-token denominator for audit logs.
    """

    if token_losses.shape != response_masks.shape:
        raise ValueError("token losses and response masks must have equal shape")
    if loss_masks.ndim != 1 or loss_masks.numel() != token_losses.size(0):
        raise ValueError("loss_masks must provide one value per response")
    if not bool(torch.isfinite(token_losses).all()):
        raise ValueError("DAPO token losses must be finite")
    active = (
        response_masks.to(token_losses.dtype)
        * loss_masks.to(token_losses.dtype)[:, None]
    )
    denominator = active.sum()
    if not bool(denominator > 0):
        raise ValueError("DAPO token-level loss has no active token")
    return (token_losses * active).sum() / denominator, denominator.detach()
