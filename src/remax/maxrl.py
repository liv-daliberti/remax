"""On-policy binary MaxRL advantages for grouped terminal rewards."""

from __future__ import annotations

import torch


def binary_maxrl_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Return centered practical MaxRL advantages for each prompt group.

    Rewards must be [prompt_count, samples_per_prompt] and exactly binary.
    For a group with K > 0 successes and width N, the returned coefficient is
    N * r_i / K - 1. All-failure groups receive zero.
    """

    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must be [prompts, at least two samples]")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("MaxRL rewards must be finite")
    if not bool(((rewards == 0) | (rewards == 1)).all()):
        raise ValueError("binary MaxRL requires rewards in {0, 1}")

    success_counts = rewards.sum(dim=1, keepdim=True)
    width = float(rewards.shape[1])
    return torch.where(
        success_counts > 0,
        rewards * width / success_counts.clamp_min(1.0) - 1.0,
        torch.zeros_like(rewards),
    )
