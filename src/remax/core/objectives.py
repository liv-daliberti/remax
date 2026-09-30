"""Fresh MaxRL advantages and uniform verified-likelihood replay.

Re:Dr uses centered task rewards; Re:Max uses binary MaxRL advantages.
Both use the same replay objective and explicit coefficient/rollout scaling.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .replay_types import CanonicalReplayLoss


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


def canonical_replay_uniform_verified_likelihood_loss(
    mode_scores: torch.Tensor,
    group_sizes: Sequence[int],
    mass_weights: torch.Tensor | None = None,
) -> CanonicalReplayLoss:
    """Raise common verified-mode score while weighting observed modes equally.

    The bank-conditioned reverse KL in :func:`canonical_replay_uniform_loss`
    has score gradients ``q_bank - U_bank``.  They sum to zero within every
    prompt, so that objective can redistribute score among retained modes but
    cannot raise their common score against invalid outputs.

    This target-free successor instead minimizes

    ``mean_g mean_{k in B_g} -score(g, k)``.

    It is uniform maximum likelihood over only validator-positive exemplars
    the policy has actually discovered.  Its score gradients are
    ``-1 / (number_of_groups * modes_in_group)`` and therefore retain a
    non-zero common-mass component.  The conditioned-bank entropy and reverse
    KL remain detached diagnostics for audit. No
    exhaustive support size or evaluation signal appears in the objective.
    """

    sizes = tuple(int(value) for value in group_sizes)
    if mode_scores.ndim != 1:
        raise ValueError("canonical replay mode_scores must be one-dimensional")
    if not sizes or any(value < 1 for value in sizes):
        raise ValueError(
            "verified-likelihood replay groups must each contain at least one mode"
        )
    if sum(sizes) != int(mode_scores.numel()):
        raise ValueError("canonical replay group sizes do not partition scores")
    if not torch.isfinite(mode_scores).all():
        raise ValueError("canonical replay mode scores must be finite")
    if mass_weights is None:
        mass_weights = torch.ones_like(mode_scores.detach())
    if (
        mass_weights.ndim != 1
        or mass_weights.shape != mode_scores.shape
        or not bool(torch.isfinite(mass_weights).all())
        or bool((mass_weights <= 0.0).any())
    ):
        raise ValueError(
            "verified-likelihood mass weights must be finite, positive, and "
            "aligned with mode scores"
        )

    losses: list[torch.Tensor] = []
    balance_losses: list[torch.Tensor] = []
    entropy_ratios: list[torch.Tensor] = []
    score_gradients: list[torch.Tensor] = []
    entropy_eligible_modes = 0
    start = 0
    for size in sizes:
        stop = start + size
        scores = mode_scores[start:stop].float()
        weights = mass_weights[start:stop].to(
            device=scores.device,
            dtype=scores.dtype,
        )
        weight_sum = weights.sum()
        losses.append(-(scores * weights).sum() / weight_sum)
        score_gradients.append(
            -weights.detach() / (float(len(sizes)) * weight_sum.detach())
        )
        if size >= 2:
            log_probabilities = torch.log_softmax(scores, dim=0)
            log_support = math.log(size)
            balance_losses.append(-log_probabilities.mean() - log_support)
            sensor_log_probabilities = torch.log_softmax(
                mode_scores[start:stop].detach().double(),
                dim=0,
            )
            sensor_probabilities = sensor_log_probabilities.exp()
            entropy_ratios.append(
                -(sensor_probabilities * sensor_log_probabilities).sum() / log_support
            )
            entropy_eligible_modes += size
        start = stop

    if balance_losses:
        balance_loss = torch.stack(balance_losses).mean().detach()
        normalized_entropy = torch.stack(entropy_ratios).mean().detach()
    else:
        # Singleton banks have no canonical entropy. Keep the finite value as
        # a finite telemetry placeholder when no balance group is eligible.
        balance_loss = mode_scores.detach().float().sum() * 0.0
        normalized_entropy = mode_scores.detach().double().new_tensor(1.0)

    return CanonicalReplayLoss(
        loss=torch.stack(losses).mean(),
        normalized_entropy=normalized_entropy,
        cross_entropy_excess=balance_loss,
        score_gradients=torch.cat(score_gradients),
        eligible_groups=len(balance_losses),
        retained_modes=entropy_eligible_modes,
        actuator_groups=len(sizes),
        actuator_modes=sum(sizes),
    )
