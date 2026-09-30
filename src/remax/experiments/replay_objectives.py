"""Retained balance, split-mass and frequency-weighted replay comparators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence
import torch

from ..core.replay_types import CanonicalReplayLoss, CanonicalReplaySplitLoss
from ..core.objectives import canonical_replay_uniform_verified_likelihood_loss


def canonical_replay_key_target_weights(
    fresh_observation_counts: torch.Tensor,
    group_sizes: Sequence[int],
    *,
    weighting: str,
) -> torch.Tensor:
    """Return group-budget-preserving weights over fixed replay rows.

    The uniform mode intentionally returns literal ones and therefore preserves
    the pre-ablation replay path exactly. Fresh-frequency mode normalizes
    cumulative counts from validator-positive fresh policy rollouts to sum to
    the number of retained keys in each prompt bank. Neither replay nor
    proposal rows are admitted as frequency observations.
    """

    sizes = tuple(int(value) for value in group_sizes)
    if fresh_observation_counts.ndim != 1:
        raise ValueError("fresh replay counts must be one-dimensional")
    if not sizes or any(value < 1 for value in sizes):
        raise ValueError("replay target groups must be non-empty")
    if sum(sizes) != int(fresh_observation_counts.numel()):
        raise ValueError("replay target group sizes do not partition counts")
    if weighting == "uniform":
        return torch.ones_like(fresh_observation_counts, dtype=torch.float32)
    if weighting != "fresh_frequency":
        raise ValueError("replay key weighting must be uniform or fresh_frequency")
    if bool((fresh_observation_counts <= 0).any()):
        raise ValueError(
            "frequency replay requires a positive fresh observation count "
            "for every materialized key"
        )

    weights: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        stop = start + size
        counts = fresh_observation_counts[start:stop].to(torch.float64)
        weights.append(counts * (float(size) / counts.sum()))
        start = stop
    return torch.cat(weights).to(torch.float32)


def canonical_replay_uniform_loss(
    mode_scores: torch.Tensor,
    group_sizes: Sequence[int],
) -> CanonicalReplayLoss:
    """Balance current model scores across each observed prompt-local bank.

    ``mode_scores`` contains one length-normalized teacher-forced log score per
    retained verified mode. ``group_sizes`` partitions those scores by prompt.
    Every group must contain at least two modes.  The loss is

    ``mean_g KL(U_g || softmax(scores_g))``.

    This reverse-direction KL retains a non-vanishing restorative gradient for
    a mode whose current probability is very small.  It uses only modes that
    the validator has actually observed; no gold support size appears.
    """

    if mode_scores.ndim != 1:
        raise ValueError("canonical replay mode_scores must be one-dimensional")
    sizes = tuple(int(value) for value in group_sizes)
    if not sizes or any(value < 2 for value in sizes):
        raise ValueError("canonical replay groups must each contain at least two modes")
    if sum(sizes) != int(mode_scores.numel()):
        raise ValueError("canonical replay group sizes do not partition scores")
    if not torch.isfinite(mode_scores).all():
        raise ValueError("canonical replay mode scores must be finite")

    losses: list[torch.Tensor] = []
    entropy_ratios: list[torch.Tensor] = []
    score_gradients: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        stop = start + size
        scores = mode_scores[start:stop].float()
        log_probabilities = torch.log_softmax(scores, dim=0)
        log_support = math.log(size)
        cross_entropy_excess = -log_probabilities.mean() - log_support
        score_gradients.append(
            (log_probabilities.detach().exp() - (1.0 / float(size))) / float(len(sizes))
        )
        # Keep the actuator in the model's ordinary precision, but compute the
        # collapse sensor in float64 so a very unlikely retained mode does not
        # disappear merely through float32 exponent underflow.
        sensor_log_probabilities = torch.log_softmax(
            mode_scores[start:stop].detach().double(),
            dim=0,
        )
        sensor_probabilities = sensor_log_probabilities.exp()
        entropy = -(sensor_probabilities * sensor_log_probabilities).sum()
        losses.append(cross_entropy_excess)
        entropy_ratios.append(entropy / log_support)
        start = stop

    loss = torch.stack(losses).mean()
    normalized_entropy = torch.stack(entropy_ratios).mean().detach()
    return CanonicalReplayLoss(
        loss=loss,
        normalized_entropy=normalized_entropy,
        cross_entropy_excess=loss.detach(),
        score_gradients=torch.cat(score_gradients),
        eligible_groups=len(sizes),
        retained_modes=sum(sizes),
        actuator_groups=len(sizes),
        actuator_modes=sum(sizes),
    )


def canonical_replay_split_mass_balance_loss(
    mode_scores: torch.Tensor,
    group_sizes: Sequence[int],
    mass_weights: torch.Tensor | None = None,
) -> CanonicalReplaySplitLoss:
    """Return actuator-aligned mass and balance terms from one score pass.

    The mass term is uniform verified likelihood over every group, including
    singletons. The balance term is ``KL(U || softmax(scores))`` over only
    groups with at least two modes. Its gradient is scattered back into the
    full replay-row layout with exact zeros for singleton rows.
    """

    mass = canonical_replay_uniform_verified_likelihood_loss(
        mode_scores,
        group_sizes,
        mass_weights,
    )
    sizes = tuple(int(value) for value in group_sizes)
    eligible_slices: list[tuple[int, int]] = []
    eligible_sizes: list[int] = []
    start = 0
    for size in sizes:
        stop = start + size
        if size >= 2:
            eligible_slices.append((start, stop))
            eligible_sizes.append(size)
        start = stop

    if eligible_slices:
        eligible_scores = torch.cat(
            [mode_scores[start:stop] for start, stop in eligible_slices]
        )
        balance = canonical_replay_uniform_loss(
            eligible_scores,
            eligible_sizes,
        )
        balance_gradients = torch.zeros_like(mode_scores.detach())
        source_start = 0
        for (target_start, target_stop), size in zip(
            eligible_slices,
            eligible_sizes,
        ):
            source_stop = source_start + size
            balance_gradients[target_start:target_stop] = balance.score_gradients[
                source_start:source_stop
            ]
            source_start = source_stop
        balance_loss = balance.loss
        normalized_entropy = balance.normalized_entropy
        balance_groups = balance.eligible_groups
        balance_modes = balance.retained_modes
    else:
        balance_loss = mode_scores.float().sum() * 0.0
        normalized_entropy = mode_scores.detach().double().new_tensor(1.0)
        balance_gradients = torch.zeros_like(mode_scores.detach())
        balance_groups = 0
        balance_modes = 0

    return CanonicalReplaySplitLoss(
        mass_loss=mass.loss,
        balance_loss=balance_loss,
        normalized_entropy=normalized_entropy,
        mass_score_gradients=mass.score_gradients,
        balance_score_gradients=balance_gradients,
        actuator_groups=mass.actuator_groups,
        actuator_modes=mass.actuator_modes,
        balance_eligible_groups=balance_groups,
        balance_retained_modes=balance_modes,
    )


def project_retention_safe_score_gradients(
    raw_score_gradients: torch.Tensor,
    group_sizes: Sequence[int],
) -> torch.Tensor:
    """Keep rare-mode emphasis without assigning downward verified-score pressure.

    Each prompt-local raw gradient is projected onto the non-positive orthant
    and renormalized to preserve that group's total verified-mass gradient.
    The input is treated as a detached score-space update, matching replay's
    existing exact-gradient surrogate.
    """

    if raw_score_gradients.ndim != 1:
        raise ValueError("replay score gradients must be one-dimensional")
    sizes = tuple(int(value) for value in group_sizes)
    if not sizes or any(value < 1 for value in sizes):
        raise ValueError("retention-safe projection requires non-empty groups")
    if sum(sizes) != int(raw_score_gradients.numel()):
        raise ValueError("replay group sizes do not partition score gradients")
    if not torch.isfinite(raw_score_gradients).all():
        raise ValueError("replay score gradients must be finite")

    detached = raw_score_gradients.detach()
    projected = torch.empty_like(detached)
    start = 0
    for size in sizes:
        stop = start + size
        chunk = detached[start:stop]
        preserved_mass = -chunk.sum()
        if not bool(preserved_mass > 0.0):
            raise ValueError("each replay group must have negative total mass gradient")
        upward_pressure = torch.clamp(-chunk, min=0.0)
        normalizer = upward_pressure.sum()
        if not bool(normalizer > 0.0):
            raise RuntimeError(
                "negative total mass gradient must leave upward pressure"
            )
        projected[start:stop] = -upward_pressure * (preserved_mass / normalizer)
        start = stop

    if not torch.isfinite(projected).all():
        raise RuntimeError("retention-safe projection produced non-finite values")
    if bool((projected > 0.0).any()):
        raise RuntimeError("retention-safe projection assigned downward pressure")
    return projected


def cap_retention_safe_balance_score_gradients(
    weighted_mass_score_gradients: torch.Tensor,
    weighted_balance_score_gradients: torch.Tensor,
    group_sizes: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the largest prompt-local balance scale with no positive gradient."""

    if weighted_mass_score_gradients.ndim != 1:
        raise ValueError("replay score gradients must be one-dimensional")
    if weighted_balance_score_gradients.shape != weighted_mass_score_gradients.shape:
        raise ValueError("mass and balance score gradients must have equal shape")
    sizes = tuple(int(value) for value in group_sizes)
    if not sizes or any(value < 1 for value in sizes):
        raise ValueError("retention-safe balance requires non-empty groups")
    if sum(sizes) != int(weighted_mass_score_gradients.numel()):
        raise ValueError("replay group sizes do not partition score gradients")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            weighted_mass_score_gradients,
            weighted_balance_score_gradients,
        )
    ):
        raise ValueError("replay score gradients must be finite")

    mass = weighted_mass_score_gradients.detach()
    balance = weighted_balance_score_gradients.detach()
    if bool((mass > 0.0).any()):
        raise ValueError("verified-mass gradients must be non-positive")

    combined = torch.empty_like(mass)
    balance_scales: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        stop = start + size
        mass_chunk = mass[start:stop]
        balance_chunk = balance[start:stop]
        if not bool(mass_chunk.sum() < 0.0):
            raise ValueError("each replay group requires negative mass pressure")
        positive_balance = balance_chunk > 0.0
        scale = mass_chunk.new_tensor(1.0)
        if bool(positive_balance.any()):
            limit = (
                -mass_chunk[positive_balance] / balance_chunk[positive_balance]
            ).min()
            scale = torch.clamp(limit, min=0.0, max=1.0)
            if bool(scale < 1.0):
                scale = torch.nextafter(scale, torch.zeros_like(scale))
        safe_chunk = mass_chunk + scale * balance_chunk
        if bool((safe_chunk > 0.0).any()):
            raise RuntimeError("safe balance assigned downward score pressure")
        combined[start:stop] = safe_chunk
        balance_scales.append(scale)
        start = stop

    return combined, torch.stack(balance_scales)
