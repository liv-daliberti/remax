"""Differentiable retention over verified canonical-mode exemplars.

The online count bank can shape modes that occur in the current rollout, but
it cannot assign positive probability to an observed mode that has disappeared
from the on-policy sample.  Canonical replay closes that actuator gap:

* one validator-positive token sequence is retained per observed prompt/mode;
* the current model assigns a length-normalized log score to every retained
  mode for a replayed prompt;
* a selectable target-free actuator either balances scores within that
  *observed* bank or raises the common likelihood of every observed verified
  exemplar;
* neither loss consults exhaustive support or evaluation feedback.

Replay and optional balance use fixed coefficients supplied by the run
configuration. This module intentionally contains no adaptive coefficient
state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

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


def materialize_canonical_replay_batch(
    groups: Sequence[Any],
    *,
    pad_token_id: int,
    device: torch.device | int | str,
) -> CanonicalReplayBatch:
    """Build causal-LM labels without changing any stored exemplar tokens."""

    if not groups:
        raise ValueError("canonical replay requires at least one group")
    if isinstance(pad_token_id, bool) or int(pad_token_id) < 0:
        raise ValueError("canonical replay pad_token_id must be non-negative")

    sequences: list[tuple[int, ...]] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    group_sizes: list[int] = []
    fresh_observation_counts: list[int] = []
    mass_weights: list[float] = []
    priority_modes = 0
    for group in groups:
        prompt = tuple(int(value) for value in group.prompt_token_ids)
        responses = tuple(
            tuple(int(value) for value in response)
            for response in group.response_token_ids
        )
        if not prompt or not responses or any(not row for row in responses):
            raise ValueError(
                "canonical replay groups require a prompt and at least one "
                "non-empty response"
            )
        if any(value < 0 for value in prompt) or any(
            value < 0 for row in responses for value in row
        ):
            raise ValueError("canonical replay token ids must be non-negative")
        group_sizes.append(len(responses))
        raw_fresh_counts = tuple(
            int(value)
            for value in getattr(group, "fresh_observation_counts", ())
        )
        if not raw_fresh_counts:
            # Compatibility for external/offline replay groups. Frequency
            # weighting rejects these sentinel zeros in the learner.
            raw_fresh_counts = tuple(0 for _ in responses)
        if len(raw_fresh_counts) != len(responses) or any(
            value < 0 for value in raw_fresh_counts
        ):
            raise ValueError(
                "canonical replay fresh observation counts must be "
                "non-negative and align with response rows"
            )
        raw_mass_weights = tuple(
            float(value) for value in getattr(group, "mass_weights", ())
        )
        if not raw_mass_weights:
            raw_mass_weights = tuple(1.0 for _ in responses)
        if len(raw_mass_weights) != len(responses):
            raise ValueError(
                "canonical replay mass weights must align with response rows"
            )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in raw_mass_weights
        ):
            raise ValueError(
                "canonical replay mass weights must be finite and positive"
            )
        if not math.isclose(
            sum(raw_mass_weights),
            float(len(raw_mass_weights)),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "canonical replay mass weights must preserve the group budget"
            )
        priority_modes += int(getattr(group, "priority_modes", 0))
        for response, fresh_count, mass_weight in zip(
            responses, raw_fresh_counts, raw_mass_weights
        ):
            sequences.append(prompt + response)
            prompt_lengths.append(len(prompt))
            response_lengths.append(len(response))
            fresh_observation_counts.append(fresh_count)
            mass_weights.append(mass_weight)

    max_length = max(len(row) for row in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    response_masks = torch.zeros(
        (len(sequences), max_length - 1),
        dtype=torch.bool,
        device=device,
    )
    for row_index, (sequence, prompt_length, response_length) in enumerate(
        zip(sequences, prompt_lengths, response_lengths)
    ):
        row_length = len(sequence)
        input_ids[row_index, :row_length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row_index, :row_length] = 1
        response_start = prompt_length - 1
        response_masks[
            row_index,
            response_start : response_start + response_length,
        ] = True

    return CanonicalReplayBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_masks=response_masks,
        group_sizes=tuple(group_sizes),
        fresh_observation_counts=torch.tensor(
            fresh_observation_counts,
            dtype=torch.int64,
            device=device,
        ),
        mass_weights=torch.tensor(
            mass_weights,
            dtype=torch.float32,
            device=device,
        ),
        priority_modes=priority_modes,
    )


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
        raise ValueError(
            "canonical replay groups must each contain at least two modes"
        )
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
            (
                log_probabilities.detach().exp()
                - (1.0 / float(size))
            )
            / float(len(sizes))
        )
        # Keep the actuator in the model's ordinary precision, but compute the
        # collapse sensor in float64 so a very unlikely retained mode does not
        # disappear merely through float32 exponent underflow.
        sensor_log_probabilities = torch.log_softmax(
            mode_scores[start:stop].detach().double(),
            dim=0,
        )
        sensor_probabilities = sensor_log_probabilities.exp()
        entropy = -(
            sensor_probabilities * sensor_log_probabilities
        ).sum()
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
            balance_losses.append(
                -log_probabilities.mean() - log_support
            )
            sensor_log_probabilities = torch.log_softmax(
                mode_scores[start:stop].detach().double(),
                dim=0,
            )
            sensor_probabilities = sensor_log_probabilities.exp()
            entropy_ratios.append(
                -(
                    sensor_probabilities * sensor_log_probabilities
                ).sum()
                / log_support
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
            balance_gradients[target_start:target_stop] = (
                balance.score_gradients[source_start:source_stop]
            )
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
            raise ValueError(
                "each replay group must have negative total mass gradient"
            )
        upward_pressure = torch.clamp(-chunk, min=0.0)
        normalizer = upward_pressure.sum()
        if not bool(normalizer > 0.0):
            raise RuntimeError(
                "negative total mass gradient must leave upward pressure"
            )
        projected[start:stop] = (
            -upward_pressure * (preserved_mass / normalizer)
        )
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
    if (
        weighted_balance_score_gradients.shape
        != weighted_mass_score_gradients.shape
    ):
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
