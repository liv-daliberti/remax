"""Prefix-ratio surrogate for direct on-policy completion entropy.

Rollout prefixes come from the frozen behavior policy, while the direct
MaxEnt objective contains the entropy of the updated policy.  The exclusive
prefix importance ratio changes the prefix distribution from the former to
the latter.  Its derivative supplies the causal state-visitation term; the
full categorical entropy supplies the local policy derivative.
"""

from __future__ import annotations

import math

import torch


def mean_active_token_entropy_by_response(
    token_entropy: torch.Tensor,
    response_masks: torch.Tensor,
) -> torch.Tensor:
    """Give every response one mean-entropy observation, independent of length."""

    if token_entropy.ndim != 2:
        raise ValueError("token_entropy must have shape [rows, tokens]")
    if response_masks.shape != token_entropy.shape:
        raise ValueError("response_masks must match token_entropy")
    mask_bool = response_masks.to(torch.bool)
    if not bool(torch.isfinite(token_entropy[mask_bool]).all()):
        raise ValueError("token_entropy must be finite on response tokens")
    mask = mask_bool.to(token_entropy.dtype)
    active_tokens = mask.sum(dim=1).clamp_min(1.0)
    return (torch.where(mask_bool, token_entropy, torch.zeros_like(token_entropy)).sum(
        dim=1
    ) / active_tokens)


def standard_maxent_loss(
    raw_entropy_surrogate: torch.Tensor,
    *,
    alpha: float,
    reward_estimator_scale: float,
    update_normalizer: float,
) -> torch.Tensor:
    r"""Apply only Dr.GRPO's common outer scale to standard MaxEnt.

    The optimized objective is ``E[R] + alpha * H(pi)``. Dr.GRPO represents
    its reward gradient with one shared ``1 / T_max`` update normalization, so
    the raw sequence-entropy surrogate receives that same single factor. This
    helper deliberately contains no objective-level entropy normalization.
    """

    coefficient = float(alpha)
    estimator_scale = float(reward_estimator_scale)
    normalizer = float(update_normalizer)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(estimator_scale) or estimator_scale <= 0:
        raise ValueError("reward_estimator_scale must be finite and positive")
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("update_normalizer must be finite and positive")
    return -coefficient * estimator_scale * raw_entropy_surrogate / normalizer


def standard_maxent_length_penalty_loss(
    raw_length_surrogate: torch.Tensor,
    *,
    length_lambda: float,
    reward_estimator_scale: float,
    update_normalizer: float,
) -> torch.Tensor:
    r"""Apply the positive Lagrangian cost with one shared outer scale.

    For ``E[R] + alpha H(pi) - lambda (E[L] - L_target)``, the target is
    constant with respect to the actor.  The minimized actor loss therefore
    receives ``+lambda E[L]``.  As for reward and entropy, Dr.GRPO's single
    shared ``1 / T_max`` normalization is applied exactly once.
    """

    multiplier = float(length_lambda)
    estimator_scale = float(reward_estimator_scale)
    normalizer = float(update_normalizer)
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError("length_lambda must be finite and non-negative")
    if not math.isfinite(estimator_scale) or estimator_scale <= 0:
        raise ValueError("reward_estimator_scale must be finite and positive")
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("update_normalizer must be finite and positive")
    return multiplier * estimator_scale * raw_length_surrogate / normalizer


def prefix_ratio_expected_length_surrogate(
    new_selected_log_probs: torch.Tensor,
    old_selected_log_probs: torch.Tensor,
    response_masks: torch.Tensor,
    *,
    cliprange: float | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Estimate new-policy expected response length on behavior rollouts.

    With the exclusive prefix ratio

    .. math::

       W_{t-1}=\prod_{j<t}\frac{\pi_{new}(a_j\mid s_j)}
                                  {\pi_{old}(a_j\mid s_j)},

    the exact finite-horizon identity is

    .. math::

       \mathbb E_{\pi_{new}}[L]
       =\mathbb E_{\pi_{old}}\sum_{t\text{ active}} W_{t-1}.

    The current action is excluded because it cannot alter whether the
    current token was already generated; it changes survival only at later
    prefixes.  Length is a cost in the actor objective, so PPO's conservative
    branch is the opposite of the positive entropy-reward branch:
    ``max(W, clip(W))``.  The detached controller observation remains the
    unclipped importance estimate.

    Returns per-row tensors for the differentiable conservative surrogate,
    detached unclipped expected length, detached sampled behavior length,
    detached prefix-ratio mean, maximum, and out-of-range fraction.
    """

    if new_selected_log_probs.ndim != 2:
        raise ValueError("new_selected_log_probs must have shape [rows, tokens]")
    for name, value in (
        ("old_selected_log_probs", old_selected_log_probs),
        ("response_masks", response_masks),
    ):
        if value.shape != new_selected_log_probs.shape:
            raise ValueError(f"{name} must match new_selected_log_probs")
    if cliprange is not None:
        epsilon = float(cliprange)
        if not math.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError("cliprange must be in (0, 1) or None")
    else:
        epsilon = 0.0

    mask_bool = response_masks.to(torch.bool)
    mask = mask_bool.to(dtype=new_selected_log_probs.dtype)
    for name, value in (
        ("new_selected_log_probs", new_selected_log_probs),
        ("old_selected_log_probs", old_selected_log_probs),
    ):
        if not bool(torch.isfinite(value[mask_bool]).all()):
            raise ValueError(f"{name} must be finite on response tokens")

    token_log_ratio = torch.where(
        mask_bool,
        new_selected_log_probs - old_selected_log_probs,
        torch.zeros_like(new_selected_log_probs),
    )
    inclusive_log_ratio = torch.cumsum(token_log_ratio, dim=1)
    exclusive_log_ratio = inclusive_log_ratio - token_log_ratio
    prefix_ratio = torch.exp(exclusive_log_ratio.clamp(-40.0, 40.0))
    unclipped_terms = prefix_ratio * mask

    if cliprange is None:
        surrogate_terms = unclipped_terms
        outside = torch.zeros_like(mask_bool)
    else:
        clipped_ratio = torch.clamp(
            prefix_ratio,
            min=1.0 - epsilon,
            max=1.0 + epsilon,
        )
        surrogate_terms = torch.maximum(
            unclipped_terms,
            clipped_ratio * mask,
        )
        outside = (prefix_ratio < 1.0 - epsilon) | (
            prefix_ratio > 1.0 + epsilon
        )

    active_tokens = mask.sum(dim=1).clamp_min(1.0)
    prefix_ratio_masked = prefix_ratio * mask
    return (
        surrogate_terms.sum(dim=1),
        unclipped_terms.detach().sum(dim=1),
        mask.detach().sum(dim=1),
        prefix_ratio_masked.detach().sum(dim=1) / active_tokens,
        prefix_ratio_masked.detach().amax(dim=1),
        (outside.to(mask.dtype) * mask).detach().sum(dim=1) / active_tokens,
    )


def prefix_ratio_maxent_surrogate(
    token_entropy: torch.Tensor,
    new_selected_log_probs: torch.Tensor,
    old_selected_log_probs: torch.Tensor,
    response_masks: torch.Tensor,
    *,
    normalization_constant: float,
    cliprange: float | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Return a PPO-style surrogate for new-policy sequence entropy.

    For an autoregressive response and a prefix state ``s_t``, let

    .. math::

       h_t = H(\pi_{new}(\cdot\mid s_t)),\qquad
       W_{t-1}=\prod_{j<t}
       \frac{\pi_{new}(a_j\mid s_j)}{\pi_{old}(a_j\mid s_j)}.

    Importance sampling gives the exact finite-horizon identity

    .. math::

       H(\pi_{new})=
       \mathbb E_{\tau\sim\pi_{old}}\sum_t W_{t-1}h_t.

    The ratio is exclusive of the current action because ``h_t`` is defined
    before that action is sampled. Differentiating ``W`` produces the causal
    state-visitation derivative without a separately constructed score term.

    When ``cliprange`` is not ``None``, each positive entropy term uses
    ``min(W*h, clip(W)*h)``, matching PPO's conservative upper branch. The
    function returns per-row tensors for: differentiable clipped surrogate,
    detached unclipped sequence-entropy estimate, detached entropy at sampled
    behavior prefixes, detached prefix-ratio mean, detached prefix-ratio
    maximum, and detached out-of-range fraction. Entropy quantities are
    divided by the caller-supplied normalization constant; the standard
    sequence-MaxEnt learner passes one and applies only Dr.GRPO's common outer
    update scale afterward.
    """

    if token_entropy.ndim != 2:
        raise ValueError("token_entropy must have shape [rows, tokens]")
    for name, value in (
        ("new_selected_log_probs", new_selected_log_probs),
        ("old_selected_log_probs", old_selected_log_probs),
        ("response_masks", response_masks),
    ):
        if value.shape != token_entropy.shape:
            raise ValueError(f"{name} must match token_entropy")
    scale = float(normalization_constant)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("normalization_constant must be finite and positive")
    if cliprange is not None:
        epsilon = float(cliprange)
        if not math.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError("cliprange must be in (0, 1) or None")
    else:
        epsilon = 0.0

    mask_bool = response_masks.to(torch.bool)
    mask = mask_bool.to(dtype=token_entropy.dtype)
    for name, value in (
        ("token_entropy", token_entropy),
        ("new_selected_log_probs", new_selected_log_probs),
        ("old_selected_log_probs", old_selected_log_probs),
    ):
        if not bool(torch.isfinite(value[mask_bool]).all()):
            raise ValueError(f"{name} must be finite on response tokens")

    active_entropy = torch.where(
        mask_bool, token_entropy, torch.zeros_like(token_entropy)
    )
    token_log_ratio = torch.where(
        mask_bool,
        new_selected_log_probs - old_selected_log_probs,
        torch.zeros_like(new_selected_log_probs),
    )
    inclusive_log_ratio = torch.cumsum(token_log_ratio, dim=1)
    exclusive_log_ratio = inclusive_log_ratio - token_log_ratio
    # Extreme products are not numerically meaningful for a PPO update. The
    # clamp is far outside the active PPO interval and only prevents overflow.
    prefix_ratio = torch.exp(exclusive_log_ratio.clamp(-40.0, 40.0))
    unclipped_terms = prefix_ratio * active_entropy

    if cliprange is None:
        surrogate_terms = unclipped_terms
        outside = torch.zeros_like(mask, dtype=torch.bool)
    else:
        clipped_ratio = torch.clamp(
            prefix_ratio,
            min=1.0 - epsilon,
            max=1.0 + epsilon,
        )
        surrogate_terms = torch.minimum(
            unclipped_terms,
            clipped_ratio * active_entropy,
        )
        outside = (prefix_ratio < 1.0 - epsilon) | (
            prefix_ratio > 1.0 + epsilon
        )

    active_tokens = mask.sum(dim=1).clamp_min(1.0)
    prefix_ratio_masked = prefix_ratio * mask
    return (
        surrogate_terms.sum(dim=1) / scale,
        unclipped_terms.detach().sum(dim=1) / scale,
        active_entropy.detach().sum(dim=1) / scale,
        prefix_ratio_masked.detach().sum(dim=1) / active_tokens,
        prefix_ratio_masked.detach().amax(dim=1),
        (outside.to(mask.dtype) * mask).detach().sum(dim=1) / active_tokens,
    )
