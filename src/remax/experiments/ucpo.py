"""UCPO advantage-mass redistribution over verifier-positive rollouts.

This implements Equations 47, 51, and 16 of Lochab et al. (2026). The rollout
policy supplies samples from the conditional correct distribution. UCPO
estimates that distribution from sequence probabilities, forms self-normalized
inverse-probability weights, and reallocates (without changing) the group's
total positive advantage mass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class UCPODiagnostics:
    groups: int
    eligible_groups: int
    correct_rows: int
    weight_min: float
    weight_max: float
    mass_error_max: float


def redistribute_ucpo_advantages(
    advantages: torch.Tensor,
    sequence_log_probs: torch.Tensor,
    task_rewards: torch.Tensor,
    loss_masks: torch.Tensor,
    *,
    num_samples: int,
    tau: float,
) -> tuple[torch.Tensor, UCPODiagnostics]:
    """Return UCPO-reweighted advantages and audit diagnostics.

    Incorrect and loss-masked rows are unchanged. Correct rows in each prompt
    group receive A_i = n A+ ((1-tau)/n + tau v_i/sum_j v_j), where
    v_i = 1/qhat_i and qhat is computed from the rollout-policy sequence
    probabilities of that group's verifier-positive rows.
    """

    if isinstance(num_samples, bool) or int(num_samples) != num_samples:
        raise ValueError("num_samples must be an integer")
    num_samples = int(num_samples)
    if num_samples <= 1:
        raise ValueError("num_samples must exceed one")
    tau = float(tau)
    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be finite and in [0, 1]")

    original_shape = advantages.shape
    adv = advantages.detach().reshape(-1)
    seq = sequence_log_probs.detach().reshape(-1)
    rewards = task_rewards.detach().reshape(-1)
    active = loss_masks.detach().reshape(-1) > 0
    rows = int(adv.numel())
    if not (seq.numel() == rewards.numel() == active.numel() == rows):
        raise ValueError("UCPO tensors must contain the same number of rows")
    if rows == 0 or rows % num_samples:
        raise ValueError("UCPO rows must be non-empty complete prompt groups")
    if not bool(torch.isfinite(adv).all()) or not bool(torch.isfinite(seq).all()):
        raise ValueError(
            "UCPO advantages and sequence log-probabilities must be finite"
        )

    result = adv.clone()
    eligible_groups = 0
    correct_rows = 0
    weight_min = math.inf
    weight_max = -math.inf
    mass_error_max = 0.0

    for start in range(0, rows, num_samples):
        stop = start + num_samples
        correct = (rewards[start:stop] > 0) & active[start:stop]
        indices = torch.nonzero(correct, as_tuple=False).reshape(-1)
        n_correct = int(indices.numel())
        if n_correct == 0:
            continue

        group_adv = adv[start:stop][indices]
        # Binary-reward GRPO/Dr.GRPO gives every correct row in a prompt the
        # same base advantage. Refuse a future reward surface that violates the
        # UCPO estimator instead of silently reinterpreting Equation 16.
        if not bool(
            torch.allclose(
                group_adv,
                group_adv[:1].expand_as(group_adv),
                rtol=1e-5,
                atol=1e-7,
            )
        ):
            raise ValueError(
                "UCPO requires one shared base advantage for correct rows"
            )

        log_qhat = torch.log_softmax(seq[start:stop][indices], dim=0)
        inverse_weights = torch.softmax(-log_qhat, dim=0)
        blended = (1.0 - tau) / float(n_correct) + tau * inverse_weights
        scales = float(n_correct) * blended
        replacement = group_adv[:1] * scales.to(group_adv.dtype)
        result[start + indices] = replacement

        old_mass = group_adv.sum().double()
        new_mass = replacement.sum().double()
        mass_error_max = max(
            mass_error_max,
            float(torch.abs(new_mass - old_mass).cpu().item()),
        )
        weight_min = min(weight_min, float(blended.min().cpu().item()))
        weight_max = max(weight_max, float(blended.max().cpu().item()))
        eligible_groups += 1
        correct_rows += n_correct

    diagnostics = UCPODiagnostics(
        groups=rows // num_samples,
        eligible_groups=eligible_groups,
        correct_rows=correct_rows,
        weight_min=0.0 if eligible_groups == 0 else weight_min,
        weight_max=0.0 if eligible_groups == 0 else weight_max,
        mass_error_max=mass_error_max,
    )
    return result.reshape(original_shape), diagnostics
