"""Shared fresh-group diagnostics."""

from __future__ import annotations

import torch


def aggregation_group_diagnostics(
    row_weights: torch.Tensor, final_rewards: torch.Tensor, *, num_samples: int
) -> dict[str, torch.Tensor]:
    """Candidate-aggregation diagnostics from the realized row weights.

    Defined identically for every quartet arm: pass the realized per-row
    aggregation weights — uniform ones (Dr.GRPO, Token-MaxEnt), the
    prompt-rescaled uniform SEED weights, or the tempered xDr.GRPO softmax
    weights — with loss-masked rows already zeroed. Each prompt group is
    normalized to a distribution over its valid rows; all-masked groups are
    dropped from the batch means.

    Returns the batch means of the effective number of active rollouts
    exp(H(w)) per prompt (n_valid for uniform weights) and the incorrect-mass
    share sum_i w_i * 1{r_i <= 0} per prompt.
    """
    with torch.no_grad():
        w = row_weights.detach().reshape(-1, num_samples).float()
        totals = w.sum(dim=1, keepdim=True)
        valid_groups = totals.reshape(-1) > 0
        if not bool(valid_groups.any()):
            zero = w.new_tensor(0.0)
            return {"agg_eff_rollouts": zero, "agg_incorrect_mass": zero}
        w = w[valid_groups] / totals[valid_groups]
        entropy = -(w * w.clamp_min(1e-12).log()).sum(dim=1)
        eff_rollouts = entropy.exp().mean()
        incorrect = (
            final_rewards.detach().reshape(-1, num_samples)[valid_groups] <= 0
        ).float()
        incorrect_mass = (w * incorrect).sum(dim=1).mean()
    return {"agg_eff_rollouts": eff_rollouts, "agg_incorrect_mass": incorrect_mass}
