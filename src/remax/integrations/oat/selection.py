"""Explicit boundary between maintained replay and historical experiments."""

from __future__ import annotations

import math
from typing import Any

# These features change admission, rewards, reductions, or replay gradients.
# Inert tuning fields of a disabled comparator do not select that comparator.
EXPERIMENT_SWITCHES = (
    "reinforce_update",
    "dapo_enabled",
    "gapo_enabled",
    "rlep_replay_count",
    "diayn_num_options",
    "outcome_collision_coef",
    "semantic_shannon_coef",
    "ucpo_tau",
    "setpo_coefficient",
    "seed_entropy_alpha",
    "maxent_alpha",
    "policy_entropy_coef",
    "beta",
    "kl_penalty_coef",
    "maxent_control_target_ratio",
    "maxent_dual_target_ratio",
    "maxent_inverse_adaptation",
    "maxent_length_target",
    "xdr_tau_control_target_ratio",
    "xdr_sac_dual_target_ratio",
    "online_canonical_bank_alpha",
    "online_canonical_dual_target_ratio",
    "online_canonical_policy_entropy_adaptation",
    "online_canonical_replay_bank_normalized",
    "online_canonical_replay_retention_safe_balance",
    "online_canonical_counterfactual_proposals",
    "online_canonical_counterfactual_fixed_control_groups",
    "online_canonical_proposal_retention_tracking",
    "online_canonical_proposal_adaptive_retention_priority",
    "online_canonical_counterfactual_separate_objective_support",
    "online_canonical_proposal_replay_priority_visits",
    "math_strategy_gate_task_reward",
)
EXPERIMENT_STATE = (
    "_diayn_mi_tracker",
    "_semantic_shannon_tracker",
    "_maxent_alpha_controller",
    "_maxent_length_controller",
    "_online_canonical_alpha_controller",
    "_xdr_tau_controller",
)


def historical_reasons(learner: Any) -> tuple[str, ...]:
    """Explain why an invocation requires historical behavior; no silent pruning."""
    args = learner.args
    reasons = [name for name in EXPERIMENT_SWITCHES if getattr(args, name, False)]
    reasons.extend(
        name for name in EXPERIMENT_STATE if getattr(learner, name, None) is not None
    )
    if getattr(args, "critic_type", "drgrpo") != "drgrpo":
        reasons.append("critic_type")
    if math.isfinite(float(getattr(args, "xdr_tau", math.inf))):
        reasons.append("xdr_tau")
    if getattr(learner, "ref_model", None) is not None:
        reasons.append("ref_model")
    if (
        getattr(args, "online_canonical_key_mode", "modebench_outcome")
        != "modebench_outcome"
    ):
        reasons.append("online_canonical_key_mode")
    if getattr(args, "online_canonical_replay", False):
        for name, expected in (
            ("online_canonical_replay_objective", "verified_likelihood_per_rollout"),
            ("online_canonical_replay_key_weighting", "uniform"),
        ):
            if getattr(args, name, expected) != expected:
                reasons.append(name)
    bank = getattr(learner, "_online_canonical_bank", None)
    if bank is not None:
        if bank.objective_active:
            reasons.append("bank.objective_active")
        if bank.proposal_retention_tracking_enabled:
            reasons.append("bank.proposal_retention_tracking_enabled")
    return tuple(reasons)
