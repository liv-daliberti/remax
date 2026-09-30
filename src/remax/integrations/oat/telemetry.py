"""Historical metric compatibility for the maintained replay adapter."""

from __future__ import annotations
import hashlib
import math
import torch


def record_replay_metrics(
    *,
    actuator_loss,
    balance_loss,
    balance_scales,
    canonical_replay_groups,
    detached_scores,
    fresh_advantage_fingerprint,
    infos,
    input_ids,
    normalized_entropy,
    raw_score_gradients,
    replay,
    replay_alpha,
    replay_applied_weighted_loss,
    replay_bank_normalized,
    replay_banked_modes,
    replay_chunk_size,
    replay_compute_only,
    replay_key_weighting,
    replay_mass_alpha,
    replay_objective,
    replay_objective_scale,
    replay_result,
    replay_target_weights,
    replay_weighted_loss,
    requested_score_gradients,
    retention_safe_balance,
    reward_estimator_scale,
    score_gradients,
    self,
    stats,
):
    """Retain established metric names without coupling telemetry to method arithmetic."""
    target_entropy_ratios: list[torch.Tensor] = []
    target_ginis: list[torch.Tensor] = []
    target_rare_allocations: list[torch.Tensor] = []
    target_common_allocations: list[torch.Tensor] = []
    target_start = 0
    for target_size in replay.group_sizes:
        target_stop = target_start + target_size
        target_slice = replay_target_weights[target_start:target_stop].to(torch.float64)
        target_probabilities = target_slice / target_slice.sum()
        target_counts = replay.fresh_observation_counts[target_start:target_stop]
        if target_size >= 2:
            target_entropy_ratios.append(
                -(target_probabilities * target_probabilities.log()).sum()
                / math.log(target_size)
            )
        else:
            target_entropy_ratios.append(target_slice.new_tensor(1.0))
        target_ginis.append(
            torch.abs(
                target_probabilities[:, None] - target_probabilities[None, :]
            ).sum()
            / (2.0 * target_size)
        )
        target_rare_allocations.append(
            target_probabilities[target_counts == target_counts.min()].sum()
        )
        target_common_allocations.append(
            target_probabilities[target_counts == target_counts.max()].sum()
        )
        target_start = target_stop
    target_normalized_entropy = torch.stack(target_entropy_ratios).mean()
    target_gini = torch.stack(target_ginis).mean()
    target_rare_allocation = torch.stack(target_rare_allocations).mean()
    target_common_allocation = torch.stack(target_common_allocations).mean()
    infos.update(
        {
            "canonical_replay_actuator_loss": actuator_loss.detach(),
            "canonical_replay_balance_loss": balance_loss.detach(),
            "canonical_replay_weighted_loss": replay_applied_weighted_loss.detach(),
            "canonical_replay_raw_weighted_loss": replay_weighted_loss.detach(),
            "canonical_replay_compute_only": torch.tensor(
                float(replay_compute_only), dtype=torch.float32, device=input_ids.device
            ),
            "canonical_replay_backward_scale": torch.tensor(
                float(self.strategy.grad_acc_step),
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "canonical_replay_chunk_size": torch.tensor(
                replay_chunk_size, dtype=torch.float32, device=input_ids.device
            ),
            "canonical_replay_score_passes": torch.tensor(
                2.0, dtype=torch.float32, device=input_ids.device
            ),
            "canonical_replay_normalized_model_entropy": normalized_entropy.detach(),
            "canonical_replay_cross_entropy_excess": balance_loss.detach(),
            "canonical_replay_alpha_used": torch.tensor(
                replay_alpha, dtype=torch.float64, device=input_ids.device
            ),
            "canonical_replay_per_mode_pressure": torch.tensor(
                replay_alpha / replay_banked_modes if replay_banked_modes > 0 else 0.0,
                dtype=torch.float64,
                device=input_ids.device,
            ),
            "canonical_replay_bank_normalized": torch.tensor(
                1.0 if replay_bank_normalized else 0.0,
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "canonical_replay_eligible_groups": torch.tensor(
                replay_result.eligible_groups,
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "canonical_replay_retained_modes": torch.tensor(
                replay_result.retained_modes,
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "canonical_replay_actuator_groups": torch.tensor(
                replay_result.actuator_groups,
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "canonical_replay_actuator_modes": torch.tensor(
                replay_result.actuator_modes,
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "canonical_replay_reward_estimator_scale": torch.tensor(
                reward_estimator_scale, dtype=torch.float32, device=input_ids.device
            ),
            "canonical_replay_score_gradient_sum": raw_score_gradients.sum(),
            "canonical_replay_objective_scale": torch.tensor(
                replay_objective_scale, dtype=torch.float32, device=input_ids.device
            ),
            "canonical_replay_applied_score_gradient_sum": score_gradients.sum()
            * replay_objective_scale,
            "canonical_replay_verified_likelihood_active": torch.tensor(
                float(
                    replay_objective
                    in {
                        "verified_likelihood",
                        "verified_likelihood_per_rollout",
                        "split_mass_balance_per_rollout",
                    }
                ),
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "canonical_replay_mass_alpha_used": torch.tensor(
                replay_mass_alpha, dtype=torch.float64, device=input_ids.device
            ),
            "canonical_replay_balance_alpha_used": torch.tensor(
                replay_alpha, dtype=torch.float64, device=input_ids.device
            ),
            "canonical_replay_mass_score_gradient_sum": detached_scores.new_tensor(
                0.0
            ).to(input_ids.device),
            "canonical_replay_balance_score_gradient_sum": detached_scores.new_tensor(
                0.0
            ).to(input_ids.device),
            "canonical_replay_mass_score_gradient_l2": detached_scores.new_tensor(
                0.0
            ).to(input_ids.device),
            "canonical_replay_balance_score_gradient_l2": detached_scores.new_tensor(
                0.0
            ).to(input_ids.device),
            "canonical_replay_applied_score_gradient_l2": torch.linalg.vector_norm(
                score_gradients
            ),
            "canonical_replay_retention_safe_balance": torch.tensor(
                float(retention_safe_balance),
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "canonical_replay_balance_scale_min": balance_scales.min().to(
                input_ids.device
            ),
            "canonical_replay_balance_scale_mean": balance_scales.mean().to(
                input_ids.device
            ),
            "canonical_replay_balance_capped_group_fraction": (balance_scales < 1.0)
            .float()
            .mean()
            .to(input_ids.device),
            "canonical_replay_requested_positive_gradient_max": torch.clamp(
                requested_score_gradients, min=0.0
            ).max(),
            "canonical_replay_applied_positive_gradient_max": torch.clamp(
                raw_score_gradients, min=0.0
            ).max(),
            "canonical_replay_priority_modes": torch.tensor(
                replay.priority_modes, dtype=torch.float32, device=input_ids.device
            ),
            "canonical_replay_mass_weight_min": replay.mass_weights.min().to(
                input_ids.device
            ),
            "canonical_replay_mass_weight_max": replay.mass_weights.max().to(
                input_ids.device
            ),
        }
    )
    infos.update(
        {
            "canonical_replay_key_weighting_frequency": torch.tensor(
                float(replay_key_weighting == "fresh_frequency"),
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "canonical_replay_target_normalized_entropy": target_normalized_entropy.to(
                input_ids.device
            ),
            "canonical_replay_target_gini": target_gini.to(input_ids.device),
            "canonical_replay_rare_key_gradient_allocation": target_rare_allocation.to(
                input_ids.device
            ),
            "canonical_replay_common_key_gradient_allocation": target_common_allocation.to(
                input_ids.device
            ),
            "canonical_replay_target_weight_sum": replay_target_weights.sum().to(
                input_ids.device
            ),
            "canonical_replay_target_weight_min": replay_target_weights.min().to(
                input_ids.device
            ),
            "canonical_replay_target_weight_max": replay_target_weights.max().to(
                input_ids.device
            ),
            "canonical_replay_fresh_observation_count_sum": replay.fresh_observation_counts.sum().to(
                input_ids.device
            ),
            "canonical_replay_fresh_observation_count_min": replay.fresh_observation_counts.min().to(
                input_ids.device
            ),
            "canonical_replay_fresh_observation_count_max": replay.fresh_observation_counts.max().to(
                input_ids.device
            ),
            "canonical_replay_frequency_count_fresh_only": torch.tensor(
                1.0, dtype=torch.float32, device=input_ids.device
            ),
            "canonical_replay_frequency_count_from_replay": torch.tensor(
                0.0, dtype=torch.float32, device=input_ids.device
            ),
            "canonical_replay_frequency_count_from_proposals": torch.tensor(
                0.0, dtype=torch.float32, device=input_ids.device
            ),
            "canonical_replay_fresh_advantage_fingerprint": torch.tensor(
                fresh_advantage_fingerprint,
                dtype=torch.float64,
                device=input_ids.device,
            ),
        }
    )
    flat_outcome_keys = [
        outcome_key
        for replay_group in canonical_replay_groups
        for outcome_key in replay_group.outcome_keys
    ]
    flat_prompt_fingerprints = [
        int(
            hashlib.sha256(
                repr(replay_group.prompt_token_ids).encode("utf-8")
            ).hexdigest()[:13],
            16,
        )
        for replay_group in canonical_replay_groups
        for _outcome_key in replay_group.outcome_keys
    ]
    replay_token_counts = replay.response_masks.sum(dim=1).to(detached_scores.dtype)
    for row_index, (
        target_weight,
        fresh_count,
        outcome_key,
        prompt_fingerprint,
        mean_logprob,
        token_count,
    ) in enumerate(
        zip(
            replay_target_weights,
            replay.fresh_observation_counts,
            flat_outcome_keys,
            flat_prompt_fingerprints,
            detached_scores,
            replay_token_counts,
        )
    ):
        infos[f"canonical_replay_target_weight_row_{row_index:02d}"] = target_weight.to(
            input_ids.device
        )
        infos[f"canonical_replay_fresh_count_row_{row_index:02d}"] = fresh_count.to(
            input_ids.device
        )
        infos[f"canonical_replay_outcome_fingerprint_row_{row_index:02d}"] = (
            torch.tensor(
                float(
                    int(
                        hashlib.sha256(outcome_key.encode("utf-8")).hexdigest()[:13], 16
                    )
                ),
                dtype=torch.float64,
                device=input_ids.device,
            )
        )
        infos[f"canonical_replay_prompt_fingerprint_row_{row_index:02d}"] = (
            torch.tensor(
                float(prompt_fingerprint), dtype=torch.float64, device=input_ids.device
            )
        )
        infos[f"canonical_replay_exemplar_mean_logprob_row_{row_index:02d}"] = (
            mean_logprob.to(input_ids.device)
        )
        infos[f"canonical_replay_exemplar_sequence_logprob_row_{row_index:02d}"] = (
            mean_logprob * token_count
        ).to(input_ids.device)
    for group_index, replay_group in enumerate(canonical_replay_groups):
        prompt_fingerprint = int(
            hashlib.sha256(
                repr(replay_group.prompt_token_ids).encode("utf-8")
            ).hexdigest()[:13],
            16,
        )
        membership_fingerprint = int(
            hashlib.sha256(repr(replay_group.outcome_keys).encode("utf-8")).hexdigest()[
                :13
            ],
            16,
        )
        infos[f"canonical_replay_prompt_fingerprint_group_{group_index:02d}"] = (
            torch.tensor(
                float(prompt_fingerprint), dtype=torch.float64, device=input_ids.device
            )
        )
        infos[f"canonical_replay_membership_fingerprint_group_{group_index:02d}"] = (
            torch.tensor(
                float(membership_fingerprint),
                dtype=torch.float64,
                device=input_ids.device,
            )
        )
    for key, value in (
        ("canonical_replay_actuator_loss", actuator_loss),
        ("canonical_replay_balance_loss", balance_loss),
        ("canonical_replay_weighted_loss", replay_applied_weighted_loss),
        ("canonical_replay_normalized_model_entropy", normalized_entropy),
        ("canonical_replay_cross_entropy_excess", balance_loss),
    ):
        stats[key].append(float(value.detach().cpu().item()))
    stats["canonical_replay_alpha_used"].append(replay_alpha)
    stats["canonical_replay_mass_alpha_used"].append(replay_mass_alpha)
    stats["canonical_replay_banked_modes"].append(float(replay_banked_modes))
    stats["canonical_replay_per_mode_pressure"].append(
        replay_alpha / replay_banked_modes if replay_banked_modes > 0 else 0.0
    )
    stats["canonical_replay_eligible_groups"].append(
        float(replay_result.eligible_groups)
    )
    stats["canonical_replay_retained_modes"].append(float(replay_result.retained_modes))


def record_admission_metrics(
    *,
    actor_positive_validator_negative_rows,
    args,
    bank_diagnostics,
    canonical_replay_groups,
    canonical_replay_used_global_scheduler,
    canonical_replay_used_prompt_local_scheduler,
    disagreement_rows,
    final_rewards,
    online_canonical_bank,
    online_canonical_replay_active,
    raw_task_final_rewards,
    replay_bank_freeze_step,
    replay_bank_membership_frozen,
    task_final_rewards,
    validator_positive_actor_negative_rows,
):
    """Retain established metric names without coupling telemetry to method arithmetic."""
    online_canonical_infos = {
        f"online_canonical_{name}": torch.tensor(
            getattr(bank_diagnostics, name), device=final_rewards.device
        )
        for name in (
            "entropy_estimate_mean",
            "normalized_entropy_mean",
            "normalized_entropy_ratio_mean",
            "normalized_entropy_ratio_eligible_fraction",
            "log_support_mean",
            "entropy_alpha_used",
            "entropy_advantage_mean",
            "entropy_advantage_rms",
            "combined_advantage_mean",
            "combined_advantage_rms",
            "eligible_fraction",
            "reward_positive_fraction",
            "canonicalizable_correct_fraction",
            "new_outcome_count",
            "new_outcome_row_fraction",
            "bank_size_before_mean",
            "bank_size_after_mean",
            "tracked_prompts",
            "tracked_outcomes",
            "support_at_least_two_prompt_fraction",
        )
    }
    online_canonical_infos.update(
        {
            f"online_canonical_proposal_retention_{name}": torch.tensor(
                value, dtype=torch.float32, device=final_rewards.device
            )
            for (
                name,
                value,
            ) in online_canonical_bank.proposal_retention_diagnostics().items()
        }
    )
    if online_canonical_replay_active:
        online_canonical_infos.update(
            {
                "canonical_replay_available_groups": torch.tensor(
                    len(canonical_replay_groups),
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_available_modes": torch.tensor(
                    sum((len(group.outcome_keys) for group in canonical_replay_groups)),
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_capacity": torch.tensor(
                    int(args.online_canonical_replay_capacity),
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_bank_freeze_step": torch.tensor(
                    replay_bank_freeze_step,
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_bank_membership_frozen": torch.tensor(
                    float(replay_bank_membership_frozen),
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_compute_only_configured": torch.tensor(
                    float(
                        bool(
                            getattr(args, "online_canonical_replay_compute_only", False)
                        )
                    ),
                    device=final_rewards.device,
                ),
                "canonical_replay_realized_prompt_tokens": torch.tensor(
                    sum(
                        (
                            len(group.prompt_token_ids) * len(group.response_token_ids)
                            for group in canonical_replay_groups
                        )
                    ),
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_realized_response_tokens": torch.tensor(
                    sum(
                        (
                            len(response)
                            for group in canonical_replay_groups
                            for response in group.response_token_ids
                        )
                    ),
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_charged_response_token_budget": torch.tensor(
                    int(args.online_canonical_replay_capacity)
                    * int(args.generate_max_length)
                    * int(max(1, args.online_canonical_replay_global_groups_per_step)),
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_gold_support_feedback": torch.tensor(
                    0.0, device=final_rewards.device
                ),
                "canonical_replay_global_scheduler_active": torch.tensor(
                    float(int(args.online_canonical_replay_global_groups_per_step) > 0),
                    device=final_rewards.device,
                ),
                "canonical_replay_global_groups_per_step": torch.tensor(
                    int(args.online_canonical_replay_global_groups_per_step),
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_global_bootstrap_steps": torch.tensor(
                    int(args.online_canonical_replay_global_bootstrap_steps),
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_global_bootstrap_updates": torch.tensor(
                    online_canonical_bank.global_replay_updates,
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_global_bootstrap_active": torch.tensor(
                    float(online_canonical_bank.global_replay_bootstrap_active),
                    device=final_rewards.device,
                ),
                "canonical_replay_prompt_local_phase_active": torch.tensor(
                    float(
                        int(args.online_canonical_replay_global_bootstrap_steps) > 0
                        and (not online_canonical_bank.global_replay_bootstrap_active)
                    ),
                    device=final_rewards.device,
                ),
                "canonical_replay_schedule_used_global": torch.tensor(
                    float(canonical_replay_used_global_scheduler),
                    device=final_rewards.device,
                ),
                "canonical_replay_schedule_used_prompt_local": torch.tensor(
                    float(canonical_replay_used_prompt_local_scheduler),
                    device=final_rewards.device,
                ),
                "canonical_replay_priority_remaining_visits": torch.tensor(
                    online_canonical_bank.proposal_priority_remaining_visits,
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_priority_replay_groups_cumulative": torch.tensor(
                    online_canonical_bank.proposal_priority_replay_groups,
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
                "canonical_replay_priority_replay_modes_cumulative": torch.tensor(
                    online_canonical_bank.proposal_priority_replay_modes,
                    dtype=torch.float32,
                    device=final_rewards.device,
                ),
            }
        )
    online_canonical_infos["math_strategy_raw_task_reward_mean"] = (
        raw_task_final_rewards.detach().mean()
    )
    online_canonical_infos["math_strategy_gated_task_reward_mean"] = (
        task_final_rewards.detach().mean()
    )
    online_canonical_infos["math_strategy_task_reward_gate_active"] = torch.tensor(
        float(bool(getattr(args, "math_strategy_gate_task_reward", False))),
        device=final_rewards.device,
    )
    online_canonical_infos["online_canonical_task_reward_mean"] = (
        task_final_rewards.detach().mean()
    )
    online_canonical_infos["online_canonical_reward_sent_to_centering_mean"] = (
        final_rewards.detach().mean()
    )
    online_canonical_infos.update(
        {
            "online_canonical_validator_positive_actor_negative_rows": torch.tensor(
                len(validator_positive_actor_negative_rows), device=final_rewards.device
            ),
            "online_canonical_actor_positive_validator_negative_rows": torch.tensor(
                len(actor_positive_validator_negative_rows), device=final_rewards.device
            ),
            "online_canonical_validator_task_disagreement_rows": torch.tensor(
                len(disagreement_rows), device=final_rewards.device
            ),
            "verified_discovery_cumulative_outcomes": torch.tensor(
                bank_diagnostics.tracked_outcomes, device=final_rewards.device
            ),
            "verified_discovery_tracked_prompts": torch.tensor(
                bank_diagnostics.tracked_prompts, device=final_rewards.device
            ),
            "verified_discovery_mean_support_per_prompt": torch.tensor(
                online_canonical_bank.mean_support_per_prompt,
                device=final_rewards.device,
            ),
        }
    )
    return online_canonical_infos
