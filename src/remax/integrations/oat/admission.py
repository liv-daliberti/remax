"""Translate verified trajectory rows into prompt-local banks and replay draws."""

from __future__ import annotations
from .telemetry import record_admission_metrics
import logging
from typing import Any
import torch
from ...math_grader import validated_modebench_outcome_key
from ...core.bank import OnlineCanonicalBank


def _task_bound_canonicalization_surfaces(
    decoded_surfaces: list[str],
    trajectory: dict[str, Any],
    *,
    canonical_task: str,
    expected_count: int | None = None,
) -> list[str]:
    """Use verifier-ready witnesses while retaining canonical action token IDs."""
    if canonical_task == "none":
        return decoded_surfaces
    surfaces = [str(value) for value in list(trajectory.get("responses") or [])]
    required = len(decoded_surfaces) if expected_count is None else expected_count
    if len(surfaces) != required:
        raise RuntimeError(
            "canonical ModeBench tracking requires the exact task-decoded response for every sampled action sequence"
        )
    return surfaces


class OatAdmissionMixin:

    def _admit_and_schedule_replay(
        self,
        *,
        args,
        canonical_actions,
        canonical_replay_groups,
        canonical_task,
        final_rewards,
        input_ids,
        loss_masks,
        online_canonical_bank,
        online_canonical_infos,
        prompt_id_lens,
        raw_task_final_rewards,
        response_masks,
        task_final_rewards,
        trajectory,
    ):
        if online_canonical_bank is not None:
            if not isinstance(online_canonical_bank, OnlineCanonicalBank):
                raise RuntimeError("invalid online canonical bank")
            online_canonical_replay_active = bool(
                getattr(args, "online_canonical_replay", False)
            )
            num_rows = int(input_ids.size(0))
            response_texts: list[str] = []
            response_token_ids: list[list[int]] = []
            label_ids = input_ids[:, 1:]
            for row_ids, row_mask in zip(label_ids, response_masks):
                token_ids = row_ids[row_mask.to(torch.bool)].detach().cpu().tolist()
                response_token_ids.append(token_ids)
                response_texts.append(
                    self.tokenizer.decode(token_ids, skip_special_tokens=True)
                )
            key_mode = str(
                getattr(args, "online_canonical_key_mode", "modebench_outcome")
            )
            if key_mode == "modebench_outcome" and canonical_actions:
                response_texts = _task_bound_canonicalization_surfaces(
                    response_texts, trajectory, canonical_task=canonical_task
                )
            prompt_token_ids = [
                input_ids[row_index, : int(prompt_id_lens[row_index])]
                .detach()
                .cpu()
                .tolist()
                for row_index in range(num_rows)
            ]
            task_reward_positive = (
                task_final_rewards.detach().view(-1).gt(0).cpu().tolist()
            )
            references = list(trajectory.get("references") or [])
            references = (references + [None] * num_rows)[:num_rows]
            outcome_keys = [
                validated_modebench_outcome_key(text, references[row_index])
                for (row_index, text) in enumerate(response_texts)
            ]
            validator_admitted = [key is not None for key in outcome_keys]
            validator_positive_actor_negative_rows = [
                index
                for (index, (verified, rewarded)) in enumerate(
                    zip(validator_admitted, task_reward_positive)
                )
                if verified and (not rewarded)
            ]
            actor_positive_validator_negative_rows = [
                index
                for (index, (verified, rewarded)) in enumerate(
                    zip(validator_admitted, task_reward_positive)
                )
                if rewarded and (not verified)
            ]
            disagreement_rows = (
                validator_positive_actor_negative_rows
                + actor_positive_validator_negative_rows
            )
            if disagreement_rows:
                logging.warning(
                    "online canonical validator/task-reward disagreement at rows %s; fail-closed intersection excludes them from bank admission",
                    sorted(disagreement_rows),
                )
            outcome_keys = [
                key if key is not None and rewarded else None
                for (key, rewarded) in zip(outcome_keys, task_reward_positive)
            ]
            admitted = [key is not None for key in outcome_keys]
            replay_bank_freeze_step = int(
                getattr(args, "online_canonical_replay_bank_freeze_step", 0)
            )
            replay_bank_membership_frozen = (
                replay_bank_freeze_step > 0
                and int(getattr(self, "steps", 0)) >= replay_bank_freeze_step
            )
            (bank_advantages, bank_diagnostics) = (
                online_canonical_bank.score_and_update(
                    prompt_token_ids=prompt_token_ids,
                    outcome_keys=outcome_keys,
                    task_rewards=task_final_rewards.detach().view(-1).cpu().tolist(),
                    active_mask=loss_masks.detach().view(-1).cpu().tolist(),
                    num_samples=int(args.num_samples),
                    entropy_alpha_override=(
                        getattr(
                            self, "_online_canonical_alpha_controller", None
                        ).current_alpha
                        if getattr(self, "_online_canonical_alpha_controller", None)
                        is not None
                        else None
                    ),
                    response_token_ids=(
                        response_token_ids if online_canonical_replay_active else None
                    ),
                    update_bank=not replay_bank_membership_frozen,
                )
            )
            canonical_replay_used_global_scheduler = False
            canonical_replay_used_prompt_local_scheduler = False
            if online_canonical_replay_active:
                replay_min_modes = (
                    1
                    if str(args.online_canonical_replay_objective)
                    in {
                        "verified_likelihood_per_rollout",
                        "split_mass_balance_per_rollout",
                    }
                    else 2
                )
                if int(args.online_canonical_replay_global_groups_per_step) > 0:
                    if int(
                        args.online_canonical_replay_global_bootstrap_steps
                    ) > 0 and (
                        not online_canonical_bank.global_replay_bootstrap_active
                    ):
                        canonical_replay_used_prompt_local_scheduler = True
                        canonical_replay_groups = online_canonical_bank.replay_groups(
                            prompt_token_ids,
                            min_modes=replay_min_modes,
                            consume_priority=True,
                        )
                    else:
                        canonical_replay_used_global_scheduler = True
                        canonical_replay_groups = (
                            online_canonical_bank.scheduled_global_replay_groups(
                                min_modes=replay_min_modes
                            )
                        )
                else:
                    canonical_replay_used_prompt_local_scheduler = True
                    canonical_replay_groups = online_canonical_bank.replay_groups(
                        prompt_token_ids,
                        min_modes=replay_min_modes,
                        consume_priority=True,
                    )
            if getattr(args, "_remax_resume_identity", None) is not None:
                from dataclasses import asdict

                self._resume_replay_decisions = [
                    asdict(group) for group in canonical_replay_groups
                ]
            online_canonical_infos = record_admission_metrics(
                actor_positive_validator_negative_rows=actor_positive_validator_negative_rows,
                args=args,
                bank_diagnostics=bank_diagnostics,
                canonical_replay_groups=canonical_replay_groups,
                canonical_replay_used_global_scheduler=canonical_replay_used_global_scheduler,
                canonical_replay_used_prompt_local_scheduler=canonical_replay_used_prompt_local_scheduler,
                disagreement_rows=disagreement_rows,
                final_rewards=final_rewards,
                online_canonical_bank=online_canonical_bank,
                online_canonical_replay_active=online_canonical_replay_active,
                raw_task_final_rewards=raw_task_final_rewards,
                replay_bank_freeze_step=replay_bank_freeze_step,
                replay_bank_membership_frozen=replay_bank_membership_frozen,
                task_final_rewards=task_final_rewards,
                validator_positive_actor_negative_rows=validator_positive_actor_negative_rows,
            )
        return (canonical_replay_groups, online_canonical_infos)
