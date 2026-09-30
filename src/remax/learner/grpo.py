"""Baseline GRPO learner helpers."""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from oat.utils.ops import masked_mean

from ..benchmark import require_scorable, failure
from ..args import resolve_canonical_action_task
from ..answer_options import coerce_option_id, conditional_answer_repr
from ..canonical_actions import (
    canonical_action_code_from_token_ids,
    canonical_action_code_from_verified_response,
    canonical_action_code_token_ids,
    decode_canonical_action_response,
    canonical_behavior_overlap_diagnostics,
    materialize_canonical_behavior_policy,
    materialize_position_canonical_behavior_policy,
    restricted_action_log_probs_entropy_and_distribution,
    restricted_position_action_log_probs_entropy_and_distribution,
)
from ..canonical_replay import (
    CanonicalReplayBatch,
    cap_retention_safe_balance_score_gradients,
    canonical_replay_key_target_weights,
    canonical_replay_split_mass_balance_loss,
    canonical_replay_uniform_loss,
    canonical_replay_uniform_verified_likelihood_loss,
    materialize_canonical_replay_batch,
)
from ..dapo import (
    dapo_soft_overlong_penalty,
    dapo_token_level_policy_loss,
)
from ..math_grader import (
    extract_normalized_final_answer,
    validated_exploration_identity,
    validated_modebench_outcome_key,
)
from ..maxrl import binary_maxrl_advantages
from ..math_strategy_canonicalizer import MathStrategyCanonicalizer
from ..on_policy_maxent import (
    mean_active_token_entropy_by_response,
    prefix_ratio_expected_length_surrogate,
    prefix_ratio_maxent_surrogate,
    standard_maxent_length_penalty_loss,
    standard_maxent_loss,
)
from ..online_canonical_bank import (
    OnlineCanonicalBank,
    VerifiedCanonicalReplayGroup,
)
from ..replicated_group import (
    replicated_group_permutation_seed,
    validate_replicated_group_layout,
)
from ..rlep import (
    OnlineRLEPExperiencePool,
    RLEPExperiencePool,
    RLEPReplayGroup,
    rlep_mixed_advantages,
)
from ..outcome_collision import (
    add_outcome_collision_outside_centering_advantage,
    compute_outcome_collision_bonuses,
)
from ..semantic_shannon import (
    SemanticShannonTracker,
    add_semantic_shannon_separate_advantage,
    success_conditioned_semantic_metric_values,
)
from ..seed_weights import compute_seed_row_weights
from ..tensor_utils import cap_last_valid_token_pos_for_zero_advantage
from ..verified_route_library import VerifiedRouteLibrary
from ..xdr import aggregation_group_diagnostics, compute_xdr_row_weights
from ..gapo import GAPOSupportIndex, gapo_group_rewards
from ..setpo import shape_setpo_advantages
from ..setpo_embedder import SetPOEmbedder
from ..ucpo import redistribute_ucpo_advantages


MATH_VERIFIED_ANSWER_OUTCOME_KEY = "math_verified_answer:correct"


def math_verified_answer_outcome_keys(
    reward_positive: list[bool],
) -> list[str | None]:
    """Collapse validator-positive MATH answers to one prompt-local outcome."""

    return [
        MATH_VERIFIED_ANSWER_OUTCOME_KEY if positive else None
        for positive in reward_positive
    ]


def apply_math_strategy_task_reward_gate(
    final_rewards: torch.Tensor,
    task_final_rewards: torch.Tensor,
    admitted: list[bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero both learning reward views for non-admitted strategy executions."""

    if final_rewards.shape != task_final_rewards.shape:
        raise ValueError("MATH strategy reward views differ in shape")
    if final_rewards.numel() != len(admitted):
        raise ValueError("MATH strategy admission mask differs in length")
    contract_mask = torch.tensor(
        admitted,
        dtype=final_rewards.dtype,
        device=final_rewards.device,
    ).reshape_as(final_rewards)
    return (
        final_rewards * contract_mask,
        task_final_rewards * contract_mask,
    )


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
            "canonical ModeBench tracking requires the exact task-decoded "
            "response for every sampled action sequence"
        )
    return surfaces


@contextmanager
def _temporary_eval_mode(model: torch.nn.Module, *, enabled: bool):
    """Temporarily disable training-only model behavior and restore it exactly."""

    was_training = bool(model.training)
    if enabled:
        model.eval()
    try:
        yield
    finally:
        if enabled:
            model.train(was_training)


class ZeroMathGrpoMixin:
    """Dr.GRPO update with optional candidate or control-arm weighting."""

    def _use_instrumented_grpo_learning_step(self) -> bool:
        return int(getattr(self.args, "zero_stage", 0) or 0) >= 3 or bool(
            getattr(self.args, "adam_offload", False)
        )

    def _seed_answer_keys_grouped(
        self,
        input_ids: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
        references_grouped: list[list[Any | None]] | None = None,
        response_surfaces: list[str] | None = None,
    ) -> list[list[str | None]]:
        """Extract canonical final-answer keys for the SEED control."""

        label_ids = input_ids[:, 1:]
        if response_surfaces is None:
            rows = []
            for row_ids, row_mask in zip(label_ids, response_masks):
                token_ids = row_ids[row_mask.to(torch.bool)].detach().cpu().tolist()
                rows.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        else:
            rows = [str(value) for value in response_surfaces]
            if len(rows) != int(input_ids.size(0)):
                raise RuntimeError(
                    "task-bound semantic surfaces differ from sampled row count"
                )
        grouped_rows = [
            rows[index : index + group_size]
            for index in range(0, len(rows), group_size)
        ]
        keys_grouped = []
        for prompt_index, prompt_rows in enumerate(grouped_rows):
            refs = (
                [None] * len(prompt_rows)
                if references_grouped is None
                else references_grouped[prompt_index]
            )
            keys_grouped.append(
                [
                    extract_normalized_final_answer(
                        text,
                        template=str(self.args.prompt_template),
                        gt_answer=refs[row_index],
                    )
                    for row_index, text in enumerate(prompt_rows)
                ]
            )
        return keys_grouped

    def _gapo_support_index(self) -> GAPOSupportIndex:
        """Return the frozen enumerated-support index, loaded once.

        The index is read-only and identical for every step, so it is cached
        on the learner rather than re-read per batch: the shared filesystem
        should see one open, not one per optimizer step.
        """

        index = getattr(self, "_gapo_support_index_cache", None)
        if index is None:
            index = GAPOSupportIndex.load(str(self.args.gapo_support_index))
            self._gapo_support_index_cache = index
        return index

    def _setpo_embedder(self) -> SetPOEmbedder:
        """Return the frozen SetPO sentence embedder, loaded once."""

        embedder = getattr(self, "_setpo_embedder_cache", None)
        if embedder is None:
            embedder = SetPOEmbedder.from_path(
                str(self.args.setpo_embedder_path),
                device=torch.cuda.current_device(),
                batch_size=int(getattr(self.args, "setpo_embed_batch_size", 64)),
            )
            self._setpo_embedder_cache = embedder
        return embedder

    def _setpo_response_surfaces(
        self,
        input_ids: torch.Tensor,
        response_masks: torch.Tensor,
    ) -> list[str]:
        """Decode one completion string per row for SetPO's kernel.

        SetPO's similarity is defined over trajectories, not over canonical
        answer keys, so this deliberately embeds the generated text rather than
        the grader's key. Collapsing onto keys first would turn SetPO into a
        different method --- one much closer to this campaign's own.
        """

        label_ids = input_ids[:, 1:]
        return [
            self.tokenizer.decode(
                row_ids[row_mask.to(torch.bool)].detach().cpu().tolist(),
                skip_special_tokens=True,
            )
            for row_ids, row_mask in zip(label_ids, response_masks)
        ]

    def _should_skip_baseline_grad_norm_logging(self) -> bool:
        return self._use_instrumented_grpo_learning_step()

    def _baseline_progress_log_interval(self, total_micro_batches: int) -> int:
        if total_micro_batches <= 0:
            return 1
        return max(1, total_micro_batches // 8)

    def _baseline_should_log_progress(
        self,
        local_grad_step: int,
        total_micro_batches: int,
    ) -> bool:
        if not self.strategy.is_rank_0():
            return False
        if local_grad_step <= 1 or local_grad_step >= total_micro_batches:
            return True
        interval = self._baseline_progress_log_interval(total_micro_batches)
        return (local_grad_step % interval) == 0

    def _materialize_canonical_replay(
        self,
        groups: list[VerifiedCanonicalReplayGroup],
        *,
        device: torch.device | int,
    ) -> CanonicalReplayBatch:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError("canonical replay requires a tokenizer pad or EOS token")
        return materialize_canonical_replay_batch(
            groups,
            pad_token_id=int(pad_token_id),
            device=device,
        )

    def _score_canonical_replay_rows(
        self,
        replay: CanonicalReplayBatch,
        *,
        start: int,
        stop: int,
        policy_vocab_upper_bound: int,
    ) -> torch.Tensor:
        """Return length-neutral model scores for a memory-safe row chunk."""

        replay_input_ids = self._sanitize_scoring_token_ids(
            replay.input_ids[start:stop],
            upper_bound=policy_vocab_upper_bound,
            context="canonical_replay_policy_input",
        )
        replay_logits = self.model(
            replay_input_ids,
            attention_mask=replay.attention_mask[start:stop],
        )["logits"]
        if self.args.temperature != 1:
            replay_logits = replay_logits / self.args.temperature
        replay_logits = self._mask_invalid_scoring_logit_columns(
            replay_logits,
            valid_vocab_size=policy_vocab_upper_bound,
            context="canonical_replay_policy_logits",
        )
        replay_logps, _ = self._policy_logps_and_optional_entropy(
            replay_logits,
            replay_input_ids,
            replay.response_masks[start:stop],
            need_entropy=False,
        )
        replay_mask = replay.response_masks[start:stop].to(replay_logps.dtype)
        token_counts = replay_mask.sum(dim=1)
        if not bool(token_counts.gt(0).all()):
            raise RuntimeError("canonical replay materialized an empty response")
        return (replay_logps * replay_mask).sum(dim=1) / token_counts

    def _score_rlep_rows(
        self,
        replay: CanonicalReplayBatch,
        *,
        start: int,
        stop: int,
        policy_vocab_upper_bound: int,
    ) -> torch.Tensor:
        """Return Dr.GRPO-normalized sequence log-probabilities for RLEP."""

        replay_input_ids = self._sanitize_scoring_token_ids(
            replay.input_ids[start:stop],
            upper_bound=policy_vocab_upper_bound,
            context="rlep_replay_policy_input",
        )
        replay_logits = self.model(
            replay_input_ids,
            attention_mask=replay.attention_mask[start:stop],
        )["logits"]
        if self.args.temperature != 1:
            replay_logits = replay_logits / self.args.temperature
        replay_logits = self._mask_invalid_scoring_logit_columns(
            replay_logits,
            valid_vocab_size=policy_vocab_upper_bound,
            context="rlep_replay_policy_logits",
        )
        replay_logps, _ = self._policy_logps_and_optional_entropy(
            replay_logits,
            replay_input_ids,
            replay.response_masks[start:stop],
            need_entropy=False,
        )
        replay_mask = replay.response_masks[start:stop].to(replay_logps.dtype)
        if not bool(replay_mask.sum(dim=1).gt(0).all()):
            raise RuntimeError("RLEP materialized an empty response")
        return (replay_logps * replay_mask).sum(dim=1) / float(
            self.args.generate_max_length
        )

    def _baseline_update_with_precomputed_advantages(
        self,
        *,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        prompt_id_lens,
        loss_masks: torch.Tensor,
        response_masks: torch.Tensor,
        logps: torch.Tensor,
        ref_logps: torch.Tensor | None,
        advantages: torch.Tensor,
        final_rewards: torch.Tensor,
        returns: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        policy_vocab_upper_bound: int | None = None,
        row_weights: torch.Tensor | None = None,
        extra_infos: dict[str, torch.Tensor] | None = None,
        canonical_replay_groups: (list[VerifiedCanonicalReplayGroup] | None) = None,
        rlep_replay_groups: list[RLEPReplayGroup] | None = None,
        rlep_replay_advantage: float | None = None,
    ) -> dict[str, torch.Tensor]:
        args = self.args
        canonical_task = resolve_canonical_action_task(args)
        canonical_actions = canonical_task != "none"
        infos: dict[str, torch.Tensor] = {}
        if extra_infos:
            infos.update(extra_infos)
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        fresh_advantage_fingerprint = float(
            int(
                hashlib.sha256(
                    advantages.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()[:13],
                16,
            )
        )

        if policy_vocab_upper_bound is None:
            policy_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(
                self.model
            )

        replicated_group = bool(
            getattr(args, "canonical_graph_learner_sampling", False)
            or getattr(args, "replicated_freeform_sampling", False)
        )
        learner_world_size = dist.get_world_size() if replicated_group else 1
        if replicated_group:
            if len(input_ids) != int(args.num_samples):
                raise RuntimeError(
                    "replicated update requires one complete candidate group"
                )
            try:
                replicated_layout = validate_replicated_group_layout(
                    num_samples=int(args.num_samples),
                    learner_world_size=learner_world_size,
                    train_batch_size=int(args.train_batch_size),
                    train_batch_size_per_device=int(args.train_batch_size_per_device),
                )
            except ValueError as error:
                raise RuntimeError(
                    f"invalid replicated update layout: {error}"
                ) from error
            local_candidate_count = replicated_layout.local_candidate_count
            if int(self.strategy.grad_acc_step) != int(
                replicated_layout.micro_batches_per_rank
            ):
                raise RuntimeError(
                    "replicated update accumulation width does not match the "
                    "exact logical candidate group"
                )
        else:
            local_candidate_count = len(input_ids)
        total_micro_batches = args.num_ppo_epochs * math.ceil(
            local_candidate_count / max(args.train_batch_size_per_device, 1)
        )
        logging.info(
            "grpo prep done: logps=%s ref_logps=%s advantages=%s total_micro_batches=%s",
            tuple(logps.shape),
            None if ref_logps is None else tuple(ref_logps.shape),
            tuple(advantages.shape),
            total_micro_batches,
        )

        stats = defaultdict(list)
        local_grad_step = 0
        for ppo_epoch in range(args.num_ppo_epochs):
            if replicated_group:
                permutation_seed = replicated_group_permutation_seed(
                    experiment_seed=int(args.seed),
                    learner_step=int(getattr(self, "steps", 0)),
                    ppo_epoch=int(ppo_epoch),
                )
                permutation = np.random.RandomState(permutation_seed).permutation(
                    len(input_ids)
                )
                rank = dist.get_rank()
                shard_start = rank * local_candidate_count
                batch_inds = permutation[
                    shard_start : shard_start + local_candidate_count
                ]
            else:
                batch_inds = np.random.permutation(len(input_ids))
            for b_st in range(0, len(batch_inds), args.train_batch_size_per_device):
                local_grad_step += 1
                mini_batch_inds = batch_inds[
                    b_st : b_st + args.train_batch_size_per_device
                ]
                mb_advantage = advantages[mini_batch_inds]
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]
                mb_logps = logps[mini_batch_inds]
                mb_loss_masks = loss_masks[mini_batch_inds]
                mb_row_weights = (
                    row_weights[mini_batch_inds] if row_weights is not None else None
                )

                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                if (
                    args.beta <= 0
                    and float(getattr(args, "maxent_alpha", 0.0) or 0.0) <= 0
                    and self.args.critic_type in ["grpo", "drgrpo"]
                    and len(mb_advantage) == 1
                    and bool(torch.count_nonzero(mb_advantage).item() == 0)
                ):
                    prompt_len = int(prompt_id_lens[int(mini_batch_inds[0])])
                    mb_last_valid_token_pos = (
                        cap_last_valid_token_pos_for_zero_advantage(
                            prompt_len=prompt_len,
                            last_valid_token_pos=int(mb_last_valid_token_pos),
                            response_token_budget=int(
                                getattr(
                                    self.args, "baseline_zero_adv_response_tokens", 8
                                )
                            ),
                        )
                    )
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]
                mb_logps = mb_logps[:, : mb_last_valid_token_pos - 1]

                if self.args.critic_type == "ppo":
                    if returns is None or values is None:
                        raise ValueError(
                            "ppo baseline updates require returns and values."
                        )
                    mb_return = returns[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_values = values[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_advantage = mb_advantage[:, : mb_last_valid_token_pos - 1]

                logits = self.model(mb_input_ids, attention_mask=mb_att_mask)["logits"]
                if args.temperature != 1:
                    logits = logits / args.temperature
                logits = self._mask_invalid_scoring_logit_columns(
                    logits,
                    valid_vocab_size=policy_vocab_upper_bound,
                    context="baseline_policy_update_logits",
                )
                maxent_objective = str(getattr(args, "maxent_objective", "sequence"))
                (
                    new_logps,
                    policy_token_entropy,
                ) = self._policy_logps_and_optional_entropy(
                    logits,
                    mb_input_ids,
                    mb_response_masks,
                    # ``train/entropy`` retains one full-policy definition
                    # across control and treatment arms. E21 computes its
                    # separately labeled conditional-content objective in
                    # addition to this shared diagnostic.
                    need_entropy=True,
                )
                if args.reinforce_update:
                    pg_loss_max = -mb_advantage * new_logps
                else:
                    logprobs_diff = new_logps - mb_logps
                    ratio = torch.exp(logprobs_diff)
                    dapo_enabled = bool(getattr(args, "dapo_enabled", False))
                    clip_low = (
                        float(args.dapo_clip_low)
                        if dapo_enabled
                        else float(args.cliprange)
                    )
                    clip_high = (
                        float(args.dapo_clip_high)
                        if dapo_enabled
                        else float(args.cliprange)
                    )
                    pg_losses = -mb_advantage * ratio
                    pg_losses2 = -mb_advantage * torch.clamp(
                        ratio, 1.0 - clip_low, 1.0 + clip_high
                    )
                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    stats["logprobs_diff_max"].append(
                        torch.amax(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["logprobs_diff_min"].append(
                        torch.amin(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["zero_pg_loss_count"].append(
                        (pg_loss_max == 0).detach().sum().item()
                    )

                if bool(getattr(args, "dapo_enabled", False)):
                    if mb_row_weights is not None:
                        raise RuntimeError(
                            "DAPO token-level reduction cannot use row weights"
                        )
                    base_pg_loss, active_tokens = dapo_token_level_policy_loss(
                        pg_loss_max,
                        mb_response_masks,
                        mb_loss_masks,
                    )
                    infos["dapo_token_level_active_tokens"] = active_tokens
                    infos["dapo_clip_low"] = torch.tensor(
                        float(args.dapo_clip_low), device=base_pg_loss.device
                    )
                    infos["dapo_clip_high"] = torch.tensor(
                        float(args.dapo_clip_high), device=base_pg_loss.device
                    )
                else:
                    base_pg_loss = self.masked_aggregator(
                        pg_loss_max, mb_response_masks, axis=1
                    )
                    if mb_row_weights is None:
                        base_pg_loss = (base_pg_loss * mb_loss_masks).mean()
                    else:
                        # xDr.GRPO: per-candidate tempered aggregation weights
                        # (G * softmax(U/tau) per prompt group, detached). Only
                        # the pg-loss aggregation is reweighted.
                        base_pg_loss = (
                            base_pg_loss * mb_loss_masks * mb_row_weights
                        ).mean()
                pg_loss = base_pg_loss
                infos["pg_loss"] = pg_loss.detach()
                loss = pg_loss
                maxent_alpha = float(getattr(args, "maxent_alpha", 0.0) or 0.0)
                maxent_controller = getattr(self, "_maxent_alpha_controller", None)
                if maxent_controller is not None:
                    maxent_alpha = float(maxent_controller.current_alpha)
                maxent_length_controller = getattr(
                    self, "_maxent_length_controller", None
                )
                maxent_length_lambda = (
                    float(maxent_length_controller.current_lambda)
                    if maxent_length_controller is not None
                    else 0.0
                )
                policy_entropy_coef = float(
                    getattr(args, "policy_entropy_coef", 0.0) or 0.0
                )
                entropy_for_loss = None
                token_entropy = policy_token_entropy
                if maxent_alpha > 0.0:
                    objective_token_entropy = token_entropy
                    if maxent_objective == "conditional_token_mean":
                        if canonical_actions:
                            raise RuntimeError(
                                "conditional-token MaxEnt cannot use a canonical policy"
                            )
                        objective_token_entropy = (
                            self._chunked_conditional_content_entropy_from_logits(
                                logits
                            )
                        )
                    if objective_token_entropy is None:
                        raise RuntimeError("policy entropy was not computed")
                    if objective_token_entropy.shape != mb_response_masks.shape:
                        objective_token_entropy = objective_token_entropy[
                            ..., : mb_response_masks.shape[-1]
                        ]
                    if maxent_objective == "conditional_token_mean":
                        # E21 deliberately does not importance-differentiate
                        # the sampled state distribution. Each response gets
                        # one equal-weight mean over active positions, and EOS
                        # has already been removed from the local categorical
                        # distribution. Thus neither extra states nor the EOS
                        # decision earn a direct entropy reward.
                        conditional_entropy_by_row = (
                            mean_active_token_entropy_by_response(
                                objective_token_entropy,
                                mb_response_masks,
                            )
                        )
                        entropy_surrogate_by_row = conditional_entropy_by_row
                        sequence_entropy_by_row = conditional_entropy_by_row.detach()
                        sampled_prefix_entropy_by_row = (
                            conditional_entropy_by_row.detach()
                        )
                        prefix_ratio_mean_by_row = torch.ones_like(
                            conditional_entropy_by_row
                        )
                        prefix_ratio_max_by_row = torch.ones_like(
                            conditional_entropy_by_row
                        )
                        prefix_ratio_clipfrac_by_row = torch.zeros_like(
                            conditional_entropy_by_row
                        )
                    else:
                        (
                            entropy_surrogate_by_row,
                            sequence_entropy_by_row,
                            sampled_prefix_entropy_by_row,
                            prefix_ratio_mean_by_row,
                            prefix_ratio_max_by_row,
                            prefix_ratio_clipfrac_by_row,
                        ) = prefix_ratio_maxent_surrogate(
                            objective_token_entropy,
                            new_logps,
                            mb_logps,
                            mb_response_masks,
                            # Standard MaxEnt uses raw completion-sequence entropy.
                            # Dr.GRPO's shared outer 1/T_max is applied below.
                            normalization_constant=1.0,
                            # The three-action canonical pilot is short enough to
                            # retain the exact direct-entropy identity. Legacy
                            # free-form smokes retain their preregistered clipped
                            # stability surrogate.
                            cliprange=(
                                None if canonical_actions else float(args.cliprange)
                            ),
                        )
                    active_rows = mb_loss_masks.sum().clamp_min(1.0)
                    entropy_surrogate = (
                        entropy_surrogate_by_row * mb_loss_masks
                    ).sum() / active_rows
                    sequence_entropy = (
                        sequence_entropy_by_row * mb_loss_masks
                    ).sum() / active_rows
                    sampled_prefix_entropy = (
                        sampled_prefix_entropy_by_row * mb_loss_masks
                    ).sum() / active_rows
                    prefix_ratio_mean = (
                        prefix_ratio_mean_by_row * mb_loss_masks
                    ).sum() / active_rows
                    prefix_ratio_max = (prefix_ratio_max_by_row * mb_loss_masks).max()
                    prefix_ratio_clipfrac = (
                        prefix_ratio_clipfrac_by_row * mb_loss_masks
                    ).sum() / active_rows
                    sequence_entropy_per_tmax = sequence_entropy / float(
                        args.generate_max_length
                    )
                    sampled_prefix_entropy_per_tmax = sampled_prefix_entropy / float(
                        args.generate_max_length
                    )
                    # Dr.GRPO divides its reward-policy gradient by T_max.
                    # The active objective is standard E[R] + alpha H, so raw
                    # sequence entropy receives exactly that one shared outer
                    # scale. Do not normalize H by T_max a second time.
                    # Its self-including group-mean reward baseline attenuates
                    # the expected reward gradient by (G-1)/G; applying the
                    # same factor here preserves alpha's stated objective
                    # units without changing the maintained Dr.GRPO update.
                    reward_estimator_scale = float(args.num_samples - 1) / float(
                        args.num_samples
                    )
                    entropy_loss = standard_maxent_loss(
                        entropy_surrogate,
                        alpha=maxent_alpha,
                        reward_estimator_scale=reward_estimator_scale,
                        update_normalizer=(
                            1.0
                            if maxent_objective == "conditional_token_mean"
                            else float(args.generate_max_length)
                        ),
                    )
                    loss = loss + entropy_loss
                    if maxent_length_controller is not None:
                        if not bool((mb_loss_masks > 0).all()):
                            raise RuntimeError(
                                "MaxEnt length control cannot exclude response rows"
                            )
                        (
                            length_surrogate_by_row,
                            expected_length_by_row,
                            sampled_prefix_length_by_row,
                            length_prefix_ratio_mean_by_row,
                            length_prefix_ratio_max_by_row,
                            length_prefix_ratio_clipfrac_by_row,
                        ) = prefix_ratio_expected_length_surrogate(
                            new_logps,
                            mb_logps,
                            mb_response_masks,
                            cliprange=float(args.cliprange),
                        )
                        length_surrogate = (
                            length_surrogate_by_row * mb_loss_masks
                        ).sum() / active_rows
                        expected_length = (
                            expected_length_by_row * mb_loss_masks
                        ).sum() / active_rows
                        sampled_prefix_length = (
                            sampled_prefix_length_by_row * mb_loss_masks
                        ).sum() / active_rows
                        length_prefix_ratio_mean = (
                            length_prefix_ratio_mean_by_row * mb_loss_masks
                        ).sum() / active_rows
                        length_prefix_ratio_max = (
                            length_prefix_ratio_max_by_row * mb_loss_masks
                        ).max()
                        length_prefix_ratio_clipfrac = (
                            length_prefix_ratio_clipfrac_by_row * mb_loss_masks
                        ).sum() / active_rows
                        length_loss = standard_maxent_length_penalty_loss(
                            length_surrogate,
                            length_lambda=maxent_length_lambda,
                            reward_estimator_scale=reward_estimator_scale,
                            update_normalizer=float(args.generate_max_length),
                        )
                        loss = loss + length_loss
                        infos["maxent_length_lambda_used"] = torch.tensor(
                            maxent_length_lambda, device=loss.device
                        )
                        infos["maxent_expected_length"] = expected_length.detach()
                        infos[
                            "maxent_sampled_prefix_length"
                        ] = sampled_prefix_length.detach()
                        infos["maxent_length_surrogate"] = length_surrogate.detach()
                        infos["maxent_length_loss"] = length_loss.detach()
                        infos[
                            "maxent_length_prefix_ratio_mean"
                        ] = length_prefix_ratio_mean.detach()
                        infos[
                            "maxent_length_prefix_ratio_max"
                        ] = length_prefix_ratio_max.detach()
                        infos[
                            "maxent_length_prefix_ratio_clipfrac"
                        ] = length_prefix_ratio_clipfrac.detach()
                        stats["maxent_length_lambda_used"].append(maxent_length_lambda)
                        stats["maxent_expected_length"].append(
                            float(expected_length.detach().cpu().item())
                        )
                        stats["maxent_sampled_prefix_length"].append(
                            float(sampled_prefix_length.detach().cpu().item())
                        )
                        stats["maxent_length_surrogate"].append(
                            float(length_surrogate.detach().cpu().item())
                        )
                        stats["maxent_length_loss"].append(
                            float(length_loss.detach().cpu().item())
                        )
                        stats["maxent_length_prefix_ratio_mean"].append(
                            float(length_prefix_ratio_mean.detach().cpu().item())
                        )
                        stats["maxent_length_prefix_ratio_max"].append(
                            float(length_prefix_ratio_max.detach().cpu().item())
                        )
                        stats["maxent_length_prefix_ratio_clipfrac"].append(
                            float(length_prefix_ratio_clipfrac.detach().cpu().item())
                        )
                    infos["maxent_alpha_used"] = torch.tensor(
                        maxent_alpha, device=loss.device
                    )
                    if maxent_objective == "conditional_token_mean":
                        infos[
                            "maxent_conditional_token_entropy"
                        ] = sequence_entropy.detach()
                        infos["maxent_state_distribution_detached"] = torch.tensor(
                            1.0, device=loss.device
                        )
                        infos["maxent_eos_excluded"] = torch.tensor(
                            1.0, device=loss.device
                        )
                        infos["maxent_response_equal_weight"] = torch.tensor(
                            1.0, device=loss.device
                        )
                    else:
                        infos["maxent_sequence_entropy"] = sequence_entropy.detach()
                        infos[
                            "maxent_sequence_entropy_per_tmax"
                        ] = sequence_entropy_per_tmax.detach()
                        infos[
                            "maxent_sampled_prefix_entropy"
                        ] = sampled_prefix_entropy.detach()
                        infos[
                            "maxent_sampled_prefix_entropy_per_tmax"
                        ] = sampled_prefix_entropy_per_tmax.detach()
                        infos["maxent_prefix_ratio_mean"] = prefix_ratio_mean.detach()
                        infos["maxent_prefix_ratio_max"] = prefix_ratio_max.detach()
                        infos[
                            "maxent_prefix_ratio_clipfrac"
                        ] = prefix_ratio_clipfrac.detach()
                    infos["maxent_entropy_surrogate"] = entropy_surrogate.detach()
                    infos["maxent_entropy_loss"] = entropy_loss.detach()
                    infos["maxent_reward_estimator_scale"] = torch.tensor(
                        reward_estimator_scale, device=loss.device
                    )
                    infos["maxent_valid_row_fraction"] = (
                        (mb_loss_masks > 0).float().mean().detach()
                    )
                    if maxent_objective == "conditional_token_mean":
                        stats["maxent_conditional_token_entropy"].append(
                            float(sequence_entropy.detach().cpu().item())
                        )
                    else:
                        stats["maxent_sequence_entropy"].append(
                            float(sequence_entropy.detach().cpu().item())
                        )
                        stats["maxent_sequence_entropy_per_tmax"].append(
                            float(sequence_entropy_per_tmax.detach().cpu().item())
                        )
                        stats["maxent_sampled_prefix_entropy"].append(
                            float(sampled_prefix_entropy.detach().cpu().item())
                        )
                        stats["maxent_sampled_prefix_entropy_per_tmax"].append(
                            float(sampled_prefix_entropy_per_tmax.detach().cpu().item())
                        )
                        stats["maxent_prefix_ratio_mean"].append(
                            float(prefix_ratio_mean.detach().cpu().item())
                        )
                        stats["maxent_prefix_ratio_max"].append(
                            float(prefix_ratio_max.detach().cpu().item())
                        )
                        stats["maxent_prefix_ratio_clipfrac"].append(
                            float(prefix_ratio_clipfrac.detach().cpu().item())
                        )
                    stats["maxent_entropy_surrogate"].append(
                        float(entropy_surrogate.detach().cpu().item())
                    )
                    stats["maxent_entropy_loss"].append(
                        float(entropy_loss.detach().cpu().item())
                    )
                elif policy_entropy_coef != 0.0:
                    if token_entropy is None:
                        raise RuntimeError("policy entropy was not computed")
                    if token_entropy.shape != mb_response_masks.shape:
                        token_entropy = token_entropy[
                            ..., : mb_response_masks.shape[-1]
                        ]
                    entropy_by_row = self.masked_aggregator(
                        token_entropy, mb_response_masks, axis=1
                    )
                    entropy_for_loss = (entropy_by_row * mb_loss_masks).mean()
                    entropy_loss = -policy_entropy_coef * entropy_for_loss
                    infos["policy_entropy_coef"] = torch.tensor(
                        policy_entropy_coef,
                        device=loss.device,
                    )
                    infos["policy_entropy_loss"] = entropy_loss.detach()
                    loss = loss + entropy_loss
                if args.beta > 0:
                    if ref_logps is None:
                        raise ValueError(
                            "beta > 0 baseline updates require reference log-probs."
                        )
                    mb_ref_logps = ref_logps[mini_batch_inds]
                    mb_ref_logps = mb_ref_logps[:, : mb_last_valid_token_pos - 1]
                    log_ratio = (mb_ref_logps - new_logps).clamp(-40.0, 40.0)
                    kl3 = torch.expm1(log_ratio) - log_ratio
                    infos["kl3"] = (kl3 * mb_response_masks).detach().sum(1).mean()

                    reg_loss = self.masked_aggregator(kl3, mb_response_masks, axis=1)
                    reg_loss = args.beta * (reg_loss * mb_loss_masks).mean()
                    infos["reg_loss"] = reg_loss.detach()
                    loss += reg_loss

                with torch.no_grad():
                    # Always log token-level masked-mean entropy so the
                    # train/entropy key has identical semantics across arms
                    # regardless of whether the entropy bonus is active.
                    if token_entropy is None:
                        raise RuntimeError("policy entropy was not computed")
                    token_entropy_for_log = token_entropy.detach()
                    if token_entropy_for_log.shape != mb_response_masks.shape:
                        token_entropy_for_log = token_entropy_for_log[
                            ..., : mb_response_masks.shape[-1]
                        ]
                    entropy = masked_mean(token_entropy_for_log, mb_response_masks)
                    infos["entropy"] = entropy
                    if canonical_actions:
                        canonical_sequence_entropy_by_row = (
                            token_entropy_for_log * mb_response_masks
                        ).sum(dim=1)
                        canonical_sequence_entropy = (
                            canonical_sequence_entropy_by_row * mb_loss_masks
                        ).sum() / mb_loss_masks.sum().clamp_min(1.0)
                        if self._canonical_action_space is None:
                            raise RuntimeError("canonical action space is missing")
                        canonical_max_entropy = float(
                            self._canonical_action_space.max_sequence_entropy
                        )
                        canonical_sequence_entropy_value = float(
                            canonical_sequence_entropy.detach().cpu().item()
                        )
                        if (
                            not math.isfinite(canonical_sequence_entropy_value)
                            or canonical_sequence_entropy_value < -1e-6
                            or canonical_sequence_entropy_value
                            > canonical_max_entropy + 1e-5
                        ):
                            raise RuntimeError(
                                "canonical sampled-prefix entropy left its "
                                f"support bound: {canonical_sequence_entropy_value}"
                            )
                        infos["canonical_token_entropy_mean"] = entropy
                        # This is a behavior-prefix diagnostic. After a
                        # learner update it is not H(q_new) unless prefix
                        # ratios are applied; reserve that name for the IS
                        # estimator and exact 27-leaf endpoint audit.
                        infos[
                            "canonical_sampled_prefix_entropy_sum"
                        ] = canonical_sequence_entropy
                        infos["canonical_sampled_prefix_entropy_ratio"] = (
                            canonical_sequence_entropy / canonical_max_entropy
                        )
                        infos["canonical_action_count"] = torch.tensor(
                            int(self._canonical_action_space.horizon),
                            device=loss.device,
                        )
                        infos["canonical_action_vocab_size"] = torch.tensor(
                            len(self._canonical_action_token_ids or ()),
                            device=loss.device,
                        )

                self.strategy.backward(loss, self.model, self.optimizer)

                if (
                    canonical_replay_groups
                    and local_grad_step % self.strategy.grad_acc_step == 0
                ):
                    replay = self._materialize_canonical_replay(
                        canonical_replay_groups,
                        device=input_ids.device,
                    )
                    replay_row_count = int(replay.input_ids.size(0))
                    replay_chunk_size = max(
                        1,
                        int(args.train_batch_size_per_device),
                    )
                    with _temporary_eval_mode(self.model, enabled=True):
                        with torch.no_grad():
                            detached_scores = torch.cat(
                                [
                                    self._score_canonical_replay_rows(
                                        replay,
                                        start=start,
                                        stop=min(
                                            start + replay_chunk_size,
                                            replay_row_count,
                                        ),
                                        policy_vocab_upper_bound=(
                                            policy_vocab_upper_bound
                                        ),
                                    ).detach()
                                    for start in range(
                                        0,
                                        replay_row_count,
                                        replay_chunk_size,
                                    )
                                ]
                            )
                        retention_bank = getattr(
                            self,
                            "_online_canonical_bank",
                            None,
                        )
                        if (
                            isinstance(retention_bank, OnlineCanonicalBank)
                            and retention_bank.proposal_retention_tracking_enabled
                        ):
                            retention_token_counts = replay.response_masks.sum(
                                dim=1
                            ).to(detached_scores.dtype)
                            retention_score_tensor = torch.stack(
                                (
                                    detached_scores,
                                    detached_scores * retention_token_counts,
                                ),
                                dim=1,
                            ).detach()
                            if (
                                dist.is_available()
                                and dist.is_initialized()
                                and dist.get_world_size() > 1
                            ):
                                dist.broadcast(retention_score_tensor, src=0)
                            retention_score_rows = (
                                retention_score_tensor.float().cpu().tolist()
                            )
                            retention_diagnostics = (
                                retention_bank.observe_replay_retention_scores(
                                    groups=canonical_replay_groups,
                                    mean_logprobs=[
                                        row[0] for row in retention_score_rows
                                    ],
                                    sequence_logprobs=[
                                        row[1] for row in retention_score_rows
                                    ],
                                )
                            )
                            infos.update(
                                {
                                    "canonical_replay_proposal_retention_"
                                    f"{name}": torch.tensor(
                                        value,
                                        dtype=torch.float32,
                                        device=input_ids.device,
                                    )
                                    for name, value in (retention_diagnostics.items())
                                }
                            )
                        replay_objective = str(args.online_canonical_replay_objective)
                        replay_alpha = float(args.online_canonical_replay_alpha)
                        # Uniform replay splits the dose across banked modes, so
                        # each mode receives alpha/n and a prompt that has found
                        # more modes protects each one less. Scaling the dose by
                        # n holds per-mode pressure at the registered constant.
                        # n is already bounded by the replay capacity, so this
                        # introduces no ceiling that is not already registered.
                        replay_banked_modes = int(sum(replay.group_sizes))
                        replay_key_weighting = str(
                            getattr(
                                args,
                                "online_canonical_replay_key_weighting",
                                "uniform",
                            )
                        )
                        if replay_key_weighting == "uniform":
                            # Alias the historical tensor exactly: the default
                            # path performs no new arithmetic.
                            replay_target_weights = replay.mass_weights
                        else:
                            if not bool(
                                replay.mass_weights.eq(1.0).all()
                            ):
                                raise RuntimeError(
                                    "fresh-frequency key weighting cannot be "
                                    "combined with replay priority weights"
                                )
                            replay_target_weights = (
                                canonical_replay_key_target_weights(
                                    replay.fresh_observation_counts,
                                    replay.group_sizes,
                                    weighting=replay_key_weighting,
                                )
                            )
                        replay_bank_normalized = bool(
                            getattr(
                                args,
                                "online_canonical_replay_bank_normalized",
                                False,
                            )
                        )
                        if replay_bank_normalized:
                            replay_alpha = (
                                float(args.online_canonical_replay_per_mode_coefficient)
                                * replay_banked_modes
                            )
                        replay_mass_alpha = 0.0
                        split_result = None
                        if replay_objective == "bank_balance":
                            replay_result = canonical_replay_uniform_loss(
                                detached_scores,
                                replay.group_sizes,
                            )
                        elif replay_objective == "verified_likelihood":
                            replay_result = (
                                canonical_replay_uniform_verified_likelihood_loss(
                                    detached_scores,
                                    replay.group_sizes,
                                    replay_target_weights,
                                )
                            )
                        elif replay_objective == ("verified_likelihood_per_rollout"):
                            replay_result = (
                                canonical_replay_uniform_verified_likelihood_loss(
                                    detached_scores,
                                    replay.group_sizes,
                                    replay_target_weights,
                                )
                            )
                        elif replay_objective == ("split_mass_balance_per_rollout"):
                            split_result = canonical_replay_split_mass_balance_loss(
                                detached_scores,
                                replay.group_sizes,
                                replay_target_weights,
                            )
                            replay_mass_alpha = float(
                                args.online_canonical_replay_mass_alpha
                            )
                            replay_result = None
                        else:
                            raise RuntimeError(
                                "unsupported canonical replay objective: "
                                f"{replay_objective}"
                            )
                        actuator_loss = (
                            split_result.mass_loss
                            if split_result is not None
                            else replay_result.loss
                        )
                        balance_loss = (
                            split_result.balance_loss
                            if split_result is not None
                            else replay_result.cross_entropy_excess
                        )
                        normalized_entropy = (
                            split_result.normalized_entropy
                            if split_result is not None
                            else replay_result.normalized_entropy
                        )
                        if not all(
                            bool(torch.isfinite(value))
                            for value in (
                                actuator_loss,
                                balance_loss,
                                normalized_entropy,
                            )
                        ):
                            raise RuntimeError(
                                "canonical replay produced a non-finite loss or sensor"
                            )
                        reward_estimator_scale = float(args.num_samples - 1) / float(
                            args.num_samples
                        )
                        replay_objective_scale = (
                            1.0 / float(args.num_samples)
                            if replay_objective
                            in {
                                "verified_likelihood_per_rollout",
                                "split_mass_balance_per_rollout",
                            }
                            else 1.0
                        )
                        weighted_replay_objective = (
                            (
                                actuator_loss * replay_mass_alpha
                                + balance_loss * replay_alpha
                            )
                            if split_result is not None
                            else actuator_loss * replay_alpha
                        )
                        replay_weighted_loss = (
                            weighted_replay_objective
                            * reward_estimator_scale
                            * replay_objective_scale
                        )
                        if not bool(torch.isfinite(replay_weighted_loss)):
                            raise RuntimeError(
                                "fixed canonical replay coefficient produced "
                                "a non-finite weighted loss"
                            )
                        requested_score_gradients = (
                            (
                                (
                                    split_result.mass_score_gradients
                                    * replay_mass_alpha
                                    + split_result.balance_score_gradients
                                    * replay_alpha
                                )
                                if split_result is not None
                                else replay_result.score_gradients * replay_alpha
                            )
                            .to(input_ids.device)
                            .detach()
                        )
                        retention_safe_balance = bool(
                            getattr(
                                args,
                                "online_canonical_replay_retention_safe_balance",
                                False,
                            )
                        )
                        balance_scales = detached_scores.new_ones(
                            len(replay.group_sizes)
                        )
                        raw_score_gradients = requested_score_gradients
                        if retention_safe_balance:
                            if split_result is None:
                                raise RuntimeError(
                                    "retention-safe balance requires a split replay loss"
                                )
                            (
                                raw_score_gradients,
                                balance_scales,
                            ) = cap_retention_safe_balance_score_gradients(
                                (
                                    split_result.mass_score_gradients
                                    * replay_mass_alpha
                                ).to(input_ids.device),
                                (
                                    split_result.balance_score_gradients * replay_alpha
                                ).to(input_ids.device),
                                replay.group_sizes,
                            )
                        if retention_safe_balance and bool(
                            (raw_score_gradients > 1e-7).any()
                        ):
                            raise RuntimeError(
                                "retention-safe replay assigned downward verified-mode "
                                "pressure"
                            )
                        replay_compute_only = bool(
                            getattr(
                                args,
                                "online_canonical_replay_compute_only",
                                False,
                            )
                        )
                        score_gradients = (
                            torch.zeros_like(raw_score_gradients)
                            if replay_compute_only
                            else raw_score_gradients
                        )
                        replay_applied_weighted_loss = (
                            replay_weighted_loss.new_zeros(())
                            if replay_compute_only
                            else replay_weighted_loss
                        )
                        replay_backward_scale = (
                            reward_estimator_scale
                            * replay_objective_scale
                            * float(self.strategy.grad_acc_step)
                        )
                        for start in range(
                            0,
                            replay_row_count,
                            replay_chunk_size,
                        ):
                            stop = min(
                                start + replay_chunk_size,
                                replay_row_count,
                            )
                            live_scores = self._score_canonical_replay_rows(
                                replay,
                                start=start,
                                stop=stop,
                                policy_vocab_upper_bound=(policy_vocab_upper_bound),
                            )
                            exact_gradient_surrogate = (
                                live_scores * score_gradients[start:stop]
                            ).sum()
                            replay_backward_loss = (
                                exact_gradient_surrogate * replay_backward_scale
                            )
                            if not bool(torch.isfinite(replay_backward_loss)):
                                raise RuntimeError(
                                    "canonical replay produced a non-finite "
                                    "chunked backward scalar"
                                )
                            self.strategy.backward(
                                replay_backward_loss,
                                self.model,
                                self.optimizer,
                            )
                    target_entropy_ratios: list[torch.Tensor] = []
                    target_ginis: list[torch.Tensor] = []
                    target_rare_allocations: list[torch.Tensor] = []
                    target_common_allocations: list[torch.Tensor] = []
                    target_start = 0
                    for target_size in replay.group_sizes:
                        target_stop = target_start + target_size
                        target_slice = replay_target_weights[
                            target_start:target_stop
                        ].to(torch.float64)
                        target_probabilities = (
                            target_slice / target_slice.sum()
                        )
                        target_counts = replay.fresh_observation_counts[
                            target_start:target_stop
                        ]
                        if target_size >= 2:
                            target_entropy_ratios.append(
                                -(
                                    target_probabilities
                                    * target_probabilities.log()
                                ).sum()
                                / math.log(target_size)
                            )
                        else:
                            target_entropy_ratios.append(
                                target_slice.new_tensor(1.0)
                            )
                        target_ginis.append(
                            torch.abs(
                                target_probabilities[:, None]
                                - target_probabilities[None, :]
                            ).sum()
                            / (2.0 * target_size)
                        )
                        target_rare_allocations.append(
                            target_probabilities[
                                target_counts == target_counts.min()
                            ].sum()
                        )
                        target_common_allocations.append(
                            target_probabilities[
                                target_counts == target_counts.max()
                            ].sum()
                        )
                        target_start = target_stop
                    target_normalized_entropy = torch.stack(
                        target_entropy_ratios
                    ).mean()
                    target_gini = torch.stack(target_ginis).mean()
                    target_rare_allocation = torch.stack(
                        target_rare_allocations
                    ).mean()
                    target_common_allocation = torch.stack(
                        target_common_allocations
                    ).mean()
                    # DeepSpeed divides every backward call by the configured
                    # accumulation width. Replay is evaluated once at the
                    # boundary (rather than repeating a full exemplar bank for
                    # every one-row policy microbatch), so cancel exactly that
                    # mechanical division. The detached q-minus-uniform weights
                    # above are the exact reverse-KL score derivative at this
                    # unchanged parameter snapshot; chunking changes peak
                    # memory, not the objective gradient.
                    infos.update(
                        {
                            "canonical_replay_actuator_loss": (actuator_loss.detach()),
                            "canonical_replay_balance_loss": (balance_loss.detach()),
                            "canonical_replay_weighted_loss": (
                                replay_applied_weighted_loss.detach()
                            ),
                            "canonical_replay_raw_weighted_loss": (
                                replay_weighted_loss.detach()
                            ),
                            "canonical_replay_compute_only": torch.tensor(
                                float(replay_compute_only),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_backward_scale": torch.tensor(
                                float(self.strategy.grad_acc_step),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_chunk_size": torch.tensor(
                                replay_chunk_size,
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_score_passes": torch.tensor(
                                2.0,
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_normalized_model_entropy": (
                                normalized_entropy.detach()
                            ),
                            "canonical_replay_cross_entropy_excess": (
                                balance_loss.detach()
                            ),
                            "canonical_replay_alpha_used": torch.tensor(
                                replay_alpha,
                                dtype=torch.float64,
                                device=input_ids.device,
                            ),
                            # The quantity the bank-normalized arm holds fixed.
                            # Logged for both arms so the fixed arm's 16x spread
                            # and the adaptive arm's constant are read off the
                            # same field rather than reconstructed.
                            "canonical_replay_per_mode_pressure": torch.tensor(
                                (
                                    replay_alpha / replay_banked_modes
                                    if replay_banked_modes > 0
                                    else 0.0
                                ),
                                dtype=torch.float64,
                                device=input_ids.device,
                            ),
                            "canonical_replay_bank_normalized": torch.tensor(
                                1.0 if replay_bank_normalized else 0.0,
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_eligible_groups": torch.tensor(
                                (
                                    split_result.balance_eligible_groups
                                    if split_result is not None
                                    else replay_result.eligible_groups
                                ),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_retained_modes": torch.tensor(
                                (
                                    split_result.balance_retained_modes
                                    if split_result is not None
                                    else replay_result.retained_modes
                                ),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_actuator_groups": torch.tensor(
                                (
                                    split_result.actuator_groups
                                    if split_result is not None
                                    else replay_result.actuator_groups
                                ),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_actuator_modes": torch.tensor(
                                (
                                    split_result.actuator_modes
                                    if split_result is not None
                                    else replay_result.actuator_modes
                                ),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_reward_estimator_scale": (
                                torch.tensor(
                                    reward_estimator_scale,
                                    dtype=torch.float32,
                                    device=input_ids.device,
                                )
                            ),
                            "canonical_replay_score_gradient_sum": (
                                raw_score_gradients.sum()
                            ),
                            "canonical_replay_objective_scale": torch.tensor(
                                replay_objective_scale,
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_applied_score_gradient_sum": (
                                score_gradients.sum() * replay_objective_scale
                            ),
                            "canonical_replay_verified_likelihood_active": (
                                torch.tensor(
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
                                )
                            ),
                            "canonical_replay_mass_alpha_used": torch.tensor(
                                replay_mass_alpha,
                                dtype=torch.float64,
                                device=input_ids.device,
                            ),
                            "canonical_replay_balance_alpha_used": torch.tensor(
                                replay_alpha,
                                dtype=torch.float64,
                                device=input_ids.device,
                            ),
                            "canonical_replay_mass_score_gradient_sum": (
                                (
                                    split_result.mass_score_gradients.sum()
                                    if split_result is not None
                                    else detached_scores.new_tensor(0.0)
                                ).to(input_ids.device)
                            ),
                            "canonical_replay_balance_score_gradient_sum": (
                                (
                                    split_result.balance_score_gradients.sum()
                                    if split_result is not None
                                    else detached_scores.new_tensor(0.0)
                                ).to(input_ids.device)
                            ),
                            "canonical_replay_mass_score_gradient_l2": (
                                (
                                    torch.linalg.vector_norm(
                                        split_result.mass_score_gradients
                                    )
                                    if split_result is not None
                                    else detached_scores.new_tensor(0.0)
                                ).to(input_ids.device)
                            ),
                            "canonical_replay_balance_score_gradient_l2": (
                                (
                                    torch.linalg.vector_norm(
                                        split_result.balance_score_gradients
                                    )
                                    if split_result is not None
                                    else detached_scores.new_tensor(0.0)
                                ).to(input_ids.device)
                            ),
                            "canonical_replay_applied_score_gradient_l2": (
                                torch.linalg.vector_norm(score_gradients)
                            ),
                            "canonical_replay_retention_safe_balance": torch.tensor(
                                float(retention_safe_balance),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_balance_scale_min": (
                                balance_scales.min().to(input_ids.device)
                            ),
                            "canonical_replay_balance_scale_mean": (
                                balance_scales.mean().to(input_ids.device)
                            ),
                            "canonical_replay_balance_capped_group_fraction": (
                                (balance_scales < 1.0)
                                .float()
                                .mean()
                                .to(input_ids.device)
                            ),
                            "canonical_replay_requested_positive_gradient_max": (
                                torch.clamp(
                                    requested_score_gradients,
                                    min=0.0,
                                ).max()
                            ),
                            "canonical_replay_applied_positive_gradient_max": (
                                torch.clamp(
                                    raw_score_gradients,
                                    min=0.0,
                                ).max()
                            ),
                            "canonical_replay_priority_modes": torch.tensor(
                                replay.priority_modes,
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_mass_weight_min": (
                                replay.mass_weights.min().to(input_ids.device)
                            ),
                            "canonical_replay_mass_weight_max": (
                                replay.mass_weights.max().to(input_ids.device)
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
                            "canonical_replay_target_normalized_entropy": (
                                target_normalized_entropy.to(input_ids.device)
                            ),
                            "canonical_replay_target_gini": target_gini.to(
                                input_ids.device
                            ),
                            "canonical_replay_rare_key_gradient_allocation": (
                                target_rare_allocation.to(input_ids.device)
                            ),
                            "canonical_replay_common_key_gradient_allocation": (
                                target_common_allocation.to(input_ids.device)
                            ),
                            "canonical_replay_target_weight_sum": (
                                replay_target_weights.sum().to(input_ids.device)
                            ),
                            "canonical_replay_target_weight_min": (
                                replay_target_weights.min().to(input_ids.device)
                            ),
                            "canonical_replay_target_weight_max": (
                                replay_target_weights.max().to(input_ids.device)
                            ),
                            "canonical_replay_fresh_observation_count_sum": (
                                replay.fresh_observation_counts.sum().to(
                                    input_ids.device
                                )
                            ),
                            "canonical_replay_fresh_observation_count_min": (
                                replay.fresh_observation_counts.min().to(
                                    input_ids.device
                                )
                            ),
                            "canonical_replay_fresh_observation_count_max": (
                                replay.fresh_observation_counts.max().to(
                                    input_ids.device
                                )
                            ),
                            "canonical_replay_frequency_count_fresh_only": torch.tensor(
                                1.0,
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_frequency_count_from_replay": torch.tensor(
                                0.0,
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_frequency_count_from_proposals": torch.tensor(
                                0.0,
                                dtype=torch.float32,
                                device=input_ids.device,
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
                    replay_token_counts = replay.response_masks.sum(dim=1).to(
                        detached_scores.dtype
                    )
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
                        infos[
                            f"canonical_replay_target_weight_row_{row_index:02d}"
                        ] = target_weight.to(input_ids.device)
                        infos[
                            f"canonical_replay_fresh_count_row_{row_index:02d}"
                        ] = fresh_count.to(input_ids.device)
                        infos[
                            f"canonical_replay_outcome_fingerprint_row_{row_index:02d}"
                        ] = torch.tensor(
                            float(
                                int(
                                    hashlib.sha256(
                                        outcome_key.encode("utf-8")
                                    ).hexdigest()[:13],
                                    16,
                                )
                            ),
                            dtype=torch.float64,
                            device=input_ids.device,
                        )
                        infos[
                            f"canonical_replay_prompt_fingerprint_row_{row_index:02d}"
                        ] = torch.tensor(
                            float(prompt_fingerprint),
                            dtype=torch.float64,
                            device=input_ids.device,
                        )
                        infos[
                            f"canonical_replay_exemplar_mean_logprob_row_{row_index:02d}"
                        ] = mean_logprob.to(input_ids.device)
                        infos[
                            f"canonical_replay_exemplar_sequence_logprob_row_{row_index:02d}"
                        ] = (mean_logprob * token_count).to(input_ids.device)
                    for group_index, replay_group in enumerate(
                        canonical_replay_groups
                    ):
                        prompt_fingerprint = int(
                            hashlib.sha256(
                                repr(replay_group.prompt_token_ids).encode("utf-8")
                            ).hexdigest()[:13],
                            16,
                        )
                        membership_fingerprint = int(
                            hashlib.sha256(
                                repr(replay_group.outcome_keys).encode("utf-8")
                            ).hexdigest()[:13],
                            16,
                        )
                        infos[
                            f"canonical_replay_prompt_fingerprint_group_{group_index:02d}"
                        ] = torch.tensor(
                            float(prompt_fingerprint),
                            dtype=torch.float64,
                            device=input_ids.device,
                        )
                        infos[
                            f"canonical_replay_membership_fingerprint_group_{group_index:02d}"
                        ] = torch.tensor(
                            float(membership_fingerprint),
                            dtype=torch.float64,
                            device=input_ids.device,
                        )
                    for key, value in (
                        (
                            "canonical_replay_actuator_loss",
                            actuator_loss,
                        ),
                        (
                            "canonical_replay_balance_loss",
                            balance_loss,
                        ),
                        (
                            "canonical_replay_weighted_loss",
                            replay_applied_weighted_loss,
                        ),
                        (
                            "canonical_replay_normalized_model_entropy",
                            normalized_entropy,
                        ),
                        (
                            "canonical_replay_cross_entropy_excess",
                            balance_loss,
                        ),
                    ):
                        stats[key].append(float(value.detach().cpu().item()))
                    stats["canonical_replay_alpha_used"].append(replay_alpha)
                    stats["canonical_replay_mass_alpha_used"].append(replay_mass_alpha)
                    stats["canonical_replay_banked_modes"].append(
                        float(replay_banked_modes)
                    )
                    stats["canonical_replay_per_mode_pressure"].append(
                        replay_alpha / replay_banked_modes
                        if replay_banked_modes > 0
                        else 0.0
                    )
                    stats["canonical_replay_eligible_groups"].append(
                        float(
                            split_result.balance_eligible_groups
                            if split_result is not None
                            else replay_result.eligible_groups
                        )
                    )
                    stats["canonical_replay_retained_modes"].append(
                        float(
                            split_result.balance_retained_modes
                            if split_result is not None
                            else replay_result.retained_modes
                        )
                    )

                if (
                    rlep_replay_groups
                    and local_grad_step % self.strategy.grad_acc_step == 0
                ):
                    if rlep_replay_advantage is None:
                        raise RuntimeError("RLEP replay rows lack a mixed advantage")
                    replay = self._materialize_canonical_replay(
                        rlep_replay_groups,
                        device=input_ids.device,
                    )
                    replay_row_count = int(replay.input_ids.size(0))
                    expected_rows = int(getattr(args, "rlep_replay_count", 0) or 0)
                    if replay_row_count != expected_rows:
                        raise RuntimeError(
                            "RLEP materialized a replay dose different from its "
                            "registered count"
                        )
                    replay_chunk_size = max(
                        1,
                        int(args.train_batch_size_per_device),
                    )
                    mixed_count = int(args.num_samples) + replay_row_count
                    score_coefficient = -float(rlep_replay_advantage) / float(
                        mixed_count
                    )
                    replay_losses: list[torch.Tensor] = []
                    for start in range(0, replay_row_count, replay_chunk_size):
                        stop = min(start + replay_chunk_size, replay_row_count)
                        live_scores = self._score_rlep_rows(
                            replay,
                            start=start,
                            stop=stop,
                            policy_vocab_upper_bound=policy_vocab_upper_bound,
                        )
                        replay_backward_loss = (
                            live_scores.sum()
                            * score_coefficient
                            * float(self.strategy.grad_acc_step)
                        )
                        if not bool(torch.isfinite(replay_backward_loss)):
                            raise RuntimeError(
                                "RLEP produced a non-finite replay backward scalar"
                            )
                        self.strategy.backward(
                            replay_backward_loss,
                            self.model,
                            self.optimizer,
                        )
                        replay_losses.append(
                            (live_scores.detach().sum() * score_coefficient)
                        )
                    infos.update(
                        {
                            "rlep_replay_rows": torch.tensor(
                                replay_row_count, device=input_ids.device
                            ),
                            "rlep_replay_advantage": torch.tensor(
                                rlep_replay_advantage, device=input_ids.device
                            ),
                            "rlep_replay_loss": torch.stack(replay_losses).sum(),
                            "rlep_replay_score_coefficient": torch.tensor(
                                score_coefficient, device=input_ids.device
                            ),
                            "rlep_backward_scale": torch.tensor(
                                float(self.strategy.grad_acc_step),
                                device=input_ids.device,
                            ),
                        }
                    )

                if local_grad_step % self.strategy.grad_acc_step == 0:
                    if self._should_skip_baseline_grad_norm_logging():
                        if not self._baseline_grad_norm_logging_disabled_warned:
                            logging.warning(
                                "Skipping baseline policy_grad_norm logging for the "
                                "ZeRO-3/offload slow path on node-local 7B runs."
                            )
                            self._baseline_grad_norm_logging_disabled_warned = True
                        stats["policy_grad_norm"].append(0.0)
                        stats["get_grad_norm_time"].append(0.0)
                    else:
                        _st = time.time()
                        stats["policy_grad_norm"].append(
                            self.strategy.get_gradient_norm(self.model)
                        )
                        stats["get_grad_norm_time"].append(time.time() - _st)

                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                if self.args.critic_type == "ppo":
                    value_pred = self.critic(
                        input_ids=mb_input_ids, attention_mask=mb_att_mask
                    )[:, :-1]

                    value_pred_clipped = torch.clamp(
                        value_pred,
                        mb_values - args.cliprange_value,
                        mb_values + args.cliprange_value,
                    )
                    vf_losses1 = torch.square(value_pred - mb_return)
                    vf_losses2 = torch.square(value_pred_clipped - mb_return)
                    vf_loss_max = torch.max(vf_losses1, vf_losses2)

                    vf_loss = 0.5 * self.masked_aggregator(
                        vf_loss_max, mb_response_masks, axis=1
                    )
                    critic_loss = args.vf_coef * (vf_loss * mb_loss_masks).mean()

                    self.strategy.backward(
                        critic_loss, self.critic, self.critic_optimizer
                    )
                    self.strategy.optimizer_step(
                        self.critic_optimizer, self.critic, self.critic_scheduler
                    )
                    infos["critic_loss"] = critic_loss.detach()
                    infos["vf_clipfrac"] = masked_mean(
                        (vf_losses2 > vf_losses1).float(), mb_response_masks
                    ).detach()

                if self._baseline_should_log_progress(
                    local_grad_step, total_micro_batches
                ):
                    logging.info(
                        "grpo progress: microbatch=%s/%s seq_len=%s pg_loss=%.6f loss_mask_mean=%.3f",
                        local_grad_step,
                        total_micro_batches,
                        int(mb_input_ids.shape[1]),
                        float(pg_loss.detach().cpu().item()),
                        float(mb_loss_masks.float().mean().item()),
                    )

                with torch.no_grad():
                    if not args.reinforce_update:
                        pg_clipfrac = masked_mean(
                            (pg_losses2 > pg_losses).float(), mb_response_masks, axis=1
                        )
                        stats["pg_clipfrac"].append(pg_clipfrac.mean().min().item())

        infos.update(
            {f"{k}_nan": torch.tensor(stats[k]).isnan().sum() for k in stats.keys()}
        )
        infos.update(
            {f"{k}_inf": torch.tensor(stats[k]).isinf().sum() for k in stats.keys()}
        )
        infos["policy_grad_norm"] = torch.tensor(
            stats["policy_grad_norm"] or [0.0]
        ).max()
        infos["get_grad_norm_time"] = torch.tensor(
            sum(stats["get_grad_norm_time"] or [0.0])
        )
        for key in (
            "maxent_conditional_token_entropy",
            "maxent_sequence_entropy",
            "maxent_sequence_entropy_per_tmax",
            "maxent_entropy_surrogate",
            "maxent_sampled_prefix_entropy",
            "maxent_sampled_prefix_entropy_per_tmax",
            "maxent_prefix_ratio_mean",
            "maxent_prefix_ratio_clipfrac",
            "maxent_entropy_loss",
            "maxent_length_lambda_used",
            "maxent_expected_length",
            "maxent_sampled_prefix_length",
            "maxent_length_surrogate",
            "maxent_length_loss",
            "maxent_length_prefix_ratio_mean",
            "maxent_length_prefix_ratio_clipfrac",
            "canonical_replay_actuator_loss",
            "canonical_replay_balance_loss",
            "canonical_replay_weighted_loss",
            "canonical_replay_normalized_model_entropy",
            "canonical_replay_cross_entropy_excess",
            "canonical_replay_alpha_used",
            "canonical_replay_banked_modes",
            "canonical_replay_per_mode_pressure",
            "canonical_replay_eligible_groups",
            "canonical_replay_retained_modes",
        ):
            if stats[key]:
                infos[key] = torch.tensor(stats[key]).mean()
        if stats["maxent_prefix_ratio_max"]:
            infos["maxent_prefix_ratio_max"] = torch.tensor(
                stats["maxent_prefix_ratio_max"]
            ).max()
        if stats["maxent_length_prefix_ratio_max"]:
            infos["maxent_length_prefix_ratio_max"] = torch.tensor(
                stats["maxent_length_prefix_ratio_max"]
            ).max()
        if not args.reinforce_update:
            infos["logprobs_diff_max"] = torch.tensor(stats["logprobs_diff_max"]).max()
            infos["logprobs_diff_min"] = torch.tensor(stats["logprobs_diff_min"]).min()
            infos["zero_pg_loss_count"] = (
                torch.tensor(stats["zero_pg_loss_count"]).float().mean()
            )
            infos["pg_clipfrac"] = torch.tensor(stats["pg_clipfrac"]).mean()
        infos["adv_mean"] = advantages.mean().cpu()
        infos["adv_min"] = advantages.min().cpu()
        infos["adv_max"] = advantages.max().cpu()
        infos["all_zero_rewards_count"] = (
            (final_rewards.view(-1, self.args.num_samples).mean(-1) == 0).sum().cpu()
        )
        infos["all_one_rewards_count"] = (
            (final_rewards.view(-1, self.args.num_samples).mean(-1) == 1).sum().cpu()
        )
        return infos

    def _grpo_learning_step_with_progress(self, trajectory):
        # Check transported diagnostics before bank mutation, scheduling or loss.
        diagnostics = trajectory.get("verifier_diagnostics")
        if diagnostics is not None:
            if len(diagnostics) != len(trajectory["input_ids"]):
                raise failure("worker_failure", "incomplete trajectory diagnostics")
            for diagnostic in diagnostics:
                if diagnostic is not None:
                    require_scorable(diagnostic)
        args = self.args
        canonical_task = resolve_canonical_action_task(args)
        canonical_actions = canonical_task != "none"
        device = torch.cuda.current_device()
        input_ids = trajectory["input_ids"].to(device)
        att_mask = trajectory["attention_mask"].to(device)
        final_rewards = (
            torch.tensor([r[-1] for r in trajectory["rewards"]])
            .to(device)
            .reshape(-1, 1)
        ).float() * args.reward_scale
        task_final_rewards = final_rewards.detach().clone()
        raw_task_final_rewards = task_final_rewards.detach().clone()
        prompt_id_lens = trajectory["prompt_ids_lens"]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]
        dapo_infos: dict[str, torch.Tensor] = {}
        if bool(getattr(args, "dapo_enabled", False)):
            response_lengths = response_masks.sum(dim=1)
            penalties, diagnostics = dapo_soft_overlong_penalty(
                response_lengths,
                max_length=int(args.generate_max_length),
                buffer_ratio=float(args.dapo_overlong_buffer_ratio),
                penalty_factor=float(args.dapo_overlong_penalty_factor),
            )
            final_rewards = final_rewards + penalties.reshape(-1, 1).to(
                final_rewards.device
            )
            dapo_infos = {
                "dapo_enabled": torch.tensor(1.0, device=device),
                "dapo_overlong_buffer_ratio": torch.tensor(
                    float(args.dapo_overlong_buffer_ratio), device=device
                ),
                "dapo_overlong_penalty_factor": torch.tensor(
                    float(args.dapo_overlong_penalty_factor), device=device
                ),
                "dapo_overlong_shaped_rows": torch.tensor(
                    diagnostics.shaped_rows, device=device
                ),
                "dapo_overlong_truncated_rows": torch.tensor(
                    diagnostics.truncated_rows, device=device
                ),
                "dapo_overlong_penalty_min": torch.tensor(
                    diagnostics.penalty_min, device=device
                ),
                "dapo_overlong_penalty_mean": torch.tensor(
                    diagnostics.penalty_mean, device=device
                ),
            }
        diayn_infos: dict[str, torch.Tensor] = {}
        outcome_collision_infos: dict[str, torch.Tensor] = {}
        outcome_collision_outside_advantage: torch.Tensor | None = None
        semantic_shannon_infos: dict[str, torch.Tensor] = {}
        semantic_shannon_separate_advantage: torch.Tensor | None = None
        online_canonical_infos: dict[str, torch.Tensor] = {}
        online_canonical_advantage: torch.Tensor | None = None
        canonical_replay_groups: list[VerifiedCanonicalReplayGroup] = []
        rlep_replay_groups: list[RLEPReplayGroup] = []
        rlep_replay_advantage: float | None = None
        rlep_infos: dict[str, torch.Tensor] = {}
        rlep_step_count = 0
        canonical_behavior_infos: dict[str, torch.Tensor] = {}
        if canonical_actions:
            if self._canonical_action_space is None:
                raise RuntimeError("canonical learner action space is missing")
            expected_count = int(self._canonical_action_space.horizon)
            observed_counts = response_masks.sum(dim=1)
            if not bool(observed_counts.eq(expected_count).all()):
                raise RuntimeError(
                    "canonical rows must contain exactly "
                    f"{expected_count} actions; got "
                    f"{observed_counts.detach().cpu().tolist()}"
                )
        logging.info(f"learn data size {input_ids.shape}")

        rlep_count = int(getattr(args, "rlep_replay_count", 0) or 0)
        rlep_online = bool(getattr(args, "rlep_online_pool", False))
        if rlep_count:
            pool = getattr(self, "_rlep_experience_pool", None)
            if pool is None:
                if rlep_online:
                    pool = OnlineRLEPExperiencePool(minimum=rlep_count)
                else:
                    pool = RLEPExperiencePool.from_directory(
                        str(args.rlep_experience_root),
                        allow_sparse=bool(
                            getattr(args, "rlep_sparse_fallback", False)
                        ),
                    )
                self._rlep_experience_pool = pool
            if rlep_online and not isinstance(pool, OnlineRLEPExperiencePool):
                raise RuntimeError("RLEP online pool is not an online pool")
            if not isinstance(pool, (RLEPExperiencePool, OnlineRLEPExperiencePool)):
                raise RuntimeError("invalid RLEP experience pool")
            num_rows = int(input_ids.size(0))
            references = list(trajectory.get("references") or [])
            if len(references) != num_rows:
                raise RuntimeError("RLEP requires one reference per fresh row")
            if num_rows != int(args.num_samples):
                raise RuntimeError("RLEP requires one complete fresh prompt group")
            first_reference = references[0]
            if any(reference != first_reference for reference in references[1:]):
                raise RuntimeError("RLEP fresh rows do not share one prompt reference")
            rlep_step_count = (
                rlep_count if pool.can_sample(first_reference, count=rlep_count) else 0
            )
            sampled_responses = (
                pool.sample(
                    first_reference,
                    count=rlep_step_count,
                    experiment_seed=int(args.seed),
                    learner_step=int(getattr(self, "steps", 0)),
                )
                if rlep_step_count
                else ()
            )
            max_response_tokens = int(args.generate_max_length)
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            replay_response_ids: list[tuple[int, ...]] = []
            for response in sampled_responses:
                if canonical_actions:
                    action_code = canonical_action_code_from_verified_response(
                        canonical_task,
                        response,
                        first_reference,
                    )
                    token_ids = list(
                        canonical_action_code_token_ids(
                            self._canonical_action_space,
                            action_code,
                        )
                    )
                else:
                    token_ids = [
                        int(value)
                        for value in self.tokenizer.encode(
                            response,
                            add_special_tokens=False,
                        )
                    ]
                    if eos_token_id is not None:
                        eos = int(eos_token_id)
                        if not token_ids or token_ids[-1] != eos:
                            token_ids = token_ids[: max_response_tokens - 1] + [eos]
                        else:
                            token_ids = token_ids[:max_response_tokens]
                    else:
                        token_ids = token_ids[:max_response_tokens]
                if not token_ids:
                    raise RuntimeError("RLEP tokenized an empty verified response")
                replay_response_ids.append(tuple(token_ids))
            if replay_response_ids:
                prompt_length = int(prompt_id_lens[0])
                prompt_tokens = tuple(
                    int(value)
                    for value in input_ids[0, :prompt_length].detach().cpu().tolist()
                )
                rlep_replay_groups = [
                    RLEPReplayGroup(
                        prompt_token_ids=prompt_tokens,
                        response_token_ids=tuple(replay_response_ids),
                    )
                ]
            rlep_online_added = 0
            if rlep_online:
                # Commit this group's verified rows only after the replay draw
                # above, so a replayed success is always a *past* success and
                # the draw depends on nothing the current group produced.
                label_ids = input_ids[:, 1:]
                fresh_rows = [
                    row_ids[row_mask.to(torch.bool)].detach().cpu().tolist()
                    for row_ids, row_mask in zip(label_ids, response_masks)
                ]
                if canonical_actions:
                    # A canonical row is one registered token per action, and
                    # the tokenizer's text for those placeholders is not the
                    # response the grader saw. Store the benchmark response
                    # instead, which is what the offline pool holds and what
                    # the replay draw above projects back onto the surface.
                    fresh_texts = [
                        decode_canonical_action_response(
                            canonical_task,
                            canonical_action_code_from_token_ids(
                                self._canonical_action_space, row
                            ),
                            first_reference,
                        )
                        for row in fresh_rows
                    ]
                else:
                    fresh_texts = [
                        self.tokenizer.decode(row, skip_special_tokens=True)
                        for row in fresh_rows
                    ]
                rlep_online_added = pool.observe(
                    first_reference,
                    fresh_texts,
                    task_final_rewards.detach().view(-1).cpu().tolist(),
                )
            diagnostics = pool.diagnostics
            rlep_infos = {
                "rlep_online_pool": torch.tensor(float(rlep_online), device=device),
                "rlep_online_pool_added_rows": torch.tensor(
                    rlep_online_added, device=device
                ),
                "rlep_pool_prompts": torch.tensor(diagnostics.prompts, device=device),
                "rlep_pool_trajectories": torch.tensor(
                    diagnostics.trajectories, device=device
                ),
                "rlep_pool_min_trajectories": torch.tensor(
                    diagnostics.minimum_trajectories_per_prompt, device=device
                ),
                "rlep_pool_max_trajectories": torch.tensor(
                    diagnostics.maximum_trajectories_per_prompt, device=device
                ),
                "rlep_pool_eligible_prompts": torch.tensor(
                    diagnostics.eligible_prompts, device=device
                ),
                "rlep_pool_ineligible_prompts": torch.tensor(
                    diagnostics.ineligible_prompts, device=device
                ),
                "rlep_replay_eligible": torch.tensor(
                    float(rlep_step_count > 0), device=device
                ),
                "rlep_replay_rows": torch.tensor(rlep_step_count, device=device),
            }

        mi_tracker = getattr(self, "_diayn_mi_tracker", None)
        if mi_tracker is not None:
            num_rows = int(input_ids.size(0))
            option_ids = [
                coerce_option_id(value)
                for value in list(trajectory.get("diayn_option_ids") or [])
            ]
            if len(option_ids) != num_rows or any(
                value is None for value in option_ids
            ):
                raise RuntimeError(
                    "DIAYN rollout is missing one valid option id per candidate"
                )
            num_options = int(args.diayn_num_options)
            for group_start in range(0, num_rows, int(args.num_samples)):
                group_options = option_ids[
                    group_start : group_start + int(args.num_samples)
                ]
                expected_per_option = int(args.num_samples) // num_options
                observed = [
                    sum(int(value == option) for value in group_options)
                    for option in range(num_options)
                ]
                if observed != [expected_per_option] * num_options:
                    raise RuntimeError(
                        "DIAYN candidate group is not balanced across options: "
                        f"observed={observed} expected={expected_per_option}"
                    )

            references = list(trajectory.get("references") or [])
            references = (references + [None] * num_rows)[:num_rows]
            refs_grouped = [
                references[index : index + int(args.num_samples)]
                for index in range(0, num_rows, int(args.num_samples))
            ]
            answer_keys_grouped = self._seed_answer_keys_grouped(
                input_ids,
                response_masks,
                int(args.num_samples),
                refs_grouped,
            )
            answer_keys = [key for group in answer_keys_grouped for key in group]
            conditional_keys = [
                conditional_answer_repr(reference, key)
                for reference, key in zip(references, answer_keys)
            ]
            task_correct = (task_final_rewards.detach().reshape(-1) > 0).cpu().tolist()
            task_reward_mean = task_final_rewards.detach().mean()
            bonuses, diagnostics = mi_tracker.update_and_score(
                answer_reprs=conditional_keys,
                option_ids=option_ids,
                correct=[bool(value) for value in task_correct],
                loss_masks=loss_masks.detach().cpu().tolist(),
                beta=float(args.diayn_mi_beta),
                correct_only=bool(args.diayn_mi_correct_only),
            )
            bonus_tensor = torch.tensor(
                bonuses, dtype=final_rewards.dtype, device=final_rewards.device
            ).reshape_as(final_rewards)
            final_rewards = final_rewards + bonus_tensor
            diayn_infos = {
                "diayn_task_reward_mean": task_reward_mean,
                "diayn_augmented_reward_mean": final_rewards.detach().mean(),
                "diayn_mi_bonus_mean": torch.tensor(
                    diagnostics.bonus_mean, device=final_rewards.device
                ),
                "diayn_mi_bonus_min": torch.tensor(
                    diagnostics.bonus_min, device=final_rewards.device
                ),
                "diayn_mi_bonus_max": torch.tensor(
                    diagnostics.bonus_max, device=final_rewards.device
                ),
                "diayn_mi_eligible_fraction": torch.tensor(
                    diagnostics.eligible_fraction, device=final_rewards.device
                ),
                "diayn_classifier_accuracy": torch.tensor(
                    diagnostics.classifier_accuracy, device=final_rewards.device
                ),
                "diayn_mi_lower_bound_nats": torch.tensor(
                    diagnostics.lower_bound_nats, device=final_rewards.device
                ),
                "diayn_distinct_answer_reprs": torch.tensor(
                    diagnostics.distinct_answer_reprs, device=final_rewards.device
                ),
                "diayn_loo_supported_fraction": torch.tensor(
                    diagnostics.leave_one_out_supported_fraction,
                    device=final_rewards.device,
                ),
            }

        gapo_infos: dict[str, torch.Tensor] = {}
        if bool(getattr(args, "gapo_enabled", False)):
            # GAPO replaces the task reward with its group frequency-aware
            # reward before centering. `task_final_rewards` stays the pristine
            # verifier outcome, so every downstream correctness reading --- the
            # eligible-row logic, the diagnostics, the evaluation --- is
            # unchanged and only the learning signal differs.
            num_rows = int(input_ids.size(0))
            references = list(trajectory.get("references") or [])
            references = (references + [None] * num_rows)[:num_rows]
            references_grouped = [
                references[index : index + int(args.num_samples)]
                for index in range(0, num_rows, int(args.num_samples))
            ]
            gapo_key_kwargs: dict[str, Any] = {}
            if canonical_actions:
                gapo_key_kwargs[
                    "response_surfaces"
                ] = _task_bound_canonicalization_surfaces(
                    [],
                    trajectory,
                    canonical_task=canonical_task,
                    expected_count=num_rows,
                )
            gapo_keys_grouped = self._seed_answer_keys_grouped(
                input_ids,
                response_masks,
                int(args.num_samples),
                references_grouped,
                **gapo_key_kwargs,
            )
            gapo_keys = [key for group in gapo_keys_grouped for key in group]
            support_index = self._gapo_support_index()
            support_sizes = support_index.lookup(references)
            gapo_rewards, gapo_diagnostics = gapo_group_rewards(
                gapo_keys,
                [float(value) for value in task_final_rewards.reshape(-1).tolist()],
                support_sizes,
                num_samples=int(args.num_samples),
                reward_scale=str(getattr(args, "gapo_reward_scale", "unit")),
            )
            final_rewards = torch.tensor(
                gapo_rewards, dtype=final_rewards.dtype, device=final_rewards.device
            ).reshape_as(final_rewards)
            gapo_infos = {
                f"gapo_{name}": torch.tensor(float(value), device=device)
                for name, value in vars(gapo_diagnostics).items()
            }
            gapo_infos["gapo_enabled"] = torch.tensor(1.0, device=device)

        outcome_collision_coef = float(
            getattr(args, "outcome_collision_coef", 0.0) or 0.0
        )
        outcome_collision_outside_centering = bool(
            getattr(args, "outcome_collision_outside_centering", False)
        )
        if outcome_collision_coef > 0:
            num_rows = int(input_ids.size(0))
            references = list(trajectory.get("references") or [])
            references = (references + [None] * num_rows)[:num_rows]
            references_grouped = [
                references[index : index + int(args.num_samples)]
                for index in range(0, num_rows, int(args.num_samples))
            ]
            semantic_key_kwargs: dict[str, Any] = {}
            if canonical_actions:
                semantic_key_kwargs[
                    "response_surfaces"
                ] = _task_bound_canonicalization_surfaces(
                    [],
                    trajectory,
                    canonical_task=canonical_task,
                    expected_count=num_rows,
                )
            answer_keys_grouped = self._seed_answer_keys_grouped(
                input_ids,
                response_masks,
                int(args.num_samples),
                references_grouped,
                **semantic_key_kwargs,
            )
            answer_keys = [key for group in answer_keys_grouped for key in group]
            bonuses, diagnostics = compute_outcome_collision_bonuses(
                answer_keys,
                num_samples=int(args.num_samples),
                coefficient=outcome_collision_coef,
            )
            bonus_tensor = torch.tensor(
                bonuses, dtype=final_rewards.dtype, device=final_rewards.device
            ).reshape_as(final_rewards)
            bonus_groups = bonus_tensor.reshape(-1, int(args.num_samples))
            centered_bonus_groups = bonus_groups - bonus_groups.mean(
                dim=1, keepdim=True
            )
            bonus_group_ranges = (
                bonus_groups.max(dim=1).values - bonus_groups.min(dim=1).values
            )
            augmented_rewards = final_rewards + bonus_tensor
            if outcome_collision_outside_centering:
                # E40: preserve ordinary Dr.GRPO centering for the task
                # reward. The detached collision vector is applied exactly
                # once to the precomputed sequence advantage below.
                outcome_collision_outside_advantage = bonus_tensor.detach()
            else:
                # E37: retain the established reward-shaping behavior.
                final_rewards = augmented_rewards
            outcome_collision_infos = {
                "outcome_collision_task_reward_mean": (
                    task_final_rewards.detach().mean()
                ),
                "outcome_collision_augmented_reward_mean": (
                    augmented_rewards.detach().mean()
                ),
                "outcome_collision_reward_sent_to_centering_mean": (
                    final_rewards.detach().mean()
                ),
                "outcome_collision_outside_centering_active": torch.tensor(
                    float(outcome_collision_outside_centering),
                    device=final_rewards.device,
                ),
                "outcome_collision_rate": torch.tensor(
                    diagnostics.collision_rate, device=final_rewards.device
                ),
                "outcome_collision_bonus_mean": torch.tensor(
                    diagnostics.bonus_mean, device=final_rewards.device
                ),
                "outcome_collision_bonus_min": torch.tensor(
                    diagnostics.bonus_min, device=final_rewards.device
                ),
                "outcome_collision_bonus_max": torch.tensor(
                    diagnostics.bonus_max, device=final_rewards.device
                ),
                "outcome_collision_bonus_zero_spread_group_fraction": (
                    bonus_group_ranges.eq(0).float().mean()
                ),
                "outcome_collision_centered_bonus_abs_mean": (
                    centered_bonus_groups.abs().mean()
                ),
                "outcome_collision_centered_bonus_rms": torch.sqrt(
                    centered_bonus_groups.square().mean()
                ),
                "outcome_collision_distinct_outcomes_mean": torch.tensor(
                    diagnostics.distinct_outcomes_mean,
                    device=final_rewards.device,
                ),
                "outcome_collision_distinct_fraction": torch.tensor(
                    diagnostics.distinct_fraction, device=final_rewards.device
                ),
                "outcome_collision_invalid_fraction": torch.tensor(
                    diagnostics.invalid_fraction, device=final_rewards.device
                ),
                "outcome_collision_parseable_fraction": torch.tensor(
                    diagnostics.parseable_fraction, device=final_rewards.device
                ),
            }

        semantic_shannon_tracker = getattr(self, "_semantic_shannon_tracker", None)
        if semantic_shannon_tracker is not None:
            if not isinstance(semantic_shannon_tracker, SemanticShannonTracker):
                raise RuntimeError("invalid semantic Shannon tracker")
            semantic_shannon_use_separate_advantage = bool(
                getattr(args, "semantic_shannon_separate_advantage", False)
            )
            semantic_shannon_use_quality_gate = bool(
                getattr(
                    args,
                    "semantic_shannon_quality_gated_advantage",
                    False,
                )
            )
            semantic_shannon_use_success_conditioned_signed = bool(
                getattr(
                    args,
                    ("semantic_shannon_success_conditioned_signed_advantage"),
                    False,
                )
            )
            semantic_shannon_use_success_conditioned_group_centered = bool(
                getattr(
                    args,
                    "semantic_shannon_success_conditioned_group_centered_advantage",
                    False,
                )
            )
            semantic_shannon_use_success_conditioned_verified_support = bool(
                getattr(
                    args,
                    "semantic_shannon_success_conditioned_verified_support_advantage",
                    False,
                )
            )
            semantic_shannon_verified_support_include_replay_bank = bool(
                getattr(
                    args,
                    "semantic_shannon_verified_support_include_replay_bank",
                    False,
                )
            )
            if (
                semantic_shannon_use_quality_gate
                or semantic_shannon_use_success_conditioned_signed
                or semantic_shannon_use_success_conditioned_group_centered
                or semantic_shannon_use_success_conditioned_verified_support
            ) and not semantic_shannon_use_separate_advantage:
                raise RuntimeError(
                    "semantic Shannon gated modes require the separate advantage path"
                )
            if (
                sum(
                    (
                        semantic_shannon_use_quality_gate,
                        semantic_shannon_use_success_conditioned_signed,
                        semantic_shannon_use_success_conditioned_group_centered,
                        semantic_shannon_use_success_conditioned_verified_support,
                    )
                )
                > 1
            ):
                raise RuntimeError(
                    "semantic Shannon quality-gated, success-conditioned "
                    "signed, group-centered, and verified-support modes "
                    "are mutually exclusive"
                )
            num_rows = int(input_ids.size(0))
            references = list(trajectory.get("references") or [])
            references = (references + [None] * num_rows)[:num_rows]
            references_grouped = [
                references[index : index + int(args.num_samples)]
                for index in range(0, num_rows, int(args.num_samples))
            ]
            # A canonical-action task emits a short action sequence, not a
            # free-form response with a boxed answer, so decoding its tokens and
            # running the text extractor over them yields None for every row.
            # The bank and the outcome-collision path already bind the
            # task's own canonicalization surfaces here; the semantic term must
            # do the same or it scores every row unparseable and contributes
            # exact zero for the whole run. That is precisely what happened to
            # PantryPlan in E81, E82, and E83: reward-positive fraction .544,
            # parseable fraction .000.
            semantic_key_kwargs: dict[str, Any] = {}
            if canonical_actions:
                semantic_key_kwargs[
                    "response_surfaces"
                ] = _task_bound_canonicalization_surfaces(
                    [],
                    trajectory,
                    canonical_task=canonical_task,
                    expected_count=num_rows,
                )
            answer_keys_grouped = self._seed_answer_keys_grouped(
                input_ids,
                response_masks,
                int(args.num_samples),
                references_grouped,
                **semantic_key_kwargs,
            )
            answer_keys = [key for group in answer_keys_grouped for key in group]
            semantic_task_rewards = task_final_rewards.detach().view(-1).cpu().tolist()
            if (
                str(
                    getattr(
                        args,
                        "online_canonical_key_mode",
                        "modebench_outcome",
                    )
                )
                == "math_verified_answer"
            ):
                # MATH-500 is a realistic single-answer generalization track,
                # not a reasoning-strategy benchmark. Collapse every
                # verifier-positive representation to the same prompt-local
                # outcome so formatting aliases cannot masquerade as modes.
                answer_keys = math_verified_answer_outcome_keys(
                    [float(reward) > 0.0 for reward in semantic_task_rewards]
                )
            prompt_token_ids = [
                input_ids[row_index, : int(prompt_id_lens[row_index])]
                .detach()
                .cpu()
                .tolist()
                for row_index in range(num_rows)
            ]
            verified_support_keys_by_group = None
            if semantic_shannon_verified_support_include_replay_bank:
                support_bank = getattr(self, "_online_canonical_bank", None)
                if not isinstance(support_bank, OnlineCanonicalBank):
                    raise RuntimeError(
                        "replay-bank semantic support requires an online "
                        "canonical bank"
                    )
                verified_support_keys_by_group = [
                    support_bank.verified_replay_support(prompt_token_ids[start])
                    for start in range(0, num_rows, int(args.num_samples))
                ]
            diagnostics = None
            advantage_diagnostics = None
            quality_gated_diagnostics = None
            success_conditioned_signed_diagnostics = None
            success_conditioned_group_centered_diagnostics = None
            success_conditioned_verified_support_diagnostics = None
            if (
                semantic_shannon_use_success_conditioned_signed
                or semantic_shannon_use_success_conditioned_group_centered
                or semantic_shannon_use_success_conditioned_verified_support
            ):
                (
                    separate_advantages,
                    success_conditioned_diagnostics,
                ) = semantic_shannon_tracker.score_success_conditioned_signed_advantages_and_update(
                    prompt_token_ids=prompt_token_ids,
                    answer_keys=answer_keys,
                    task_rewards=semantic_task_rewards,
                    active_mask=loss_masks.detach().view(-1).cpu().tolist(),
                    num_samples=int(args.num_samples),
                    verified_support_keys_by_group=(verified_support_keys_by_group),
                )
                if semantic_shannon_use_success_conditioned_signed:
                    success_conditioned_signed_diagnostics = (
                        success_conditioned_diagnostics
                    )
                elif semantic_shannon_use_success_conditioned_group_centered:
                    success_conditioned_group_centered_diagnostics = (
                        success_conditioned_diagnostics
                    )
                else:
                    success_conditioned_verified_support_diagnostics = (
                        success_conditioned_diagnostics
                    )
                semantic_shannon_separate_advantage = (
                    torch.tensor(
                        separate_advantages,
                        dtype=final_rewards.dtype,
                        device=final_rewards.device,
                    )
                    .reshape_as(final_rewards)
                    .detach()
                )
            elif semantic_shannon_use_quality_gate:
                (
                    separate_advantages,
                    quality_gated_diagnostics,
                ) = semantic_shannon_tracker.score_quality_gated_advantages_and_update(
                    prompt_token_ids=prompt_token_ids,
                    answer_keys=answer_keys,
                    task_rewards=semantic_task_rewards,
                    active_mask=loss_masks.detach().view(-1).cpu().tolist(),
                    num_samples=int(args.num_samples),
                )
                semantic_shannon_separate_advantage = (
                    torch.tensor(
                        separate_advantages,
                        dtype=final_rewards.dtype,
                        device=final_rewards.device,
                    )
                    .reshape_as(final_rewards)
                    .detach()
                )
            elif semantic_shannon_use_separate_advantage:
                (
                    separate_advantages,
                    diagnostics,
                    advantage_diagnostics,
                ) = semantic_shannon_tracker.score_separate_advantages_and_update(
                    prompt_token_ids=prompt_token_ids,
                    answer_keys=answer_keys,
                    num_samples=int(args.num_samples),
                )
                semantic_shannon_separate_advantage = (
                    torch.tensor(
                        separate_advantages,
                        dtype=final_rewards.dtype,
                        device=final_rewards.device,
                    )
                    .reshape_as(final_rewards)
                    .detach()
                )
            else:
                bonuses, diagnostics = semantic_shannon_tracker.score_and_update(
                    prompt_token_ids=prompt_token_ids,
                    answer_keys=answer_keys,
                    num_samples=int(args.num_samples),
                )
                bonus_tensor = torch.tensor(
                    bonuses,
                    dtype=final_rewards.dtype,
                    device=final_rewards.device,
                ).reshape_as(final_rewards)
                final_rewards = final_rewards + bonus_tensor
            semantic_shannon_infos = {
                "semantic_shannon_task_reward_mean": (
                    task_final_rewards.detach().mean()
                ),
                "semantic_shannon_augmented_reward_mean": (
                    final_rewards.detach().mean()
                ),
                "semantic_shannon_reward_sent_to_centering_mean": (
                    final_rewards.detach().mean()
                ),
                "semantic_shannon_separate_advantage_active": torch.tensor(
                    float(semantic_shannon_use_separate_advantage),
                    device=final_rewards.device,
                ),
                "semantic_shannon_quality_gated_advantage_active": torch.tensor(
                    float(semantic_shannon_use_quality_gate),
                    device=final_rewards.device,
                ),
                "semantic_shannon_success_conditioned_signed_advantage_active": (
                    torch.tensor(
                        float(semantic_shannon_use_success_conditioned_signed),
                        device=final_rewards.device,
                    )
                ),
                "semantic_shannon_success_conditioned_group_centered_advantage_active": (
                    torch.tensor(
                        float(semantic_shannon_use_success_conditioned_group_centered),
                        device=final_rewards.device,
                    )
                ),
                "semantic_shannon_success_conditioned_verified_support_advantage_active": (
                    torch.tensor(
                        float(
                            semantic_shannon_use_success_conditioned_verified_support
                        ),
                        device=final_rewards.device,
                    )
                ),
                "semantic_shannon_verified_support_include_replay_bank_active": (
                    torch.tensor(
                        float(semantic_shannon_verified_support_include_replay_bank),
                        device=final_rewards.device,
                    )
                ),
                "semantic_rms_controller_active": torch.tensor(
                    float(getattr(self, "_semantic_rms_controller", None) is not None),
                    device=final_rewards.device,
                ),
            }
            if diagnostics is not None:
                semantic_shannon_infos.update(
                    {
                        "semantic_shannon_bonus_mean": torch.tensor(
                            diagnostics.bonus_mean, device=final_rewards.device
                        ),
                        "semantic_shannon_bonus_min": torch.tensor(
                            diagnostics.bonus_min, device=final_rewards.device
                        ),
                        "semantic_shannon_bonus_max": torch.tensor(
                            diagnostics.bonus_max, device=final_rewards.device
                        ),
                        "semantic_shannon_surprisal_mean": torch.tensor(
                            diagnostics.surprisal_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_normalized_surprisal_mean": torch.tensor(
                            diagnostics.normalized_surprisal_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_entropy_mean": torch.tensor(
                            diagnostics.entropy_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_clipped_surprisal_mean": torch.tensor(
                            diagnostics.clipped_surprisal_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_clip_fraction": torch.tensor(
                            diagnostics.clip_fraction,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_predictive_probability_mean": torch.tensor(
                            diagnostics.predictive_probability_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_predictive_probability_min": torch.tensor(
                            diagnostics.predictive_probability_min,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_predictive_probability_max": torch.tensor(
                            diagnostics.predictive_probability_max,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_normalization_error_max": torch.tensor(
                            diagnostics.normalization_error_max,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_unseen_fraction": torch.tensor(
                            diagnostics.unseen_fraction,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_history_total_mean": torch.tensor(
                            diagnostics.history_total_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_distinct_outcomes_mean": torch.tensor(
                            diagnostics.distinct_outcomes_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_distinct_fraction": torch.tensor(
                            diagnostics.distinct_fraction,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_invalid_fraction": torch.tensor(
                            diagnostics.invalid_fraction,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_parseable_fraction": torch.tensor(
                            diagnostics.parseable_fraction,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_tracked_prompts": torch.tensor(
                            diagnostics.tracked_prompts,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_tracked_outcomes": torch.tensor(
                            diagnostics.tracked_outcomes,
                            device=final_rewards.device,
                        ),
                    }
                )
            if quality_gated_diagnostics is not None:
                quality_values = {
                    field_name: getattr(quality_gated_diagnostics, field_name)
                    for field_name in (
                        "raw_all_row_advantage_mean",
                        "raw_all_row_advantage_min",
                        "raw_all_row_advantage_max",
                        "raw_all_row_advantage_abs_mean",
                        "raw_all_row_advantage_rms",
                        "raw_all_row_advantage_positive_fraction",
                        "raw_all_row_advantage_negative_fraction",
                        "raw_all_row_advantage_zero_fraction",
                        "effective_advantage_mean",
                        "effective_advantage_min",
                        "effective_advantage_max",
                        "effective_advantage_abs_mean",
                        "effective_advantage_rms",
                        "effective_advantage_positive_fraction",
                        "effective_advantage_zero_fraction",
                        "eligible_fraction",
                        "gated_fraction",
                        "active_fraction",
                        "reward_positive_fraction",
                        "parseable_fraction",
                        "positive_only_zeroed_fraction",
                        "cap_fraction",
                        "advantage_cap",
                        "predictive_baseline_mean",
                        "predictive_centering_error_max",
                        "predictive_probability_mean",
                        "predictive_probability_min",
                        "predictive_probability_max",
                        "normalization_error_max",
                        "history_total_before_mean",
                        "history_rows_added",
                        "history_groups_updated",
                        "history_groups_skipped",
                        "tracked_prompts",
                        "tracked_outcomes",
                    )
                }
                semantic_shannon_infos.update(
                    {
                        f"semantic_shannon_quality_gated_{name}": torch.tensor(
                            value, device=final_rewards.device
                        )
                        for name, value in quality_values.items()
                    }
                )
            success_conditioned_diagnostic_streams = (
                (
                    "semantic_shannon_success_conditioned_signed",
                    success_conditioned_signed_diagnostics,
                ),
                (
                    "semantic_shannon_success_conditioned_group_centered",
                    success_conditioned_group_centered_diagnostics,
                ),
                (
                    "semantic_shannon_success_conditioned_verified_support",
                    success_conditioned_verified_support_diagnostics,
                ),
            )
            for (
                diagnostic_prefix,
                success_conditioned_diagnostics,
            ) in success_conditioned_diagnostic_streams:
                if success_conditioned_diagnostics is None:
                    continue
                semantic_values = success_conditioned_semantic_metric_values(
                    diagnostic_prefix,
                    success_conditioned_diagnostics,
                )
                semantic_shannon_infos.update(
                    {
                        name: torch.tensor(value, device=final_rewards.device)
                        for name, value in semantic_values.items()
                    }
                )
            if (
                semantic_shannon_use_separate_advantage
                and advantage_diagnostics is not None
            ):
                semantic_shannon_infos.update(
                    {
                        "semantic_shannon_separate_predictive_baseline_mean": (
                            torch.tensor(
                                advantage_diagnostics.predictive_baseline_mean,
                                device=final_rewards.device,
                            )
                        ),
                        "semantic_shannon_separate_predictive_baseline_min": (
                            torch.tensor(
                                advantage_diagnostics.predictive_baseline_min,
                                device=final_rewards.device,
                            )
                        ),
                        "semantic_shannon_separate_predictive_baseline_max": (
                            torch.tensor(
                                advantage_diagnostics.predictive_baseline_max,
                                device=final_rewards.device,
                            )
                        ),
                        "semantic_shannon_separate_predictive_baseline_normalized_mean": (
                            torch.tensor(
                                advantage_diagnostics.predictive_baseline_normalized_mean,
                                device=final_rewards.device,
                            )
                        ),
                        "semantic_shannon_separate_predictive_centering_error_max": (
                            torch.tensor(
                                advantage_diagnostics.predictive_centering_error_max,
                                device=final_rewards.device,
                            )
                        ),
                        "semantic_shannon_separate_advantage_scale": torch.tensor(
                            advantage_diagnostics.advantage_scale,
                            device=final_rewards.device,
                        ),
                    }
                )

        online_canonical_bank = getattr(self, "_online_canonical_bank", None)
        if online_canonical_bank is not None:
            if not isinstance(online_canonical_bank, OnlineCanonicalBank):
                raise RuntimeError("invalid online canonical bank")
            online_canonical_bank_objective_active = (
                online_canonical_bank.objective_active
            )
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
                getattr(
                    args,
                    "online_canonical_key_mode",
                    "modebench_outcome",
                )
            )
            if key_mode in {"math_strategy_qwen72", "verified_route"}:
                response_texts = [
                    str(value) for value in list(trajectory.get("responses") or [])
                ]
                if len(response_texts) != num_rows:
                    raise RuntimeError(
                        "MATH strategy canonicalization requires the exact "
                        "validator-graded response for every row"
                    )
            elif key_mode == "modebench_outcome" and canonical_actions:
                response_texts = _task_bound_canonicalization_surfaces(
                    response_texts,
                    trajectory,
                    canonical_task=canonical_task,
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
            math_strategy_diagnostics = None
            route_verifier_ids: list[str | None] = [None] * num_rows
            route_signatures: list[str | None] = [None] * num_rows
            if key_mode == "modebench_outcome":
                references = list(trajectory.get("references") or [])
                references = (references + [None] * num_rows)[:num_rows]
                outcome_keys = [
                    validated_modebench_outcome_key(
                        text,
                        references[row_index],
                    )
                    for row_index, text in enumerate(response_texts)
                ]
            elif key_mode == "math_verified_answer":
                # The ordinary full MATH verifier has already produced
                # task_reward_positive. Use that validation as the complete
                # canonical contract and intentionally expose exactly one
                # accepted outcome per prompt. This enables verified-mass
                # replay while making multi-mode balance structurally
                # ineligible; it does not claim to verify reasoning routes.
                outcome_keys = math_verified_answer_outcome_keys(task_reward_positive)
            elif key_mode == "math_strategy_qwen72":
                canonicalizer = getattr(self, "_math_strategy_canonicalizer", None)
                if not isinstance(canonicalizer, MathStrategyCanonicalizer):
                    raise RuntimeError("MATH strategy key mode lacks its canonicalizer")
                prompt_texts = [
                    str(value) for value in list(trajectory.get("prompts") or [])
                ]
                if len(prompt_texts) != num_rows:
                    raise RuntimeError(
                        "MATH strategy canonicalization requires one raw "
                        "problem per trajectory row"
                    )
                outcome_keys, math_strategy_diagnostics = canonicalizer.canonicalize(
                    prompt_token_ids=prompt_token_ids,
                    prompt_texts=prompt_texts,
                    response_texts=response_texts,
                    task_reward_positive=task_reward_positive,
                    active_mask=(loss_masks.detach().view(-1).cpu().tolist()),
                    num_samples=int(args.num_samples),
                )
            elif key_mode == "verified_route":
                references = list(trajectory.get("references") or [])
                references = (references + [None] * num_rows)[:num_rows]
                prompt_texts = [
                    str(value) for value in list(trajectory.get("prompts") or [])
                ]
                if len(prompt_texts) != num_rows:
                    raise RuntimeError(
                        "verified-route canonicalization requires one raw "
                        "problem per trajectory row"
                    )
                identities = [
                    validated_exploration_identity(
                        response_texts[row_index],
                        prompt_texts[row_index],
                        references[row_index],
                        fast=(str(args.verifier_version) != "math_verify"),
                        task_verified=bool(task_reward_positive[row_index]),
                    )
                    for row_index in range(num_rows)
                ]
                outcome_keys = [
                    (identity.endpoint_key if identity is not None else None)
                    for identity in identities
                ]
                route_verifier_ids = [
                    (identity.verifier if identity is not None else None)
                    for identity in identities
                ]
                route_signatures = [
                    (identity.route_signature if identity is not None else None)
                    for identity in identities
                ]
            else:
                raise RuntimeError(
                    f"unsupported online canonical key mode: {key_mode!r}"
                )
            validator_admitted = [key is not None for key in outcome_keys]
            validator_positive_actor_negative_rows = [
                index
                for index, (verified, rewarded) in enumerate(
                    zip(validator_admitted, task_reward_positive)
                )
                if verified and not rewarded
            ]
            actor_positive_validator_negative_rows = [
                index
                for index, (verified, rewarded) in enumerate(
                    zip(validator_admitted, task_reward_positive)
                )
                if rewarded and not verified
            ]
            disagreement_rows = (
                validator_positive_actor_negative_rows
                + actor_positive_validator_negative_rows
            )
            if disagreement_rows:
                logging.warning(
                    "online canonical validator/task-reward disagreement "
                    "at rows %s; fail-closed intersection excludes them "
                    "from bank admission",
                    sorted(disagreement_rows),
                )
            # A canonical key is eligible only when both independent gates
            # agree: the executable canonicalizer validated the response and
            # the rollout received positive task reward.  In particular, a
            # parseable/valid-looking completion that was task-reward-zero
            # (for example because the rollout contract rejected truncation)
            # must not influence either the passive discovery tracker or the
            # active canonical objective.  Disagreement is telemetry, not a
            # fatal training condition.
            outcome_keys = [
                key if (key is not None and rewarded) else None
                for key, rewarded in zip(outcome_keys, task_reward_positive)
            ]
            if key_mode == "verified_route":
                verified_route_library = getattr(
                    self,
                    "_verified_route_library",
                    None,
                )
                if not isinstance(
                    verified_route_library,
                    VerifiedRouteLibrary,
                ):
                    raise RuntimeError(
                        "verified-route key mode lacks its route library"
                    )
                action_logprob_rows = list(trajectory.get("action_logprobs") or [])
                if len(action_logprob_rows) != num_rows:
                    raise RuntimeError(
                        "verified-route tracking requires actor log "
                        "probabilities for every row"
                    )
                model_mean_logprobs: list[float] = []
                for row_index, values in enumerate(action_logprob_rows):
                    row_values = [float(value) for value in list(values)]
                    if route_signatures[row_index] is not None and not row_values:
                        raise RuntimeError(
                            "verified neutral route lacks behavior log probabilities"
                        )
                    model_mean_logprobs.append(
                        (sum(row_values) / len(row_values) if row_values else 0.0)
                    )
                route_signatures = [
                    (route if endpoint_key is not None else None)
                    for route, endpoint_key in zip(
                        route_signatures,
                        outcome_keys,
                    )
                ]
                route_verifier_ids = [
                    (verifier if endpoint_key is not None else None)
                    for verifier, endpoint_key in zip(
                        route_verifier_ids,
                        outcome_keys,
                    )
                ]
                verified_route_library.observe_neutral(
                    prompt_token_ids=prompt_token_ids,
                    verifier_ids=route_verifier_ids,
                    endpoint_keys=outcome_keys,
                    route_signatures=route_signatures,
                    response_token_ids=response_token_ids,
                    model_mean_logprobs=model_mean_logprobs,
                    task_verified=task_reward_positive,
                    active_mask=(loss_masks.detach().view(-1).cpu().tolist()),
                )
            admitted = [key is not None for key in outcome_keys]
            if bool(getattr(args, "math_strategy_gate_task_reward", False)):
                if key_mode != "math_strategy_qwen72":
                    raise RuntimeError(
                        "MATH strategy reward gate reached a non-MATH key mode"
                    )
                (
                    final_rewards,
                    task_final_rewards,
                ) = apply_math_strategy_task_reward_gate(
                    final_rewards,
                    task_final_rewards,
                    admitted,
                )
            replay_bank_freeze_step = int(
                getattr(args, "online_canonical_replay_bank_freeze_step", 0)
            )
            replay_bank_membership_frozen = (
                replay_bank_freeze_step > 0
                and int(getattr(self, "steps", 0)) >= replay_bank_freeze_step
            )
            bank_advantages, bank_diagnostics = online_canonical_bank.score_and_update(
                prompt_token_ids=prompt_token_ids,
                outcome_keys=outcome_keys,
                task_rewards=(task_final_rewards.detach().view(-1).cpu().tolist()),
                active_mask=loss_masks.detach().view(-1).cpu().tolist(),
                num_samples=int(args.num_samples),
                entropy_alpha_override=(
                    getattr(
                        self,
                        "_online_canonical_alpha_controller",
                        None,
                    ).current_alpha
                    if getattr(
                        self,
                        "_online_canonical_alpha_controller",
                        None,
                    )
                    is not None
                    else None
                ),
                response_token_ids=(
                    response_token_ids if online_canonical_replay_active else None
                ),
                update_bank=not replay_bank_membership_frozen,
            )
            if online_canonical_bank_objective_active:
                online_canonical_advantage = (
                    torch.tensor(
                        bank_advantages,
                        dtype=final_rewards.dtype,
                        device=final_rewards.device,
                    )
                    .reshape_as(final_rewards)
                    .detach()
                )
            canonical_replay_used_global_scheduler = False
            canonical_replay_used_prompt_local_scheduler = False
            verified_route_replay_used = False
            verified_route_endpoint_fallback_used = False
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
                if key_mode == "verified_route":
                    verified_route_library = getattr(
                        self,
                        "_verified_route_library",
                        None,
                    )
                    if not isinstance(
                        verified_route_library,
                        VerifiedRouteLibrary,
                    ):
                        raise RuntimeError(
                            "verified-route replay lacks its route library"
                        )
                    canonical_replay_used_global_scheduler = True
                    canonical_replay_groups = (
                        verified_route_library.scheduled_cross_prompt_replay_groups(
                            prompt_token_ids,
                        )
                    )
                    verified_route_replay_used = bool(canonical_replay_groups)
                    if not canonical_replay_groups:
                        # Graph has no domain-independent executable route;
                        # early cold-start batches in other domains may not yet
                        # have a route recurring on two neutral prompts. Keep
                        # the exact fixed compute budget with the independently
                        # verified endpoint bank until route replay is eligible.
                        canonical_replay_groups = (
                            online_canonical_bank.scheduled_global_replay_groups(
                                min_modes=1,
                            )
                        )
                        verified_route_endpoint_fallback_used = bool(
                            canonical_replay_groups
                        )
                elif int(args.online_canonical_replay_global_groups_per_step) > 0:
                    if int(
                        args.online_canonical_replay_global_bootstrap_steps
                    ) > 0 and not (
                        online_canonical_bank.global_replay_bootstrap_active
                    ):
                        canonical_replay_used_prompt_local_scheduler = True
                        canonical_replay_groups = online_canonical_bank.replay_groups(
                            prompt_token_ids,
                            min_modes=(replay_min_modes),
                            consume_priority=True,
                        )
                    else:
                        canonical_replay_used_global_scheduler = True
                        canonical_replay_groups = (
                            online_canonical_bank.scheduled_global_replay_groups(
                                min_modes=replay_min_modes,
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
            online_canonical_infos = {
                f"online_canonical_{name}": torch.tensor(
                    getattr(bank_diagnostics, name),
                    device=final_rewards.device,
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
            if key_mode == "verified_route":
                verified_route_library = getattr(
                    self,
                    "_verified_route_library",
                    None,
                )
                if not isinstance(
                    verified_route_library,
                    VerifiedRouteLibrary,
                ):
                    raise RuntimeError(
                        "verified-route telemetry lacks its route library"
                    )
                route_diagnostics = verified_route_library.diagnostics()
                online_canonical_infos.update(
                    {
                        f"verified_route_{name}": torch.tensor(
                            float(getattr(route_diagnostics, name)),
                            device=final_rewards.device,
                        )
                        for name in (
                            "neutral_rows_observed",
                            "neutral_routes_observed",
                            "proposal_rows_admitted",
                            "proposal_rows_rejected_trust",
                            "proposal_graduations",
                            "distinct_routes",
                            "recurring_routes",
                            "distinct_source_prompts",
                            "cross_prompt_neutral_reproductions",
                            "post_replay_cross_prompt_neutral_reproductions",
                            "cross_prompt_replay_updates",
                            "cross_prompt_replay_groups",
                            "cross_prompt_replay_rows",
                        )
                    }
                )
                online_canonical_infos.update(
                    {
                        "verified_route_replay_used": torch.tensor(
                            float(verified_route_replay_used),
                            device=final_rewards.device,
                        ),
                        "verified_route_endpoint_fallback_used": torch.tensor(
                            float(verified_route_endpoint_fallback_used),
                            device=final_rewards.device,
                        ),
                        "verified_route_proposal_rows_to_ppo": torch.tensor(
                            0.0,
                            device=final_rewards.device,
                        ),
                        "verified_route_gold_support_feedback": torch.tensor(
                            0.0,
                            device=final_rewards.device,
                        ),
                        "verified_route_eval_feedback": torch.tensor(
                            0.0,
                            device=final_rewards.device,
                        ),
                    }
                )
            online_canonical_infos.update(
                {
                    "online_canonical_proposal_retention_"
                    f"{name}": torch.tensor(
                        value,
                        dtype=torch.float32,
                        device=final_rewards.device,
                    )
                    for name, value in (
                        online_canonical_bank.proposal_retention_diagnostics().items()
                    )
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
                            sum(
                                len(group.outcome_keys)
                                for group in canonical_replay_groups
                            ),
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
                                    getattr(
                                        args,
                                        "online_canonical_replay_compute_only",
                                        False,
                                    )
                                )
                            ),
                            device=final_rewards.device,
                        ),
                        "canonical_replay_realized_prompt_tokens": torch.tensor(
                            sum(
                                len(group.prompt_token_ids)
                                * len(group.response_token_ids)
                                for group in canonical_replay_groups
                            ),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_realized_response_tokens": torch.tensor(
                            sum(
                                len(response)
                                for group in canonical_replay_groups
                                for response in group.response_token_ids
                            ),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_charged_response_token_budget": torch.tensor(
                            int(args.online_canonical_replay_capacity)
                            * int(args.generate_max_length)
                            * int(
                                max(
                                    1,
                                    args.online_canonical_replay_global_groups_per_step,
                                )
                            ),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_gold_support_feedback": torch.tensor(
                            0.0,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_global_scheduler_active": torch.tensor(
                            float(
                                int(args.online_canonical_replay_global_groups_per_step)
                                > 0
                            ),
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
                                int(args.online_canonical_replay_global_bootstrap_steps)
                                > 0
                                and not (
                                    online_canonical_bank.global_replay_bootstrap_active
                                )
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
                        "canonical_replay_priority_replay_groups_cumulative": (
                            torch.tensor(
                                online_canonical_bank.proposal_priority_replay_groups,
                                dtype=torch.float32,
                                device=final_rewards.device,
                            )
                        ),
                        "canonical_replay_priority_replay_modes_cumulative": (
                            torch.tensor(
                                online_canonical_bank.proposal_priority_replay_modes,
                                dtype=torch.float32,
                                device=final_rewards.device,
                            )
                        ),
                    }
                )
            if math_strategy_diagnostics is not None:
                online_canonical_infos.update(
                    {
                        f"math_strategy_{name}": torch.tensor(
                            getattr(math_strategy_diagnostics, name),
                            device=final_rewards.device,
                        )
                        for name in (
                            "judge_calls",
                            "validator_positive_rows",
                            "accepted_rows",
                            "rejected_integrity_rows",
                            "rejected_ambiguous_rows",
                            "rejected_disagreement_rows",
                            "matched_existing_rows",
                            "new_strategy_rows",
                            "new_strategy_count",
                            "judge_format_failure_rows",
                            "rejected_contract_rows",
                            "inferred_unstructured_rows",
                            "rejected_strategy_inference_rows",
                        )
                    }
                )
            online_canonical_infos[
                "math_strategy_raw_task_reward_mean"
            ] = raw_task_final_rewards.detach().mean()
            online_canonical_infos[
                "math_strategy_gated_task_reward_mean"
            ] = task_final_rewards.detach().mean()
            online_canonical_infos[
                "math_strategy_task_reward_gate_active"
            ] = torch.tensor(
                float(
                    bool(
                        getattr(
                            args,
                            "math_strategy_gate_task_reward",
                            False,
                        )
                    )
                ),
                device=final_rewards.device,
            )
            online_canonical_infos[
                "online_canonical_task_reward_mean"
            ] = task_final_rewards.detach().mean()
            online_canonical_infos[
                "online_canonical_reward_sent_to_centering_mean"
            ] = final_rewards.detach().mean()
            online_canonical_infos.update(
                {
                    "online_canonical_validator_positive_actor_negative_rows": (
                        torch.tensor(
                            len(validator_positive_actor_negative_rows),
                            device=final_rewards.device,
                        )
                    ),
                    "online_canonical_actor_positive_validator_negative_rows": (
                        torch.tensor(
                            len(actor_positive_validator_negative_rows),
                            device=final_rewards.device,
                        )
                    ),
                    "online_canonical_validator_task_disagreement_rows": (
                        torch.tensor(
                            len(disagreement_rows),
                            device=final_rewards.device,
                        )
                    ),
                    "verified_discovery_cumulative_outcomes": torch.tensor(
                        bank_diagnostics.tracked_outcomes,
                        device=final_rewards.device,
                    ),
                    "verified_discovery_tracked_prompts": torch.tensor(
                        bank_diagnostics.tracked_prompts,
                        device=final_rewards.device,
                    ),
                    "verified_discovery_mean_support_per_prompt": torch.tensor(
                        online_canonical_bank.mean_support_per_prompt,
                        device=final_rewards.device,
                    ),
                }
            )

        indices = torch.arange(
            response_masks.size(1), device=response_masks.device
        ).expand_as(response_masks)
        masked_indices = torch.where(
            response_masks, indices, torch.full_like(indices, -1)
        )
        eos_indices = masked_indices.max(dim=1).values

        logps = torch.zeros(
            input_ids.shape[0], input_ids.shape[1] - 1, device=input_ids.device
        )
        canonical_learner_full_logps: torch.Tensor | None = None
        canonical_learner_support_mask: torch.Tensor | None = None
        if canonical_actions:
            canonical_learner_full_logps = torch.zeros(
                (
                    input_ids.shape[0],
                    input_ids.shape[1] - 1,
                    len(self._canonical_action_token_ids or ()),
                ),
                dtype=torch.float32,
                device=input_ids.device,
            )
            canonical_learner_support_mask = torch.zeros_like(
                canonical_learner_full_logps, dtype=torch.bool
            )
        policy_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(self.model)
        # E14's behavior trace is produced in eval mode.  Score its frozen old
        # policy in the same mode as well; otherwise training-only behavior can
        # invalidate what should be a same-snapshot identity check.  This is
        # intentionally scoped to canonical actions so retained baselines are
        # unchanged.
        with (
            _temporary_eval_mode(self.model, enabled=canonical_actions),
            torch.no_grad(),
        ):
            for i in range(0, len(input_ids), args.train_batch_size_per_device):
                batch_end = min(i + args.train_batch_size_per_device, len(input_ids))
                mini_batch_inds = torch.arange(i, batch_end, device=input_ids.device)
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]

                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]
                mb_input_ids = self._sanitize_scoring_token_ids(
                    mb_input_ids,
                    upper_bound=policy_vocab_upper_bound,
                    context="baseline_policy_input",
                )

                batch_logits = self.model(mb_input_ids, attention_mask=mb_att_mask)[
                    "logits"
                ]
                if args.temperature != 1:
                    batch_logits = batch_logits / args.temperature
                batch_logits = self._mask_invalid_scoring_logit_columns(
                    batch_logits,
                    valid_vocab_size=policy_vocab_upper_bound,
                    context="baseline_policy_logits",
                )
                if canonical_actions:
                    if canonical_task == "graph_coloring":
                        (
                            batch_logps,
                            _,
                            batch_full_logps,
                        ) = restricted_action_log_probs_entropy_and_distribution(
                            batch_logits,
                            mb_input_ids,
                            mb_response_masks,
                            allowed_token_ids=(self._canonical_action_token_ids or ()),
                        )
                        batch_support_mask = (
                            mb_response_masks.to(torch.bool)
                            .unsqueeze(-1)
                            .expand_as(batch_full_logps)
                        )
                    else:
                        (
                            batch_logps,
                            _,
                            batch_full_logps,
                            batch_support_mask,
                        ) = restricted_position_action_log_probs_entropy_and_distribution(
                            batch_logits,
                            mb_input_ids,
                            mb_response_masks,
                            allowed_token_ids_by_position=(
                                self._canonical_action_token_ids_by_position or ()
                            ),
                        )
                    assert canonical_learner_full_logps is not None
                    assert canonical_learner_support_mask is not None
                    canonical_learner_full_logps[
                        mini_batch_inds, : mb_last_valid_token_pos - 1
                    ] = batch_full_logps
                    canonical_learner_support_mask[
                        mini_batch_inds, : mb_last_valid_token_pos - 1
                    ] = batch_support_mask
                else:
                    batch_logps, _ = self._policy_logps_and_optional_entropy(
                        batch_logits,
                        mb_input_ids,
                        mb_response_masks,
                        need_entropy=False,
                    )
                logps[mini_batch_inds, : mb_last_valid_token_pos - 1] = batch_logps

        old_logps = logps
        if canonical_actions:
            canonical_support = tuple(self._canonical_action_token_ids or ())
            positional_supports = tuple(
                self._canonical_action_token_ids_by_position or ()
            )
            expected_horizon = (
                int(self._canonical_action_space.horizon)
                if self._canonical_action_space is not None
                else 0
            )
            if len(positional_supports) != expected_horizon:
                raise RuntimeError(
                    "canonical learner did not retain every positional support: "
                    f"expected={expected_horizon} observed={len(positional_supports)}"
                )
            behavior_support_mask = None
            if canonical_task == "graph_coloring":
                (
                    behavior_selected_logps,
                    behavior_full_logps,
                    behavior_norm_error,
                    behavior_selected_echo_diff,
                ) = materialize_canonical_behavior_policy(
                    input_ids[:, 1:],
                    response_masks,
                    action_ids=list(trajectory.get("action_ids") or []),
                    selected_log_probs=list(trajectory.get("action_logprobs") or []),
                    full_log_probs=list(
                        trajectory.get("canonical_behavior_action_logprobs") or []
                    ),
                    behavior_action_token_ids=list(
                        trajectory.get("canonical_behavior_action_token_ids") or []
                    ),
                    allowed_token_ids=canonical_support,
                    normalizer_atol=1e-6,
                )
            else:
                (
                    behavior_selected_logps,
                    behavior_full_logps,
                    behavior_support_mask,
                    behavior_norm_error,
                    behavior_selected_echo_diff,
                ) = materialize_position_canonical_behavior_policy(
                    input_ids[:, 1:],
                    response_masks,
                    action_ids=list(trajectory.get("action_ids") or []),
                    selected_log_probs=list(trajectory.get("action_logprobs") or []),
                    full_log_probs=list(
                        trajectory.get("canonical_behavior_action_logprobs") or []
                    ),
                    behavior_action_token_ids_by_position=list(
                        trajectory.get(
                            "canonical_behavior_action_token_ids_by_position"
                        )
                        or []
                    ),
                    allowed_token_ids_by_position=positional_supports,
                    normalizer_atol=1e-6,
                )
            if canonical_learner_full_logps is None:
                raise RuntimeError("canonical learner full policy trace is missing")
            if behavior_support_mask is not None and not torch.equal(
                behavior_support_mask, canonical_learner_support_mask
            ):
                raise RuntimeError(
                    "canonical behavior and learner positional supports disagree"
                )
            canonical_behavior_infos.update(
                canonical_behavior_overlap_diagnostics(
                    behavior_selected_logps,
                    behavior_full_logps,
                    logps,
                    canonical_learner_full_logps,
                    response_masks,
                    support_mask=behavior_support_mask,
                )
            )
            selected_diff = torch.abs(logps - behavior_selected_logps)[
                response_masks.to(torch.bool)
            ].max()
            canonical_behavior_infos.update(
                {
                    "canonical_behavior_denominator_actor": torch.tensor(
                        1.0, device=input_ids.device
                    ),
                    "canonical_behavior_q_row_count": response_masks.sum(),
                    "canonical_behavior_q_support_min": torch.tensor(
                        min(len(support) for support in positional_supports),
                        device=input_ids.device,
                    ),
                    "canonical_behavior_q_support_max": torch.tensor(
                        max(len(support) for support in positional_supports),
                        device=input_ids.device,
                    ),
                    "canonical_behavior_q_norm_error_max": torch.tensor(
                        behavior_norm_error,
                        dtype=torch.float32,
                        device=input_ids.device,
                    ),
                    "canonical_behavior_selected_echo_diff_max": torch.tensor(
                        behavior_selected_echo_diff,
                        dtype=torch.float32,
                        device=input_ids.device,
                    ),
                    # Retain the old name as a diagnostic only; the gate now
                    # checks full behavior overlap rather than equality.
                    "canonical_actor_logp_diff_max": selected_diff,
                }
            )
            old_logps = behavior_selected_logps
            logging.info(
                "canonical behavior overlap: ratio=[%.6f, %.6f] tv_max=%.6f "
                "kl_actor_learner_max=%.6f kl_learner_actor_max=%.6f "
                "sequence_ess=%.6f prefix_ess_min=%.6f selected_diff=%.6f",
                float(canonical_behavior_infos["canonical_behavior_ratio_min"]),
                float(canonical_behavior_infos["canonical_behavior_ratio_max"]),
                float(canonical_behavior_infos["canonical_behavior_tv_max"]),
                float(
                    canonical_behavior_infos["canonical_behavior_kl_actor_learner_max"]
                ),
                float(
                    canonical_behavior_infos["canonical_behavior_kl_learner_actor_max"]
                ),
                float(
                    canonical_behavior_infos["canonical_behavior_sequence_ess_fraction"]
                ),
                float(
                    canonical_behavior_infos[
                        "canonical_behavior_prefix_ess_fraction_min"
                    ]
                ),
                float(selected_diff),
            )

        if self.ref_model is not None:
            all_ref_logps = []
            ref_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(
                self.ref_model
            )
            with torch.no_grad():
                for i in range(0, len(input_ids), args.train_batch_size_per_device):
                    batch_end = min(
                        i + args.train_batch_size_per_device, len(input_ids)
                    )
                    batch_inds = torch.arange(i, batch_end, device=input_ids.device)
                    batch_input_ids = self._sanitize_scoring_token_ids(
                        input_ids[batch_inds],
                        upper_bound=ref_vocab_upper_bound,
                        context="baseline_reference_input",
                    )

                    batch_ref_logits = self.ref_model(
                        batch_input_ids, attention_mask=att_mask[batch_inds]
                    )["logits"]
                    if args.temperature != 1:
                        batch_ref_logits = batch_ref_logits / args.temperature
                    batch_ref_logits = self._mask_invalid_scoring_logit_columns(
                        batch_ref_logits,
                        valid_vocab_size=ref_vocab_upper_bound,
                        context="baseline_reference_logits",
                    )
                    batch_ref_logps, _ = self._policy_logps_and_optional_entropy(
                        batch_ref_logits,
                        batch_input_ids,
                        response_masks[batch_inds],
                        need_entropy=False,
                    )
                    all_ref_logps.append(batch_ref_logps)
            ref_logps = torch.cat(all_ref_logps)

            kl_rewards = -args.kl_penalty_coef * (logps - ref_logps) * response_masks
            rewards = kl_rewards.clone()
            del all_ref_logps
            torch.cuda.empty_cache()
            gc.collect()
        else:
            ref_logps = None
            rewards = torch.zeros_like(response_masks).float()

        rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()

        if self.args.critic_type == "ppo":
            advantages, returns, values = self.compute_ppo_advantages(
                rewards, input_ids, att_mask, response_masks
            )
        elif self.args.critic_type in ["grpo", "drgrpo"]:
            advantages = self.compute_monte_carlo_advantages(rewards, response_masks)[
                :, None
            ]
        if rlep_step_count:
            mixed = rlep_mixed_advantages(
                task_final_rewards,
                replay_count=rlep_step_count,
            )
            advantages = mixed.fresh.to(
                dtype=advantages.dtype,
                device=advantages.device,
            )
            rlep_replay_advantage = mixed.replay_advantage
            rlep_infos.update(
                {
                    "rlep_fresh_rows": torch.tensor(
                        mixed.fresh_count, device=final_rewards.device
                    ),
                    "rlep_mixed_rows": torch.tensor(
                        mixed.fresh_count + mixed.replay_count,
                        device=final_rewards.device,
                    ),
                    "rlep_mixed_reward_mean": torch.tensor(
                        mixed.mixed_reward_mean, device=final_rewards.device
                    ),
                }
            )
        ucpo_tau = float(getattr(args, "ucpo_tau", 0.0) or 0.0)
        ucpo_infos: dict[str, torch.Tensor] = {}
        if ucpo_tau > 0.0:
            sequence_log_probs = (old_logps * response_masks.float()).sum(dim=1)
            advantages, ucpo_diagnostics = redistribute_ucpo_advantages(
                advantages,
                sequence_log_probs,
                task_final_rewards,
                loss_masks,
                num_samples=int(args.num_samples),
                tau=ucpo_tau,
            )
            ucpo_infos = {
                "ucpo_tau": torch.tensor(ucpo_tau, device=final_rewards.device),
                "ucpo_groups": torch.tensor(
                    ucpo_diagnostics.groups, device=final_rewards.device
                ),
                "ucpo_eligible_groups": torch.tensor(
                    ucpo_diagnostics.eligible_groups, device=final_rewards.device
                ),
                "ucpo_correct_rows": torch.tensor(
                    ucpo_diagnostics.correct_rows, device=final_rewards.device
                ),
                "ucpo_weight_min": torch.tensor(
                    ucpo_diagnostics.weight_min, device=final_rewards.device
                ),
                "ucpo_weight_max": torch.tensor(
                    ucpo_diagnostics.weight_max, device=final_rewards.device
                ),
                "ucpo_advantage_mass_error_max": torch.tensor(
                    ucpo_diagnostics.mass_error_max, device=final_rewards.device
                ),
            }
        setpo_coefficient = float(getattr(args, "setpo_coefficient", 0.0) or 0.0)
        setpo_infos: dict[str, torch.Tensor] = {}
        if setpo_coefficient > 0.0:
            # SetPO adds each row's leave-one-out marginal contribution to the
            # group's kernelized set diversity. The embedder is frozen and the
            # term is detached, so the kernel shapes the advantage without ever
            # receiving a gradient.
            setpo_texts = self._setpo_response_surfaces(
                input_ids, response_masks
            )
            setpo_embeddings = self._setpo_embedder().embed(setpo_texts)
            advantages, setpo_diagnostics = shape_setpo_advantages(
                advantages,
                setpo_embeddings.to(advantages.device),
                num_samples=int(args.num_samples),
                coefficient=setpo_coefficient,
            )
            setpo_infos = {
                f"setpo_{name}": torch.tensor(float(value), device=device)
                for name, value in vars(setpo_diagnostics).items()
            }
            setpo_infos["setpo_coefficient"] = torch.tensor(
                setpo_coefficient, device=device
            )

        # Freeze the ordinary task-centered advantage before any separately
        # added semantic term. E44 uses this only to form detached xDr row
        # weights, while the actor below still receives the combined advantage.
        task_advantages_for_xdr = advantages.detach()
        if outcome_collision_outside_advantage is not None:
            base_advantages = advantages.detach()
            semantic_advantages = outcome_collision_outside_advantage
            advantages = add_outcome_collision_outside_centering_advantage(
                advantages,
                semantic_advantages,
            )
            combined_advantages = advantages.detach()
            base_advantage_rms = torch.sqrt(base_advantages.square().mean())
            semantic_advantage_rms = torch.sqrt(semantic_advantages.square().mean())
            combined_advantage_rms = torch.sqrt(combined_advantages.square().mean())
            outcome_collision_infos.update(
                {
                    "outcome_collision_outside_base_advantage_mean": (
                        base_advantages.mean()
                    ),
                    "outcome_collision_outside_base_advantage_abs_mean": (
                        base_advantages.abs().mean()
                    ),
                    "outcome_collision_outside_base_advantage_rms": (
                        base_advantage_rms
                    ),
                    "outcome_collision_outside_base_advantage_nonzero_fraction": (
                        base_advantages.ne(0).float().mean()
                    ),
                    "outcome_collision_outside_semantic_advantage_mean": (
                        semantic_advantages.mean()
                    ),
                    "outcome_collision_outside_semantic_advantage_min": (
                        semantic_advantages.min()
                    ),
                    "outcome_collision_outside_semantic_advantage_max": (
                        semantic_advantages.max()
                    ),
                    "outcome_collision_outside_semantic_advantage_abs_mean": (
                        semantic_advantages.abs().mean()
                    ),
                    "outcome_collision_outside_semantic_advantage_rms": (
                        semantic_advantage_rms
                    ),
                    "outcome_collision_outside_semantic_advantage_nonzero_fraction": (
                        semantic_advantages.ne(0).float().mean()
                    ),
                    "outcome_collision_outside_combined_advantage_mean": (
                        combined_advantages.mean()
                    ),
                    "outcome_collision_outside_combined_advantage_abs_mean": (
                        combined_advantages.abs().mean()
                    ),
                    "outcome_collision_outside_combined_advantage_rms": (
                        combined_advantage_rms
                    ),
                    "outcome_collision_outside_combined_advantage_nonzero_fraction": (
                        combined_advantages.ne(0).float().mean()
                    ),
                }
            )
        if semantic_shannon_separate_advantage is not None:
            base_advantages = advantages.detach()
            semantic_advantages = semantic_shannon_separate_advantage
            advantages = add_semantic_shannon_separate_advantage(
                advantages,
                semantic_advantages,
            )
            combined_advantages = advantages.detach()

            # Adaptive semantic MaxEnt observes the dose it just applied and
            # sets the coefficient for the next update. It reads only the two
            # advantage magnitudes and the eligible fraction, never evaluation
            # behavior, and refuses a degenerate observation instead of
            # extrapolating from it.
            rms_controller = getattr(self, "_semantic_rms_controller", None)
            if rms_controller is not None:
                task_rms = float(torch.sqrt(base_advantages.square().mean()).item())
                sem_rms = float(torch.sqrt(semantic_advantages.square().mean()).item())
                eligible = (
                    float(success_conditioned_signed_diagnostics.eligible_fraction)
                    if success_conditioned_signed_diagnostics is not None
                    else 0.0
                )
                coefficient = rms_controller.observe(
                    semantic_rms=sem_rms,
                    task_rms=task_rms,
                    eligible_fraction=eligible,
                )
                semantic_shannon_tracker.coefficient = float(coefficient)
                semantic_shannon_infos.update(
                    {
                        key: torch.tensor(value, device=final_rewards.device)
                        for key, value in rms_controller.diagnostics().items()
                    }
                )
            semantic_shannon_infos.update(
                {
                    "semantic_shannon_separate_base_advantage_mean": (
                        base_advantages.mean()
                    ),
                    "semantic_shannon_separate_base_advantage_abs_mean": (
                        base_advantages.abs().mean()
                    ),
                    "semantic_shannon_separate_base_advantage_rms": torch.sqrt(
                        base_advantages.square().mean()
                    ),
                    "semantic_shannon_separate_base_advantage_nonzero_fraction": (
                        base_advantages.ne(0).float().mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_mean": (
                        semantic_advantages.mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_min": (
                        semantic_advantages.min()
                    ),
                    "semantic_shannon_separate_semantic_advantage_max": (
                        semantic_advantages.max()
                    ),
                    "semantic_shannon_separate_semantic_advantage_abs_mean": (
                        semantic_advantages.abs().mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_rms": (
                        torch.sqrt(semantic_advantages.square().mean())
                    ),
                    "semantic_shannon_separate_semantic_advantage_positive_fraction": (
                        semantic_advantages.gt(0).float().mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_negative_fraction": (
                        semantic_advantages.lt(0).float().mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_zero_fraction": (
                        semantic_advantages.eq(0).float().mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_nonzero_fraction": (
                        semantic_advantages.ne(0).float().mean()
                    ),
                    "semantic_shannon_separate_combined_advantage_mean": (
                        combined_advantages.mean()
                    ),
                    "semantic_shannon_separate_combined_advantage_abs_mean": (
                        combined_advantages.abs().mean()
                    ),
                    "semantic_shannon_separate_combined_advantage_rms": torch.sqrt(
                        combined_advantages.square().mean()
                    ),
                    "semantic_shannon_separate_combined_advantage_nonzero_fraction": (
                        combined_advantages.ne(0).float().mean()
                    ),
                }
            )
        if online_canonical_advantage is not None:
            base_advantages = advantages.detach()
            advantages = advantages + online_canonical_advantage
            combined_advantages = advantages.detach()
            online_canonical_infos.update(
                {
                    "online_canonical_separate_base_advantage_mean": (
                        base_advantages.mean()
                    ),
                    "online_canonical_separate_base_advantage_rms": torch.sqrt(
                        base_advantages.square().mean()
                    ),
                    "online_canonical_separate_combined_advantage_mean": (
                        combined_advantages.mean()
                    ),
                    "online_canonical_separate_combined_advantage_rms": (
                        torch.sqrt(combined_advantages.square().mean())
                    ),
                    "online_canonical_advantage_applied_after_task_centering": (
                        torch.tensor(1.0, device=final_rewards.device)
                    ),
                }
            )
        row_weights = None
        extra_infos: dict[str, torch.Tensor] = {
            **canonical_behavior_infos,
            **diayn_infos,
            **outcome_collision_infos,
            **semantic_shannon_infos,
            **online_canonical_infos,
            **ucpo_infos,
            **gapo_infos,
            **setpo_infos,
            **rlep_infos,
            **dapo_infos,
        }
        configured_xdr_tau = float(getattr(args, "xdr_tau", math.inf))
        tau_controller = getattr(self, "_xdr_tau_controller", None)
        xdr_tau = (
            float(tau_controller.current_tau)
            if tau_controller is not None
            else configured_xdr_tau
        )
        seed_alpha = float(getattr(args, "seed_entropy_alpha", 0.0) or 0.0)
        if math.isfinite(xdr_tau) and self.args.critic_type == "drgrpo":
            # xDr.GRPO: per-candidate Dr.GRPO utilities at the rollout policy
            # (ratio=1): U_i = A_i * T_i / T_max. Weights are computed once per
            # rollout batch and frozen for the update, like the advantages.
            extra_infos["xdr_tau_used"] = torch.tensor(
                xdr_tau, dtype=torch.float32, device=final_rewards.device
            )
            per_group_tau = None
            if bool(getattr(args, "xdr_mode_adaptive", False)):
                # Mode-adaptive tempering: tau_x = tau0 / log(1 + kappa_x),
                # with kappa_x the number of distinct canonical answer modes
                # observed among the group's correct candidates. Prompts where
                # more modes are already in play get sharper attenuation of
                # negative-advantage gradients (they have the most to lose).
                num_rows = int(input_ids.size(0))
                refs = list(trajectory.get("references") or [])
                refs = (refs + [None] * num_rows)[:num_rows]
                refs_grouped = [
                    refs[i : i + args.num_samples]
                    for i in range(0, num_rows, args.num_samples)
                ]
                keys_grouped = self._seed_answer_keys_grouped(
                    input_ids, response_masks, args.num_samples, refs_grouped
                )
                correct = final_rewards.detach().reshape(-1, args.num_samples) > 0
                kappas = []
                for g, group_keys in enumerate(keys_grouped):
                    modes = {
                        key if key is not None else ("__u__", g, i)
                        for i, key in enumerate(group_keys)
                        if bool(correct[g, i])
                    }
                    kappas.append(max(len(modes), 1))
                per_group_tau = configured_xdr_tau / torch.log1p(
                    torch.tensor(
                        kappas, dtype=torch.float32, device=final_rewards.device
                    )
                )
                extra_infos["xdr_adaptive_tau_mean"] = per_group_tau.mean().detach()
            xdr_task_advantage_weights = bool(
                getattr(args, "xdr_task_advantage_weights", False)
            )
            xdr_weight_advantages = (
                task_advantages_for_xdr if xdr_task_advantage_weights else advantages
            )
            extra_infos["xdr_task_advantage_weights_active"] = torch.tensor(
                float(xdr_task_advantage_weights),
                dtype=torch.float32,
                device=final_rewards.device,
            )
            extra_infos[
                "xdr_weight_advantage_mean"
            ] = xdr_weight_advantages.detach().mean()
            extra_infos["xdr_weight_advantage_rms"] = torch.sqrt(
                xdr_weight_advantages.detach().square().mean()
            )
            row_weights = compute_xdr_row_weights(
                xdr_weight_advantages,
                response_masks.sum(dim=1),
                num_samples=args.num_samples,
                tau=xdr_tau,
                t_max=int(args.generate_max_length),
                loss_masks=loss_masks,
                per_group_tau=per_group_tau,
            )
        elif seed_alpha > 0 and self.args.critic_type == "drgrpo":
            # SEED-Dr.GRPO: per-prompt semantic-entropy scaling. Cluster the
            # group by canonical final answer, weight clusters by the policy's
            # length-normalized sequence likelihood, and scale the prompt's
            # rows by (1 + (alpha/log G) * H_sem)^{-1}. The per-row references
            # must reach the clusterer so modebench answers (countdown
            # expressions, colorings) reduce to canonical mode keys rather
            # than surface forms.
            num_rows = int(input_ids.size(0))
            references = list(trajectory.get("references") or [])
            references = (references + [None] * num_rows)[:num_rows]
            references_grouped = [
                references[i : i + args.num_samples]
                for i in range(0, num_rows, args.num_samples)
            ]
            seq_logp_sums = (logps * response_masks.float()).sum(dim=1)
            token_counts_raw = response_masks.sum(dim=1)
            token_counts = token_counts_raw.clamp(min=1).float()
            answer_keys_grouped = self._seed_answer_keys_grouped(
                input_ids,
                response_masks,
                args.num_samples,
                references_grouped,
            )
            answer_keys = [key for group in answer_keys_grouped for key in group]
            # Rows with no response tokens would otherwise take the maximal
            # normalized logp of exactly 0; exclude them from the cluster
            # softmax alongside loss-masked rows.
            effective_masks = loss_masks * (token_counts_raw > 0).float()
            row_weights = compute_seed_row_weights(
                seq_logp_sums / token_counts,
                answer_keys,
                num_samples=args.num_samples,
                alpha=seed_alpha,
                loss_masks=effective_masks,
            )
            extra_infos["seed_prompt_scale_mean"] = (
                row_weights.view(-1, args.num_samples)[:, 0].mean().detach()
            )
        if self.args.critic_type in ("grpo", "drgrpo"):
            # Aggregation diagnostics (effective active rollouts, incorrect-
            # mass share) are defined identically for every quartet arm from
            # the realized per-row aggregation weights: uniform for Dr.GRPO
            # and Token-MaxEnt, prompt-rescaled uniform for SEED, tempered
            # softmax for xDr.GRPO. Masking by loss_masks restricts each
            # group's distribution to the rows that actually train (a no-op
            # for xdr weights, which already zero masked rows).
            diag_weights = (
                row_weights if row_weights is not None else torch.ones_like(loss_masks)
            ) * loss_masks
            extra_infos.update(
                aggregation_group_diagnostics(
                    diag_weights, final_rewards, num_samples=args.num_samples
                )
            )
        return self._baseline_update_with_precomputed_advantages(
            input_ids=input_ids,
            att_mask=att_mask,
            prompt_id_lens=prompt_id_lens,
            loss_masks=loss_masks,
            response_masks=response_masks,
            logps=old_logps,
            ref_logps=ref_logps,
            advantages=advantages,
            final_rewards=task_final_rewards,
            returns=returns if self.args.critic_type == "ppo" else None,
            values=values if self.args.critic_type == "ppo" else None,
            policy_vocab_upper_bound=policy_vocab_upper_bound,
            row_weights=row_weights,
            extra_infos=extra_infos,
            canonical_replay_groups=canonical_replay_groups,
            rlep_replay_groups=rlep_replay_groups,
            rlep_replay_advantage=rlep_replay_advantage,
        )

    # Dr. GRPO Modification 2: remove difficulty bias by computing the MC
    # advantage without dividing by std, except for standard GRPO compatibility.
    def compute_monte_carlo_advantages(
        self,
        rewards: torch.Tensor,
        response_masks=None,
    ) -> torch.Tensor:
        del response_masks
        rewards = rewards.sum(-1)
        grouped_rewards = rewards.view(-1, self.args.num_samples)
        if bool(getattr(self.args, "maxrl_task_objective", False)):
            return binary_maxrl_advantages(grouped_rewards).reshape(-1)
        values = grouped_rewards.mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if getattr(self.args, "critic_type", "grpo") == "grpo":
            std_grouped_rewards = grouped_rewards.std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples,
                dim=0,
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
        return advantages
