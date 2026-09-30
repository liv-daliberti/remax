"""Fresh clipped-policy loss and the OAT optimizer boundary."""

from __future__ import annotations
import hashlib
import logging
import math
import time
from collections import defaultdict
from typing import Any
import numpy as np
import torch
import torch.distributed as dist
from oat.utils.ops import masked_mean
from ...args import resolve_canonical_action_task
from ...core.bank_types import VerifiedCanonicalReplayGroup
from ...replicated_group import (
    replicated_group_permutation_seed,
    validate_replicated_group_layout,
)
from ...tensor_utils import cap_last_valid_token_pos_for_zero_advantage


class OatUpdateMixin:

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
        canonical_replay_groups: list[VerifiedCanonicalReplayGroup] | None = None,
        rlep_replay_groups: list[Any] | None = None,
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
        from ...core.execution import validate_update_partition, replay_contract_digest

        actual_world_size = dist.get_world_size() if dist.is_initialized() else 1
        if actual_world_size > 1:
            signature = replay_contract_digest(
                groups=canonical_replay_groups or [],
                settings={
                    "replicated": replicated_group,
                    "samples": args.num_samples,
                    "batch": args.train_batch_size,
                    "microbatch": args.train_batch_size_per_device,
                    "accumulation": self.strategy.grad_acc_step,
                    "alpha": args.online_canonical_replay_alpha,
                    "objective": args.online_canonical_replay_objective,
                    "control": args.online_canonical_replay_compute_only,
                    "temperature": args.temperature,
                    "epochs": args.num_ppo_epochs,
                },
                tensors=(input_ids, att_mask, response_masks, loss_masks, advantages, logps),
            )
            signatures = [None] * actual_world_size
            dist.all_gather_object(signatures, signature)
            if len(set(signatures)) != 1:
                raise RuntimeError("replicated learner inputs, replay groups or execution settings differ across ranks")
            if not replicated_group:
                raise RuntimeError("maintained multi-rank updates require replicated candidate groups")
        learner_world_size = actual_world_size if replicated_group else 1
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
                    "replicated update accumulation width does not match the exact logical candidate group"
                )
        else:
            local_candidate_count = len(input_ids)
        validate_update_partition(
            local_rows=local_candidate_count,
            microbatch=args.train_batch_size_per_device,
            accumulation=self.strategy.grad_acc_step,
            global_batch=args.train_batch_size,
            world_size=actual_world_size,
        )
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
                    and (self.args.critic_type in ["grpo", "drgrpo"])
                    and (len(mb_advantage) == 1)
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
                logits = self.model(mb_input_ids, attention_mask=mb_att_mask)["logits"]
                if args.temperature != 1:
                    logits = logits / args.temperature
                logits = self._mask_invalid_scoring_logit_columns(
                    logits,
                    valid_vocab_size=policy_vocab_upper_bound,
                    context="baseline_policy_update_logits",
                )
                (new_logps, policy_token_entropy) = (
                    self._policy_logps_and_optional_entropy(
                        logits, mb_input_ids, mb_response_masks, need_entropy=True
                    )
                )
                logprobs_diff = new_logps - mb_logps
                ratio = torch.exp(logprobs_diff)
                clip_low = float(args.cliprange)
                clip_high = float(args.cliprange)
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
                base_pg_loss = self.masked_aggregator(
                    pg_loss_max, mb_response_masks, axis=1
                )
                base_pg_loss = (base_pg_loss * mb_loss_masks).mean()
                pg_loss = base_pg_loss
                infos["pg_loss"] = pg_loss.detach()
                loss = pg_loss
                maxent_alpha = float(getattr(args, "maxent_alpha", 0.0) or 0.0)
                token_entropy = policy_token_entropy
                with torch.no_grad():
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
                            or canonical_sequence_entropy_value < -1e-06
                            or canonical_sequence_entropy_value
                            > canonical_max_entropy + 1e-05
                        ):
                            raise RuntimeError(
                                f"canonical sampled-prefix entropy left its support bound: {canonical_sequence_entropy_value}"
                            )
                        infos["canonical_token_entropy_mean"] = entropy
                        infos["canonical_sampled_prefix_entropy_sum"] = (
                            canonical_sequence_entropy
                        )
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
                # Replay supplements accumulated fresh gradients before the step.
                if (
                    canonical_replay_groups
                    and local_grad_step % self.strategy.grad_acc_step == 0
                ):
                    self._backward_verified_replay(
                        args=args,
                        canonical_replay_groups=canonical_replay_groups,
                        fresh_advantage_fingerprint=fresh_advantage_fingerprint,
                        infos=infos,
                        input_ids=input_ids,
                        policy_vocab_upper_bound=policy_vocab_upper_bound,
                        stats=stats,
                    )
                if local_grad_step % self.strategy.grad_acc_step == 0:
                    if self._should_skip_baseline_grad_norm_logging():
                        if not self._baseline_grad_norm_logging_disabled_warned:
                            logging.warning(
                                "Skipping baseline policy_grad_norm logging for the ZeRO-3/offload slow path on node-local 7B runs."
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
