"""Maintained admission → scoring → fresh advantage → optimizer update flow."""

from __future__ import annotations
import logging
import torch
from ...benchmark import require_scorable, failure
from ...args import resolve_canonical_action_task
from ...core.objectives import binary_maxrl_advantages
from ...core.bank_types import VerifiedCanonicalReplayGroup
from ...core.metrics import aggregation_group_diagnostics

from .admission import OatAdmissionMixin
from .scoring import OatScoringMixin
from .replay import OatReplayMixin
from .update import OatUpdateMixin


class ZeroMathGrpoMixin(
    OatAdmissionMixin, OatScoringMixin, OatReplayMixin, OatUpdateMixin
):

    def _use_instrumented_grpo_learning_step(self) -> bool:
        return int(getattr(self.args, "zero_stage", 0) or 0) >= 3 or bool(
            getattr(self.args, "adam_offload", False)
        )

    def _should_skip_baseline_grad_norm_logging(self) -> bool:
        return self._use_instrumented_grpo_learning_step()

    def _baseline_progress_log_interval(self, total_micro_batches: int) -> int:
        if total_micro_batches <= 0:
            return 1
        return max(1, total_micro_batches // 8)

    def _baseline_should_log_progress(
        self, local_grad_step: int, total_micro_batches: int
    ) -> bool:
        if not self.strategy.is_rank_0():
            return False
        if local_grad_step <= 1 or local_grad_step >= total_micro_batches:
            return True
        interval = self._baseline_progress_log_interval(total_micro_batches)
        return local_grad_step % interval == 0

    def _grpo_learning_step_with_progress(self, trajectory):
        """Execute one verified rollout group for Re:Dr/Re:Max or a control."""
        # Fatal diagnostics precede every bank mutation and optimizer action.
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
            .float()
            * args.reward_scale
        )
        task_final_rewards = final_rewards.detach().clone()
        raw_task_final_rewards = task_final_rewards.detach().clone()
        prompt_id_lens = trajectory["prompt_ids_lens"]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]
        online_canonical_infos: dict[str, torch.Tensor] = {}
        canonical_replay_groups: list[VerifiedCanonicalReplayGroup] = []
        canonical_behavior_infos: dict[str, torch.Tensor] = {}
        if canonical_actions:
            if self._canonical_action_space is None:
                raise RuntimeError("canonical learner action space is missing")
            expected_count = int(self._canonical_action_space.horizon)
            observed_counts = response_masks.sum(dim=1)
            if not bool(observed_counts.eq(expected_count).all()):
                raise RuntimeError(
                    f"canonical rows must contain exactly {expected_count} actions; got {observed_counts.detach().cpu().tolist()}"
                )
        logging.info(f"learn data size {input_ids.shape}")
        online_canonical_bank = getattr(self, "_online_canonical_bank", None)
        (canonical_replay_groups, online_canonical_infos) = (
            self._admit_and_schedule_replay(
                args=args,
                canonical_actions=canonical_actions,
                canonical_replay_groups=canonical_replay_groups,
                canonical_task=canonical_task,
                final_rewards=final_rewards,
                input_ids=input_ids,
                loss_masks=loss_masks,
                online_canonical_bank=online_canonical_bank,
                online_canonical_infos=online_canonical_infos,
                prompt_id_lens=prompt_id_lens,
                raw_task_final_rewards=raw_task_final_rewards,
                response_masks=response_masks,
                task_final_rewards=task_final_rewards,
                trajectory=trajectory,
            )
        )
        (eos_indices, old_logps, policy_vocab_upper_bound, ref_logps, rewards) = (
            self._score_rollout_policy(
                args=args,
                att_mask=att_mask,
                canonical_actions=canonical_actions,
                canonical_behavior_infos=canonical_behavior_infos,
                canonical_task=canonical_task,
                input_ids=input_ids,
                response_masks=response_masks,
                trajectory=trajectory,
            )
        )
        rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()
        if self.args.critic_type in ["grpo", "drgrpo"]:
            advantages = self.compute_monte_carlo_advantages(rewards, response_masks)[
                :, None
            ]
        row_weights = None
        extra_infos: dict[str, torch.Tensor] = {
            **canonical_behavior_infos,
            **online_canonical_infos,
        }
        if self.args.critic_type in ("grpo", "drgrpo"):
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
            returns=None,
            values=None,
            policy_vocab_upper_bound=policy_vocab_upper_bound,
            row_weights=row_weights,
            extra_infos=extra_infos,
            canonical_replay_groups=canonical_replay_groups,
        )

    def compute_monte_carlo_advantages(
        self, rewards: torch.Tensor, response_masks=None
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
                self.args.num_samples, dim=0
            )
            advantages = advantages / (std_grouped_rewards + 1e-08)
        return advantages
