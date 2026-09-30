"""Two-pass teacher-forced replay and exact accumulation compensation."""

from __future__ import annotations
from .telemetry import record_replay_metrics
from ...core.objectives import canonical_replay_uniform_verified_likelihood_loss
from contextlib import contextmanager
import torch


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


class OatReplayMixin:

    def _backward_verified_replay(
        self,
        *,
        args,
        canonical_replay_groups,
        fresh_advantage_fingerprint,
        infos,
        input_ids,
        policy_vocab_upper_bound,
        stats,
    ):
        replay = self._materialize_canonical_replay(
            canonical_replay_groups, device=input_ids.device
        )
        replay_row_count = int(replay.input_ids.size(0))
        replay_chunk_size = max(1, int(args.train_batch_size_per_device))
        # Both passes use the same eval-mode model snapshot; dropout must agree.
        with _temporary_eval_mode(self.model, enabled=True):
            with torch.no_grad():
                detached_scores = torch.cat(
                    [
                        self._score_canonical_replay_rows(
                            replay,
                            start=start,
                            stop=min(start + replay_chunk_size, replay_row_count),
                            policy_vocab_upper_bound=policy_vocab_upper_bound,
                        ).detach()
                        for start in range(0, replay_row_count, replay_chunk_size)
                    ]
                )
            replay_objective = str(args.online_canonical_replay_objective)
            replay_alpha = float(args.online_canonical_replay_alpha)
            replay_banked_modes = int(sum(replay.group_sizes))
            replay_key_weighting = str(
                getattr(args, "online_canonical_replay_key_weighting", "uniform")
            )
            replay_target_weights = replay.mass_weights
            replay_bank_normalized = bool(
                getattr(args, "online_canonical_replay_bank_normalized", False)
            )
            replay_mass_alpha = 0.0
            replay_result = canonical_replay_uniform_verified_likelihood_loss(
                detached_scores, replay.group_sizes, replay_target_weights
            )
            actuator_loss = replay_result.loss
            balance_loss = replay_result.cross_entropy_excess
            normalized_entropy = replay_result.normalized_entropy
            if not all(
                (
                    bool(torch.isfinite(value))
                    for value in (actuator_loss, balance_loss, normalized_entropy)
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
                in {"verified_likelihood_per_rollout", "split_mass_balance_per_rollout"}
                else 1.0
            )
            weighted_replay_objective = actuator_loss * replay_alpha
            replay_weighted_loss = (
                weighted_replay_objective
                * reward_estimator_scale
                * replay_objective_scale
            )
            if not bool(torch.isfinite(replay_weighted_loss)):
                raise RuntimeError(
                    "fixed canonical replay coefficient produced a non-finite weighted loss"
                )
            requested_score_gradients = (
                (replay_result.score_gradients * replay_alpha)
                .to(input_ids.device)
                .detach()
            )
            retention_safe_balance = bool(
                getattr(args, "online_canonical_replay_retention_safe_balance", False)
            )
            balance_scales = detached_scores.new_ones(len(replay.group_sizes))
            raw_score_gradients = requested_score_gradients
            replay_compute_only = bool(
                getattr(args, "online_canonical_replay_compute_only", False)
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
            # OAT divides each backward by grad_acc_step. Replay runs once per
            # optimizer boundary, so compensate here, after the registered N scales.
            replay_backward_scale = (
                reward_estimator_scale
                * replay_objective_scale
                * float(self.strategy.grad_acc_step)
            )
            # Controls execute this live pass even with exactly zero score gradients.
            for start in range(0, replay_row_count, replay_chunk_size):
                stop = min(start + replay_chunk_size, replay_row_count)
                live_scores = self._score_canonical_replay_rows(
                    replay,
                    start=start,
                    stop=stop,
                    policy_vocab_upper_bound=policy_vocab_upper_bound,
                )
                exact_gradient_surrogate = (
                    live_scores * score_gradients[start:stop]
                ).sum()
                replay_backward_loss = exact_gradient_surrogate * replay_backward_scale
                if not bool(torch.isfinite(replay_backward_loss)):
                    raise RuntimeError(
                        "canonical replay produced a non-finite chunked backward scalar"
                    )
                self.strategy.backward(replay_backward_loss, self.model, self.optimizer)
        record_replay_metrics(
            actuator_loss=actuator_loss,
            balance_loss=balance_loss,
            balance_scales=balance_scales,
            canonical_replay_groups=canonical_replay_groups,
            detached_scores=detached_scores,
            fresh_advantage_fingerprint=fresh_advantage_fingerprint,
            infos=infos,
            input_ids=input_ids,
            normalized_entropy=normalized_entropy,
            raw_score_gradients=raw_score_gradients,
            replay=replay,
            replay_alpha=replay_alpha,
            replay_applied_weighted_loss=replay_applied_weighted_loss,
            replay_bank_normalized=replay_bank_normalized,
            replay_banked_modes=replay_banked_modes,
            replay_chunk_size=replay_chunk_size,
            replay_compute_only=replay_compute_only,
            replay_key_weighting=replay_key_weighting,
            replay_mass_alpha=replay_mass_alpha,
            replay_objective=replay_objective,
            replay_objective_scale=replay_objective_scale,
            replay_result=replay_result,
            replay_target_weights=replay_target_weights,
            replay_weighted_loss=replay_weighted_loss,
            requested_score_gradients=requested_score_gradients,
            retention_safe_balance=retention_safe_balance,
            reward_estimator_scale=reward_estimator_scale,
            score_gradients=score_gradients,
            self=self,
            stats=stats,
        )
        return None
