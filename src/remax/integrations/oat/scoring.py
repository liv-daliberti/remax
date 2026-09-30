"""Rollout behavior-policy checks and length-normalized replay scores."""

from __future__ import annotations
from ...core.replay_types import CanonicalReplayBatch
from ...core.scoring import materialize_canonical_replay_batch
import logging
from contextlib import contextmanager
import torch
from ...canonical_actions import (
    canonical_behavior_overlap_diagnostics,
    materialize_canonical_behavior_policy,
    materialize_position_canonical_behavior_policy,
    restricted_action_log_probs_entropy_and_distribution,
    restricted_position_action_log_probs_entropy_and_distribution,
)
from ...core.bank_types import VerifiedCanonicalReplayGroup


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


class OatScoringMixin:

    def _score_rollout_policy(
        self,
        *,
        args,
        att_mask,
        canonical_actions,
        canonical_behavior_infos,
        canonical_task,
        input_ids,
        response_masks,
        trajectory,
    ):
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
                        (batch_logps, _, batch_full_logps) = (
                            restricted_action_log_probs_entropy_and_distribution(
                                batch_logits,
                                mb_input_ids,
                                mb_response_masks,
                                allowed_token_ids=self._canonical_action_token_ids
                                or (),
                            )
                        )
                        batch_support_mask = (
                            mb_response_masks.to(torch.bool)
                            .unsqueeze(-1)
                            .expand_as(batch_full_logps)
                        )
                    else:
                        (batch_logps, _, batch_full_logps, batch_support_mask) = (
                            restricted_position_action_log_probs_entropy_and_distribution(
                                batch_logits,
                                mb_input_ids,
                                mb_response_masks,
                                allowed_token_ids_by_position=self._canonical_action_token_ids_by_position
                                or (),
                            )
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
                    (batch_logps, _) = self._policy_logps_and_optional_entropy(
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
                    f"canonical learner did not retain every positional support: expected={expected_horizon} observed={len(positional_supports)}"
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
                    normalizer_atol=1e-06,
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
                    normalizer_atol=1e-06,
                )
            if canonical_learner_full_logps is None:
                raise RuntimeError("canonical learner full policy trace is missing")
            if behavior_support_mask is not None and (
                not torch.equal(behavior_support_mask, canonical_learner_support_mask)
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
                        min((len(support) for support in positional_supports)),
                        device=input_ids.device,
                    ),
                    "canonical_behavior_q_support_max": torch.tensor(
                        max((len(support) for support in positional_supports)),
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
                    "canonical_actor_logp_diff_max": selected_diff,
                }
            )
            old_logps = behavior_selected_logps
            logging.info(
                "canonical behavior overlap: ratio=[%.6f, %.6f] tv_max=%.6f kl_actor_learner_max=%.6f kl_learner_actor_max=%.6f sequence_ess=%.6f prefix_ess_min=%.6f selected_diff=%.6f",
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
        ref_logps = None
        rewards = torch.zeros_like(response_masks).float()
        return (eos_indices, old_logps, policy_vocab_upper_bound, ref_logps, rewards)

    def _materialize_canonical_replay(
        self, groups: list[VerifiedCanonicalReplayGroup], *, device: torch.device | int
    ) -> CanonicalReplayBatch:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError("canonical replay requires a tokenizer pad or EOS token")
        return materialize_canonical_replay_batch(
            groups, pad_token_id=int(pad_token_id), device=device
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
            replay_input_ids, attention_mask=replay.attention_mask[start:stop]
        )["logits"]
        if self.args.temperature != 1:
            replay_logits = replay_logits / self.args.temperature
        replay_logits = self._mask_invalid_scoring_logit_columns(
            replay_logits,
            valid_vocab_size=policy_vocab_upper_bound,
            context="canonical_replay_policy_logits",
        )
        (replay_logps, _) = self._policy_logps_and_optional_entropy(
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
