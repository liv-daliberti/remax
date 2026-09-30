"""OAT experimental responsibilities, preserving historical state and ordering."""
from __future__ import annotations
import logging
import time
from typing import Any, Callable
import torch.distributed as dist
from oat.types import TrajectoryData
from ...args import resolve_canonical_action_task
from ..dapo import dapo_group_diagnostics, dapo_resample_prompt_index
from ...math_grader import VerifiedExplorationIdentity, validated_exploration_identity, validated_modebench_outcome_key
from ...online_canonical_bank import OnlineCanonicalBank
from ...replicated_group import validate_replicated_group_layout
from ...verified_transformations import derive_countdown_action_neighborhood_counterfactuals, derive_validator_preserving_counterfactuals
from ...verified_route_library import VerifiedRouteLibrary
from ...integrations.oat.support import _derive_freeform_request_seed, _exploration_support_key, _trajectory_mean_logprob

class HistoricalRunMixin:
    def _generate_verified_counterfactual_proposals(
        self,
        *,
        actor: Any,
        raw_prompts: list[str],
        processed_prompts: list[str],
        refs: list[str],
        neutral_feedback: list[TrajectoryData],
        precomputed_proposal_groups: list[list[TrajectoryData]] | None = None,
        proposal_group_factory: (
            Callable[[int, float, int], list[TrajectoryData]] | None
        ) = None,
    ) -> tuple[dict[str, list[Any]], dict[str, float]]:
        """Search a fixed temperature sweep for novel valid modes.

        The returned admission payload contains no proposal trajectory object,
        log probability, reward advantage, or loss mask. This makes it
        impossible for the caller to accidentally append proposal samples to
        the neutral PPO batch.
        """

        key_mode = str(
            getattr(
                self.args,
                "online_canonical_key_mode",
                "modebench_outcome",
            )
        )
        verified_route_mode = key_mode == "verified_route"
        transform_proposals = bool(
            getattr(
                self.args,
                "online_canonical_counterfactual_transform_proposals",
                True,
            )
        ) and not verified_route_mode
        exact_grammar_transforms = bool(
            getattr(
                self.args,
                "online_canonical_counterfactual_exact_grammar_transforms",
                False,
            )
        ) and not verified_route_mode
        if transform_proposals and exact_grammar_transforms:
            raise RuntimeError(
                "legacy and exact-grammar proposal transforms are mutually exclusive"
            )
        metrics = {
            "actor/counterfactual_proposal_enabled": 1.0,
            "actor/counterfactual_proposal_transform_enabled": float(
                transform_proposals
            ),
            "actor/counterfactual_proposal_exact_grammar_transform_enabled": float(
                exact_grammar_transforms
            ),
            "actor/counterfactual_proposal_anchor_available": 0.0,
            "actor/counterfactual_proposal_anchor_from_current_neutral": 0.0,
            "actor/counterfactual_proposal_anchor_from_prior_bank": 0.0,
            "actor/counterfactual_proposal_neutral_task_reward_positive_rows": 0.0,
            "actor/counterfactual_proposal_neutral_validator_positive_rows": 0.0,
            "actor/counterfactual_proposal_neutral_validator_task_disagreement_rows": 0.0,
            "actor/counterfactual_proposal_neutral_unique_outcomes": 0.0,
            "actor/counterfactual_proposal_groups_generated": 0.0,
            "actor/counterfactual_proposal_rows_generated": 0.0,
            "actor/counterfactual_proposal_task_reward_positive_rows": 0.0,
            "actor/counterfactual_proposal_validator_positive_rows": 0.0,
            "actor/counterfactual_proposal_validator_task_disagreement_rows": 0.0,
            "actor/counterfactual_proposal_same_anchor_rows": 0.0,
            "actor/counterfactual_proposal_known_alternate_rows": 0.0,
            "actor/counterfactual_proposal_novel_candidate_rows": 0.0,
            "actor/counterfactual_proposal_novel_unique_outcomes": 0.0,
            "actor/counterfactual_proposal_generate_time": 0.0,
            "actor/counterfactual_proposal_success_attempt": 0.0,
            "actor/counterfactual_proposal_attempts_exhausted": 0.0,
            "actor/counterfactual_proposal_original_prompt_groups": 0.0,
            "actor/counterfactual_proposal_conditioned_prompt_groups": 0.0,
            "actor/counterfactual_proposal_gold_support_feedback": 0.0,
            "actor/counterfactual_proposal_desired_mode_count_feedback": 0.0,
            "actor/counterfactual_proposal_eval_feedback": 0.0,
            "actor/counterfactual_proposal_transform_candidate_surfaces": 0.0,
            "actor/counterfactual_proposal_transform_validator_positive": 0.0,
            "actor/counterfactual_proposal_transform_tokenization_rejections": 0.0,
            "actor/counterfactual_proposal_transform_known_outcomes": 0.0,
            "actor/counterfactual_proposal_transform_replay_capacity_slots": 0.0,
            "actor/counterfactual_proposal_transform_replay_capacity_exhausted": 0.0,
            "actor/counterfactual_proposal_transform_novel_unique_outcomes": 0.0,
            "actor/counterfactual_proposal_transform_success": 0.0,
            "actor/counterfactual_proposal_transform_rows_sent_to_ppo": 0.0,
            "actor/counterfactual_proposal_transform_gold_support_feedback": 0.0,
            "actor/counterfactual_proposal_transform_desired_mode_count_feedback": 0.0,
            "actor/counterfactual_proposal_transform_eval_feedback": 0.0,
            "actor/counterfactual_proposal_singleton_only_enabled": 0.0,
            "actor/counterfactual_proposal_singleton_only_known_support": 0.0,
            "actor/counterfactual_proposal_singleton_only_active": 0.0,
            "actor/counterfactual_proposal_verified_route_mode": float(
                verified_route_mode
            ),
            "actor/counterfactual_proposal_trust_checked_rows": 0.0,
            "actor/counterfactual_proposal_trust_rejected_rows": 0.0,
            "actor/counterfactual_proposal_max_admissions_per_update": (
                1.0 if verified_route_mode else 0.0
            ),
        }
        starvation_controller = getattr(
            self,
            "_proposal_starvation_controller",
            None,
        )
        starvation_diagnostics = (
            starvation_controller.diagnostics()
            if starvation_controller is not None
            else {}
        )
        metrics.update(
            {
                "actor/counterfactual_proposal_starvation_fallback_enabled": float(
                    starvation_controller is not None
                ),
                "actor/counterfactual_proposal_starvation_eligible_update": 0.0,
                "actor/counterfactual_proposal_starvation_fallback_active": 0.0,
                "actor/counterfactual_proposal_starvation_fallback_activation": 0.0,
                "actor/counterfactual_proposal_starvation_fallback_extra_groups": 0.0,
                "actor/counterfactual_proposal_starvation_stalled_updates_before": float(
                    starvation_diagnostics.get("stalled_eligible_updates", 0)
                ),
                "actor/counterfactual_proposal_starvation_fallback_updates_cumulative": float(
                    starvation_diagnostics.get("fallback_updates", 0)
                ),
                "actor/counterfactual_proposal_starvation_fallback_activations_cumulative": float(
                    starvation_diagnostics.get("fallback_activations", 0)
                ),
                "actor/counterfactual_proposal_starvation_gold_support_feedback": 0.0,
                "actor/counterfactual_proposal_starvation_desired_mode_count_feedback": 0.0,
                "actor/counterfactual_proposal_starvation_eval_feedback": 0.0,
            }
        )
        empty_payload: dict[str, list[Any]] = {
            "prompt_token_ids": [],
            "outcome_keys": [],
            "response_token_ids": [],
        }
        if verified_route_mode:
            empty_payload.update(
                {
                    "endpoint_keys": [],
                    "route_signatures": [],
                    "verifier_ids": [],
                    "proposal_mean_logprobs": [],
                    "anchor_mean_logprobs": [],
                }
            )
        if len(raw_prompts) != 1 or len(processed_prompts) != 1 or len(refs) != 1:
            raise RuntimeError("counterfactual proposals require one current prompt")
        if len(neutral_feedback) != int(self.args.num_samples):
            raise RuntimeError(
                "counterfactual proposal source group is not one neutral "
                "rollout-width group"
            )
        if not bool(getattr(self.args, "online_evaluation", False)):
            raise RuntimeError("counterfactual proposals require validator references")
        bank = getattr(self, "_online_canonical_bank", None)
        if not isinstance(bank, OnlineCanonicalBank):
            raise RuntimeError(
                "counterfactual proposals require an online canonical bank"
            )
        metrics["actor/counterfactual_proposal_max_admissions_per_update"] = (
            1.0 if verified_route_mode else float(bank.replay_capacity)
        )
        route_library = getattr(self, "_verified_route_library", None)
        if verified_route_mode and not isinstance(
            route_library,
            VerifiedRouteLibrary,
        ):
            raise RuntimeError(
                "verified-route proposals require a verified route library"
            )

        neutral_prompt_rows = [
            tuple(int(token_id) for token_id in trajectory.prompt_ids)
            for trajectory in neutral_feedback
        ]
        if not neutral_prompt_rows[0] or (len(set(neutral_prompt_rows)) != 1):
            raise RuntimeError(
                "neutral proposal source rows do not share one non-empty prompt"
            )
        neutral_prompt_token_ids = neutral_prompt_rows[0]
        groups = bank.replay_groups(
            [neutral_prompt_token_ids],
            min_modes=1,
        )
        if len(groups) > 1:
            raise RuntimeError(
                "counterfactual proposal lookup returned multiple prompt banks"
            )
        prior_candidates: dict[str, tuple[int, ...]] = {}
        prior_candidate_identities: dict[
            str,
            VerifiedExplorationIdentity,
        ] = {}
        prior_candidate_mean_logprobs: dict[str, float] = {}
        if groups:
            group = groups[0]
            if len(group.outcome_keys) != len(group.response_token_ids):
                raise RuntimeError(
                    "counterfactual proposal bank keys/exemplars are misaligned"
                )
            prior_candidates = {
                str(key): tuple(int(token_id) for token_id in response_ids)
                for key, response_ids in zip(
                    group.outcome_keys,
                    group.response_token_ids,
                )
            }
        if verified_route_mode:
            assert isinstance(route_library, VerifiedRouteLibrary)
            route_exemplars = route_library.prompt_exemplars(neutral_prompt_token_ids)
            if route_exemplars:
                # Once a prompt has executable route support, singleton and
                # novelty decisions are made in route space rather than
                # double-counting its prompt-local endpoint.
                prior_candidates = {}
            for exemplar in route_exemplars:
                support_key = exemplar.route_signature
                prior_candidates[support_key] = exemplar.response_token_ids
                prior_candidate_identities[support_key] = VerifiedExplorationIdentity(
                    verifier=exemplar.verifier,
                    endpoint_key=exemplar.endpoint_key,
                    route_signature=exemplar.route_signature,
                )
                prior_candidate_mean_logprobs[support_key] = exemplar.model_mean_logprob

        # Bootstrap on the current neutral group. A one-pass training set does
        # not revisit a prompt, so requiring a bank entry from an earlier
        # update would make the actuator structurally inert for the entire
        # first pass. These rows were generated by the same unconditioned
        # policy that PPO will train on; they are independently validated and
        # only supply an exemplar to the separate proposal query.
        current_candidates: dict[str, tuple[int, ...]] = {}
        current_candidate_surfaces: dict[str, str] = {}
        current_candidate_identities: dict[
            str,
            VerifiedExplorationIdentity,
        ] = {}
        current_candidate_mean_logprobs: dict[str, float] = {}
        neutral_task_positive_rows = 0
        neutral_validator_positive_rows = 0
        neutral_disagreement_rows = 0
        for trajectory in neutral_feedback:
            rewards = list(trajectory.rewards)
            task_positive = bool(trajectory.loss_mask) and bool(
                rewards and float(rewards[-1]) > 0.0
            )
            neutral_task_positive_rows += int(task_positive)
            identity = (
                validated_exploration_identity(
                    str(trajectory.response),
                    raw_prompts[0],
                    refs[0],
                    fast=(str(self.args.verifier_version) != "math_verify"),
                    task_verified=task_positive,
                )
                if verified_route_mode
                else None
            )
            outcome_key = (
                _exploration_support_key(identity)
                if identity is not None
                else validated_modebench_outcome_key(
                    str(trajectory.response),
                    refs[0],
                )
                if not verified_route_mode
                else None
            )
            validator_positive = outcome_key is not None
            neutral_validator_positive_rows += int(validator_positive)
            neutral_disagreement_rows += int(validator_positive != task_positive)
            if not (validator_positive and task_positive):
                continue
            assert outcome_key is not None
            response_ids = tuple(int(token_id) for token_id in trajectory.response_ids)
            if not response_ids:
                raise RuntimeError(
                    "validator-positive neutral anchor has an empty token response"
                )
            previous = current_candidates.get(outcome_key)
            if previous is None or response_ids < previous:
                current_candidates[outcome_key] = response_ids
                current_candidate_surfaces[outcome_key] = str(trajectory.response)
                if identity is not None:
                    current_candidate_identities[outcome_key] = identity
                    current_candidate_mean_logprobs[outcome_key] = (
                        _trajectory_mean_logprob(trajectory)
                    )
        if verified_route_mode and any(
            identity.route_signature is not None
            for identity in current_candidate_identities.values()
        ):
            route_keys = {
                key
                for key, identity in current_candidate_identities.items()
                if identity.route_signature is not None
            }
            current_candidates = {
                key: value
                for key, value in current_candidates.items()
                if key in route_keys
            }
            current_candidate_surfaces = {
                key: value
                for key, value in current_candidate_surfaces.items()
                if key in route_keys
            }
            current_candidate_identities = {
                key: value
                for key, value in current_candidate_identities.items()
                if key in route_keys
            }
            current_candidate_mean_logprobs = {
                key: value
                for key, value in current_candidate_mean_logprobs.items()
                if key in route_keys
            }
        metrics.update(
            {
                "actor/counterfactual_proposal_neutral_task_reward_positive_rows": (
                    float(neutral_task_positive_rows)
                ),
                "actor/counterfactual_proposal_neutral_validator_positive_rows": (
                    float(neutral_validator_positive_rows)
                ),
                "actor/counterfactual_proposal_neutral_validator_task_disagreement_rows": (
                    float(neutral_disagreement_rows)
                ),
                "actor/counterfactual_proposal_neutral_unique_outcomes": float(
                    len(current_candidates)
                ),
            }
        )
        if neutral_disagreement_rows:
            logging.warning(
                "counterfactual neutral-anchor validator/task-reward "
                "disagreement on %d rows; fail-closed intersection excludes them",
                neutral_disagreement_rows,
            )

        if current_candidates:
            anchor_candidates = current_candidates
            metrics["actor/counterfactual_proposal_anchor_from_current_neutral"] = 1.0
        elif prior_candidates and not verified_route_mode:
            anchor_candidates = prior_candidates
            metrics["actor/counterfactual_proposal_anchor_from_prior_bank"] = 1.0
        else:
            return empty_payload, metrics
        anchor_keys = sorted(anchor_candidates)
        anchor_index = max(int(self._prompt_batches_consumed_total) - 1, 0) % len(
            anchor_keys
        )
        anchor_key = anchor_keys[anchor_index]
        anchor_mean_logprob = (
            current_candidate_mean_logprobs.get(anchor_key)
            if verified_route_mode
            else None
        )
        if verified_route_mode and anchor_mean_logprob is None:
            raise RuntimeError(
                "verified-route proposals require a current neutral anchor likelihood"
            )

        metrics["actor/counterfactual_proposal_anchor_available"] = 1.0
        known_keys = set(prior_candidates) | set(current_candidates)
        novel_candidates: dict[str, tuple[int, ...]] = {}
        novel_candidate_identities: dict[
            str,
            VerifiedExplorationIdentity,
        ] = {}
        novel_candidate_mean_logprobs: dict[str, float] = {}
        singleton_only = bool(
            getattr(
                self.args,
                "online_canonical_counterfactual_singleton_only",
                False,
            )
        )
        metrics[
            "actor/counterfactual_proposal_singleton_only_enabled"
        ] = float(singleton_only)
        metrics[
            "actor/counterfactual_proposal_singleton_only_known_support"
        ] = float(len(known_keys))
        metrics[
            "actor/counterfactual_proposal_singleton_only_active"
        ] = float(singleton_only and len(known_keys) == 1)
        if verified_route_mode and len(known_keys) != 1:
            # E69 keeps E68's actuator-identifiability rule but does not make
            # route admission depend on auxiliary task-reward shaping:
            # the explorer only searches from an exactly singleton verified
            # support set, while the neutral learner remains task-first.
            return empty_payload, metrics
        if singleton_only and len(known_keys) != 1:
            # Exactly one discovered valid mode is an actuator-identifiability
            # condition, not a desired task-support target. Expansion stops
            # as soon as a second independently verified mode exists.
            return empty_payload, metrics

        # A sampled policy can collapse to singleton valid support, at which
        # point repeated sampling has no usable actuator.  First derive
        # validator-preserving alternatives from the model's own verified
        # response and public instance constraints.  The derivation never
        # reads exhaustive support, a desired count, evaluation metrics, or a
        # target entropy.  Each surface is independently validated before and
        # after tokenization, and only its token ids and canonical key can
        # enter support-only replay.
        if anchor_key in current_candidate_surfaces:
            anchor_surface = current_candidate_surfaces[anchor_key]
        else:
            anchor_surface = self.tokenizer.decode(
                anchor_candidates[anchor_key],
                skip_special_tokens=True,
            )
        transformed_surfaces = (
            derive_countdown_action_neighborhood_counterfactuals(
                anchor_surface,
                refs[0],
                radius=2,
            )
            if exact_grammar_transforms
            else derive_validator_preserving_counterfactuals(
                anchor_surface,
                refs[0],
            )
            if transform_proposals
            else []
        )
        metrics["actor/counterfactual_proposal_transform_candidate_surfaces"] = float(
            len(transformed_surfaces)
        )
        transformed_known = 0
        transformed_validator_positive = 0
        transformed_tokenization_rejections = 0
        future_known_keys = set(prior_candidates) | set(current_candidates)
        transform_replay_capacity_slots = (
            1
            if verified_route_mode
            else max(
                int(bank.replay_capacity) - len(future_known_keys),
                0,
            )
        )
        if singleton_only or verified_route_mode:
            transform_replay_capacity_slots = min(
                transform_replay_capacity_slots,
                1,
            )
        metrics["actor/counterfactual_proposal_transform_replay_capacity_slots"] = (
            float(transform_replay_capacity_slots)
        )
        max_response_tokens = int(getattr(self.args, "generate_max_length", 0))
        for surface in transformed_surfaces:
            outcome_key = validated_modebench_outcome_key(
                surface,
                refs[0],
            )
            if outcome_key is None:
                continue
            transformed_validator_positive += 1
            if outcome_key in known_keys:
                transformed_known += 1
                continue
            response_ids = tuple(
                int(token_id)
                for token_id in self.tokenizer.encode(
                    surface,
                    add_special_tokens=False,
                )
            )
            if not response_ids or (
                max_response_tokens > 0 and len(response_ids) > max_response_tokens
            ):
                transformed_tokenization_rejections += 1
                continue
            roundtrip_surface = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            )
            if (
                validated_modebench_outcome_key(
                    roundtrip_surface,
                    refs[0],
                )
                != outcome_key
            ):
                transformed_tokenization_rejections += 1
                continue
            previous = novel_candidates.get(outcome_key)
            if previous is None or response_ids < previous:
                novel_candidates[outcome_key] = response_ids
        if len(novel_candidates) > transform_replay_capacity_slots:
            novel_candidates = {
                key: novel_candidates[key]
                for key in sorted(novel_candidates)[:transform_replay_capacity_slots]
            }
        metrics.update(
            {
                "actor/counterfactual_proposal_transform_validator_positive": (
                    float(transformed_validator_positive)
                ),
                "actor/counterfactual_proposal_transform_tokenization_rejections": (
                    float(transformed_tokenization_rejections)
                ),
                "actor/counterfactual_proposal_transform_known_outcomes": float(
                    transformed_known
                ),
                "actor/counterfactual_proposal_transform_novel_unique_outcomes": (
                    float(len(novel_candidates))
                ),
                "actor/counterfactual_proposal_transform_success": float(
                    bool(novel_candidates)
                ),
            }
        )
        if novel_candidates:
            outcome_keys = sorted(novel_candidates)
            return (
                {
                    "prompt_token_ids": [
                        list(neutral_prompt_token_ids) for _ in outcome_keys
                    ],
                    "outcome_keys": outcome_keys,
                    "response_token_ids": [
                        list(novel_candidates[key]) for key in outcome_keys
                    ],
                },
                metrics,
            )
        if transform_replay_capacity_slots == 0:
            metrics[
                "actor/counterfactual_proposal_transform_replay_capacity_exhausted"
            ] = 1.0
            return empty_payload, metrics

        task_positive_rows = 0
        validator_positive_rows = 0
        disagreement_rows = 0
        same_anchor_rows = 0
        known_alternate_rows = 0
        novel_candidate_rows = 0
        generation_time = 0.0
        starvation_plan = (
            starvation_controller.plan()
            if starvation_controller is not None
            else None
        )
        max_attempts = (
            int(starvation_plan.max_attempts)
            if starvation_plan is not None
            else int(self.args.online_canonical_counterfactual_max_attempts)
        )
        metrics[
            "actor/counterfactual_proposal_starvation_eligible_update"
        ] = 1.0
        if starvation_plan is not None:
            metrics[
                "actor/counterfactual_proposal_starvation_fallback_active"
            ] = float(starvation_plan.fallback_active)
            metrics[
                "actor/counterfactual_proposal_starvation_fallback_activation"
            ] = float(starvation_plan.fallback_activation)
        base_sampling_temperature = float(
            self.args.online_canonical_counterfactual_sampling_temperature
        )
        metrics["actor/counterfactual_proposal_max_attempts"] = float(max_attempts)
        metrics["actor/counterfactual_proposal_sampling_temperature"] = (
            base_sampling_temperature
        )
        metrics["actor/counterfactual_proposal_temperature_step"] = (
            0.0 if verified_route_mode else 0.2
        )
        metrics["actor/counterfactual_proposal_last_temperature"] = (
            base_sampling_temperature
        )
        proposal_request_seeds: list[int] = []
        for attempt_index in range(max_attempts):
            # Preserve the original task grammar. R4's answer-conditioned
            # query copied the anchor; R5's stronger conditioned query made
            # most samples invalid. Repeated untouched-model requests search
            # outside the collision without exposing a target mode, support
            # size, or gold answer. Legacy endpoint proposals retain their
            # registered temperature sweep. Verified-route proposals remain
            # at the neutral temperature so likelihoods are comparable.
            sampling_temperature = (
                base_sampling_temperature
                if verified_route_mode
                else base_sampling_temperature + 0.2 * attempt_index
            )
            metrics["actor/counterfactual_proposal_last_temperature"] = (
                sampling_temperature
            )
            proposal_request_seed = _derive_freeform_request_seed(
                base_seed=int(self.args.seed),
                prompt_batch_index=int(self._prompt_batches_consumed_total),
                stream=f"proposal:{attempt_index}",
            )
            proposal_request_seeds.append(proposal_request_seed)
            if proposal_group_factory is not None:
                started = time.time()
                proposal_feedback = proposal_group_factory(
                    attempt_index,
                    sampling_temperature,
                    proposal_request_seed,
                )
                generation_time += time.time() - started
            elif precomputed_proposal_groups is not None:
                if attempt_index >= len(precomputed_proposal_groups):
                    raise RuntimeError(
                        "fixed counterfactual controls do not cover every "
                        "proposal attempt"
                    )
                proposal_feedback = precomputed_proposal_groups[attempt_index]
            else:
                started = time.time()
                proposal_handle = actor.step(
                    raw_prompts,
                    processed_prompts,
                    refs,
                    sampling_temperature=sampling_temperature,
                    sampling_seed=proposal_request_seed,
                )
                proposal_feedback = self.collector.ipc_client.deserialize_ipc(
                    proposal_handle
                )
                generation_time += time.time() - started
            if len(proposal_feedback) != int(self.args.num_samples):
                raise RuntimeError(
                    "counterfactual actor returned "
                    f"{len(proposal_feedback)} rows, expected "
                    f"{self.args.num_samples}"
                )
            metrics["actor/counterfactual_proposal_groups_generated"] += 1.0
            metrics["actor/counterfactual_proposal_original_prompt_groups"] += 1.0
            metrics["actor/counterfactual_proposal_rows_generated"] += float(
                len(proposal_feedback)
            )
            for trajectory in proposal_feedback:
                rewards = list(trajectory.rewards)
                task_positive = bool(trajectory.loss_mask) and bool(
                    rewards and float(rewards[-1]) > 0.0
                )
                if task_positive:
                    task_positive_rows += 1
                identity = (
                    validated_exploration_identity(
                        str(trajectory.response),
                        raw_prompts[0],
                        refs[0],
                        fast=(str(self.args.verifier_version) != "math_verify"),
                        task_verified=task_positive,
                    )
                    if verified_route_mode
                    else None
                )
                outcome_key = (
                    _exploration_support_key(identity)
                    if identity is not None
                    else validated_modebench_outcome_key(
                        str(trajectory.response),
                        refs[0],
                    )
                    if not verified_route_mode
                    else None
                )
                validator_positive = outcome_key is not None
                if validator_positive:
                    validator_positive_rows += 1
                if validator_positive != task_positive:
                    disagreement_rows += 1
                if not (validator_positive and task_positive):
                    continue
                assert outcome_key is not None
                if outcome_key == anchor_key:
                    same_anchor_rows += 1
                    continue
                if outcome_key in known_keys:
                    known_alternate_rows += 1
                    continue
                response_ids = tuple(
                    int(token_id) for token_id in trajectory.response_ids
                )
                if not response_ids:
                    raise RuntimeError(
                        "validator-positive proposal has an empty token response"
                    )
                proposal_mean_logprob = (
                    _trajectory_mean_logprob(trajectory)
                    if verified_route_mode
                    else None
                )
                if verified_route_mode:
                    assert proposal_mean_logprob is not None
                    assert anchor_mean_logprob is not None
                    metrics["actor/counterfactual_proposal_trust_checked_rows"] += 1.0
                    max_drop = float(
                        self.args.verified_route_proposal_max_mean_logprob_drop
                    )
                    if proposal_mean_logprob < anchor_mean_logprob - max_drop:
                        metrics[
                            "actor/counterfactual_proposal_trust_rejected_rows"
                        ] += 1.0
                        continue
                novel_candidate_rows += 1
                previous = novel_candidates.get(outcome_key)
                if previous is None or response_ids < previous:
                    novel_candidates[outcome_key] = response_ids
                    if identity is not None:
                        novel_candidate_identities[outcome_key] = identity
                    if proposal_mean_logprob is not None:
                        novel_candidate_mean_logprobs[outcome_key] = (
                            proposal_mean_logprob
                        )
            if novel_candidates:
                metrics["actor/counterfactual_proposal_success_attempt"] = float(
                    attempt_index + 1
                )
                break

        metrics["actor/counterfactual_proposal_generate_time"] = generation_time
        metrics["actor/counterfactual_proposal_request_seed_min"] = float(
            min(proposal_request_seeds)
        )
        metrics["actor/counterfactual_proposal_request_seed_max"] = float(
            max(proposal_request_seeds)
        )
        metrics["actor/counterfactual_proposal_seed_isolation_active"] = 1.0
        if not novel_candidates:
            metrics["actor/counterfactual_proposal_attempts_exhausted"] = 1.0
        if starvation_plan is not None and starvation_plan.fallback_active:
            metrics[
                "actor/counterfactual_proposal_starvation_fallback_extra_groups"
            ] = float(
                max(
                    int(metrics["actor/counterfactual_proposal_groups_generated"])
                    - int(starvation_controller.base_max_attempts),
                    0,
                )
            )

        if disagreement_rows:
            logging.warning(
                "counterfactual proposal validator/task-reward disagreement "
                "on %d rows; fail-closed intersection excludes them",
                disagreement_rows,
            )
        if len(novel_candidates) > transform_replay_capacity_slots:
            retained_keys = sorted(novel_candidates)[:transform_replay_capacity_slots]
            novel_candidates = {key: novel_candidates[key] for key in retained_keys}
            novel_candidate_identities = {
                key: novel_candidate_identities[key]
                for key in retained_keys
                if key in novel_candidate_identities
            }
            novel_candidate_mean_logprobs = {
                key: novel_candidate_mean_logprobs[key]
                for key in retained_keys
                if key in novel_candidate_mean_logprobs
            }
        metrics.update(
            {
                "actor/counterfactual_proposal_task_reward_positive_rows": float(
                    task_positive_rows
                ),
                "actor/counterfactual_proposal_validator_positive_rows": float(
                    validator_positive_rows
                ),
                "actor/counterfactual_proposal_validator_task_disagreement_rows": (
                    float(disagreement_rows)
                ),
                "actor/counterfactual_proposal_same_anchor_rows": float(
                    same_anchor_rows
                ),
                "actor/counterfactual_proposal_known_alternate_rows": float(
                    known_alternate_rows
                ),
                "actor/counterfactual_proposal_novel_candidate_rows": float(
                    novel_candidate_rows
                ),
                "actor/counterfactual_proposal_novel_unique_outcomes": float(
                    len(novel_candidates)
                ),
            }
        )
        outcome_keys = sorted(novel_candidates)
        if verified_route_mode and (
            set(outcome_keys) != set(novel_candidate_identities)
            or set(outcome_keys) != set(novel_candidate_mean_logprobs)
        ):
            raise RuntimeError("verified-route proposal metadata is incomplete")
        payload: dict[str, list[Any]] = {
            "prompt_token_ids": [list(neutral_prompt_token_ids) for _ in outcome_keys],
            "outcome_keys": outcome_keys,
            "response_token_ids": [list(novel_candidates[key]) for key in outcome_keys],
        }
        if verified_route_mode:
            payload.update(
                {
                    "endpoint_keys": [
                        novel_candidate_identities[key].endpoint_key
                        for key in outcome_keys
                    ],
                    "route_signatures": [
                        novel_candidate_identities[key].route_signature
                        for key in outcome_keys
                    ],
                    "verifier_ids": [
                        novel_candidate_identities[key].verifier for key in outcome_keys
                    ],
                    "proposal_mean_logprobs": [
                        novel_candidate_mean_logprobs[key] for key in outcome_keys
                    ],
                    "anchor_mean_logprobs": [anchor_mean_logprob for _ in outcome_keys],
                }
            )
        return payload, metrics


    def _generate_counterfactual_fixed_control_groups(
        self,
        *,
        actor: Any,
        raw_prompts: list[str],
        processed_prompts: list[str],
        refs: list[str],
    ) -> tuple[list[list[TrajectoryData]], dict[str, float]]:
        """Issue a fixed proposal-shaped sampling budget for compute matching.

        The rows remain ordinary model generations, but this helper never
        validates, stores, replays, or returns them to PPO. A proposal-enabled
        arm may inspect the frozen list downstream; control arms discard every
        row. Thus request count and charged token budget are independent of
        whether a treatment finds a usable alternate early.
        """

        group_count = int(
            getattr(
                self.args,
                "online_canonical_counterfactual_fixed_control_groups",
                0,
            )
        )
        metrics = {
            "actor/counterfactual_fixed_control_groups_generated": 0.0,
            "actor/counterfactual_fixed_control_rows_generated": 0.0,
            "actor/counterfactual_fixed_control_realized_prompt_tokens": 0.0,
            "actor/counterfactual_fixed_control_realized_response_tokens": 0.0,
            "actor/counterfactual_fixed_control_charged_response_token_budget": 0.0,
            "actor/counterfactual_fixed_control_generate_time": 0.0,
            "actor/counterfactual_fixed_control_rows_sent_to_ppo": 0.0,
            "actor/counterfactual_fixed_control_groups_consumed_by_explorer": 0.0,
            "actor/counterfactual_fixed_control_groups_discarded": float(group_count),
        }
        if group_count == 0:
            return [], metrics
        if len(raw_prompts) != 1 or len(processed_prompts) != 1 or len(refs) != 1:
            raise RuntimeError(
                "fixed counterfactual controls require one current prompt"
            )
        base_temperature = float(
            self.args.online_canonical_counterfactual_sampling_temperature
        )
        verified_route_mode = (
            str(self.args.online_canonical_key_mode) == "verified_route"
        )
        groups: list[list[TrajectoryData]] = []
        request_seeds: list[int] = []
        temperatures: list[float] = []
        started = time.time()
        for attempt_index in range(group_count):
            temperature = (
                base_temperature
                if verified_route_mode
                else base_temperature + 0.2 * attempt_index
            )
            request_seed = _derive_freeform_request_seed(
                base_seed=int(self.args.seed),
                prompt_batch_index=int(self._prompt_batches_consumed_total),
                stream=f"proposal:{attempt_index}",
            )
            request_seeds.append(request_seed)
            temperatures.append(temperature)
            if self.args.online_evaluation:
                handle = actor.step(
                    raw_prompts,
                    processed_prompts,
                    refs,
                    sampling_temperature=temperature,
                    sampling_seed=request_seed,
                )
            else:
                handle = actor.step(
                    raw_prompts,
                    processed_prompts,
                    sampling_temperature=temperature,
                    sampling_seed=request_seed,
                )
            feedback = self.collector.ipc_client.deserialize_ipc(handle)
            if len(feedback) != int(self.args.num_samples):
                raise RuntimeError(
                    "fixed counterfactual control returned "
                    f"{len(feedback)} rows, expected {self.args.num_samples}"
                )
            groups.append(feedback)
        prompt_tokens = sum(
            len(trajectory.prompt_ids)
            for group in groups
            for trajectory in group
        )
        response_tokens = sum(
            len(trajectory.response_ids)
            for group in groups
            for trajectory in group
        )
        rows = group_count * int(self.args.num_samples)
        metrics.update(
            {
                "actor/counterfactual_fixed_control_groups_generated": float(
                    group_count
                ),
                "actor/counterfactual_fixed_control_rows_generated": float(rows),
                "actor/counterfactual_fixed_control_realized_prompt_tokens": float(
                    prompt_tokens
                ),
                "actor/counterfactual_fixed_control_realized_response_tokens": float(
                    response_tokens
                ),
                "actor/counterfactual_fixed_control_charged_response_token_budget": (
                    float(rows * int(self.args.generate_max_length))
                ),
                "actor/counterfactual_fixed_control_generate_time": (
                    time.time() - started
                ),
                "actor/counterfactual_fixed_control_request_seed_min": float(
                    min(request_seeds)
                ),
                "actor/counterfactual_fixed_control_request_seed_max": float(
                    max(request_seeds)
                ),
                "actor/counterfactual_fixed_control_sampling_temperature_min": float(
                    min(temperatures)
                ),
                "actor/counterfactual_fixed_control_sampling_temperature_max": float(
                    max(temperatures)
                ),
            }
        )
        return groups, metrics


    def _sample_replicated_freeform_feedback(
        self,
        raw_prompts: list[str],
        processed_prompts: list[str],
        refs: list[str],
    ) -> tuple[list[TrajectoryData], dict[str, float]]:
        """Generate one free-form group and replicate it across learner ranks."""

        if not bool(getattr(self.args, "replicated_freeform_sampling", False)):
            raise RuntimeError("replicated free-form sampling was not enabled")
        if resolve_canonical_action_task(self.args) != "none":
            raise RuntimeError("replicated free-form sampling received canonical mode")
        if len(raw_prompts) != 1 or len(processed_prompts) != 1 or len(refs) != 1:
            raise RuntimeError(
                "replicated free-form sampling requires one prompt per learner rank"
            )
        world_size = dist.get_world_size()
        if int(self.args.rollout_batch_size) != world_size:
            raise RuntimeError(
                "replicated free-form sampling requires one replicated rollout "
                "slot per learner rank"
            )
        if int(self.update_interval) != 1:
            raise RuntimeError(
                "replicated free-form sampling requires update_interval=1"
            )
        try:
            replicated_layout = validate_replicated_group_layout(
                num_samples=int(self.args.num_samples),
                learner_world_size=world_size,
                train_batch_size=int(self.args.train_batch_size),
                train_batch_size_per_device=int(self.args.train_batch_size_per_device),
            )
        except ValueError as error:
            raise RuntimeError(
                f"invalid replicated free-form group layout: {error}"
            ) from error
        if int(self.strategy.grad_acc_step) != int(
            replicated_layout.micro_batches_per_rank
        ):
            raise RuntimeError(
                "replicated free-form DeepSpeed accumulation width does not "
                "match the exact logical candidate group"
            )

        local_contract = (raw_prompts[0], processed_prompts[0], refs[0])
        gathered_contracts: list[tuple[str, str, str] | None] = [None] * world_size
        dist.all_gather_object(gathered_contracts, local_contract)
        if any(contract != local_contract for contract in gathered_contracts):
            raise RuntimeError(
                "replicated free-form learner ranks received different prompts"
            )

        payload: list[Any] = [None, None, None]
        if dist.get_rank() == 0:
            actor = self.actors[0]
            started = time.time()
            neutral_request_seed = _derive_freeform_request_seed(
                base_seed=int(self.args.seed),
                prompt_batch_index=int(self._prompt_batches_consumed_total),
                stream="neutral",
            )
            if self.args.online_evaluation:
                handle = actor.step(
                    raw_prompts,
                    processed_prompts,
                    refs,
                    sampling_seed=neutral_request_seed,
                )
            else:
                handle = actor.step(
                    raw_prompts,
                    processed_prompts,
                    sampling_seed=neutral_request_seed,
                )
            feedback_data = self.collector.ipc_client.deserialize_ipc(handle)
            if len(feedback_data) != int(self.args.num_samples):
                raise RuntimeError(
                    "replicated free-form actor returned "
                    f"{len(feedback_data)} rows, expected {self.args.num_samples}"
                )
            actor_info = self.collector.get_metrics(
                time.time() - started, feedback_data
            )
            (
                fixed_control_groups,
                fixed_control_metrics,
            ) = ZeroMathRunMixin._generate_counterfactual_fixed_control_groups(
                self,
                actor=actor,
                raw_prompts=raw_prompts,
                processed_prompts=processed_prompts,
                refs=refs,
            )
            actor_info.update(fixed_control_metrics)
            proposal_admission: dict[str, list[Any]] | None = None
            if bool(
                getattr(
                    self.args,
                    "online_canonical_counterfactual_proposals",
                    False,
                )
            ):
                (
                    proposal_admission,
                    proposal_metrics,
                ) = self._generate_verified_counterfactual_proposals(
                    actor=actor,
                    raw_prompts=raw_prompts,
                    processed_prompts=processed_prompts,
                    refs=refs,
                    neutral_feedback=feedback_data,
                    precomputed_proposal_groups=(
                        fixed_control_groups if fixed_control_groups else None
                    ),
                )
                actor_info.update(proposal_metrics)
                if fixed_control_groups:
                    consumed_groups = int(
                        proposal_metrics[
                            "actor/counterfactual_proposal_groups_generated"
                        ]
                    )
                    actor_info[
                        "actor/counterfactual_fixed_control_groups_consumed_by_explorer"
                    ] = float(consumed_groups)
                    actor_info[
                        "actor/counterfactual_fixed_control_groups_discarded"
                    ] = float(max(len(fixed_control_groups) - consumed_groups, 0))
            payload = [feedback_data, actor_info, proposal_admission]
        dist.broadcast_object_list(payload, src=0)
        feedback_data, actor_info, proposal_admission = payload
        if not isinstance(feedback_data, list) or len(feedback_data) != int(
            self.args.num_samples
        ):
            raise RuntimeError("replicated free-form feedback broadcast failed")
        if not isinstance(actor_info, dict):
            raise RuntimeError("replicated free-form metric broadcast failed")
        if bool(
            getattr(
                self.args,
                "online_canonical_counterfactual_proposals",
                False,
            )
        ):
            if not isinstance(proposal_admission, dict):
                raise RuntimeError("counterfactual proposal admission broadcast failed")
            bank = getattr(self, "_online_canonical_bank", None)
            if not isinstance(bank, OnlineCanonicalBank):
                raise RuntimeError(
                    "counterfactual proposals require an online canonical bank"
                )
            key_mode = str(
                getattr(
                    self.args,
                    "online_canonical_key_mode",
                    "modebench_outcome",
                )
            )
            verified_route_mode = key_mode == "verified_route"
            base_payload_fields = (
                "prompt_token_ids",
                "outcome_keys",
                "response_token_ids",
            )
            route_payload_fields = (
                "endpoint_keys",
                "route_signatures",
                "verifier_ids",
                "proposal_mean_logprobs",
                "anchor_mean_logprobs",
            )
            payload_fields = base_payload_fields + (
                route_payload_fields if verified_route_mode else ()
            )
            if any(
                field not in proposal_admission
                or not isinstance(proposal_admission[field], list)
                for field in payload_fields
            ):
                raise RuntimeError(
                    "counterfactual proposal admission payload is incomplete"
                )
            proposal_count = len(proposal_admission["outcome_keys"])
            if any(
                len(proposal_admission[field]) != proposal_count
                for field in payload_fields
            ):
                raise RuntimeError(
                    "counterfactual proposal admission fields are misaligned"
                )
            if verified_route_mode and proposal_count > 1:
                raise RuntimeError(
                    "verified-route actuator may admit at most one proposal per update"
                )
            admission_compute_only = bool(
                getattr(
                    self.args,
                    "online_canonical_counterfactual_admission_compute_only",
                    False,
                )
            )
            if admission_compute_only:
                actor_info.update(
                    {
                        "actor/counterfactual_proposal_admitted_new_outcomes": 0.0,
                        "actor/counterfactual_proposal_stored_exemplars": 0.0,
                        "actor/counterfactual_proposal_cumulative_new_outcomes": float(
                            bank.proposal_new_outcomes
                        ),
                        "actor/counterfactual_proposal_bank_mean_support": float(
                            bank.replay_mean_support_per_prompt
                        ),
                        "actor/counterfactual_proposal_objective_bank_mean_support": float(
                            bank.mean_support_per_prompt
                        ),
                        "actor/counterfactual_proposal_objective_outcome_delta": 0.0,
                        "actor/counterfactual_proposal_objective_support_separated": float(
                            bank.separate_proposal_objective_support
                        ),
                        "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo": 0.0,
                        "actor/counterfactual_proposal_neutral_ppo_rows": float(
                            len(feedback_data)
                        ),
                        "actor/counterfactual_proposal_route_admitted": 0.0,
                        "actor/counterfactual_proposal_route_rejected_trust": 0.0,
                        "actor/counterfactual_proposal_route_already_known": 0.0,
                        "actor/counterfactual_proposal_endpoint_fallback_admitted": 0.0,
                        "actor/counterfactual_proposal_admission_compute_only": 1.0,
                        "actor/counterfactual_proposal_candidates_discarded": float(
                            proposal_count
                        ),
                    }
                )
                return feedback_data, actor_info
            objective_outcomes_before = bank.tracked_outcome_count
            admitted_new_outcomes = 0
            stored_exemplars = 0
            route_admitted = 0
            route_rejected_trust = 0
            route_already_known = 0
            endpoint_fallback_admitted = 0
            route_library: VerifiedRouteLibrary | None = None
            if verified_route_mode:
                route_library = getattr(
                    self,
                    "_verified_route_library",
                    None,
                )
                if not isinstance(route_library, VerifiedRouteLibrary):
                    raise RuntimeError(
                        "verified-route proposal admission lacks its route library"
                    )
                for row_index in range(proposal_count):
                    route_signature = proposal_admission["route_signatures"][row_index]
                    if route_signature is None:
                        endpoint_admission = bank.admit_verified_proposals(
                            prompt_token_ids=[
                                proposal_admission["prompt_token_ids"][row_index]
                            ],
                            outcome_keys=[
                                proposal_admission["endpoint_keys"][row_index]
                            ],
                            response_token_ids=[
                                proposal_admission["response_token_ids"][row_index]
                            ],
                        )
                        admitted_new_outcomes += endpoint_admission.new_outcomes
                        stored_exemplars += endpoint_admission.stored_exemplars
                        endpoint_fallback_admitted += endpoint_admission.new_outcomes
                        continue
                    verifier = proposal_admission["verifier_ids"][row_index]
                    endpoint_key = proposal_admission["endpoint_keys"][row_index]
                    proposal_mean_logprob = proposal_admission[
                        "proposal_mean_logprobs"
                    ][row_index]
                    anchor_mean_logprob = proposal_admission["anchor_mean_logprobs"][
                        row_index
                    ]
                    if not all(
                        isinstance(value, str) and value
                        for value in (
                            verifier,
                            endpoint_key,
                            route_signature,
                        )
                    ) or not all(
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                        for value in (
                            proposal_mean_logprob,
                            anchor_mean_logprob,
                        )
                    ):
                        raise RuntimeError(
                            "verified-route proposal admission metadata is invalid"
                        )
                    route_admission = route_library.admit_proposal(
                        prompt_token_ids=proposal_admission["prompt_token_ids"][
                            row_index
                        ],
                        verifier=verifier,
                        endpoint_key=endpoint_key,
                        route_signature=route_signature,
                        response_token_ids=proposal_admission["response_token_ids"][
                            row_index
                        ],
                        proposal_mean_logprob=float(proposal_mean_logprob),
                        anchor_mean_logprob=float(anchor_mean_logprob),
                    )
                    route_admitted += int(route_admission.admitted)
                    route_rejected_trust += int(route_admission.rejected_trust)
                    route_already_known += int(route_admission.already_known)
                    admitted_new_outcomes += int(route_admission.admitted)
                    stored_exemplars += int(route_admission.admitted)
            else:
                admission = bank.admit_verified_proposals(
                    prompt_token_ids=proposal_admission["prompt_token_ids"],
                    outcome_keys=proposal_admission["outcome_keys"],
                    response_token_ids=proposal_admission["response_token_ids"],
                )
                admitted_new_outcomes = admission.new_outcomes
                stored_exemplars = admission.stored_exemplars
            objective_outcome_delta = (
                bank.tracked_outcome_count - objective_outcomes_before
            )
            if verified_route_mode and objective_outcome_delta != 0:
                raise RuntimeError(
                    "verified-route proposals changed the neutral objective support"
                )
            route_diagnostics = (
                route_library.diagnostics() if route_library is not None else None
            )
            actor_info.update(
                {
                    "actor/counterfactual_proposal_admitted_new_outcomes": (
                        float(admitted_new_outcomes)
                    ),
                    "actor/counterfactual_proposal_stored_exemplars": float(
                        stored_exemplars
                    ),
                    "actor/counterfactual_proposal_cumulative_new_outcomes": (
                        float(
                            bank.proposal_new_outcomes
                            + (
                                route_diagnostics.proposal_rows_admitted
                                if route_diagnostics is not None
                                else 0
                            )
                        )
                    ),
                    "actor/counterfactual_proposal_bank_mean_support": float(
                        bank.replay_mean_support_per_prompt
                    ),
                    "actor/counterfactual_proposal_objective_bank_mean_support": float(
                        bank.mean_support_per_prompt
                    ),
                    "actor/counterfactual_proposal_objective_outcome_delta": float(
                        objective_outcome_delta
                    ),
                    "actor/counterfactual_proposal_objective_support_separated": float(
                        bank.separate_proposal_objective_support
                    ),
                    "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo": (0.0),
                    "actor/counterfactual_proposal_neutral_ppo_rows": float(
                        len(feedback_data)
                    ),
                    "actor/counterfactual_proposal_route_admitted": float(
                        route_admitted
                    ),
                    "actor/counterfactual_proposal_route_rejected_trust": float(
                        route_rejected_trust
                    ),
                    "actor/counterfactual_proposal_route_already_known": float(
                        route_already_known
                    ),
                    "actor/counterfactual_proposal_endpoint_fallback_admitted": (
                        float(endpoint_fallback_admitted)
                    ),
                    "actor/counterfactual_proposal_admission_compute_only": 0.0,
                    "actor/counterfactual_proposal_candidates_discarded": 0.0,
                }
            )
            self._record_counterfactual_starvation_outcome(
                actor_info,
                admitted_new_outcomes=admitted_new_outcomes,
            )
        return feedback_data, actor_info


    def _collect_dapo_dynamic_feedback(
        self,
        raw_prompts: list[str],
        processed_prompts: list[str],
        refs: list[str],
    ) -> tuple[list[TrajectoryData], dict[str, float]]:
        """Fill one DAPO group, discarding reward-constant generations.

        The official DAPO recipe oversamples generation batches and retains
        only prompts whose sampled rewards are neither all zero nor all one.
        This campaign trains on one prompt group at a time, so every rejected
        group is charged to the query budget and a deterministic fresh prompt
        is drawn from the frozen training dataset for the next attempt. A
        rejected row never enters the PPO buffer.
        """

        if not bool(getattr(self.args, "dapo_enabled", False)):
            raise RuntimeError("DAPO dynamic sampling was not enabled")
        if len(raw_prompts) != 1 or len(processed_prompts) != 1 or len(refs) != 1:
            raise RuntimeError("DAPO dynamic sampling requires one prompt group")
        if int(self.update_interval) != 1:
            raise RuntimeError("DAPO dynamic sampling requires update_interval=1")
        if len(self.prompts_dataset) <= 0:
            raise RuntimeError("DAPO dynamic sampling received an empty dataset")

        current_raw = list(raw_prompts)
        current_processed = list(processed_prompts)
        current_refs = list(refs)
        max_generation_batches = int(self.args.dapo_max_num_gen_batches)
        rejected_groups = 0
        all_zero_groups = 0
        all_one_groups = 0
        total_actor_time = 0.0

        for generation_batch in range(1, max_generation_batches + 1):
            feedback_data, actor_info = self.collector.collect_feedback(
                current_raw,
                current_processed,
                current_refs,
                self._same_actor_group,
            )
            if feedback_data is None:
                raise RuntimeError(
                    "DAPO actor returned no feedback while filling a training group"
                )
            if len(feedback_data) != int(self.args.num_samples):
                raise RuntimeError(
                    "DAPO actor returned "
                    f"{len(feedback_data)} rows, expected {self.args.num_samples}"
                )
            rewards = [float(trajectory.rewards[-1]) for trajectory in feedback_data]
            diagnostics = dapo_group_diagnostics(
                rewards,
                num_samples=int(self.args.num_samples),
            )
            if diagnostics.groups != 1:
                raise RuntimeError("DAPO one-prompt adapter produced multiple groups")
            all_zero_groups += diagnostics.all_zero_groups
            all_one_groups += diagnostics.all_one_groups
            total_actor_time += float(actor_info.get("actor/total_time", 0.0))

            if diagnostics.eligible_groups == 1:
                metrics = dict(actor_info)
                if total_actor_time > 0.0:
                    metrics["actor/total_time"] = total_actor_time
                metrics.update(
                    {
                        "actor/dapo_dynamic_sampling_enabled": 1.0,
                        "actor/dapo_generation_batches": float(generation_batch),
                        "actor/dapo_rejected_groups": float(rejected_groups),
                        "actor/dapo_all_zero_groups": float(all_zero_groups),
                        "actor/dapo_all_one_groups": float(all_one_groups),
                        "actor/dapo_accepted_groups": 1.0,
                        "actor/dapo_max_num_gen_batches": float(
                            max_generation_batches
                        ),
                    }
                )
                return feedback_data, metrics

            # Charge every discarded generation to the actual sampling budget,
            # but never expose it to process_feedback_data or the PPO buffer.
            self.query_step += len(feedback_data)
            self.prompt_consumed += len(feedback_data)
            rejected_groups += 1
            if generation_batch < max_generation_batches:
                prompt_index = dapo_resample_prompt_index(
                    base_seed=int(self.args.seed),
                    learner_step=int(self.steps),
                    generation_batch=generation_batch,
                    dataset_size=len(self.prompts_dataset),
                )
                processed_prompt, raw_prompt, reference = self.prompts_dataset[
                    prompt_index
                ]
                current_processed = [processed_prompt]
                current_raw = [raw_prompt]
                current_refs = [reference]

        raise RuntimeError(
            "DAPO dynamic sampling could not produce a non-constant reward group "
            f"within {max_generation_batches} generation batches at learner "
            f"step {self.steps}; rejected={rejected_groups}, "
            f"all_zero={all_zero_groups}, all_one={all_one_groups}"
        )


    def _record_counterfactual_starvation_outcome(
        self,
        metrics: dict[str, float],
        *,
        admitted_new_outcomes: int,
    ) -> None:
        """Commit the rank-consistent fallback transition after bank admission."""

        controller = getattr(self, "_proposal_starvation_controller", None)
        if controller is None:
            return
        eligible = bool(
            metrics.get(
                "actor/counterfactual_proposal_starvation_eligible_update",
                0.0,
            )
        )
        fallback_active = False
        if eligible:
            plan = controller.plan()
            registered_attempts = int(
                metrics.get("actor/counterfactual_proposal_max_attempts", -1.0)
            )
            registered_active = bool(
                metrics.get(
                    "actor/counterfactual_proposal_starvation_fallback_active",
                    0.0,
                )
            )
            registered_activation = bool(
                metrics.get(
                    "actor/counterfactual_proposal_starvation_fallback_activation",
                    0.0,
                )
            )
            if (
                registered_attempts != plan.max_attempts
                or registered_active != plan.fallback_active
                or registered_activation != plan.fallback_activation
            ):
                raise RuntimeError(
                    "proposal starvation plan diverged across learner ranks"
                )
            fallback_active = plan.fallback_active
            controller.observe(
                plan,
                admitted_new_outcomes=admitted_new_outcomes,
            )
        elif admitted_new_outcomes > 0:
            controller.observe_admission_without_opportunity(
                admitted_new_outcomes=admitted_new_outcomes,
            )

        diagnostics = controller.diagnostics()
        metrics.update(
            {
                "actor/counterfactual_proposal_starvation_stalled_updates_after": float(
                    diagnostics["stalled_eligible_updates"]
                ),
                "actor/counterfactual_proposal_starvation_burst_remaining": float(
                    diagnostics["burst_remaining"]
                ),
                "actor/counterfactual_proposal_starvation_cooldown_remaining": float(
                    diagnostics["cooldown_remaining"]
                ),
                "actor/counterfactual_proposal_starvation_eligible_updates_cumulative": float(
                    diagnostics["eligible_updates"]
                ),
                "actor/counterfactual_proposal_starvation_fallback_updates_cumulative": float(
                    diagnostics["fallback_updates"]
                ),
                "actor/counterfactual_proposal_starvation_fallback_activations_cumulative": float(
                    diagnostics["fallback_activations"]
                ),
                "actor/counterfactual_proposal_starvation_admissions_cumulative": float(
                    diagnostics["admitted_new_outcomes"]
                ),
                "actor/counterfactual_proposal_starvation_fallback_admitted_new_outcomes": float(
                    admitted_new_outcomes if fallback_active else 0
                ),
            }
        )


    def _sample_and_admit_canonical_counterfactual_proposals(
        self,
        *,
        raw_prompts: list[str],
        processed_prompts: list[str],
        refs: list[str],
        neutral_feedback: list[TrajectoryData],
        actor_info: dict[str, float],
    ) -> dict[str, float]:
        """Discover Pantry modes with a second learner sample, never PPO rows."""

        if resolve_canonical_action_task(self.args) != "pantry_support_mask":
            raise RuntimeError(
                "canonical counterfactual proposals are implemented only for "
                "the validator-complete Pantry support-mask policy"
            )
        bank = getattr(self, "_online_canonical_bank", None)
        if not isinstance(bank, OnlineCanonicalBank):
            raise RuntimeError("canonical proposals require an online canonical bank")
        if not bank.separate_proposal_objective_support:
            raise RuntimeError("canonical proposals require separate objective support")

        proposal_sampler_infos: list[dict[str, float]] = []

        def sample_group(
            _attempt_index: int,
            sampling_temperature: float,
            sampling_seed: int,
        ) -> list[TrajectoryData]:
            proposal_feedback, sampler_info = (
                self._sample_canonical_feedback_with_learner(
                    raw_prompts,
                    processed_prompts,
                    refs,
                    sampling_seed=sampling_seed,
                    sampling_temperature=sampling_temperature,
                    set_entropy_pending=False,
                )
            )
            proposal_sampler_infos.append(sampler_info)
            return proposal_feedback

        proposal_admission, proposal_metrics = (
            self._generate_verified_counterfactual_proposals(
                actor=None,
                raw_prompts=raw_prompts,
                processed_prompts=processed_prompts,
                refs=refs,
                neutral_feedback=neutral_feedback,
                proposal_group_factory=sample_group,
            )
        )
        for field in ("prompt_token_ids", "outcome_keys", "response_token_ids"):
            if field not in proposal_admission or not isinstance(
                proposal_admission[field], list
            ):
                raise RuntimeError("canonical proposal admission payload is incomplete")
        proposal_count = len(proposal_admission["outcome_keys"])
        if any(
            len(proposal_admission[field]) != proposal_count
            for field in ("prompt_token_ids", "outcome_keys", "response_token_ids")
        ):
            raise RuntimeError("canonical proposal admission fields are misaligned")

        admission_compute_only = bool(
            getattr(
                self.args,
                "online_canonical_counterfactual_admission_compute_only",
                False,
            )
        )
        if admission_compute_only:
            proposal_metrics.update(
                {
                    "actor/counterfactual_proposal_admitted_new_outcomes": 0.0,
                    "actor/counterfactual_proposal_stored_exemplars": 0.0,
                    "actor/counterfactual_proposal_cumulative_new_outcomes": float(
                        bank.proposal_new_outcomes
                    ),
                    "actor/counterfactual_proposal_bank_mean_support": float(
                        bank.replay_mean_support_per_prompt
                    ),
                    "actor/counterfactual_proposal_objective_bank_mean_support": float(
                        bank.mean_support_per_prompt
                    ),
                    "actor/counterfactual_proposal_objective_outcome_delta": 0.0,
                    "actor/counterfactual_proposal_objective_support_separated": 1.0,
                    "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo": 0.0,
                    "actor/counterfactual_proposal_neutral_ppo_rows": float(
                        len(neutral_feedback)
                    ),
                    "actor/counterfactual_proposal_route_admitted": 0.0,
                    "actor/counterfactual_proposal_route_rejected_trust": 0.0,
                    "actor/counterfactual_proposal_route_already_known": 0.0,
                    "actor/counterfactual_proposal_endpoint_fallback_admitted": 0.0,
                    "actor/counterfactual_proposal_canonical_learner_sampler": 1.0,
                    "actor/counterfactual_proposal_canonical_sampler_groups": float(
                        len(proposal_sampler_infos)
                    ),
                    "actor/counterfactual_proposal_admission_compute_only": 1.0,
                    "actor/counterfactual_proposal_candidates_discarded": float(
                        proposal_count
                    ),
                }
            )
            actor_info.update(proposal_metrics)
            return actor_info

        objective_outcomes_before = bank.tracked_outcome_count
        admission = bank.admit_verified_proposals(
            prompt_token_ids=proposal_admission["prompt_token_ids"],
            outcome_keys=proposal_admission["outcome_keys"],
            response_token_ids=proposal_admission["response_token_ids"],
        )
        objective_outcome_delta = bank.tracked_outcome_count - objective_outcomes_before
        if objective_outcome_delta != 0:
            raise RuntimeError(
                "canonical proposal changed the neutral objective support"
            )
        proposal_metrics.update(
            {
                "actor/counterfactual_proposal_admitted_new_outcomes": float(
                    admission.new_outcomes
                ),
                "actor/counterfactual_proposal_stored_exemplars": float(
                    admission.stored_exemplars
                ),
                "actor/counterfactual_proposal_cumulative_new_outcomes": float(
                    bank.proposal_new_outcomes
                ),
                "actor/counterfactual_proposal_bank_mean_support": float(
                    bank.replay_mean_support_per_prompt
                ),
                "actor/counterfactual_proposal_objective_bank_mean_support": float(
                    bank.mean_support_per_prompt
                ),
                "actor/counterfactual_proposal_objective_outcome_delta": float(
                    objective_outcome_delta
                ),
                "actor/counterfactual_proposal_objective_support_separated": 1.0,
                "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo": 0.0,
                "actor/counterfactual_proposal_neutral_ppo_rows": float(
                    len(neutral_feedback)
                ),
                "actor/counterfactual_proposal_route_admitted": 0.0,
                "actor/counterfactual_proposal_route_rejected_trust": 0.0,
                "actor/counterfactual_proposal_route_already_known": 0.0,
                "actor/counterfactual_proposal_endpoint_fallback_admitted": 0.0,
                "actor/counterfactual_proposal_canonical_learner_sampler": 1.0,
                "actor/counterfactual_proposal_canonical_sampler_groups": float(
                    len(proposal_sampler_infos)
                ),
                "actor/counterfactual_proposal_admission_compute_only": 0.0,
                "actor/counterfactual_proposal_candidates_discarded": 0.0,
            }
        )
        actor_info.update(proposal_metrics)
        self._record_counterfactual_starvation_outcome(
            actor_info,
            admitted_new_outcomes=admission.new_outcomes,
        )
        return actor_info


    def _update_xdr_tau_controller(self, train_info: dict[str, Any]) -> None:
        """Advance the optional controller from a global entropy observation."""

        tau_controller = getattr(self, "_xdr_tau_controller", None)
        if tau_controller is None:
            return
        local_entropy = self._coerce_log_float(train_info.get("entropy"))
        if local_entropy is None:
            raise RuntimeError(
                "xDr tau control requires a finite train/entropy observation"
            )
        # All ranks must use the same tau on the next update. Reduce the
        # observation now rather than relying on the less-frequent logging
        # reduction in eval_and_log().
        observation_key = getattr(
            tau_controller,
            "observation_metric_key",
            "xdr_tau_control_observed_entropy",
        )
        reduced = self.strategy.all_reduce({observation_key: local_entropy})
        global_entropy = self._coerce_log_float(reduced.get(observation_key))
        if global_entropy is None:
            raise RuntimeError(
                "xDr entropy controller received an invalid distributed entropy"
            )
        train_info.update(tau_controller.observe(global_entropy))


    def _update_maxent_alpha_controller(self, train_info: dict[str, Any]) -> None:
        """Advance direct MaxEnt control from its own entropy estimator."""

        controller = getattr(self, "_maxent_alpha_controller", None)
        if controller is None:
            return
        metric_key = getattr(
            controller, "observation_metric_key", "maxent_sequence_entropy"
        )
        local_entropy = self._coerce_log_float(train_info.get(metric_key))
        if local_entropy is None:
            raise RuntimeError(
                "direct MaxEnt control requires a finite sequence-entropy observation"
            )
        reduced = self.strategy.all_reduce({metric_key: local_entropy})
        global_entropy = self._coerce_log_float(reduced.get(metric_key))
        if global_entropy is None:
            raise RuntimeError(
                "direct MaxEnt controller received invalid distributed entropy"
            )
        train_info.update(controller.observe(global_entropy))


    def _update_maxent_length_controller(self, train_info: dict[str, Any]) -> None:
        """Advance the expected-length multiplier from its own IS estimate."""

        controller = getattr(self, "_maxent_length_controller", None)
        if controller is None:
            return
        metric_key = getattr(
            controller, "observation_metric_key", "maxent_expected_length"
        )
        local_length = self._coerce_log_float(train_info.get(metric_key))
        if local_length is None:
            raise RuntimeError(
                "MaxEnt length control requires a finite expected-length observation"
            )
        reduced = self.strategy.all_reduce({metric_key: local_length})
        global_length = self._coerce_log_float(reduced.get(metric_key))
        if global_length is None:
            raise RuntimeError(
                "MaxEnt length controller received invalid distributed length"
            )
        train_info.update(controller.observe(global_length))


    def _update_online_canonical_alpha_controller(
        self, train_info: dict[str, Any]
    ) -> None:
        """Advance canonical alpha from the configured detached sensor."""

        controller = getattr(self, "_online_canonical_alpha_controller", None)
        if controller is None:
            return
        if getattr(controller, "observation_metric_key", None) == "entropy":
            local_entropy = self._coerce_log_float(train_info.get("entropy"))
            if local_entropy is None:
                raise RuntimeError(
                    "online canonical policy-entropy adaptation requires a "
                    "finite train/entropy observation"
                )
            reduced = self.strategy.all_reduce(
                {"online_canonical_policy_entropy_observed": (local_entropy)}
            )
            global_entropy = self._coerce_log_float(
                reduced.get("online_canonical_policy_entropy_observed")
            )
            if global_entropy is None:
                raise RuntimeError(
                    "online canonical policy-entropy controller received an "
                    "invalid distributed observation"
                )
            train_info.update(controller.observe(global_entropy))
            return
        ratio_key = controller.observation_metric_key
        eligibility_key = controller.eligibility_metric_key
        local_ratio = self._coerce_log_float(train_info.get(ratio_key))
        local_eligibility = self._coerce_log_float(train_info.get(eligibility_key))
        if local_ratio is None or local_eligibility is None:
            raise RuntimeError(
                "online canonical dual control requires normalized bank "
                "entropy and eligibility diagnostics"
            )
        if local_eligibility < 0 or local_eligibility > 1:
            raise RuntimeError(
                "online canonical normalized-entropy eligibility left [0, 1]"
            )
        reduced = self.strategy.all_reduce(
            {
                "online_canonical_dual_ratio_weighted": (
                    local_ratio * local_eligibility
                ),
                "online_canonical_dual_eligibility_weight": local_eligibility,
            }
        )
        global_weight = self._coerce_log_float(
            reduced.get("online_canonical_dual_eligibility_weight")
        )
        global_weighted_ratio = self._coerce_log_float(
            reduced.get("online_canonical_dual_ratio_weighted")
        )
        if global_weight is None or global_weighted_ratio is None:
            raise RuntimeError(
                "online canonical dual controller received invalid "
                "distributed diagnostics"
            )
        if global_weight <= 0:
            train_info.update(controller.idle_diagnostics())
            return
        global_ratio = global_weighted_ratio / global_weight
        diagnostics = controller.observe(global_ratio)
        diagnostics["online_canonical_dual_observation_skipped"] = 0.0
        diagnostics["online_canonical_dual_global_eligibility_weight"] = global_weight
        train_info.update(diagnostics)


