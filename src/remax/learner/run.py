"""Learner run loop, data setup, evaluation, and logging helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from oat.interface import lp
from oat.types import TrajectoryData
from oat.utils.data import PromptDataset, load_data_from_disk_or_hf
from oat.utils.distributed import (
    init_process_group,
    node_ip_address_from_perspective,
    torch_type_codec,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ..answer_options import conditional_answer_repr, crossfit_option_answer_mi
from ..args import resolve_canonical_action_task
from ..canonical_actions import (
    CanonicalActionSpace,
    _sample_canonical_actions_from_logits,
    decode_canonical_action_response,
)
from ..dapo import dapo_group_diagnostics, dapo_resample_prompt_index
from ..logging_utils import filter_wandb_logs
from ..math_grader import (
    VerifiedExplorationIdentity,
    boxed_reward_fn,
    validated_exploration_identity,
    validated_modebench_outcome_key,
)
from ..online_canonical_bank import OnlineCanonicalBank
from ..rlep import OnlineRLEPExperiencePool
from ..replicated_group import validate_replicated_group_layout
from ..resume_state import (
    discover_local_wandb_resume_run,
    resolve_resume_progress_state,
)
from modebench.templates import (
    CANONICAL_TASK_PROMPT_TEMPLATES,
    apply_prompt_template_to_example,
    collate_eval_prompt_items,
    validate_canonical_prompt_materialization,
)
from ..verified_transformations import (
    derive_countdown_action_neighborhood_counterfactuals,
    derive_validator_preserving_counterfactuals,
)
from ..verified_route_library import VerifiedRouteLibrary


def _derive_freeform_request_seed(
    *,
    base_seed: int,
    prompt_batch_index: int,
    stream: str,
) -> int:
    """Derive an isolated deterministic vLLM request stream.

    Extra proposal requests must not advance or otherwise perturb subsequent
    neutral rollouts. Every replicated free-form neutral and proposal request
    therefore has an explicit seed derived from disjoint named streams.
    """

    if isinstance(base_seed, bool) or int(base_seed) != base_seed:
        raise ValueError("free-form request base seed must be an integer")
    if (
        isinstance(prompt_batch_index, bool)
        or int(prompt_batch_index) != prompt_batch_index
        or int(prompt_batch_index) < 0
    ):
        raise ValueError(
            "free-form request prompt-batch index must be a non-negative integer"
        )
    if not isinstance(stream, str) or not stream:
        raise ValueError("free-form request stream must be a non-empty string")
    payload = (
        f"replicated-freeform-v1|{int(base_seed)}|{int(prompt_batch_index)}|{stream}"
    ).encode("utf-8")
    # Keep the value exactly representable in IEEE-754 telemetry while leaving
    # a collision-resistant namespace for long multi-domain campaigns.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 52) - 1)


def _parse_answer_mode_count(ref: str) -> int:
    """Extract the number of valid answer modes from a modebench reference JSON."""
    try:
        spec = json.loads(ref) if isinstance(ref, str) else ref
        if bool(spec.get("support_is_open", False)):
            return 0
        return max(int(spec.get("num_completions", 1)), 1)
    except Exception:
        return 1


def _parse_public_seed_key(ref: str) -> str | None:
    """Return a public MathIR seed strategy key when the prompt declares one."""

    try:
        spec = json.loads(ref) if isinstance(ref, str) else ref
        key = spec.get("public_seed_key")
        return str(key) if key else None
    except Exception:
        return None


def _trajectory_mean_logprob(trajectory: TrajectoryData) -> float:
    values = [float(value) for value in trajectory.response_logprobs]
    if not values or any(not math.isfinite(value) or value > 1e-8 for value in values):
        raise RuntimeError(
            "proposal trust checking requires finite non-positive response "
            "log probabilities"
        )
    return float(sum(values) / len(values))


def _exploration_support_key(
    identity: VerifiedExplorationIdentity,
) -> str:
    """Prefer transferable route identity, with endpoint fallback for Graph."""

    return str(identity.route_signature or identity.endpoint_key)


def _compute_mode_coverage_metrics(
    rewards: list[float],
    answer_keys: list,
    answer_mode_count: int,
    public_seed_key: str | None = None,
) -> dict[str, float]:
    k = len(rewards)
    correct_keys = {
        str(key)
        for reward, key in zip(rewards, answer_keys)
        if float(reward) > 0.0 and key is not None
    }
    distinct = len(correct_keys)
    nonseed_keys = (
        correct_keys - {str(public_seed_key)} if public_seed_key is not None else set()
    )
    total = int(answer_mode_count)
    return {
        "any_correct_at_k": float(any(float(r) > 0.0 for r in rewards)),
        "mean_at_k": float(sum(float(r) for r in rewards) / k),
        "distinct_correct_modes_at_k": float(distinct),
        "distinct_nonseed_correct_modes_at_k": float(len(nonseed_keys)),
        "any_nonseed_correct_at_k": float(bool(nonseed_keys)),
        # A growing-support task has no known exhaustive denominator.  Preserve
        # distinct validated outcomes while refusing to report fake coverage.
        "mode_coverage_at_k": (
            float(distinct) / float(total) if total > 0 else float("nan")
        ),
    }


MODE_COVERAGE_METRICS = (
    "mode_coverage_at_k",
    "any_correct_at_k",
    "mean_at_k",
    "distinct_correct_modes_at_k",
    "distinct_nonseed_correct_modes_at_k",
    "any_nonseed_correct_at_k",
)

OPTION_BINDING_METRICS = (
    "option_answer_mi_lower_bound_nats",
    "option_classifier_accuracy",
    "option_eligible_fraction",
    "option_correct_rate_range",
)


def _mode_coverage_log_key(
    benchmark_name: str,
    metric: str,
    k: int,
    *,
    metric_namespace: str = "neutral",
) -> str:
    names = {
        "mode_coverage_at_k": "sampled_mode_coverage",
        "any_correct_at_k": "sampled_any_correct",
        "mean_at_k": "sampled_mean",
        "distinct_correct_modes_at_k": "sampled_distinct_correct",
        "distinct_nonseed_correct_modes_at_k": "sampled_distinct_nonseed_correct",
        "any_nonseed_correct_at_k": "sampled_any_nonseed_correct",
        "option_answer_mi_lower_bound_nats": (
            "sampled_option_answer_mi_lower_bound_nats"
        ),
        "option_classifier_accuracy": "sampled_option_classifier_accuracy",
        "option_eligible_fraction": "sampled_option_eligible_fraction",
        "option_correct_rate_range": "sampled_option_correct_rate_range",
    }
    name = names[metric]
    if metric_namespace == "latent":
        if metric not in MODE_COVERAGE_METRICS:
            raise ValueError("only quality metrics support the latent namespace")
        name = name.replace("sampled_", "sampled_latent_", 1)
    elif metric_namespace != "neutral":
        raise ValueError(f"unknown mode-coverage metric namespace: {metric_namespace}")
    return f"eval/{benchmark_name}/{name}_at_{k}"


def _summarize_mode_coverage_draws(
    draw_means: list[dict[str, float]],
    benchmark_name: str,
    k: int,
    *,
    metric_namespace: str = "neutral",
) -> dict[str, float]:
    """Flatten raw draw means and their Monte Carlo spread into log metrics."""

    if not draw_means:
        return {}
    summary: dict[str, float] = {}
    for metric in draw_means[0]:
        key = _mode_coverage_log_key(
            benchmark_name,
            metric,
            k,
            metric_namespace=metric_namespace,
        )
        values = np.asarray([draw[metric] for draw in draw_means], dtype=float)
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary[key] = float(np.mean(values))
        summary[f"{key}_draw_count"] = float(len(values))
        summary[f"{key}_draw_std"] = standard_deviation
        summary[f"{key}_draw_se"] = standard_deviation / math.sqrt(len(values))
        summary[f"{key}_draw_min"] = float(np.min(values))
        summary[f"{key}_draw_max"] = float(np.max(values))
        for draw_index, value in enumerate(values):
            summary[f"{key}_draw_{draw_index}"] = float(value)
    return summary


class ZeroMathRunMixin:
    """Data, training-loop, checkpoint, W&B, and eval orchestration."""

    def _ensure_training_progress_state(self) -> None:
        if hasattr(self, "_progress_metric_baselines"):
            return
        self._progress_metric_baselines: dict[str, float] = {}
        self._progress_metric_previous: dict[str, float] = {}
        self._progress_metric_bests: dict[str, float] = {}
        self._progress_metric_best_steps: dict[str, int] = {}
        self._actor_reward_ema: float | None = None
        self._actor_reward_ema_start: float | None = None

    def _init_local_actor_weight_sync(self) -> None:
        """Replace rank-zero fanout with four concurrent learner/actor pairs."""

        if not bool(getattr(self.args, "local_actor_weight_sync", False)):
            return
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if int(self.args.num_gpus_per_actor) != 1:
            raise RuntimeError("local actor sync requires one GPU per actor")
        if len(self.actors) != world_size:
            raise RuntimeError(
                "local actor sync requires exactly one actor per learner rank"
            )
        master_addr = node_ip_address_from_perspective()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        group_name = f"oat_local_actor_sync_{rank}"
        actor = self.actors[rank]
        actor_future = actor.futures.init_process_group(
            master_addr,
            master_port,
            1,
            2,
            group_name,
            "gloo",
        )
        local_group = init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=2,
            rank=0,
            group_name=group_name,
        )
        actor_future.result()
        self._local_sync_actor = actor
        self._local_model_update_group = local_group
        dist.barrier()
        logging.info(
            "local actor weight-sync group initialized learner_rank=%s actor=%s",
            rank,
            rank,
        )

    def sync_params_to_actors(self):
        if not bool(getattr(self.args, "local_actor_weight_sync", False)):
            return super().sync_params_to_actors()

        started = time.time()
        dist.barrier()
        actor = self._local_sync_actor
        reset_future = (
            actor.futures.reset_prefix_cache()
            if self.args.enable_prefix_caching
            else None
        )
        model = self.model.model.module
        parameters = list(model.named_parameters())
        torch.cuda.empty_cache()
        for index, (name, param) in enumerate(parameters, start=1):
            actor_future = actor.futures.update_weight(
                name,
                dtype=torch_type_codec(param.dtype),
                shape=param.shape,
                empty_cache=index == len(parameters),
            )
            dist.broadcast(
                param.data,
                0,
                group=self._local_model_update_group,
            )
            actor_future.result()
        if reset_future is not None:
            reset_future.result()
        if int(getattr(self.args, "vllm_sleep_level", 1)) == 2:
            actor.wake_up(["kv_cache"])
        torch.cuda.empty_cache()
        dist.barrier()
        self.pi_beta_version += 1
        self.pi_beta_lags_behind = False
        self.weight_sync_elapse = time.time() - started
        logging.info(
            "weights @version=%s broadcasted through local actor pair in %.3fs",
            self.pi_beta_version,
            self.weight_sync_elapse,
        )

    def _pre_learning(self):
        if int(getattr(self.args, "vllm_sleep_level", 1)) != 2:
            return super()._pre_learning()
        started = time.time()
        torch.cuda.synchronize()
        dist.barrier()
        if self.strategy.is_group_rank_0():
            backup_futures = [
                actor.futures.backup_model_buffers() for actor in self.actors
            ]
            backup_reports = [future.result() for future in backup_futures]
            logging.info("backed up vLLM model buffers: %s", backup_reports)
            futures = [actor.futures.sleep(2) for actor in self.actors]
            _ = [future.result() for future in futures]
        torch.cuda.synchronize()
        dist.barrier()
        self.vllm_go_sleep_time = time.time() - started
        logging.info(
            "vLLM actors entered discard-mode sleep in %.3fs",
            self.vllm_go_sleep_time,
        )

    def _post_learning(self):
        if int(getattr(self.args, "vllm_sleep_level", 1)) != 2:
            return super()._post_learning()
        started = time.time()
        torch.cuda.synchronize()
        dist.barrier()
        if self.strategy.is_group_rank_0():
            futures = [actor.futures.wake_up(["weights"]) for actor in self.actors]
            _ = [future.result() for future in futures]
            restore_futures = [
                actor.futures.restore_model_buffers() for actor in self.actors
            ]
            restore_reports = [future.result() for future in restore_futures]
            logging.info("restored vLLM model buffers: %s", restore_reports)
        torch.cuda.synchronize()
        dist.barrier()
        self.vllm_wake_up_time = time.time() - started
        self.pi_beta_lags_behind = True
        logging.info(
            "vLLM actor weight storage remapped in %.3fs",
            self.vllm_wake_up_time,
        )

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

    def _sample_canonical_feedback_with_learner(
        self,
        raw_prompts: list[str],
        processed_prompts: list[str],
        refs: list[str],
        *,
        sampling_seed: int | None = None,
        sampling_temperature: float | None = None,
        set_entropy_pending: bool = True,
    ) -> tuple[list[TrajectoryData], dict[str, float]]:
        """Sample a three-position canonical policy with the HF learner.

        This deliberately bypasses vLLM for training rollouts.  Each action is
        sampled after a fresh autoregressive learner forward, and the complete
        three-way behavior distribution is transported into the PPO dataset.

        Every forward has exactly the same ``[microbatch, prompt + horizon]``
        shape and all-ones attention mask as subsequent teacher-forced
        old-policy scoring.  Unsampled suffix positions contain a canonical
        support token and are causally invisible to the prefix logit being
        read.  Keeping sequence width, microbatch layout, and model mode fixed
        avoids BF16 kernel drift from otherwise equivalent variable-width
        prefix forwards.
        """

        canonical_task = resolve_canonical_action_task(self.args)
        if canonical_task == "none":
            raise RuntimeError(
                "learner-side canonical sampling requires canonical mode"
            )
        if not bool(getattr(self.args, "canonical_graph_learner_sampling", False)):
            raise RuntimeError("learner-side canonical sampling was not enabled")
        if not bool(getattr(self.args, "canonical_graph_fixed_shape_sampling", False)):
            raise RuntimeError(
                "learner-side canonical sampling requires the frozen fixed-shape "
                "causal-placeholder path"
            )
        if len(raw_prompts) != len(processed_prompts) or len(raw_prompts) != len(refs):
            raise RuntimeError("canonical prompt/reference batch lengths do not match")
        if not raw_prompts:
            return [], {}
        if len(raw_prompts) != 1:
            raise RuntimeError(
                "canonical learner-side sampling requires exactly one prompt per "
                "rollout so sampler and old-policy microbatches have identical "
                "row order"
            )
        if dist.get_world_size() != 1:
            raise RuntimeError(
                "canonical learner-side sampling is frozen to one learner rank"
            )
        if int(self.update_interval) != 1:
            raise RuntimeError(
                "canonical learner-side sampling requires update_interval=1 so the "
                "sampled behavior policy is the immediately updated policy"
            )

        action_space = getattr(self, "_canonical_action_space", None)
        if action_space is None and canonical_task == "graph_coloring":
            # Backward-compatible construction for retained E14 harnesses and
            # old callers that initialized only the legacy union support.
            legacy_support = tuple(
                int(value) for value in self._canonical_action_token_ids or ()
            )
            if len(legacy_support) == 3:
                action_space = CanonicalActionSpace(
                    task="graph_coloring",
                    action_strings_by_position=(("1", "2", "3"),) * 3,
                    token_ids_by_position=(legacy_support,) * 3,
                )
                self._canonical_action_space = action_space
                self._canonical_action_token_ids_by_position = (
                    action_space.token_ids_by_position
                )
        if action_space is None:
            raise RuntimeError("canonical learner action space was not initialized")
        supports = tuple(
            tuple(int(value) for value in support)
            for support in action_space.token_ids_by_position
        )
        union_support = tuple(int(value) for value in action_space.union_token_ids)
        horizon = action_space.horizon
        num_samples = int(self.args.num_samples)
        if horizon <= 0 or any(not support for support in supports):
            raise RuntimeError(
                "canonical learner sampler requires a positive horizon and "
                "nonempty positional supports"
            )
        micro_batch_size = int(self.args.train_batch_size_per_device)
        if micro_batch_size <= 0:
            raise RuntimeError("canonical learner sampling needs a positive microbatch")

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cuda", torch.cuda.current_device())
        request_seed = (
            int(sampling_seed)
            if sampling_seed is not None
            else (
                int(self.args.seed)
                + 1_000_003 * int(self.steps)
                + int(self._prompt_batches_consumed_total)
            )
        )
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(request_seed)
        uniforms = torch.rand(
            (len(raw_prompts), num_samples, horizon),
            generator=cpu_generator,
            dtype=torch.float32,
        )

        records: list[dict[str, Any]] = []
        normalization_error_max = 0.0
        generate_start = time.time()
        model_was_training = bool(self.model.training)
        self.model.eval()
        try:
            with torch.no_grad():
                for prompt_index, (raw_prompt, processed_prompt, ref) in enumerate(
                    zip(raw_prompts, processed_prompts, refs)
                ):
                    prompt_ids = list(self.tokenizer.encode(processed_prompt))
                    if not prompt_ids:
                        raise RuntimeError("canonical learner received an empty prompt")
                    if len(prompt_ids) > int(self.args.prompt_max_length):
                        raise RuntimeError(
                            "canonical learner prompt exceeds prompt_max_length: "
                            f"{len(prompt_ids)} > {self.args.prompt_max_length}"
                        )
                    # A support-token suffix is semantically inert at the prefix
                    # position under causal attention.  Unlike a zero attention
                    # mask, an all-ones mask also preserves the exact attention
                    # kernel selected by full teacher forcing.
                    sequences = [
                        list(prompt_ids) + [support[0] for support in supports]
                        for _ in range(num_samples)
                    ]
                    responses: list[list[int]] = [[] for _ in range(num_samples)]
                    selected_traces: list[list[float]] = [
                        [] for _ in range(num_samples)
                    ]
                    full_traces: list[list[list[float]]] = [
                        [] for _ in range(num_samples)
                    ]

                    for action_position in range(horizon):
                        position_support = supports[action_position]
                        for batch_start in range(0, num_samples, micro_batch_size):
                            batch_end = min(batch_start + micro_batch_size, num_samples)
                            batch_sequences = sequences[batch_start:batch_end]
                            input_ids = torch.tensor(
                                batch_sequences, dtype=torch.long, device=device
                            )
                            attention_mask = torch.ones_like(input_ids)
                            prefix_last_position = len(prompt_ids) + action_position - 1
                            next_logits = self.model(
                                input_ids, attention_mask=attention_mask
                            )["logits"][:, prefix_last_position, :]
                            temperature = float(
                                self.args.temperature
                                if sampling_temperature is None
                                else sampling_temperature
                            )
                            if not np.isfinite(temperature) or temperature <= 0:
                                raise RuntimeError(
                                    "canonical learner sampling requires a finite "
                                    "positive temperature"
                                )
                            if temperature != 1.0:
                                next_logits = next_logits / temperature
                            (
                                sampled_token_ids,
                                selected_log_probs,
                                full_log_probs,
                            ) = _sample_canonical_actions_from_logits(
                                next_logits,
                                allowed_token_ids=position_support,
                                uniforms=uniforms[
                                    prompt_index,
                                    batch_start:batch_end,
                                    action_position,
                                ],
                            )
                            normalization_error_max = max(
                                normalization_error_max,
                                float(
                                    torch.abs(
                                        torch.logsumexp(full_log_probs.double(), dim=-1)
                                    )
                                    .max()
                                    .detach()
                                    .cpu()
                                    .item()
                                ),
                            )
                            for local_index, sample_index in enumerate(
                                range(batch_start, batch_end)
                            ):
                                token_id = int(sampled_token_ids[local_index].item())
                                responses[sample_index].append(token_id)
                                sequences[sample_index][
                                    len(prompt_ids) + action_position
                                ] = token_id
                                selected_traces[sample_index].append(
                                    float(selected_log_probs[local_index].item())
                                )
                                full_traces[sample_index].append(
                                    [
                                        float(value)
                                        for value in full_log_probs[local_index]
                                        .detach()
                                        .cpu()
                                        .tolist()
                                    ]
                                )

                    for sample_index in range(num_samples):
                        records.append(
                            {
                                "prompt": raw_prompt,
                                "prompt_ids": prompt_ids,
                                "reference": ref,
                                "response_ids": responses[sample_index],
                                "selected_log_probs": selected_traces[sample_index],
                                "full_log_probs": full_traces[sample_index],
                            }
                        )
        finally:
            if model_was_training:
                self.model.train()

        generate_time = time.time() - generate_start
        verify_start = time.time()
        formatted_values: list[float] = []
        rewards: list[float] = []
        for record in records:
            response_code = self.tokenizer.decode(
                record["response_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            expected_strings = action_space.action_strings_by_position
            if len(response_code) != horizon or any(
                action not in expected_strings[position]
                for position, action in enumerate(response_code)
            ):
                raise RuntimeError(
                    "canonical learner serialization escaped the frozen action "
                    f"space: {response_code!r}"
                )
            response = decode_canonical_action_response(
                canonical_task, response_code, record["reference"]
            )
            record["response"] = response
            record["response_code"] = response_code
            oracle_info, reward = boxed_reward_fn(
                response,
                record["reference"],
                fast=self.args.verifier_version == "fast",
            )
            formatted = bool(oracle_info.get("formatted", False))
            numeric_reward = float(reward)
            if not formatted:
                raise RuntimeError(
                    "canonical learner verifier rejected a valid action serialization"
                )
            if not np.isfinite(numeric_reward) or numeric_reward not in {0.0, 1.0}:
                raise RuntimeError(
                    "canonical learner verifier must return a finite binary reward; "
                    f"got {reward!r}"
                )
            rewards.append(numeric_reward)
            formatted_values.append(1.0)
        verify_time = time.time() - verify_start

        info: dict[str, float] = {
            "actor/generate_time": generate_time,
            "actor/verify_time": verify_time,
            "actor/rewards": float(np.mean(rewards)),
            "actor/num_data": float(len(records)),
            "actor/formatted": float(np.mean(formatted_values)),
            "actor/response_tok_len": float(horizon),
            "actor/generate_avg_str_len": float(horizon),
            "actor/sampling_max_tokens": float(horizon),
            "actor/sampling_temperature": float(
                self.args.temperature
                if sampling_temperature is None
                else sampling_temperature
            ),
            "actor/no_eos_count": 0.0,
            "actor/canonical_graph_actions": float(canonical_task == "graph_coloring"),
            "actor/canonical_countdown_actions": float(canonical_task == "countdown"),
            "actor/canonical_pantry_support_mask_actions": float(
                canonical_task == "pantry_support_mask"
            ),
            "actor/canonical_action_count": float(horizon),
            "actor/canonical_action_support_size": float(len(union_support)),
            "actor/canonical_sequence_support_size": float(action_space.sequence_count),
            "actor/canonical_max_sequence_entropy": float(
                action_space.max_sequence_entropy
            ),
            "actor/canonical_invalid_count": 0.0,
            "actor/canonical_finish_length_count": float(len(records)),
            "actor/canonical_finish_unexpected_count": 0.0,
            "actor/canonical_behavior_q_row_count": float(len(records) * horizon),
            "actor/canonical_behavior_q_support_min": float(
                min(len(support) for support in supports)
            ),
            "actor/canonical_behavior_q_support_max": float(
                max(len(support) for support in supports)
            ),
            "actor/canonical_behavior_q_norm_error_max": normalization_error_max,
            "actor/canonical_request_seed": float(request_seed),
            "actor/canonical_sampler_learner": 1.0,
            "actor/canonical_sampler_fixed_shape": 1.0,
        }
        info["actor/total_time"] = generate_time + verify_time

        trajectories: list[TrajectoryData] = []
        for record, reward in zip(records, rewards):
            dense_rewards = [0.0] * horizon
            dense_rewards[-1] = reward
            trajectory = TrajectoryData(
                prompt=record["prompt"],
                prompt_ids=record["prompt_ids"],
                response=record["response"],
                response_ids=record["response_ids"],
                response_logprobs=record["selected_log_probs"],
                rewards=dense_rewards,
                loss_mask=True,
                info=info,
            )
            setattr(trajectory, "reference", record["reference"])
            setattr(
                trajectory,
                "canonical_behavior_action_logprobs",
                record["full_log_probs"],
            )
            setattr(
                trajectory,
                "canonical_behavior_action_token_ids",
                list(union_support),
            )
            setattr(
                trajectory,
                "canonical_behavior_action_token_ids_by_position",
                [list(support) for support in supports],
            )
            trajectories.append(trajectory)
        if set_entropy_pending:
            self._canonical_entropy_prompt_pending = processed_prompts[0]
        logging.info(
            "canonical learner sampler finished data_len=%s seed=%s "
            "normalization_error_max=%.3g fixed_shape=1",
            len(trajectories),
            request_seed,
            normalization_error_max,
        )
        return trajectories, info

    @staticmethod
    def _coerce_log_float(value: Any) -> float | None:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "item"):
                value = value.item()
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        if scalar != scalar or scalar in {float("inf"), float("-inf")}:
            return None
        return scalar

    def _restore_training_progress_state(
        self,
        resume_states: dict[str, Any] | None,
    ) -> None:
        self._ensure_training_progress_state()
        if not isinstance(resume_states, dict):
            return
        for attr, key in (
            ("_progress_metric_baselines", "progress_metric_baselines"),
            ("_progress_metric_previous", "progress_metric_previous"),
            ("_progress_metric_bests", "progress_metric_bests"),
            ("_progress_metric_best_steps", "progress_metric_best_steps"),
        ):
            saved = resume_states.get(key)
            if isinstance(saved, dict):
                setattr(
                    self,
                    attr,
                    {
                        str(saved_key): (
                            int(saved_value)
                            if attr == "_progress_metric_best_steps"
                            else float(saved_value)
                        )
                        for saved_key, saved_value in saved.items()
                    },
                )
        self._actor_reward_ema = self._coerce_log_float(
            resume_states.get("actor_reward_ema")
        )
        self._actor_reward_ema_start = self._coerce_log_float(
            resume_states.get("actor_reward_ema_start")
        )
        tau_controller = getattr(self, "_xdr_tau_controller", None)
        saved_controller = resume_states.get("xdr_tau_controller_state")
        if tau_controller is not None and isinstance(saved_controller, dict):
            tau_controller.load_state_dict(saved_controller)
        maxent_controller = getattr(self, "_maxent_alpha_controller", None)
        saved_maxent_controller = resume_states.get("maxent_alpha_controller_state")
        if maxent_controller is not None:
            if not isinstance(saved_maxent_controller, dict):
                raise ValueError(
                    "adaptive MaxEnt checkpoint is missing alpha-controller state"
                )
            maxent_controller.load_state_dict(saved_maxent_controller)
        elif saved_maxent_controller is not None:
            raise ValueError(
                "fixed-alpha run cannot resume an entropy-adaptive checkpoint"
            )
        length_controller = getattr(self, "_maxent_length_controller", None)
        length_state_key = "maxent_length_controller_state"
        saved_length_controller = resume_states.get(length_state_key)
        if length_controller is not None:
            if not isinstance(saved_length_controller, dict):
                raise ValueError(
                    "constrained MaxEnt checkpoint is missing length-controller state"
                )
            length_controller.load_state_dict(saved_length_controller)
        elif saved_length_controller is not None:
            raise ValueError(
                "unconstrained run cannot resume a length-constrained checkpoint"
            )
        semantic_shannon_tracker = getattr(self, "_semantic_shannon_tracker", None)
        semantic_shannon_state_key = "semantic_shannon_tracker_state"
        saved_semantic_shannon = resume_states.get(semantic_shannon_state_key)
        if semantic_shannon_tracker is not None:
            if not isinstance(saved_semantic_shannon, dict):
                raise ValueError(
                    "semantic Shannon run cannot resume without estimator state"
                )
            rms_controller = getattr(self, "_semantic_rms_controller", None)
            saved_rms = resume_states.get("semantic_rms_controller_state")
            if rms_controller is not None:
                if not isinstance(saved_rms, dict):
                    raise ValueError(
                        "adaptive semantic run cannot resume without controller state"
                    )
                # The tracker stores the coefficient that was active when the
                # checkpoint was written, while a fresh adaptive run starts at
                # the registered base coefficient. Restore the controller
                # first so the tracker's strict contract check compares the
                # two checkpoint states rather than comparing evolved state to
                # the base configuration.
                rms_controller.load_state_dict(saved_rms)
                semantic_shannon_tracker.coefficient = float(
                    rms_controller.current_coefficient
                )
            elif saved_rms is not None:
                raise ValueError(
                    "fixed-coefficient run cannot resume an adaptive checkpoint"
                )
            semantic_shannon_tracker.load_state_dict(saved_semantic_shannon)
        elif saved_semantic_shannon is not None:
            raise ValueError(
                "non-semantic-Shannon run cannot resume a semantic Shannon checkpoint"
            )
        online_canonical_bank = getattr(self, "_online_canonical_bank", None)
        online_canonical_state_key = "online_canonical_bank_state"
        saved_online_canonical = resume_states.get(online_canonical_state_key)
        if online_canonical_bank is not None:
            if isinstance(saved_online_canonical, dict):
                online_canonical_bank.load_state_dict(saved_online_canonical)
            elif online_canonical_bank.objective_active or bool(
                getattr(
                    online_canonical_bank,
                    "retain_exemplars",
                    False,
                )
            ):
                raise ValueError(
                    "online canonical run cannot resume without bank state"
                )
            else:
                logging.warning(
                    "legacy Dr.GRPO checkpoint has no passive verified-discovery "
                    "state; cumulative tracking restarts from zero"
                )
        elif saved_online_canonical is not None:
            raise ValueError(
                "regular run cannot resume an online canonical bank checkpoint"
            )
        saved_rlep_online_pool = resume_states.get("rlep_online_pool_state")
        if bool(getattr(self.args, "rlep_online_pool", False)):
            if not isinstance(saved_rlep_online_pool, dict):
                raise ValueError(
                    "RLEP online-pool run cannot resume without its pool state"
                )
            pool = OnlineRLEPExperiencePool(
                minimum=int(getattr(self.args, "rlep_replay_count", 0) or 0)
            )
            pool.load_state_dict(saved_rlep_online_pool)
            self._rlep_experience_pool = pool
        elif saved_rlep_online_pool is not None:
            raise ValueError("regular run cannot resume an RLEP online pool checkpoint")
        proposal_starvation_controller = getattr(
            self,
            "_proposal_starvation_controller",
            None,
        )
        saved_proposal_starvation = resume_states.get(
            "proposal_starvation_controller_state"
        )
        if proposal_starvation_controller is not None:
            if not isinstance(saved_proposal_starvation, dict):
                raise ValueError(
                    "proposal-starvation run cannot resume without controller state"
                )
            proposal_starvation_controller.load_state_dict(
                saved_proposal_starvation
            )
        elif saved_proposal_starvation is not None:
            raise ValueError(
                "fixed proposal run cannot resume a starvation-controller checkpoint"
            )
        verified_route_library = getattr(
            self,
            "_verified_route_library",
            None,
        )
        verified_route_state_key = "verified_route_library_state"
        saved_verified_routes = resume_states.get(verified_route_state_key)
        if verified_route_library is not None:
            if not isinstance(verified_route_library, VerifiedRouteLibrary):
                raise RuntimeError("invalid verified route library")
            if not isinstance(saved_verified_routes, dict):
                raise ValueError(
                    "verified-route run cannot resume without route-library state"
                )
            verified_route_library.load_state_dict(saved_verified_routes)
        elif saved_verified_routes is not None:
            raise ValueError(
                "non-route run cannot resume a verified route library checkpoint"
            )
        math_strategy_canonicalizer = getattr(
            self, "_math_strategy_canonicalizer", None
        )
        math_strategy_state_key = "math_strategy_canonicalizer_state"
        saved_math_strategy = resume_states.get(math_strategy_state_key)
        if math_strategy_canonicalizer is not None:
            if not isinstance(saved_math_strategy, dict):
                raise ValueError(
                    "MATH strategy run cannot resume without canonicalizer state"
                )
            math_strategy_canonicalizer.load_state_dict(saved_math_strategy)
        elif saved_math_strategy is not None:
            raise ValueError(
                "non-MATH-strategy run cannot resume a strategy checkpoint"
            )
        online_canonical_controller = getattr(
            self, "_online_canonical_alpha_controller", None
        )
        online_canonical_controller_state_key = (
            "online_canonical_alpha_controller_state"
        )
        saved_online_canonical_controller = resume_states.get(
            online_canonical_controller_state_key
        )
        if online_canonical_controller is not None:
            if not isinstance(saved_online_canonical_controller, dict):
                raise ValueError(
                    "adaptive online canonical checkpoint is missing "
                    "alpha-controller state"
                )
            online_canonical_controller.load_state_dict(
                saved_online_canonical_controller
            )
        elif saved_online_canonical_controller is not None:
            raise ValueError(
                "fixed-alpha online canonical run cannot resume an adaptive "
                "online canonical checkpoint"
            )

    def _record_progress_metric(
        self,
        logs_dict: dict[str, Any],
        *,
        source_key: str,
        output_prefix: str,
        higher_is_better: bool = True,
    ) -> None:
        self._ensure_training_progress_state()
        value = self._coerce_log_float(logs_dict.get(source_key))
        if value is None:
            return

        baselines = self._progress_metric_baselines
        previous = self._progress_metric_previous
        bests = self._progress_metric_bests
        best_steps = self._progress_metric_best_steps

        baseline = baselines.setdefault(source_key, value)
        prev = previous.get(source_key, value)
        best = bests.get(source_key, value)
        is_new_best = value > best if higher_is_better else value < best
        if source_key not in bests or is_new_best:
            best = value
            bests[source_key] = value
            best_steps[source_key] = int(self.steps)
        previous[source_key] = value

        logs_dict[f"{output_prefix}/value"] = value
        logs_dict[f"{output_prefix}/gain_from_start"] = value - baseline
        logs_dict[f"{output_prefix}/gain_from_prev"] = value - prev
        logs_dict[f"{output_prefix}/best"] = best
        logs_dict[f"{output_prefix}/best_gain_from_start"] = best - baseline
        logs_dict[f"{output_prefix}/best_step"] = int(best_steps[source_key])
        logs_dict[f"{output_prefix}/steps_since_best"] = int(
            self.steps - best_steps[source_key]
        )

    def _add_learning_progress_metrics(self, logs_dict: dict[str, Any]) -> None:
        self._record_progress_metric(
            logs_dict,
            source_key="eval/average/accuracy",
            output_prefix="xdr/progress/eval_accuracy",
        )
        self._record_progress_metric(
            logs_dict,
            source_key="eval/average/score",
            output_prefix="xdr/progress/eval_score",
        )
        self._record_progress_metric(
            logs_dict,
            source_key="actor/rewards",
            output_prefix="xdr/progress/rollout_reward",
        )

        actor_reward = self._coerce_log_float(logs_dict.get("actor/rewards"))
        if actor_reward is None:
            return
        self._ensure_training_progress_state()
        if self._actor_reward_ema is None:
            self._actor_reward_ema = actor_reward
            self._actor_reward_ema_start = actor_reward
        else:
            self._actor_reward_ema = 0.9 * self._actor_reward_ema + 0.1 * actor_reward
        logs_dict["xdr/progress/rollout_reward_ema"] = self._actor_reward_ema
        if self._actor_reward_ema_start is not None:
            logs_dict["xdr/progress/rollout_reward_ema_gain_from_start"] = (
                self._actor_reward_ema - self._actor_reward_ema_start
            )

    def _format_compact_training_sample(self) -> str | None:
        """Return a compact rollout sample summary for console logs."""

        if not self.pi_buffer:
            return None
        sample = np.random.choice(self.pi_buffer)
        prompt = str(getattr(sample, "prompt", "") or "")
        response = str(getattr(sample, "response", "") or "")
        response_ids = getattr(sample, "response_ids", None)
        rewards = getattr(sample, "rewards", None)
        info = getattr(sample, "info", None)
        response_tokens = len(response_ids) if response_ids is not None else None
        reward_mean = None
        if rewards is not None:
            try:
                reward_values = list(rewards)
                if reward_values:
                    reward_mean = sum(float(value) for value in reward_values) / len(
                        reward_values
                    )
            except (TypeError, ValueError):
                reward_mean = None
        pieces = [
            f"prompt_chars={len(prompt)}",
            f"response_chars={len(response)}",
        ]
        if response_tokens is not None:
            pieces.append(f"response_tokens={response_tokens}")
        if reward_mean is not None:
            pieces.append(f"reward_mean={reward_mean:.4f}")
        if isinstance(info, dict):
            actor_reward = info.get("actor/rewards")
            if actor_reward is not None:
                try:
                    pieces.append(f"actor_reward={float(actor_reward):.4f}")
                except (TypeError, ValueError):
                    pass
        return "Training sample summary: " + " ".join(pieces)

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

    def _compute_exact_canonical_sequence_entropy(
        self, processed_prompt: str
    ) -> dict[str, float]:
        """Enumerate the updated canonical policy tree for one current prompt.

        The model is evaluated only after the optimizer step.  At each depth,
        prefix probability weights multiply the categorical entropy of the
        next positional support, yielding the exact chain-rule entropy of all
        action sequences in the task's finite positional support tree.
        """

        action_space = getattr(self, "_canonical_action_space", None)
        if action_space is None:
            raise RuntimeError("exact canonical entropy requires an action space")
        if dist.get_world_size() != 1:
            raise RuntimeError("exact canonical entropy is frozen to one learner rank")
        prompt_ids = list(self.tokenizer.encode(processed_prompt))
        if not prompt_ids:
            raise RuntimeError("exact canonical entropy received an empty prompt")
        supports = action_space.token_ids_by_position
        horizon = action_space.horizon
        if horizon <= 0 or any(not support for support in supports):
            raise RuntimeError(
                "exact canonical entropy requires a positive horizon and "
                "nonempty positional supports"
            )
        temperature = float(self.args.temperature)
        if not np.isfinite(temperature) or temperature <= 0:
            raise RuntimeError(
                "exact canonical entropy requires a finite positive temperature"
            )
        micro_batch_size = int(self.args.train_batch_size_per_device)
        if micro_batch_size <= 0:
            raise RuntimeError("exact canonical entropy needs a positive microbatch")

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cuda", torch.cuda.current_device())
        prefixes: list[tuple[tuple[int, ...], float]] = [((), 1.0)]
        exact_entropy = 0.0
        prefix_row_count = 0
        model_was_training = bool(self.model.training)
        self.model.eval()
        try:
            with torch.no_grad():
                for depth, support in enumerate(supports):
                    next_prefixes: list[tuple[tuple[int, ...], float]] = []
                    placeholder_suffix = [
                        position_support[0] for position_support in supports[depth:]
                    ]
                    for batch_start in range(0, len(prefixes), micro_batch_size):
                        batch_rows = prefixes[
                            batch_start : batch_start + micro_batch_size
                        ]
                        sequences = [
                            prompt_ids + list(prefix) + placeholder_suffix
                            for prefix, _ in batch_rows
                        ]
                        input_ids = torch.tensor(
                            sequences, dtype=torch.long, device=device
                        )
                        attention_mask = torch.ones_like(input_ids)
                        logits = self.model(input_ids, attention_mask=attention_mask)[
                            "logits"
                        ][:, len(prompt_ids) + depth - 1, :]
                        if temperature != 1.0:
                            logits = logits / temperature
                        allowed = torch.tensor(support, dtype=torch.long, device=device)
                        support_logits = logits.index_select(-1, allowed).double()
                        if not bool(torch.isfinite(support_logits).all()):
                            raise FloatingPointError(
                                "exact canonical entropy found nonfinite logits"
                            )
                        log_probs = torch.log_softmax(support_logits, dim=-1)
                        probabilities = torch.exp(log_probs)
                        row_entropies = -(probabilities * log_probs).sum(dim=-1)
                        probabilities_cpu = probabilities.cpu().tolist()
                        row_entropies_cpu = row_entropies.cpu().tolist()
                        for row_index, (prefix, mass) in enumerate(batch_rows):
                            prefix_row_count += 1
                            exact_entropy += mass * float(row_entropies_cpu[row_index])
                            for token_id, probability in zip(
                                support, probabilities_cpu[row_index]
                            ):
                                next_prefixes.append(
                                    (
                                        prefix + (int(token_id),),
                                        mass * float(probability),
                                    )
                                )
                    prefixes = next_prefixes
        finally:
            if model_was_training:
                self.model.train()

        leaf_mass = sum(mass for _, mass in prefixes)
        expected_prefix_rows = 0
        prefix_count = 1
        for support in supports:
            expected_prefix_rows += prefix_count
            prefix_count *= len(support)
        max_entropy = float(action_space.max_sequence_entropy)
        if prefix_row_count != expected_prefix_rows:
            raise RuntimeError(
                "canonical entropy prefix count mismatch: "
                f"{prefix_row_count} != {expected_prefix_rows}"
            )
        if not math.isclose(leaf_mass, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(
                f"canonical entropy leaf probabilities sum to {leaf_mass}"
            )
        if (
            not math.isfinite(exact_entropy)
            or exact_entropy < -1e-10
            or exact_entropy > max_entropy + 1e-8
        ):
            raise RuntimeError(
                "exact canonical entropy left its support bound: "
                f"entropy={exact_entropy} maximum={max_entropy}"
            )
        exact_entropy = min(max(exact_entropy, 0.0), max_entropy)
        return {
            "canonical_exact_sequence_entropy": exact_entropy,
            "canonical_exact_sequence_entropy_ratio": exact_entropy / max_entropy,
            "canonical_exact_prefix_row_count": float(prefix_row_count),
            "canonical_exact_leaf_count": float(len(prefixes)),
            "canonical_exact_leaf_mass": leaf_mass,
            "canonical_max_sequence_entropy": max_entropy,
            "canonical_exact_post_update": 1.0,
        }

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

    def learn(self, learning_round: int):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier()
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.strategy,
        )
        if learning_round == 1:
            # The upstream PPO learner prints the full first training example.
            # For long math trajectories this can dominate startup time and bury
            # the actual learner-step logs without changing the optimization.
            self.strategy.print("Training example omitted for compact startup logs")

        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=(True if self.args.critic_type == "ppo" else False),
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        step_bar = tqdm(
            range(len(dataloader)),
            desc="Train steps",
            disable=not self.strategy.is_rank_0(),
        )
        learn_batch_time = []

        self.model.train()
        if self.critic is not None:
            self.critic.train()
        st = time.time()

        logging.info(
            f"start learn() buffer_len={len(self.pi_buffer)} dl_len={len(dataloader)}"
        )
        for data in dataloader:
            if local_sgd_steps > self.args.max_sgd_steps:
                break
            infos = self.learning_step(data)
            replication_factor = (
                self.strategy.world_size
                if bool(
                    getattr(self.args, "canonical_graph_learner_sampling", False)
                    or getattr(self.args, "replicated_freeform_sampling", False)
                )
                else 1
            )
            self.policy_sgd_step += (
                len(dataset)
                * self.args.num_ppo_epochs
                / self.args.train_batch_size_per_device
                / self.strategy.grad_acc_step
                / replication_factor
            )
            learn_batch_time.append(time.time() - st)
            step_bar.update()

            self.global_step += 1
            if self.global_step % self.strategy.grad_acc_step == 0:
                self.gradient_update_elapse = time.time() - self.gradient_update_st
                st = time.time()
                self.gradient_update_st = time.time()

                local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            "total_time": time.time() - st,
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        pending_canonical_prompt = getattr(
            self, "_canonical_entropy_prompt_pending", None
        )
        if pending_canonical_prompt is not None:
            train_info.update(
                self._compute_exact_canonical_sequence_entropy(pending_canonical_prompt)
            )
            self._canonical_entropy_prompt_pending = None
        self._update_xdr_tau_controller(train_info)
        self._update_maxent_alpha_controller(train_info)
        self._update_maxent_length_controller(train_info)
        self._update_online_canonical_alpha_controller(train_info)
        # Keep distributed logging reductions aligned even when optional metrics
        # are populated by different local minibatch conditions.
        train_info = {key: train_info[key] for key in sorted(train_info)}
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        logging.info("finish learn()")
        return train_info

    def learning_step(self, trajectory):
        return self._grpo_learning_step_with_progress(trajectory)

    def prepare_data(self, strategy, tokenizer):
        prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data)
        prompts_data = prompt_dataset[self.args.train_split].select(
            range(min(self.args.max_train, len(prompt_dataset[self.args.train_split])))
        )
        raw_questions = list(prompts_data[self.args.input_key])

        # Prepare the data: templated questions & gt final answers.
        # Do not persist or consume Dataset.map caches here. This repository
        # intentionally runs several prompt contracts over the same frozen raw
        # rows; a stale cached transform would silently change the policy's
        # conditioning context while leaving the data path unchanged.
        prompts_data = prompts_data.map(
            apply_prompt_template_to_example,
            fn_kwargs={
                "input_key": self.args.input_key,
                "prompt_template": self.args.prompt_template,
            },
            load_from_cache_file=False,
            keep_in_memory=True,
        )
        canonical_task = resolve_canonical_action_task(self.args)
        if canonical_task != "none":
            allowed_templates = CANONICAL_TASK_PROMPT_TEMPLATES[canonical_task]
            if self.args.prompt_template not in allowed_templates:
                raise RuntimeError(
                    f"canonical task {canonical_task} requires one of "
                    f"{sorted(allowed_templates)}; got "
                    f"{self.args.prompt_template}"
                )
            validate_canonical_prompt_materialization(
                self.args.prompt_template,
                raw_questions,
                list(prompts_data[self.args.input_key]),
            )
            logging.info(
                "canonical %s prompt materialization verified: rows=%d "
                "template=%s dataset_map_cache=disabled",
                canonical_task,
                len(prompts_data),
                self.args.prompt_template,
            )

        self.prompts_dataset = PromptDataset(
            prompts_data,
            tokenizer,
            strategy,
            input_key=self.args.input_key,
            output_key=self.args.output_key,
            apply_chat_template=False,  # Because we have applied already.
            get_reference=True,
        )
        if canonical_task != "none" and len(self.prompts_dataset) != len(prompts_data):
            raise RuntimeError(
                "canonical prompt tokenization dropped rows under "
                f"prompt_max_length={self.args.prompt_max_length}: "
                f"rendered={len(prompts_data)} retained={len(self.prompts_dataset)}"
            )
        prompt_sampler = None
        if bool(
            getattr(self.args, "canonical_graph_learner_sampling", False)
            or getattr(self.args, "replicated_freeform_sampling", False)
        ):
            # Every learner rank follows the same audited prompt order. The
            # complete candidate group is partitioned only for backpropagation.
            prompt_sampler = DistributedSampler(
                self.prompts_dataset,
                num_replicas=1,
                rank=0,
                shuffle=True,
                seed=int(self.args.seed),
                drop_last=True,
            )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            self.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
            sampler=prompt_sampler,
        )
        self.eval_prompts_dataset = self.eval_prompts_dataloader = (
            None  # We use our own `self.eval_dataset_dict`.
        )

    def _init_wandb(self, resume_states: dict[str, Any] | None = None) -> None:
        if not self._requested_use_wb or not self.strategy.is_rank_0():
            return
        if self._wandb is not None:
            return

        import wandb

        if not wandb.api.api_key and isinstance(self._requested_use_wb, str):
            wandb.login(key=self._requested_use_wb)

        env_run_id = os.environ.get("OAT_ZERO_WANDB_RUN_ID") or os.environ.get(
            "WANDB_RUN_ID"
        )
        saved_run_id = None
        saved_run_name = None
        if isinstance(resume_states, dict):
            raw_run_id = resume_states.get("wandb_run_id")
            if isinstance(raw_run_id, str) and raw_run_id:
                saved_run_id = raw_run_id
            raw_run_name = resume_states.get("wandb_run_name")
            if isinstance(raw_run_name, str) and raw_run_name:
                saved_run_name = raw_run_name

        discovered_run_id = None
        discovered_run_name = None
        if not env_run_id and not saved_run_id and self.args.resume_dir:
            wandb_run_roots = []
            env_wandb_dir = os.environ.get("WANDB_DIR")
            if env_wandb_dir:
                wandb_run_roots.append(Path(env_wandb_dir) / "runs" / "wandb")
            wandb_run_roots.extend(
                [
                    Path.cwd() / "var" / "wandb" / "runs" / "wandb",
                    Path.cwd() / "wandb" / "runs" / "wandb",
                ]
            )
            discovered_run_id, discovered_run_name = discover_local_wandb_resume_run(
                wandb_run_roots=wandb_run_roots,
                resume_dir=self.args.resume_dir,
                resume_tag=self.args.resume_tag,
                saved_run_name=saved_run_name,
                current_run_name=self._wandb_run_name,
            )
            if discovered_run_id:
                logging.info(
                    "Recovered W&B resume run id %s from local run logs for %s",
                    discovered_run_id,
                    self.args.resume_dir,
                )

        run_id = env_run_id or saved_run_id or discovered_run_id
        if env_run_id:
            if saved_run_id or discovered_run_id:
                logging.info(
                    "Using explicit W&B run id %s; skipping resumed/discovered lineage.",
                    env_run_id,
                )
            run_name = self._wandb_run_name
        else:
            run_name = saved_run_name or discovered_run_name or self._wandb_run_name
        init_kwargs: dict[str, Any] = {
            "entity": self.args.wb_org,
            "project": self.args.wb_project,
            "group": self.args.wb_group,
            "name": run_name,
            "config": self.args.__dict__,
            "reinit": True,
        }
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"

        self._wandb = wandb
        wandb.init(**init_kwargs)
        if wandb.run is not None:
            self._wandb_run_id = wandb.run.id
            self._wandb_run_name = wandb.run.name
        else:
            self._wandb_run_id = run_id
            self._wandb_run_name = run_name

        # Use the actual training step for chart alignment across resumes.
        wandb.define_metric("trainer/step")
        wandb.define_metric("*", step_metric="trainer/step")

    def _checkpoint_client_state(self) -> dict[str, Any]:
        self._ensure_training_progress_state()
        client_state: dict[str, Any] = {
            "global_step": int(self.global_step),
            "policy_sgd_step": float(self.policy_sgd_step),
            "query_step": int(self.query_step),
            "prompt_consumed": int(self.prompt_consumed),
            "prompt_epoch": int(self.prompt_epoch),
            "steps": int(self.steps),
            "prompt_batches_consumed_total": int(self._prompt_batches_consumed_total),
            "update_interval": int(self.update_interval),
            "progress_metric_baselines": dict(self._progress_metric_baselines),
            "progress_metric_previous": dict(self._progress_metric_previous),
            "progress_metric_bests": dict(self._progress_metric_bests),
            "progress_metric_best_steps": dict(self._progress_metric_best_steps),
        }
        if self._actor_reward_ema is not None:
            client_state["actor_reward_ema"] = float(self._actor_reward_ema)
        if self._actor_reward_ema_start is not None:
            client_state["actor_reward_ema_start"] = float(self._actor_reward_ema_start)
        tau_controller = getattr(self, "_xdr_tau_controller", None)
        if tau_controller is not None:
            client_state["xdr_tau_controller_state"] = tau_controller.state_dict()
        maxent_controller = getattr(self, "_maxent_alpha_controller", None)
        if maxent_controller is not None:
            client_state["maxent_alpha_controller_state"] = (
                maxent_controller.state_dict()
            )
        length_controller = getattr(self, "_maxent_length_controller", None)
        if length_controller is not None:
            client_state["maxent_length_controller_state"] = (
                length_controller.state_dict()
            )
        diayn_tracker = getattr(self, "_diayn_mi_tracker", None)
        if diayn_tracker is not None:
            client_state["diayn_mi_tracker_state"] = diayn_tracker.state_dict()
        semantic_shannon_tracker = getattr(self, "_semantic_shannon_tracker", None)
        if semantic_shannon_tracker is not None:
            client_state["semantic_shannon_tracker_state"] = (
                semantic_shannon_tracker.state_dict()
            )
        semantic_rms_controller = getattr(self, "_semantic_rms_controller", None)
        if semantic_rms_controller is not None:
            # The adapted coefficient is run state, not a hyperparameter: a
            # resume that restarted it at the base dose would silently rerun
            # the controller's warmup partway through training.
            client_state["semantic_rms_controller_state"] = (
                semantic_rms_controller.state_dict()
            )
        online_canonical_bank = getattr(self, "_online_canonical_bank", None)
        if online_canonical_bank is not None:
            client_state["online_canonical_bank_state"] = (
                online_canonical_bank.state_dict()
            )
        rlep_online_pool = getattr(self, "_rlep_experience_pool", None)
        if isinstance(rlep_online_pool, OnlineRLEPExperiencePool):
            # The pool is run state: a resume that restarted it empty would
            # silently turn the arm back into plain Dr.GRPO for a full pass.
            client_state["rlep_online_pool_state"] = rlep_online_pool.state_dict()
        proposal_starvation_controller = getattr(
            self,
            "_proposal_starvation_controller",
            None,
        )
        if proposal_starvation_controller is not None:
            client_state["proposal_starvation_controller_state"] = (
                proposal_starvation_controller.state_dict()
            )
        verified_route_library = getattr(
            self,
            "_verified_route_library",
            None,
        )
        if verified_route_library is not None:
            if not isinstance(verified_route_library, VerifiedRouteLibrary):
                raise RuntimeError("invalid verified route library")
            client_state["verified_route_library_state"] = (
                verified_route_library.state_dict()
            )
        math_strategy_canonicalizer = getattr(
            self, "_math_strategy_canonicalizer", None
        )
        if math_strategy_canonicalizer is not None:
            client_state["math_strategy_canonicalizer_state"] = (
                math_strategy_canonicalizer.state_dict()
            )
        online_canonical_controller = getattr(
            self, "_online_canonical_alpha_controller", None
        )
        if online_canonical_controller is not None:
            client_state["online_canonical_alpha_controller_state"] = (
                online_canonical_controller.state_dict()
            )
        if hasattr(self, "last_eval_query_step"):
            client_state["last_eval_query_step"] = int(self.last_eval_query_step)
        if hasattr(self, "_pending_eval"):
            client_state["_pending_eval"] = bool(self._pending_eval)
        if self._wandb_run_id:
            client_state["wandb_run_id"] = self._wandb_run_id
        if self._wandb_run_name:
            client_state["wandb_run_name"] = self._wandb_run_name
        return client_state

    def _infer_resume_step(self, resume_states: dict[str, Any] | None) -> int:
        """Recover the loaded checkpoint's step without restarting its counters."""
        saved_step = None
        if isinstance(resume_states, dict) and "steps" in resume_states:
            raw_step = resume_states["steps"]
            try:
                saved_step = int(raw_step)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("Checkpoint has invalid client-state steps") from exc
            if isinstance(raw_step, bool) or saved_step < 0 or str(raw_step) not in (
                str(saved_step), str(float(saved_step))
            ):
                raise RuntimeError("Checkpoint has invalid client-state steps")

        tag = getattr(self.args, "resume_tag", None)
        if not tag and getattr(self.args, "resume_dir", None):
            latest = Path(self.args.resume_dir) / "latest"
            if latest.is_symlink():
                tag = latest.resolve().name
            elif latest.is_file():
                tag = latest.read_text(encoding="utf-8").strip()
        tag_step = None
        if tag:
            tag_name = Path(str(tag)).name
            if tag_name.startswith("step_") and tag_name[5:].isdigit():
                tag_step = int(tag_name[5:])
        if saved_step is not None:
            if tag_step is not None and tag_step != saved_step:
                raise RuntimeError(
                    f"Checkpoint step mismatch: client state={saved_step}, tag={tag_step}"
                )
            return saved_step
        if tag_step is not None:
            return tag_step
        raise RuntimeError("Cannot infer resumed checkpoint step from client state or tag")

    def _restore_prompt_progress(
        self,
        resume_states: dict[str, Any] | None,
        resume_step: int,
    ) -> tuple[int, int, int]:
        saved_update_interval = None
        if isinstance(resume_states, dict) and "update_interval" in resume_states:
            try:
                saved_update_interval = int(resume_states["update_interval"])
            except (TypeError, ValueError):
                saved_update_interval = None
        if saved_update_interval is not None and saved_update_interval != int(
            self.update_interval
        ):
            logging.warning(
                "Checkpoint update_interval=%s does not match current update_interval=%s; "
                "continuing with the current setting.",
                saved_update_interval,
                int(self.update_interval),
            )

        progress_state = resolve_resume_progress_state(
            resume_states=resume_states,
            resume_step=resume_step,
            update_interval=int(self.update_interval),
            num_prompt_epochs=int(self.args.num_prompt_epoch),
            num_prompt_batches_per_epoch=int(len(self.prompts_dataloader)),
            rollout_batch_size=int(self.args.rollout_batch_size),
        )
        self.steps = int(progress_state["checkpoint_step"])
        self.global_step = int(progress_state["global_step"])
        self.policy_sgd_step = float(progress_state["policy_sgd_step"])
        self.query_step = int(progress_state["query_step"])
        self.prompt_consumed = int(progress_state["prompt_consumed"])
        self.prompt_epoch = int(progress_state["prompt_epoch"])
        self._prompt_batches_consumed_total = int(
            progress_state["prompt_batches_consumed_total"]
        )

        last_eval_query_step = int(progress_state["last_eval_query_step"])
        if last_eval_query_step > 0:
            self.last_eval_query_step = last_eval_query_step
        if isinstance(resume_states, dict) and "_pending_eval" in resume_states:
            self._pending_eval = bool(resume_states["_pending_eval"])
        diayn_tracker = getattr(self, "_diayn_mi_tracker", None)
        saved_diayn_state = (
            resume_states.get("diayn_mi_tracker_state")
            if isinstance(resume_states, dict)
            else None
        )
        if diayn_tracker is not None:
            if saved_diayn_state is None:
                raise RuntimeError(
                    "DIAYN run cannot resume without discriminator state"
                )
            diayn_tracker.load_state_dict(saved_diayn_state)
        elif saved_diayn_state is not None:
            raise RuntimeError(
                "non-DIAYN run cannot resume a DIAYN discriminator checkpoint"
            )
        self._restore_training_progress_state(resume_states)

        if self.steps % max(1, int(self.update_interval)) != 0:
            logging.warning(
                "Checkpoint step %s lands mid-update interval %s; optimizer step state "
                "resumes, but any in-memory rollout buffer from the unfinished interval "
                "cannot be reconstructed from checkpoints alone.",
                self.steps,
                int(self.update_interval),
            )

        used_saved_prompt_cursor = (
            isinstance(resume_states, dict)
            and "prompt_batches_consumed_total" in resume_states
        )
        log_fn = logging.info if used_saved_prompt_cursor else logging.warning
        log_fn(
            "%s prompt traversal state: checkpoint_step=%s completed_batches=%s "
            "start_epoch=%s start_batch_offset=%s query_step=%s prompt_consumed=%s",
            ("Restored" if used_saved_prompt_cursor else "Inferred fallback"),
            self.steps,
            self._prompt_batches_consumed_total,
            int(progress_state["start_prompt_epoch"]),
            int(progress_state["start_batch_offset"]),
            self.query_step,
            self.prompt_consumed,
        )

        return (
            int(progress_state["next_step"]),
            int(progress_state["start_prompt_epoch"]),
            int(progress_state["start_batch_offset"]),
        )

    def run(self):
        self._init(self.args, self.actors)
        self._init_local_actor_weight_sync()

        resume_step = 0
        next_step = 1
        start_prompt_epoch = 0
        start_batch_offset = 0
        resume_states: dict[str, Any] | None = None
        if self.args.resume_dir:
            _, resume_states = self.strategy.load_ckpt(
                self.model.model,
                self.args.resume_dir,
                self.args.resume_tag,
            )
            resume_step = self._infer_resume_step(resume_states)
            next_step, start_prompt_epoch, start_batch_offset = (
                self._restore_prompt_progress(resume_states, resume_step)
            )
        else:
            self.steps = 0
            self._prompt_batches_consumed_total = 0

        early_stop = False
        self.start_time = time.time()

        self.actor_info = {}
        train_info: dict[str, Any] = {}
        self._init_wandb(resume_states)

        if self.args.resume_dir:
            # The learner now holds the restored checkpoint, but actors were
            # initialized from ``args.pretrain``.  Synchronize before either
            # the initial evaluation or the first resumed rollout; otherwise
            # both would use stale base-model weights until the next update.
            logging.info(
                "resume actor weight sync start checkpoint_step=%s", self.steps
            )
            self.sync_params_to_actors()
            logging.info("resume actor weight sync done checkpoint_step=%s", self.steps)

        if bool(getattr(self.args, "eval_only", False)):
            # Measure the loaded policy through the ordinary training
            # evaluation path and stop. No rollout, optimizer step, export, or
            # resume checkpoint occurs, so the reported cell is a pure function
            # of the checkpoint and the decoding settings. Unlike the initial
            # evaluation below, this runs even under ``debug``: an eval-only
            # job that silently produced no evaluation would be indistinguishable
            # from a completed one.
            self.eval_and_log({}, eval=True, save=False, allow_scheduled_save=False)
            self._write_eval_only_marker()
            # Tear the program down exactly as the training path does. Without
            # this the measurement finishes but the job holds its GPU until the
            # scheduler's time limit.
            if self.strategy.is_rank_0():
                self._wandb.finish() if self._wandb else None
                lp.stop()
            return

        if not self.strategy.args.debug:
            # The checkpoint already exists at a resumed boundary. Rewriting
            # the same multi-gigabyte model/optimizer state before the initial
            # evaluation creates avoidable I/O contention across a recovery
            # cohort and cannot improve recoverability.
            self.eval_and_log(
                {},
                eval=True,
                save=False,
                allow_scheduled_save=not bool(self.args.resume_dir),
            )

        self.steps = next_step
        self.gradient_update_st = time.time()
        for p_ep in range(start_prompt_epoch, self.args.num_prompt_epoch):
            batch_offset = start_batch_offset if p_ep == start_prompt_epoch else 0
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(p_ep)
                self.strategy.print(f"Set DistributedSampler at epoch {p_ep}")
            if batch_offset > 0:
                self.strategy.print(
                    "Skipping "
                    f"{batch_offset} already-consumed prompt batches in epoch {p_ep}"
                )
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
                initial=batch_offset,
            )

            for batch_idx, (processed_prompts, raw_prompts, refs) in enumerate(
                self.prompts_dataloader
            ):
                if batch_idx < batch_offset:
                    continue
                if early_stop:
                    break
                self._prompt_batches_consumed_total += 1
                if resolve_canonical_action_task(self.args) != "none":
                    if len(processed_prompts) != 1:
                        raise RuntimeError(
                            "exact canonical entropy requires one current prompt"
                        )
                    self._canonical_entropy_prompt_pending = processed_prompts[0]

                learner_sampling = bool(
                    getattr(self.args, "canonical_graph_learner_sampling", False)
                )
                replicated_freeform_sampling = bool(
                    getattr(self.args, "replicated_freeform_sampling", False)
                )
                pre_learning_done = False
                if learner_sampling:
                    if self.steps % self.update_interval != 0:
                        raise RuntimeError(
                            "canonical learner sampling reached a rollout without an "
                            "immediate optimizer update"
                        )
                    logging.info(
                        "pre-learning start before canonical learner sampling step=%s",
                        self.steps,
                    )
                    self._pre_learning()
                    pre_learning_done = True
                    feedback_data, self.actor_info = (
                        self._sample_canonical_feedback_with_learner(
                            raw_prompts,
                            processed_prompts,
                            refs,
                        )
                    )
                    if bool(
                        getattr(
                            self.args,
                            "online_canonical_counterfactual_proposals",
                            False,
                        )
                    ):
                        self.actor_info = (
                            self._sample_and_admit_canonical_counterfactual_proposals(
                                raw_prompts=raw_prompts,
                                processed_prompts=processed_prompts,
                                refs=refs,
                                neutral_feedback=feedback_data,
                                actor_info=self.actor_info,
                            )
                        )
                elif replicated_freeform_sampling:
                    feedback_data, self.actor_info = (
                        self._sample_replicated_freeform_feedback(
                            raw_prompts,
                            processed_prompts,
                            refs,
                        )
                    )
                elif bool(getattr(self.args, "dapo_enabled", False)):
                    feedback_data, self.actor_info = (
                        self._collect_dapo_dynamic_feedback(
                            raw_prompts,
                            processed_prompts,
                            refs,
                        )
                    )
                else:
                    feedback_data, self.actor_info = self.collector.collect_feedback(
                        raw_prompts,
                        processed_prompts,
                        refs,
                        self._same_actor_group,
                    )
                dist.barrier()

                if feedback_data is None:
                    if pre_learning_done:
                        self._post_learning()
                    continue
                replication_factor = (
                    dist.get_world_size()
                    if (learner_sampling or replicated_freeform_sampling)
                    and dist.get_world_size() > 1
                    else 1
                )
                if len(feedback_data) % replication_factor != 0:
                    raise RuntimeError(
                        "replicated feedback does not divide across learner ranks"
                    )
                unique_local_count = len(feedback_data) // replication_factor
                self.prompt_consumed += unique_local_count

                self.process_feedback_data(feedback_data)
                if replication_factor > 1:
                    self.query_step -= len(feedback_data) - unique_local_count

                if (
                    self.args.dump_replay_every > 0
                    and self.steps % self.args.dump_replay_every == 0
                ):
                    if not self.strategy.is_rank_0():
                        dist.gather_object(self.pi_buffer)
                    else:
                        gather_all_buffer = [None] * self.strategy.world_size
                        dist.gather_object(self.pi_buffer, gather_all_buffer)
                        pd.to_pickle(
                            (processed_prompts, refs, gather_all_buffer),
                            os.path.join(
                                self.save_path,
                                f"buffer_step{self.steps:05}.pkl",
                            ),
                        )

                if self.steps % self.update_interval == 0:
                    if not pre_learning_done:
                        logging.info("pre-learning start step=%s", self.steps)
                        self._pre_learning()
                    logging.info("learn start step=%s", self.steps)
                    train_info = self.learn(self.steps // self.update_interval)
                    logging.info("post-learning start step=%s", self.steps)
                    self._post_learning()
                    logging.info("post-learning done step=%s", self.steps)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.sync_params_every == 0:
                        logging.info("sync params to actors start step=%s", self.steps)
                        self.sync_params_to_actors()
                        logging.info("sync params to actors done step=%s", self.steps)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                    logging.info("eval/log start step=%s", self.steps)
                    self.eval_and_log(train_info)
                    logging.info("eval/log done step=%s", self.steps)

                progress_bar.update()
                self.steps += 1

                if self.get_current_query() > self.args.max_queries:
                    early_stop = True

            self.prompt_epoch = p_ep + 1
            if early_stop:
                break

        self.eval_and_log(train_info, eval=True, save=True)

        if self.args.dump_all_buffer:  # For debug purpose.
            if not self.strategy.is_rank_0():
                dist.gather_object(self.all_buffer)
            else:
                gather_all_buffer = [None] * self.strategy.world_size
                dist.gather_object(self.all_buffer, gather_all_buffer)
                pd.to_pickle(
                    gather_all_buffer,
                    os.path.join(self.save_path, "all_buffer.pkl"),
                )

        self._finalize_successful_storage()

        if self.strategy.is_rank_0():
            self._wandb.finish() if self._wandb else None
            lp.stop()

    @staticmethod
    def _checkpoint_step(path: Path) -> int:
        try:
            return int(path.name.removeprefix("step_"))
        except ValueError:
            return -1

    @staticmethod
    def _storage_barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _save_resume_checkpoint(self) -> None:
        """Atomically replace rolling recovery state without a zero-copy gap."""
        checkpoint_root = Path(self.save_path) / "checkpoints"
        tag = "step_{:05d}".format(self.steps)
        keep = int(self.args.max_resume_num)

        # OAT rotates before writing.  Giving it one temporary extra slot keeps
        # the previous valid checkpoint alive until the new distributed write
        # has returned on every rank; only then do we enforce the real limit.
        self.strategy.save_ckpt(
            self.model.model,
            str(checkpoint_root),
            tag=tag,
            max_num=keep + 1,
            max_mem=int(self.args.max_resume_mem),
            client_state=self._checkpoint_client_state(),
        )
        self._storage_barrier()
        if self.strategy.is_rank_0() and checkpoint_root.is_dir():
            checkpoints = sorted(
                (
                    path
                    for path in checkpoint_root.iterdir()
                    if path.is_dir()
                    and not path.is_symlink()
                    and self._checkpoint_step(path) >= 0
                ),
                key=lambda path: (self._checkpoint_step(path), path.stat().st_mtime_ns),
                reverse=True,
            )
            for obsolete in checkpoints[keep:]:
                shutil.rmtree(obsolete)
                logging.info("Deleted superseded resume checkpoint %s", obsolete)
        self._storage_barrier()

    def _write_eval_only_marker(self) -> None:
        """Record that an eval-only measurement completed, and under what settings.

        The decoding-frontier aggregator refuses any cell without this marker,
        so a job that died between loading the checkpoint and finishing its
        draws cannot be mistaken for a measured point.
        """
        self._storage_barrier()
        if self.strategy.is_rank_0():
            attempt_root = Path(self.save_path).resolve()
            payload = {
                "schema": "oat_zero_eval_only_complete_v1",
                "completed_at_unix": time.time(),
                "attempt_root": str(attempt_root),
                "pretrain": str(self.args.pretrain),
                "eval_mode_coverage_k": int(self.args.eval_mode_coverage_k),
                "eval_mode_coverage_temperature": float(
                    self.args.eval_mode_coverage_temperature
                ),
                "eval_mode_coverage_top_p": float(
                    getattr(self.args, "eval_mode_coverage_top_p", 1.0)
                ),
                "eval_mode_coverage_draws": int(self.args.eval_mode_coverage_draws),
                "eval_mode_coverage_seed": int(self.args.eval_mode_coverage_seed),
                "eval_temperature": float(self.args.eval_temperature),
                "test_split": str(self.args.test_split),
                "eval_data": str(self.args.eval_data),
                "prompt_template": str(self.args.prompt_template),
                "optimizer_steps": 0,
            }
            marker = attempt_root / "EVAL_ONLY_COMPLETE.json"
            temporary = attempt_root / f".{marker.name}.{os.getpid()}.tmp"
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, marker)
        self._storage_barrier()

    def _finalize_successful_storage(self) -> None:
        """Mark a completed run and retire optimizer state from all attempts."""
        self._storage_barrier()
        if self.strategy.is_rank_0():
            run_root = Path(self.args.save_path).resolve()
            attempt_root = Path(self.save_path).resolve()
            if attempt_root.parent != run_root:
                raise RuntimeError(
                    f"attempt path {attempt_root} is not directly beneath {run_root}"
                )

            removed: list[str] = []
            cleanup_errors: list[str] = []
            terminal_export = attempt_root / "saved_models" / f"step_{self.steps:05d}"
            export_required = int(self.args.export_steps) >= 0
            export_ready = not export_required or terminal_export.is_dir()
            if not export_ready:
                cleanup_errors.append(
                    f"terminal export missing; preserved resume state: {terminal_export}"
                )
            if bool(self.args.prune_resume_on_success) and export_ready:
                for checkpoint_root in sorted(run_root.glob("*/checkpoints")):
                    resolved = checkpoint_root.resolve()
                    if (
                        checkpoint_root.is_symlink()
                        or resolved.parent.parent != run_root
                        or resolved.name != "checkpoints"
                    ):
                        cleanup_errors.append(f"refused unsafe path: {checkpoint_root}")
                        continue
                    try:
                        shutil.rmtree(resolved)
                        removed.append(str(resolved))
                    except OSError as exc:
                        cleanup_errors.append(f"{resolved}: {exc}")
                        logging.exception(
                            "Could not retire completed-run checkpoint %s", resolved
                        )

            completion = {
                "schema": "oat_zero_training_complete_v1",
                "completed_at_unix": time.time(),
                "terminal_step": int(self.steps),
                "terminal_attempt": str(attempt_root),
                "terminal_export": (str(terminal_export) if export_required else None),
                "resume_checkpoints_pruned": bool(
                    self.args.prune_resume_on_success and export_ready
                ),
                "removed_checkpoint_roots": removed,
                "cleanup_errors": cleanup_errors,
            }
            marker = run_root / "TRAINING_COMPLETE.json"
            temporary = run_root / f".{marker.name}.{os.getpid()}.tmp"
            temporary.write_text(
                json.dumps(completion, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, marker)
        self._storage_barrier()

    def _storage_actions(
        self, *, terminal: bool, allow_scheduled_save: bool
    ) -> tuple[bool, bool]:
        """Return ``(export_model, save_resume_state)`` for this boundary."""
        export_steps = int(self.args.export_steps)
        resume_steps = int(self.args.resume_steps)
        should_export = (terminal and export_steps >= 0) or (
            allow_scheduled_save
            and export_steps > 0
            and self.steps > 0
            and self._should_do(export_steps)
            and self.steps >= int(self.args.export_from)
        )
        should_resume = (
            bool(self.args.save_ckpt)
            and not terminal
            and allow_scheduled_save
            and resume_steps > 0
            and self.steps > 0
            and self._should_do(resume_steps)
            and self.steps >= int(self.args.resume_from)
        )
        return should_export, should_resume

    def eval_and_log(
        self,
        train_info,
        eval=False,
        save=False,
        allow_scheduled_save=True,
    ):
        eval_info = {}
        forced_eval = self.args.eval_steps > 0 and eval
        should_eval = forced_eval or self._should_do(self.args.eval_steps)
        duplicate_terminal_eval = (
            bool(save)
            and forced_eval
            and getattr(self, "_last_evaluated_global_step", None)
            == int(self.global_step)
        )
        if duplicate_terminal_eval:
            # The run loop increments ``steps`` after every consumed prompt,
            # including the final one.  If that final update already landed
            # on a scheduled evaluation boundary, the terminal call below
            # therefore has a new bookkeeping step but the exact same policy.
            # Keep terminal export/storage semantics while avoiding a second
            # full deterministic evaluation of unchanged weights.
            should_eval = False
            logging.info(
                "Skipping duplicate terminal evaluation at step %s; "
                "policy global_step %s was already evaluated.",
                self.steps,
                self.global_step,
            )
        should_export, should_resume = self._storage_actions(
            terminal=bool(save), allow_scheduled_save=allow_scheduled_save
        )

        if should_resume and should_eval and self.strategy.is_rank_0():
            logging.info(
                "Recovery boundary at step %s: saving checkpoint before evaluation.",
                self.steps,
            )

        if should_resume:
            self._save_resume_checkpoint()

        if should_export:
            self.strategy.save_model(
                self.model,
                self.tokenizer,
                os.path.join(self.save_path, "saved_models"),
                tag="step_{:05d}".format(self.steps),
                max_num=int(self.args.max_export_num),
                max_mem=int(self.args.max_export_mem),
            )

        if should_eval:
            eval_info = self.evaluate(self.eval_prompts_dataloader, self.steps)
            self._last_evaluated_global_step = int(self.global_step)

        if eval_info or self.steps % self.args.logging_steps == 0:
            misc_info = self.get_misc_info()
            misc_info["lr"] = self.scheduler.get_last_lr()[0]

            misc_info = {
                "misc/%s" % k: v
                for k, v in {
                    **misc_info,
                }.items()
            }
            logs_dict = {**train_info, **eval_info, **self.actor_info, **misc_info}
            logs_dict = self.strategy.all_reduce(logs_dict)
            logs_dict.update(
                self.strategy.all_reduce(
                    {
                        "misc/query_step": self.query_step,
                        "misc/prompt_consumed": self.prompt_consumed,
                    },
                    op="sum",
                )
            )
            logs_dict["trainer/step"] = int(self.steps)
            logs_dict["trainer/global_step"] = int(self.global_step)
            logs_dict["trainer/policy_sgd_step"] = float(self.policy_sgd_step)
            self._add_learning_progress_metrics(logs_dict)

            if self.strategy.is_rank_0():
                sample_summary = self._format_compact_training_sample()
                if sample_summary:
                    self.strategy.print(sample_summary)
                self.strategy.pprint(logs_dict)
                self._append_train_metrics_jsonl(logs_dict)
                if self._wandb is not None:
                    self._wandb.log(
                        filter_wandb_logs(logs_dict),
                        step=int(self.steps),
                    )

    def _append_train_metrics_jsonl(self, logs_dict: dict[str, Any]) -> None:
        """Persist per-step scalar training metrics to train_metrics.jsonl.

        The comparative recipes run with wandb disabled, but the exploration
        side-car figure needs per-step aggregation diagnostics
        (train/agg_eff_rollouts, train/agg_incorrect_mass) from every arm, so
        rank 0 appends one JSON record per logging step under save_path. The
        sink is best-effort: logging must never kill a training run, so any
        failure is warned once and then ignored.
        """
        try:
            record: dict[str, float] = {}
            for key, value in logs_dict.items():
                scalar = self._coerce_log_float(value)
                if scalar is not None:
                    record[str(key)] = scalar
            if not record:
                return
            path = os.path.join(self.save_path, "train_metrics.jsonl")
            with open(path, "a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            if not getattr(self, "_train_metrics_jsonl_failed_warned", False):
                logging.exception(
                    "Failed to append train_metrics.jsonl; continuing without "
                    "the per-step metrics sink."
                )
                self._train_metrics_jsonl_failed_warned = True

    def _append_mode_coverage_draw_jsonl(self, record: dict[str, Any]) -> None:
        """Durably retain every sampled evaluation outcome on rank zero."""

        path = os.path.join(self.save_path, "eval_mode_coverage_draws.jsonl")
        with open(path, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, sort_keys=True) + "\n")

    def eval_dataloader_collate_fn(self, item_list):
        return collate_eval_prompt_items(
            item_list,
            prompt_template=self.args.prompt_template,
        )

    def evaluate(self, dataloader, steps):
        # Discard the default eval dataloader, and run eval on multiple benchmarks.
        del dataloader
        all_metrics = {}
        accuracies = []
        scores = []
        lens = []
        total_benchmarks = len(self.eval_dataset_dict)
        for benchmark_idx, (benchmark_name, dataset) in enumerate(
            self.eval_dataset_dict.items(), start=1
        ):
            eval_prompts_dataloader = DataLoader(
                dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.eval_dataloader_collate_fn,
            )
            if self.strategy.is_rank_0():
                logging.info(
                    "Starting eval benchmark %s/%s: %s (%s prompts, %s batches) at step %s",
                    benchmark_idx,
                    total_benchmarks,
                    benchmark_name,
                    len(dataset),
                    len(eval_prompts_dataloader),
                    steps,
                )
            metrics = super().evaluate(
                eval_prompts_dataloader, f"{steps}_{benchmark_name}"
            )
            if self.strategy.is_rank_0():
                logging.info(
                    "Finished eval benchmark %s/%s: %s accuracy=%.4f score=%.4f avg_len=%.2f at step %s",
                    benchmark_idx,
                    total_benchmarks,
                    benchmark_name,
                    float(metrics["eval/accuracy"]),
                    float(metrics["eval/score"]),
                    float(metrics["eval/response_tok_len"]),
                    steps,
                )
            all_metrics.update(
                {
                    k.replace("eval/", f"eval/{benchmark_name}/"): v
                    for k, v in metrics.items()
                }
            )
            accuracies.append(metrics["eval/accuracy"])
            scores.append(metrics["eval/score"])
            lens.append(metrics["eval/response_tok_len"])
        all_metrics.update(
            {
                "eval/average/accuracy": np.mean(accuracies),
                "eval/average/score": np.mean(scores),
                "eval/average/response_tok_len": np.mean(lens),
            }
        )

        if self.args.eval_mode_coverage_k > 0:
            mc_coverages = []
            for benchmark_name, dataset in self.eval_dataset_dict.items():
                mc = self.evaluate_mode_coverage(
                    dataset,
                    benchmark_name,
                    steps,
                    k=self.args.eval_mode_coverage_k,
                    temperature=self.args.eval_mode_coverage_temperature,
                )
                all_metrics.update(mc)
                cov_key = f"eval/{benchmark_name}/sampled_mode_coverage_at_{self.args.eval_mode_coverage_k}"
                if cov_key in mc:
                    mc_coverages.append(mc[cov_key])
            if mc_coverages:
                all_metrics[
                    f"eval/average/sampled_mode_coverage_at_{self.args.eval_mode_coverage_k}"
                ] = float(np.mean(mc_coverages))

        return all_metrics

    def evaluate_mode_coverage(
        self,
        dataset,
        benchmark_name: str,
        steps: int,
        *,
        k: int,
        temperature: float,
    ) -> dict[str, float]:
        """Sampled mode-coverage eval: K completions per prompt at T=temperature.

        Runs on the live vLLM engine without saving a checkpoint. All distributed
        ranks participate in the barriers; only rank 0 drives actor inference.
        Results are broadcast from rank 0 so that the subsequent all_reduce in
        eval_and_log produces correct per-run values.
        """
        self._pre_evaluate()
        draw_count = int(getattr(self.args, "eval_mode_coverage_draws", 4))
        seed_base = int(getattr(self.args, "eval_mode_coverage_seed", 1001))
        try:
            metrics = self._run_sampled_mode_coverage(
                dataset,
                benchmark_name,
                steps,
                k=k,
                temperature=temperature,
                draw_count=draw_count,
                seed_base=seed_base,
            )
        finally:
            self._post_evaluate()
        # All ranks must call broadcast the same number of times (once per key).
        # Non-rank-0 processes return {} from _run_sampled_mode_coverage, so
        # pre-populate the canonical keys with 0.0 on every rank before broadcast.
        canonical_keys = []
        metrics_to_broadcast = [(metric, "neutral") for metric in MODE_COVERAGE_METRICS]
        run_args = getattr(self, "args", None)
        if int(getattr(run_args, "diayn_num_options", 0) or 0) > 1:
            metrics_to_broadcast.extend(
                (metric, "latent") for metric in MODE_COVERAGE_METRICS
            )
            metrics_to_broadcast.extend(
                (metric, "neutral") for metric in OPTION_BINDING_METRICS
            )
        for metric, metric_namespace in metrics_to_broadcast:
            key = _mode_coverage_log_key(
                benchmark_name,
                metric,
                k,
                metric_namespace=metric_namespace,
            )
            canonical_keys.extend(
                [
                    key,
                    f"{key}_draw_count",
                    f"{key}_draw_std",
                    f"{key}_draw_se",
                    f"{key}_draw_min",
                    f"{key}_draw_max",
                    *(f"{key}_draw_{index}" for index in range(draw_count)),
                ]
            )
        for key in canonical_keys:
            metrics.setdefault(key, 0.0)
        metrics = self.strategy.broadcast(metrics)
        return metrics

    def _run_sampled_mode_coverage(
        self,
        dataset,
        benchmark_name: str,
        steps: int,
        *,
        k: int,
        temperature: float,
        draw_count: int,
        seed_base: int,
    ) -> dict[str, float]:
        if not self.strategy.is_rank_0():
            return {}

        # Nucleus truncation is a property of the sampled draws only; the
        # greedy trace below is unaffected by it and keeps the untruncated
        # surface so its value stays comparable across a decoding sweep.
        coverage_top_p = float(
            getattr(getattr(self, "args", None), "eval_mode_coverage_top_p", 1.0) or 1.0
        )
        # Each draw reserves its own block of K child streams; see
        # ``eval_mode_coverage_disjoint_draws``.
        stride = k if bool(getattr(self.args, "eval_mode_coverage_disjoint_draws", True)) else 1
        logging.info(
            "Starting sampled mode-coverage eval %s/%s: %s "
            "(%s prompts, draws=%s, K=%s, T=%s, top_p=%s, seeds=%s..%s stride=%s) at step %s",
            benchmark_name,
            len(self.eval_dataset_dict),
            benchmark_name,
            len(dataset),
            draw_count,
            k,
            temperature,
            coverage_top_p,
            seed_base,
            seed_base + (draw_count - 1) * stride + k - 1,
            stride,
            steps,
        )
        greedy_mean, greedy_prompt_outcomes = self._run_sampled_mode_coverage_draw(
            dataset,
            k=1,
            temperature=0.0,
            seed=0,
            condition_on_answer_options=False,
        )
        if greedy_mean:
            self._append_mode_coverage_draw_jsonl(
                {
                    "schema_version": 1,
                    "evaluation_kind": "deterministic_greedy_trace_neutral",
                    "benchmark": benchmark_name,
                    "step": int(steps),
                    "draw_index": None,
                    "seed": 0,
                    "sample_count": 1,
                    "temperature": 0.0,
                    "metrics": greedy_mean,
                    "prompts": greedy_prompt_outcomes,
                }
            )
        draw_means: list[dict[str, float]] = []
        draw_prompt_outcomes: list[list[dict[str, Any]]] = []
        for draw_index in range(draw_count):
            draw_seed = seed_base + draw_index * stride
            mean, prompt_outcomes = self._run_sampled_mode_coverage_draw(
                dataset,
                k=k,
                temperature=temperature,
                seed=draw_seed,
                condition_on_answer_options=False,
                top_p=coverage_top_p,
            )
            if not mean:
                continue
            draw_means.append(mean)
            draw_prompt_outcomes.append(prompt_outcomes)
            self._append_mode_coverage_draw_jsonl(
                {
                    "schema_version": 1,
                    "evaluation_kind": "fixed_seed_sampled_k_neutral",
                    "benchmark": benchmark_name,
                    "step": int(steps),
                    "draw_index": draw_index,
                    "seed": draw_seed,
                    "sample_count": int(k),
                    "temperature": float(temperature),
                    "top_p": coverage_top_p,
                    "metrics": mean,
                    "prompts": prompt_outcomes,
                }
            )

        if not draw_means:
            return {}
        metrics = _summarize_mode_coverage_draws(draw_means, benchmark_name, k)
        run_args = getattr(self, "args", None)
        if int(getattr(run_args, "diayn_num_options", 0) or 0) > 1:
            latent_draw_means: list[dict[str, float]] = []
            latent_draw_prompt_outcomes: list[list[dict[str, Any]]] = []
            for draw_index in range(draw_count):
                draw_seed = seed_base + draw_index * stride
                latent_mean, latent_prompt_outcomes = (
                    self._run_sampled_mode_coverage_draw(
                        dataset,
                        k=k,
                        temperature=temperature,
                        seed=draw_seed,
                        condition_on_answer_options=True,
                        top_p=coverage_top_p,
                    )
                )
                if not latent_mean:
                    continue
                latent_draw_means.append(latent_mean)
                latent_draw_prompt_outcomes.append(latent_prompt_outcomes)
                self._append_mode_coverage_draw_jsonl(
                    {
                        "schema_version": 2,
                        "evaluation_kind": ("fixed_seed_sampled_k_latent_binding"),
                        "benchmark": benchmark_name,
                        "step": int(steps),
                        "draw_index": draw_index,
                        "seed": draw_seed,
                        "seed_derivation": (
                            "draw_seed*1000000 + prompt_index*num_options + z"
                        ),
                        "sample_count": int(k),
                        "temperature": float(temperature),
                        "num_options": int(run_args.diayn_num_options),
                        "metrics": latent_mean,
                        "prompts": latent_prompt_outcomes,
                    }
                )
            if not latent_draw_means:
                raise RuntimeError(
                    "DIAYN latent-conditioned evaluation produced no draws"
                )
            metrics.update(
                _summarize_mode_coverage_draws(
                    latent_draw_means,
                    benchmark_name,
                    k,
                    metric_namespace="latent",
                )
            )
            answer_reprs_by_draw: list[list[str | None]] = []
            option_ids_by_draw: list[list[int | None]] = []
            correct_by_draw: list[list[bool]] = []
            for outcomes in latent_draw_prompt_outcomes:
                answer_rows: list[str | None] = []
                option_rows: list[int | None] = []
                correct_rows: list[bool] = []
                for outcome in outcomes:
                    prompt_index = int(outcome["prompt_index"])
                    answer_rows.extend(
                        conditional_answer_repr(prompt_index, key)
                        for key in outcome["answer_keys"]
                    )
                    option_rows.extend(outcome["option_ids"])
                    correct_rows.extend(
                        float(value) > 0.0 for value in outcome["rewards"]
                    )
                answer_reprs_by_draw.append(answer_rows)
                option_ids_by_draw.append(option_rows)
                correct_by_draw.append(correct_rows)
            crossfit_metrics = crossfit_option_answer_mi(
                answer_reprs_by_draw=answer_reprs_by_draw,
                option_ids_by_draw=option_ids_by_draw,
                correct_by_draw=correct_by_draw,
                num_options=int(run_args.diayn_num_options),
                smoothing=float(run_args.diayn_mi_smoothing),
            )
            metrics.update(
                _summarize_mode_coverage_draws(
                    crossfit_metrics,
                    benchmark_name,
                    k,
                )
            )
            self._append_mode_coverage_draw_jsonl(
                {
                    "schema_version": 2,
                    "evaluation_kind": "crossfit_option_binding",
                    "benchmark": benchmark_name,
                    "step": int(steps),
                    "draw_count": len(latent_draw_means),
                    "num_options": int(run_args.diayn_num_options),
                    "metrics_by_held_out_draw": crossfit_metrics,
                }
            )
        coverage_key = _mode_coverage_log_key(benchmark_name, "mode_coverage_at_k", k)
        pass_key = _mode_coverage_log_key(benchmark_name, "any_correct_at_k", k)
        mean_key = _mode_coverage_log_key(benchmark_name, "mean_at_k", k)
        distinct_key = _mode_coverage_log_key(
            benchmark_name, "distinct_correct_modes_at_k", k
        )

        logging.info(
            "Finished sampled mode-coverage eval %s: "
            "mode_cov@%s=%.4f±%.4f any_correct@%s=%.4f "
            "mean@%s=%.4f distinct@%s=%.4f across %s fixed draws at step %s",
            benchmark_name,
            k,
            metrics[coverage_key],
            metrics[f"{coverage_key}_draw_se"],
            k,
            metrics[pass_key],
            k,
            metrics[mean_key],
            k,
            metrics[distinct_key],
            len(draw_means),
            steps,
        )
        if int(getattr(run_args, "diayn_num_options", 0) or 0) > 1:
            latent_coverage_key = _mode_coverage_log_key(
                benchmark_name,
                "mode_coverage_at_k",
                k,
                metric_namespace="latent",
            )
            binding_key = _mode_coverage_log_key(
                benchmark_name,
                "option_answer_mi_lower_bound_nats",
                k,
            )
            logging.info(
                "Finished latent-conditioned binding eval %s: "
                "latent_mode_cov@%s=%.4f MI_lb@%s=%.4f with independent "
                "(draw,prompt,z) seeds at step %s",
                benchmark_name,
                k,
                metrics[latent_coverage_key],
                k,
                metrics[binding_key],
                steps,
            )
        return metrics

    def _run_sampled_mode_coverage_draw(
        self,
        dataset,
        *,
        k: int,
        temperature: float,
        seed: int,
        condition_on_answer_options: bool = False,
        top_p: float = 1.0,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Evaluate one fixed-seed K draw and return every prompt outcome."""

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.eval_dataloader_collate_fn,
        )
        per_prompt_metrics: list[dict[str, float]] = []
        prompt_outcomes: list[dict[str, Any]] = []
        futs: list = []
        pending: list[tuple[list[str], list[int], list[str]]] = []
        prompt_offset = 0

        for batch_index, (batch_formatted, batch_raw, batch_refs) in enumerate(
            dataloader
        ):
            actor = self.actors[batch_index % len(self.actors)]
            refs_batch = list(batch_refs)
            indices = list(range(prompt_offset, prompt_offset + len(refs_batch)))
            prompt_offset += len(refs_batch)
            futs.append(
                actor.futures.generate_for_mode_coverage(
                    list(batch_formatted),
                    refs_batch,
                    k,
                    temperature,
                    seed,
                    condition_on_answer_options,
                    indices,
                    float(top_p),
                )
            )
            pending.append((refs_batch, indices, list(batch_raw)))
            if len(futs) == len(self.actors) or batch_index == len(dataloader) - 1:
                for fut, (queued_refs, queued_indices, queued_prompts) in zip(
                    futs, pending
                ):
                    result = fut.result()
                    option_rows = result.get("option_ids")
                    if option_rows is None:
                        option_rows = [
                            [None] * len(reward_rows)
                            for reward_rows in result["rewards"]
                        ]
                    request_seed_rows = result.get("request_seeds_by_prompt")
                    if request_seed_rows is None:
                        request_seed_rows = [[] for _ in result["rewards"]]
                    for (
                        rewards,
                        answer_keys,
                        responses,
                        option_ids,
                        request_seeds,
                        ref,
                        prompt_index,
                        prompt,
                    ) in zip(
                        result["rewards"],
                        result["answer_keys"],
                        result["responses"],
                        option_rows,
                        request_seed_rows,
                        queued_refs,
                        queued_indices,
                        queued_prompts,
                    ):
                        mode_count = _parse_answer_mode_count(ref)
                        prompt_metrics = _compute_mode_coverage_metrics(
                            rewards,
                            answer_keys,
                            mode_count,
                            _parse_public_seed_key(ref),
                        )
                        per_prompt_metrics.append(prompt_metrics)
                        prompt_outcomes.append(
                            {
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "reference": ref,
                                "answer_mode_count": mode_count,
                                "responses": [str(response) for response in responses],
                                "rewards": [float(value) for value in rewards],
                                "answer_keys": [
                                    None if key is None else str(key)
                                    for key in answer_keys
                                ],
                                "option_ids": [
                                    None if value is None else int(value)
                                    for value in option_ids
                                ],
                                "request_seeds_by_option": [
                                    int(value) for value in request_seeds
                                ],
                                "metrics": prompt_metrics,
                            }
                        )
                futs.clear()
                pending.clear()

        if not per_prompt_metrics:
            return {}, []
        mean = {
            key: float(
                sum(metrics[key] for metrics in per_prompt_metrics)
                / len(per_prompt_metrics)
            )
            for key in per_prompt_metrics[0]
        }
        return mean, prompt_outcomes
