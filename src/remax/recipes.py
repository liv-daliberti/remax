"""Strict, dependency-light validation of maintained training recipes."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from .recipe_types import TrainingSettings

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DOMAINS = ("countdown", "graph_coloring", "python_factors", "mathir", "pantry_plan")


def strict_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


@dataclass(frozen=True)
class Recipe:
    name: str
    domain: str
    level: int
    method: Literal["drgrpo", "redr", "maxrl", "remax"]
    model_id: str
    model_revision: str
    seed: int
    registered_seeds: tuple[int, ...]
    settings: TrainingSettings
    source_ledger: str
    source_run_stamp: str
    resolved_wrapper_defaults: Mapping[str, str]

    @classmethod
    def load(cls, path: Path) -> Recipe:
        return cls.from_dict(strict_json(path.read_text()))

    @classmethod
    def from_dict(cls, raw: dict) -> Recipe:
        expected = {f.name for f in fields(cls)} - {"settings"} | {"environment"}
        if not isinstance(raw, dict) or set(raw) != expected:
            actual = set(raw) if isinstance(raw, dict) else set()
            raise ValueError(
                f"recipe fields: unknown={sorted(actual-expected)}, missing={sorted(expected-actual)}"
            )
        for key in (
            "name",
            "domain",
            "method",
            "model_id",
            "model_revision",
            "source_ledger",
            "source_run_stamp",
        ):
            if type(raw[key]) is not str or not raw[key].strip():
                raise ValueError(f"{key} requires a nonempty string")
        if (
            type(raw["level"]) is not int
            or raw["level"] != 1
            or raw["domain"] not in DOMAINS
        ):
            raise ValueError(
                "maintained recipes require Level 1 and a supported domain"
            )
        if raw["method"] not in ("drgrpo", "redr", "maxrl", "remax"):
            raise ValueError("unknown method")
        if (raw["model_id"], raw["model_revision"]) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("model identity is not in the maintained input registry")
        seeds = raw["registered_seeds"]
        if (
            type(seeds) is not list
            or seeds != [43, 44, 45, 46, 47]
            or any(type(v) is not int for v in seeds)
        ):
            raise ValueError("registered_seeds must be [43, 44, 45, 46, 47]")
        if type(raw["seed"]) is not int or raw["seed"] not in seeds:
            raise ValueError("seed must belong to registered_seeds")
        notes = raw["resolved_wrapper_defaults"]
        if (
            not isinstance(notes, dict)
            or set(notes) != {"evaluation", "resume_from", "xdr_tau"}
            or any(type(v) is not str for v in notes.values())
        ):
            raise ValueError("unknown or malformed resolved_wrapper_defaults metadata")
        values = {
            k: v
            for k, v in raw.items()
            if k not in ("environment", "registered_seeds", "resolved_wrapper_defaults")
        }
        result = cls(
            **values,
            settings=TrainingSettings.from_environment(raw["environment"]),
            registered_seeds=tuple(seeds),
            resolved_wrapper_defaults=MappingProxyType(dict(notes)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        s = self.settings
        required = {
            "seed": self.seed,
            "maxrl_task_objective": self.method in ("maxrl", "remax"),
            "online_canonical_replay_compute_only": self.method in ("maxrl", "drgrpo"),
            "online_canonical_replay": True,
            "online_canonical_replay_objective": "verified_likelihood_per_rollout",
            "online_canonical_key_mode": "modebench_outcome",
            "verified_discovery_tracking": True,
            "critic_type": "drgrpo",
            "verifier_version": "fast",
            "input_key": "problem",
            "eval_input_key": "problem",
            "output_key": "answer",
            "eval_output_key": "answer",
            "test_split": "multi_answer",
            "canonical_graph_actions": False,
            "prompt_template": (
                "qwen_pantry_support_mask"
                if self.domain == "pantry_plan"
                else "qwen_boxed"
            ),
            "canonical_action_task": (
                "pantry_support_mask" if self.domain == "pantry_plan" else "none"
            ),
            "canonical_graph_action_count": 6 if self.domain == "pantry_plan" else 3,
            "canonical_graph_fixed_shape_sampling": self.domain == "pantry_plan",
            "canonical_graph_learner_sampling": self.domain == "pantry_plan",
            "xdr_tau": math.inf,
        }
        for name in (
            "beta",
            "maxent_alpha",
            "maxent_control_target_ratio",
            "maxent_dual_target_ratio",
            "maxent_inverse_adaptation",
            "online_canonical_bank_alpha",
            "online_canonical_counterfactual_proposals",
            "online_canonical_counterfactual_separate_objective_support",
            "online_canonical_counterfactual_singleton_only",
            "policy_entropy_coef",
            "seed_entropy_alpha",
            "semantic_shannon_coef",
            "semantic_shannon_quality_gated_advantage",
            "semantic_shannon_separate_advantage",
            "semantic_shannon_success_conditioned_signed_advantage",
            "dapo_enabled",
            "rlep_replay_count",
        ):
            required[name] = 0
        for name, value in required.items():
            if getattr(s, name) != value:
                raise ValueError(
                    f"{self.method}/{self.domain} requires {name}={value!r}"
                )
        for name in (
            "learning_rate",
            "max_norm",
            "max_train",
            "num_samples",
            "num_prompt_epoch",
            "max_prompt_epochs",
            "num_ppo_epochs",
            "max_queries",
            "train_batch_size",
            "train_batch_size_per_device",
            "rollout_batch_size",
            "rollout_batch_size_per_device",
            "pi_buffer_maxlen_per_device",
            "sync_params_every",
            "prompt_max_length",
            "generate_max_length",
            "eval_generate_max_length",
            "eval_batch_size",
            "eval_mode_coverage_k",
            "eval_mode_coverage_draws",
            "online_canonical_replay_capacity",
            "online_canonical_replay_global_groups_per_step",
            "online_canonical_replay_alpha",
        ):
            if getattr(s, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if s.num_samples < 2 or s.max_train > 384:
            raise ValueError(
                "require at least two samples and at most 384 training rows"
            )
        if s.eval_prompt_interval * s.num_samples != s.eval_steps * s.train_batch_size:
            raise ValueError(
                "eval_prompt_interval metadata disagrees with eval_steps and batch size"
            )
        if s.num_prompt_epoch > s.max_prompt_epochs:
            raise ValueError("num_prompt_epoch exceeds max_prompt_epochs")
        if s.max_model_len < s.prompt_max_length + max(
            s.generate_max_length, s.eval_generate_max_length
        ):
            raise ValueError("max_model_len cannot hold prompt and generation budgets")
        if (
            not 0 < s.top_p <= 1
            or not 0 < s.vllm_gpu_ratio < 1
            or s.zero_stage not in (0, 1, 2, 3)
        ):
            raise ValueError("invalid top_p, vllm_gpu_ratio or zero_stage")
        if (
            s.train_batch_size % s.train_batch_size_per_device
            or s.train_batch_size % s.num_samples
        ):
            raise ValueError(
                "training batch must divide into microbatches and prompt sample groups"
            )
        if s.rollout_batch_size % s.rollout_batch_size_per_device:
            raise ValueError("rollout batch must divide into device batches")
        if self.domain == "pantry_plan" and (
            s.generate_max_length != 8 or s.eval_generate_max_length != 8
        ):
            raise ValueError(
                "pantry support-mask generation requires the registered eight-token budget"
            )

    def choose_seed(self, override: int | None) -> int:
        seed = self.seed if override is None else override
        if type(seed) is not int or seed not in self.registered_seeds:
            raise ValueError("seed must belong to registered_seeds")
        return seed
