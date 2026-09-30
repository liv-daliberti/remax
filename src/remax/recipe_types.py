"""Typed wire contract for the maintained Level 1 recipes. No training imports."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import re
from typing import get_type_hints


@dataclass(frozen=True)
class TrainingSettings:
    auto_resume: bool
    beta: float
    canonical_action_task: str
    canonical_graph_actions: bool
    canonical_graph_action_count: int
    canonical_graph_fixed_shape_sampling: bool
    canonical_graph_learner_sampling: bool
    collocate: bool
    eval_batch_size: int
    eval_generate_max_length: int
    eval_input_key: str
    eval_mode_coverage_draws: int
    eval_mode_coverage_k: int
    eval_mode_coverage_seed: int
    eval_mode_coverage_temperature: float
    eval_output_key: str
    eval_prompt_interval: int
    eval_steps: int
    eval_temperature: float
    export_steps: int
    generate_max_length: int
    input_key: str
    learning_rate: float
    maxent_alpha: float
    maxent_control_target_ratio: float
    maxent_dual_target_ratio: float
    maxent_inverse_adaptation: bool
    max_model_len: int
    max_norm: float
    max_prompt_epochs: int
    max_queries: int
    max_resume_num: int
    max_save_num: int
    max_train: int
    num_ppo_epochs: int
    num_prompt_epoch: int
    num_samples: int
    online_canonical_bank_alpha: float
    online_canonical_bank_pseudocount: float
    online_canonical_bank_surprisal_clip: float
    online_canonical_counterfactual_proposals: bool
    online_canonical_counterfactual_separate_objective_support: bool
    online_canonical_counterfactual_singleton_only: bool
    online_canonical_key_mode: str
    online_canonical_replay: bool
    online_canonical_replay_alpha: float
    online_canonical_replay_capacity: int
    online_canonical_replay_compute_only: bool
    online_canonical_replay_global_bootstrap_steps: int
    online_canonical_replay_global_groups_per_step: int
    online_canonical_replay_mass_alpha: float
    online_canonical_replay_objective: str
    output_key: str
    pi_buffer_maxlen_per_device: int
    policy_entropy_coef: float
    prompt_max_length: int
    prompt_template: str
    prune_resume_on_success: bool
    resume_from: int
    resume_steps: int
    rollout_batch_size: int
    rollout_batch_size_per_device: int
    save_from: int
    save_steps: int
    seed: int
    seed_entropy_alpha: float
    semantic_shannon_coef: float
    semantic_shannon_pseudocount: float
    semantic_shannon_quality_gated_advantage: bool
    semantic_shannon_separate_advantage: bool
    semantic_shannon_success_conditioned_signed_advantage: bool
    semantic_shannon_surprisal_clip: float
    sync_params_every: int
    temperature: float
    test_split: str
    top_p: float
    train_batch_size: int
    train_batch_size_per_device: int
    use_wb: bool
    verified_discovery_tracking: bool
    verifier_version: str
    vllm_gpu_ratio: float
    xdr_tau: float
    zero_stage: int
    critic_type: str = "drgrpo"
    dapo_enabled: bool = False
    rlep_replay_count: int = 0
    maxrl_task_objective: bool = False
    save_ckpt: bool = False

    @classmethod
    def from_environment(cls, raw: dict[str, str]) -> TrainingSettings:
        if not isinstance(raw, dict):
            raise ValueError("environment must be an object")
        types = get_type_hints(cls)
        parsed = {}
        for key, value in raw.items():
            name = key.removeprefix("OAT_ZERO_").lower()
            if key != "OAT_ZERO_" + name.upper() or name not in types:
                raise ValueError(f"unknown recipe environment field: {key}")
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"{key} requires a nonempty, unpadded string")
            kind = types[name]
            if kind is bool:
                if value not in ("0", "1"):
                    raise ValueError(f"{key} must be 0 or 1")
                parsed[name] = value == "1"
            elif kind is int:
                if not re.fullmatch(r"0|[1-9][0-9]*", value):
                    raise ValueError(f"{key} requires a nonnegative integer")
                parsed[name] = int(value)
            elif kind is float:
                try:
                    number = float(value)
                except ValueError as error:
                    raise ValueError(f"{key} requires a number") from error
                if not math.isfinite(number) and not (
                    name == "xdr_tau" and value == "inf"
                ):
                    raise ValueError(f"{key} requires a finite number")
                if number < 0:
                    raise ValueError(f"{key} cannot be negative")
                parsed[name] = number
            else:
                parsed[name] = value
        try:
            return cls(**parsed)
        except TypeError as error:
            raise ValueError(f"incomplete recipe environment: {error}") from error

    def environment(self) -> dict[str, str]:
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            result["OAT_ZERO_" + field.name.upper()] = (
                str(int(value)) if type(value) is bool else str(value)
            )
        return result
