"""Actor and oracle wiring for zero-math rollouts."""

from __future__ import annotations

import functools
import itertools
import logging
import math
import copy
import time
from multiprocessing.pool import ThreadPool
from typing import Any

import numpy as np
import torch
import tree
import vllm.envs as vllm_envs
from oat.algorithms.ppo import PPOActor
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric, TrajectoryData
from transformers import AutoTokenizer

from .args import ZeroMathArgs, resolve_canonical_action_task
from .answer_options import format_answer_option_prompt, option_id_for_sample
from .canonical_actions import (
    CanonicalActionSpace,
    decode_canonical_action_response,
    resolve_canonical_action_space,
)
from .math_grader import (
    answer_tag_reward_fn,
    boxed_reward_fn,
    collect_threaded_math_rewards,
    extract_normalized_final_answer,
)
from .benchmark import validate_reward_batch
from .math_grader_process import FullMathVerifierProcess
from .modebench_guided import guided_sampling_params
from .vllm_worker import PinnedWorkerExtensionArgs


def _configure_canonical_sampling_params(
    params: Any,
    *,
    action_token_ids: tuple[int, ...],
    action_count: int,
) -> None:
    """Restrict a vLLM request to a deterministic-horizon action policy."""

    if action_count <= 0:
        raise ValueError("canonical action_count must be positive")
    if not action_token_ids or len(set(action_token_ids)) != len(action_token_ids):
        raise ValueError("canonical action_token_ids must be nonempty and unique")
    params.allowed_token_ids = list(action_token_ids)
    # vLLM V0 reports probabilities after the allowed-token processor.  Ask
    # for the whole finite support so the learner can use the distribution
    # that actually generated the action as its behavior policy.
    params.logprobs = len(action_token_ids)
    params.max_tokens = action_count
    params.min_tokens = action_count
    params.ignore_eos = True
    params.stop = None
    params.stop_token_ids = None


class _CanonicalPositionLogitsProcessor:
    """Mask vLLM logits to the support for the current action position."""

    def __init__(self, token_ids_by_position: tuple[tuple[int, ...], ...]) -> None:
        self.token_ids_by_position = token_ids_by_position

    def __call__(
        self, generated_token_ids: list[int], scores: torch.Tensor
    ) -> torch.Tensor:
        position = len(generated_token_ids)
        if position == len(self.token_ids_by_position):
            # vLLM V0 applies custom logits processors once at the terminal
            # boundary before enforcing ``max_tokens``.  No fourth token is
            # sampled, so leave this bookkeeping call unchanged; the strict
            # post-generation validator still rejects any overlong result.
            return scores
        if position > len(self.token_ids_by_position):
            raise RuntimeError("canonical generation exceeded its fixed horizon")
        allowed = self.token_ids_by_position[position]
        masked = torch.full_like(scores, -torch.inf)
        indices = torch.tensor(allowed, dtype=torch.long, device=scores.device)
        masked[indices] = scores[indices]
        return masked


def _configure_position_canonical_sampling_params(
    params: Any, *, action_space: CanonicalActionSpace
) -> None:
    """Restrict a vLLM request to a position-dependent finite grammar."""

    params.allowed_token_ids = None
    params.logits_processors = [
        _CanonicalPositionLogitsProcessor(action_space.token_ids_by_position)
    ]
    params.logprobs = max(
        len(support) for support in action_space.token_ids_by_position
    )
    params.max_tokens = action_space.horizon
    params.min_tokens = action_space.horizon
    params.ignore_eos = True
    params.stop = None
    params.stop_token_ids = None


def _canonical_action_violation(
    token_ids: list[int] | tuple[int, ...],
    *,
    action_token_ids: tuple[int, ...],
    action_count: int,
) -> str | None:
    """Return a fail-closed explanation when a sampled action leaves support."""

    if len(token_ids) != action_count:
        return f"expected {action_count} tokens, got {len(token_ids)}"
    support = set(action_token_ids)
    unsupported = sorted({int(token_id) for token_id in token_ids} - support)
    if unsupported:
        return f"unsupported token ids {unsupported}; allowed={sorted(support)}"
    return None


def _position_canonical_action_violation(
    token_ids: list[int] | tuple[int, ...],
    *,
    action_token_ids_by_position: tuple[tuple[int, ...], ...],
) -> str | None:
    """Validate one token against each position-specific support."""

    if len(token_ids) != len(action_token_ids_by_position):
        return (
            f"expected {len(action_token_ids_by_position)} tokens, got {len(token_ids)}"
        )
    for position, (token_id, support) in enumerate(
        zip(token_ids, action_token_ids_by_position)
    ):
        if int(token_id) not in support:
            return (
                f"unsupported token id {int(token_id)} at position {position}; "
                f"allowed={list(support)}"
            )
    return None


def _resolve_mode_coverage_allowed_token_ids(
    eval_sampling_params: Any,
    canonical_action_token_ids: tuple[int, ...] | None,
) -> list[int] | None:
    """Recover immutable canonical support for a fresh coverage request.

    vLLM V0 destructively translates ``allowed_token_ids`` into a logits
    processor on the first request and then clears the public field on the
    caller's ``SamplingParams`` object.  Coverage uses a fresh params object,
    so copying that cleared field would silently drop the action constraint.
    The actor's resolved canonical token tuple is the source of truth.
    """

    if canonical_action_token_ids is not None:
        return [int(token_id) for token_id in canonical_action_token_ids]
    allowed = getattr(eval_sampling_params, "allowed_token_ids", None)
    return None if allowed is None else [int(token_id) for token_id in allowed]


def _extract_selected_action_logprobs(
    token_ids: list[int], logprob_rows: Any
) -> list[float]:
    """Fail closed while extracting actor probabilities for learner parity."""

    if logprob_rows is None or len(logprob_rows) != len(token_ids):
        observed = None if logprob_rows is None else len(logprob_rows)
        raise RuntimeError(
            "actor log-prob row count does not match sampled action count: "
            f"actions={len(token_ids)} logprob_rows={observed}"
        )
    selected = []
    for position, (token_id, row) in enumerate(zip(token_ids, logprob_rows)):
        try:
            value = float(row[token_id].logprob)
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise RuntimeError(
                "actor log-prob output is missing the sampled token at "
                f"position={position} token_id={token_id}"
            ) from error
        if not math.isfinite(value):
            raise RuntimeError(
                "actor selected-token log probability is nonfinite at "
                f"position={position} token_id={token_id}: {value}"
            )
        selected.append(value)
    return selected


def _extract_canonical_behavior_logprobs(
    token_ids: list[int],
    logprob_rows: Any,
    *,
    action_token_ids: tuple[int, ...],
) -> tuple[list[float], list[list[float]], float]:
    """Extract and validate vLLM's complete restricted behavior policy.

    With ``logprobs=len(action_token_ids)`` and all other vocabulary logits
    masked to ``-inf``, every V0 row must contain exactly the finite canonical
    support.  Values are returned in the immutable action-token order.
    """

    selected = _extract_selected_action_logprobs(token_ids, logprob_rows)
    expected_support = set(action_token_ids)
    ordered_rows: list[list[float]] = []
    normalization_error_max = 0.0
    for position, row in enumerate(logprob_rows):
        try:
            observed_support = {int(token_id) for token_id in row}
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "canonical actor returned a malformed behavior log-prob row "
                f"at position={position}"
            ) from error
        if observed_support != expected_support:
            raise RuntimeError(
                "canonical actor behavior support mismatch at "
                f"position={position}: expected={sorted(expected_support)} "
                f"observed={sorted(observed_support)}"
            )
        values: list[float] = []
        for action_token_id in action_token_ids:
            try:
                value = float(row[action_token_id].logprob)
            except (KeyError, TypeError, AttributeError, ValueError) as error:
                raise RuntimeError(
                    "canonical actor behavior row is missing a finite action "
                    f"at position={position} token_id={action_token_id}"
                ) from error
            if not math.isfinite(value):
                raise RuntimeError(
                    "canonical actor behavior log probability is nonfinite at "
                    f"position={position} token_id={action_token_id}: {value}"
                )
            values.append(value)

        row_max = max(values)
        log_normalizer = row_max + math.log(
            sum(math.exp(value - row_max) for value in values)
        )
        normalization_error = abs(log_normalizer)
        if normalization_error > 1e-6:
            raise RuntimeError(
                "canonical actor behavior probabilities are not normalized at "
                f"position={position}: abs(logsumexp)={normalization_error:.6g}"
            )
        normalization_error_max = max(normalization_error_max, normalization_error)
        ordered_rows.append(values)

    return selected, ordered_rows, normalization_error_max


def _extract_position_canonical_behavior_logprobs(
    token_ids: list[int],
    logprob_rows: Any,
    *,
    action_token_ids_by_position: tuple[tuple[int, ...], ...],
) -> tuple[list[float], list[list[float]], float]:
    """Extract vLLM probabilities under each positional grammar support."""

    selected = _extract_selected_action_logprobs(token_ids, logprob_rows)
    if len(logprob_rows) != len(action_token_ids_by_position):
        raise RuntimeError("canonical actor behavior row count mismatches horizon")
    ordered_rows = []
    normalization_error_max = 0.0
    for position, (row, support) in enumerate(
        zip(logprob_rows, action_token_ids_by_position)
    ):
        observed_support = {
            int(token_id)
            for token_id, value in row.items()
            if math.isfinite(float(value.logprob))
        }
        if observed_support != set(support):
            raise RuntimeError(
                "canonical actor behavior positional support mismatch at "
                f"position={position}: expected={sorted(support)} "
                f"observed={sorted(observed_support)}"
            )
        values = [float(row[token_id].logprob) for token_id in support]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("canonical actor behavior probabilities are nonfinite")
        row_max = max(values)
        log_normalizer = row_max + math.log(
            sum(math.exp(value - row_max) for value in values)
        )
        normalization_error = abs(log_normalizer)
        if normalization_error > 1e-6:
            raise RuntimeError(
                "canonical actor behavior probabilities are not normalized at "
                f"position={position}: abs(logsumexp)={normalization_error:.6g}"
            )
        normalization_error_max = max(normalization_error_max, normalization_error)
        ordered_rows.append(values)
    return selected, ordered_rows, normalization_error_max


class MATHOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for math answer grading."""

    FULL_VERIFIER_PROCESS_TIMEOUT_SECONDS = 5
    GRADER_WORKERS = 2

    def __init__(self, template, verifier_version) -> None:
        super().__init__()
        if template == "r1":
            math_reward_fn = answer_tag_reward_fn
        else:
            math_reward_fn = boxed_reward_fn
        self.full_math_verifier = None
        self.mp_pool = None
        if verifier_version == "math_verify":
            # Execute a dedicated module in a clean interpreter.  Unlike
            # multiprocessing spawn, this never re-imports the CUDA actor's
            # __main__; unlike a thread, it can be killed if SymPy wedges.
            self.full_math_verifier = FullMathVerifierProcess(
                reward_kind="r1" if template == "r1" else "boxed",
                timeout_seconds=self.FULL_VERIFIER_PROCESS_TIMEOUT_SECONDS,
            )
        else:
            self.math_reward_fn = functools.partial(math_reward_fn, fast=True)
            self.mp_pool = ThreadPool(self.GRADER_WORKERS)

    def get_reward(
        self,
        inputs: list[str],
        responses: list[str],
        references: list[str],
        batch_size: int = 4,
    ) -> tuple[torch.Tensor, Metric]:
        del inputs, batch_size

        if self.full_math_verifier is not None:
            rewards, infos = self.full_math_verifier.grade_batch(responses, references)
        else:
            assert self.mp_pool is not None
            rewards, infos = collect_threaded_math_rewards(
                self.mp_pool,
                self.math_reward_fn,
                responses,
                references,
                timeout_seconds=1,
            )

        validate_reward_batch(
            rewards,
            infos,
            count=len(responses),
            context="actor_oracle",
            references=references,
        )
        return torch.tensor(rewards), infos

    def compare(
        self,
        inputs: list[str],
        candidates_A: list[str],
        candidates_B: list[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> tuple[list[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""

        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info


class ZeroMathActor(PPOActor):
    def __init__(self, ipc_server, vllm_args, args: ZeroMathArgs) -> None:
        if int(getattr(args, "vllm_sleep_level", 1)) == 2:
            vllm_args = PinnedWorkerExtensionArgs(vllm_args)
        super().__init__(ipc_server, vllm_args, args)
        self._prompt_token_id_cache: dict[str, list[int]] = {}
        # OAT 0.0.9 configures actor sampling in __init__, while newer OAT
        # versions populate these fields in init(actor_id, save_path).
        if hasattr(self, "sampling_params"):
            self._configure_math_actor()

    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self._configure_math_actor()

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Expose vLLM's selective wake-up needed by discard-mode syncing."""

        self.llm.wake_up(tags=tags)

    def backup_model_buffers(self):
        """CPU-back model buffers before level-2 sleep discards weight storage."""

        return self.llm.collective_rpc("backup_model_buffers")

    def restore_model_buffers(self):
        """Restore and verify model buffers after weight storage is remapped."""

        return self.llm.collective_rpc("restore_model_buffers")

    def _configure_math_actor(self) -> None:
        if not hasattr(self, "_prompt_token_id_cache"):
            self._prompt_token_id_cache = {}
        if not hasattr(self, "prompt_tokenizer"):
            try:
                self.prompt_tokenizer = AutoTokenizer.from_pretrained(
                    self.args.pretrain,
                    trust_remote_code=True,
                    use_fast=False,
                )
                logging.info(
                    "actor prompt tokenizer type=%s is_fast=%s",
                    type(self.prompt_tokenizer).__name__,
                    getattr(self.prompt_tokenizer, "is_fast", None),
                )
            except Exception:
                logging.exception(
                    "Failed to load slow prompt tokenizer; falling back to actor tokenizer"
                )
                self.prompt_tokenizer = self.tokenizer

        self.oracle = MATHOracle(
            template=self.args.prompt_template,
            verifier_version=self.args.verifier_version,
        )
        syntax_profile = str(getattr(self.args, "modebench_syntax_profile", "none"))
        modebench_domain = str(getattr(self.args, "modebench_domain", "none"))
        if syntax_profile != "none" and modebench_domain == "none":
            raise RuntimeError("active ModeBench syntax profile requires a domain")
        self._modebench_syntax_profile = syntax_profile
        self._modebench_domain = modebench_domain

        self._canonical_action_space: CanonicalActionSpace | None = None
        self._canonical_action_token_ids: tuple[int, ...] | None = None
        canonical_task = resolve_canonical_action_task(self.args)
        if canonical_task != "none":
            if bool(vllm_envs.VLLM_USE_V1):
                raise RuntimeError(
                    "canonical actions require VLLM_USE_V1=0 because "
                    "vLLM 0.8.4 V1 reports raw pre-processor log probabilities"
                )
            self._canonical_action_space = resolve_canonical_action_space(
                self.tokenizer, canonical_task
            )
            self._canonical_action_token_ids = (
                self._canonical_action_space.union_token_ids
            )
            for params in (self.sampling_params, self.eval_sampling_params):
                if canonical_task == "graph_coloring":
                    _configure_canonical_sampling_params(
                        params,
                        action_token_ids=self._canonical_action_token_ids,
                        action_count=self._canonical_action_space.horizon,
                    )
                else:
                    _configure_position_canonical_sampling_params(
                        params, action_space=self._canonical_action_space
                    )
            # OAT otherwise seeds the vLLM engine from wall-clock time.  Use
            # deterministic per-request streams so a nominal experiment seed
            # reproduces the sampled canonical prefixes as well as the learner.
            actor_id = int(getattr(self, "actor_id", 0) or 0)
            self._canonical_request_seed_base = int(self.args.seed) + actor_id * 100_000
            self._canonical_rollout_request_index = 0
            self._canonical_coverage_request_index = 0
            self.sampling_params.seed = self._canonical_request_seed_base
            self.eval_sampling_params.seed = self._canonical_request_seed_base + 50_000
            logging.info(
                "canonical %s actor configured: position_token_ids=%s "
                "sequence_count=%s max_entropy=%.9f action_count=%s "
                "tokenizer=%s tokenizer_class=%s vocab_size=%s vllm_engine=v0 "
                "request_seed_base=%s",
                canonical_task,
                self._canonical_action_space.token_ids_by_position,
                self._canonical_action_space.sequence_count,
                self._canonical_action_space.max_sequence_entropy,
                self._canonical_action_space.horizon,
                getattr(self.tokenizer, "name_or_path", self.args.pretrain),
                type(self.tokenizer).__name__,
                len(self.tokenizer),
                self._canonical_request_seed_base,
            )

        if self.args.prompt_template in [
            "qwen_boxed",
            "qwen_countdown_digits",
            "qwen_graph_digits",
            "qwen_pantry_support_mask",
            "qwen_math",
            "qwen_math_route",
            "qwen_level2_countdown",
            "qwen_level2_python_factors",
            "qwen_level2_mathir",
            "qwen_level2_pantry",
            "falcon_boxed",
            "falcon_countdown_digits",
            "falcon_graph_digits",
            "falcon_pantry_support_mask",
            "falcon_math",
            "falcon_math_route",
            "no",
        ]:
            self.sampling_params.stop = None
            self.sampling_params.stop_token_ids = None
            self.eval_sampling_params.stop = None
            self.eval_sampling_params.stop_token_ids = None
        elif self.args.prompt_template == "r1":
            self.sampling_params.stop = ["</answer>"]
            self.sampling_params.include_stop_str_in_output = True
            self.eval_sampling_params.stop = ["</answer>"]
            self.eval_sampling_params.include_stop_str_in_output = True

    def _encode_formatted_prompts(
        self, formatted_prompts: list[str]
    ) -> list[list[int]]:
        """Encode prompts once in the actor before handing them to vLLM."""

        prompt_tokenizer = getattr(self, "prompt_tokenizer", self.tokenizer)
        bos_token = getattr(prompt_tokenizer, "bos_token", None) or getattr(
            self.tokenizer, "bos_token", None
        )
        tokenized_prompts = []
        for prompt in formatted_prompts:
            if bos_token:
                prompt = prompt.removeprefix(bos_token)
            cached_ids = self._prompt_token_id_cache.get(prompt)
            if cached_ids is None:
                cached_ids = list(prompt_tokenizer.encode(prompt))
                self._prompt_token_id_cache[prompt] = cached_ids
            tokenized_prompts.append(list(cached_ids))
        return tokenized_prompts

    def generate_for_mode_coverage(
        self,
        formatted_prompts: list[str],
        refs: list[str],
        n: int,
        temperature: float,
        seed: int | None = None,
        condition_on_answer_options: bool = False,
        prompt_indices: list[int] | None = None,
        top_p: float = 1.0,
    ) -> dict[str, list]:
        """Sample n completions per prompt and return per-prompt rewards and answer keys.

        Used by the learner to compute sampled mode-coverage metrics during training
        without saving a checkpoint.  The default path is deliberately neutral even
        for a DIAYN-trained policy, so matched policies receive identical requests.
        The explicit answer-option path derives a separate deterministic request seed
        for every (draw, prompt, option) tuple.
        """
        import vllm

        if prompt_indices is None:
            prompt_indices = list(range(len(formatted_prompts)))
        if len(prompt_indices) != len(formatted_prompts):
            raise ValueError("mode-coverage prompt indices must match the prompt batch")
        if not 0 < float(top_p) <= 1:
            raise ValueError("mode-coverage top_p must lie in (0, 1]")
        canonical_action_token_ids = getattr(self, "_canonical_action_token_ids", None)
        canonical_action_space = getattr(self, "_canonical_action_space", None)
        coverage_seed = int(seed) if seed is not None else None
        if coverage_seed is None and canonical_action_token_ids is not None:
            coverage_seed = (
                int(self._canonical_request_seed_base)
                + 50_000
                + int(self._canonical_coverage_request_index)
            )
            self._canonical_coverage_request_index += 1
        diayn_option_count = int(getattr(self.args, "diayn_num_options", 0) or 0)
        use_diayn_options = bool(condition_on_answer_options)
        if use_diayn_options and diayn_option_count <= 1:
            raise ValueError(
                "answer-option-conditioned evaluation requires DIAYN options"
            )
        if use_diayn_options and n % diayn_option_count != 0:
            raise ValueError(
                "answer-option-conditioned sample count must divide across options"
            )
        if use_diayn_options and coverage_seed is None:
            raise ValueError(
                "answer-option-conditioned evaluation requires an explicit draw seed"
            )
        samples_per_request = n // diayn_option_count if use_diayn_options else n

        def make_params(request_seed: int | None):
            return vllm.SamplingParams(
                n=samples_per_request,
                temperature=temperature,
                # Untruncated (1.0) unless a decoding sweep asks otherwise, so
                # every previously reported cell is reproduced exactly.
                top_p=float(top_p),
                max_tokens=self.eval_sampling_params.max_tokens,
                min_tokens=getattr(self.eval_sampling_params, "min_tokens", 0),
                ignore_eos=getattr(self.eval_sampling_params, "ignore_eos", False),
                stop=self.eval_sampling_params.stop,
                stop_token_ids=self.eval_sampling_params.stop_token_ids,
                allowed_token_ids=_resolve_mode_coverage_allowed_token_ids(
                    self.eval_sampling_params, canonical_action_token_ids
                ),
                logits_processors=(
                    list(
                        getattr(self.eval_sampling_params, "logits_processors", None)
                        or []
                    )
                    if canonical_action_space is not None
                    and canonical_action_space.task == "countdown"
                    else None
                ),
                include_stop_str_in_output=getattr(
                    self.eval_sampling_params,
                    "include_stop_str_in_output",
                    False,
                ),
                seed=request_seed,
            )

        generation_prompts: list[str] = []
        generation_rows: list[tuple[int, int | None]] = []
        generation_seeds: list[int] = []
        request_seeds_by_prompt: list[list[int]] = [
            [] for _ in formatted_prompts
        ]
        if use_diayn_options:
            for prompt_index, (global_prompt_index, prompt) in enumerate(
                zip(prompt_indices, formatted_prompts)
            ):
                for option_id in range(diayn_option_count):
                    generation_prompts.append(
                        format_answer_option_prompt(
                            prompt, option_id, diayn_option_count
                        )
                    )
                    generation_rows.append((prompt_index, option_id))
                    request_seed = (
                        int(coverage_seed) * 1_000_000
                        + int(global_prompt_index) * diayn_option_count
                        + option_id
                    )
                    generation_seeds.append(request_seed)
                    request_seeds_by_prompt[prompt_index].append(request_seed)
        else:
            generation_prompts = list(formatted_prompts)
            generation_rows = [
                (prompt_index, None) for prompt_index in range(len(formatted_prompts))
            ]
            if coverage_seed is not None:
                request_seeds_by_prompt = [
                    [int(coverage_seed)] for _ in formatted_prompts
                ]

        sampling_params = (
            [make_params(request_seed) for request_seed in generation_seeds]
            if use_diayn_options
            else make_params(coverage_seed)
        )
        if not use_diayn_options:
            sampling_params = guided_sampling_params(
                sampling_params,
                self._modebench_syntax_profile,
                self._modebench_domain,
                refs,
            )
        outputs = self.generate(generation_prompts, sampling_params)
        if len(outputs) != len(generation_rows):
            raise RuntimeError("mode-coverage actor returned an invalid request grid")

        # Flatten: score all (prompt, response) pairs in one oracle call.
        per_prompt_responses: list[list[str | None]] = [[None] * n for _ in refs]
        per_prompt_options: list[list[int | None]] = [[None] * n for _ in refs]
        for request_index, output in enumerate(outputs):
            prompt_index, option_id = generation_rows[request_index]
            ref = refs[prompt_index]
            for sample_index, sample in enumerate(output.outputs):
                target_index = (
                    int(option_id) * samples_per_request + sample_index
                    if option_id is not None
                    else sample_index
                )
                if canonical_action_token_ids is not None:
                    assert canonical_action_space is not None
                    violation = _position_canonical_action_violation(
                        list(sample.token_ids),
                        action_token_ids_by_position=(
                            canonical_action_space.token_ids_by_position
                        ),
                    )
                    if violation is not None:
                        raise RuntimeError(
                            "canonical mode-coverage rollout violated its "
                            "fixed action support at "
                            f"prompt={prompt_index} sample={sample_index}: {violation}"
                        )
                    if sample.finish_reason != "length":
                        raise RuntimeError(
                            "canonical mode-coverage rollout terminated "
                            "unexpectedly at "
                            f"prompt={prompt_index} sample={sample_index}: "
                            f"finish_reason={sample.finish_reason!r}"
                        )
                response = sample.text.strip()
                if canonical_action_space is not None:
                    response = decode_canonical_action_response(
                        canonical_action_space.task, response, ref
                    )
                per_prompt_responses[prompt_index][target_index] = response
                per_prompt_options[prompt_index][target_index] = option_id

        if any(response is None for rows in per_prompt_responses for response in rows):
            raise RuntimeError("mode-coverage actor failed to fill every sample")
        all_responses = [
            str(response) for rows in per_prompt_responses for response in rows
        ]
        all_refs_flat = [ref for ref in refs for _ in range(n)]

        rewards_tensor, oracle_infos = self.oracle.get_reward(
            [""] * len(all_responses),
            all_responses,
            all_refs_flat,
        )
        rewards_flat = rewards_tensor.tolist()

        answer_keys_flat = [
            extract_normalized_final_answer(
                resp,
                template=self.args.prompt_template,
                gt_answer=ref,
            )
            for resp, ref in zip(all_responses, all_refs_flat)
        ]

        per_prompt_rewards = [
            rewards_flat[index : index + n] for index in range(0, len(rewards_flat), n)
        ]
        per_prompt_keys = [
            answer_keys_flat[index : index + n]
            for index in range(0, len(answer_keys_flat), n)
        ]
        realized_responses = [
            all_responses[index : index + n]
            for index in range(0, len(all_responses), n)
        ]

        return {
            "rewards": per_prompt_rewards,
            "verifier_infos": [
                oracle_infos[i : i + n] for i in range(0, len(oracle_infos), n)
            ],
            "answer_keys": per_prompt_keys,
            "responses": realized_responses,
            "option_ids": per_prompt_options,
            "request_seeds": generation_seeds if use_diayn_options else None,
            "request_seeds_by_prompt": request_seeds_by_prompt,
        }

    def generate_and_maybe_eval(
        self,
        prompts: list[str],
        formatted_prompts: list[str],
        references: list[str] | None = None,
    ):
        """Generate greedy eval responses and decode canonical actions before grading.

        OAT's inherited evaluator sends generated text directly to the oracle.  That
        is valid for ordinary text responses and canonical graph colorings, but a
        canonical Countdown response is a three-digit action code that must first
        be decoded into its arithmetic expression.  Keep the inherited path for
        non-canonical runs and mirror its sample-major grading/reshape contract for
        canonical runs.
        """

        canonical_action_space = getattr(self, "_canonical_action_space", None)
        syntax_profile = getattr(self, "_modebench_syntax_profile", "none")
        if canonical_action_space is None and syntax_profile == "none":
            return super().generate_and_maybe_eval(
                prompts, formatted_prompts, references
            )

        if not self.eval_mode:
            raise RuntimeError("canonical evaluation requires eval mode")
        if references is None or len(references) != len(prompts):
            raise RuntimeError("canonical evaluation requires one reference per prompt")

        eval_params = guided_sampling_params(
            self.eval_sampling_params,
            syntax_profile,
            self._modebench_domain,
            references,
        )
        outputs = self.generate(formatted_prompts, eval_params)
        candidates = (
            [[sample.text.strip() for sample in output.outputs] for output in outputs]
            if isinstance(eval_params, list)
            else self.extract_candidates_from_output(outputs, self.eval_sampling_params)
        )
        sample_count = int(self.eval_sampling_params.n)
        if len(candidates) != len(prompts) or any(
            len(prompt_candidates) != sample_count for prompt_candidates in candidates
        ):
            raise RuntimeError("canonical evaluation returned an invalid sample grid")

        decoded = (
            [
                [
                    decode_canonical_action_response(
                        canonical_action_space.task, candidate, reference
                    )
                    for candidate in prompt_candidates
                ]
                for prompt_candidates, reference in zip(candidates, references)
            ]
            if canonical_action_space is not None
            else candidates
        )
        responses = [
            decoded[prompt_index][sample_index]
            for sample_index in range(sample_count)
            for prompt_index in range(len(prompts))
        ]
        win_probs, _ = self.oracle.compare(
            prompts * sample_count,
            responses,
            references * sample_count,
            batch_size=self.oracle_batch_size,
            return_probs=True,
            disable_tqdm=True,
        )
        reshaped_responses = [
            [
                decoded[prompt_index][sample_index]
                for sample_index in range(sample_count)
            ]
            for prompt_index in range(len(prompts))
        ]
        reshaped_win_probs = (
            np.asarray(win_probs).reshape(sample_count, len(prompts)).transpose(1, 0)
        )
        return reshaped_responses, reshaped_win_probs

    def step(
        self,
        prompts: list[str],
        formatted_prompts: list[str],
        references: list[str] | None = None,
        *,
        sampling_temperature: float | None = None,
        sampling_seed: int | None = None,
    ) -> list[TrajectoryData]:
        """Generate trajectories and score them with the math oracle."""

        assert not self.eval_mode
        info = {}
        logging.info("actor start")

        st = time.time()
        diayn_option_count = int(getattr(self.args, "diayn_num_options", 0) or 0)
        use_diayn_options = diayn_option_count > 1
        samples_per_request = int(self.sampling_params.n)
        generation_prompt_to_original: list[int] | None = None
        generation_prompt_to_option: list[int | None] | None = None
        sampling_params = self.sampling_params
        if sampling_temperature is not None or sampling_seed is not None:
            sampling_params = copy.copy(self.sampling_params)
        if sampling_temperature is not None:
            if (
                not math.isfinite(float(sampling_temperature))
                or float(sampling_temperature) <= 0
            ):
                raise ValueError(
                    "per-request sampling temperature must be finite and positive"
                )
            sampling_params.temperature = float(sampling_temperature)
        if sampling_seed is not None:
            if (
                isinstance(sampling_seed, bool)
                or int(sampling_seed) != sampling_seed
                or int(sampling_seed) < 0
            ):
                raise ValueError(
                    "per-request sampling seed must be a non-negative integer"
                )
            sampling_params.seed = int(sampling_seed)
        if use_diayn_options:
            if getattr(self, "_canonical_action_token_ids", None) is not None:
                raise RuntimeError("DIAYN answer options require free-form rollouts")
            if int(self.sampling_params.n) % diayn_option_count != 0:
                raise RuntimeError(
                    "num_samples must divide evenly across DIAYN options"
                )
            samples_per_request = int(self.sampling_params.n) // diayn_option_count
            sampling_params = copy.copy(sampling_params)
            sampling_params.n = samples_per_request
            expanded_prompts = []
            generation_prompt_to_original = []
            generation_prompt_to_option = []
            for prompt_index, prompt in enumerate(formatted_prompts):
                for option_id in range(diayn_option_count):
                    expanded_prompts.append(
                        format_answer_option_prompt(
                            prompt, option_id, diayn_option_count
                        )
                    )
                    generation_prompt_to_original.append(prompt_index)
                    generation_prompt_to_option.append(option_id)
            formatted_prompt_token_ids = self._encode_formatted_prompts(
                expanded_prompts
            )
        else:
            formatted_prompt_token_ids = self._encode_formatted_prompts(
                formatted_prompts
            )
        canonical_request_seed = None
        if getattr(self, "_canonical_action_token_ids", None) is not None:
            canonical_request_seed = int(self._canonical_request_seed_base) + int(
                self._canonical_rollout_request_index
            )
            self._canonical_rollout_request_index += 1
            sampling_params.seed = canonical_request_seed
        if use_diayn_options and self._modebench_syntax_profile != "none":
            raise RuntimeError("Level-2 guided syntax is incompatible with DIAYN options")
        if references is None and self._modebench_syntax_profile != "none":
            raise RuntimeError("Level-2 guided syntax requires row references")
        generation_sampling_params = guided_sampling_params(
            sampling_params,
            self._modebench_syntax_profile,
            self._modebench_domain,
            [] if references is None else references,
        )
        outputs = self.generate(formatted_prompt_token_ids, generation_sampling_params)

        total_samples_per_prompt = int(self.sampling_params.n)
        candidates = [[None] * total_samples_per_prompt for _ in prompts]
        prompt_token_ids = [[None] * total_samples_per_prompt for _ in prompts]
        no_eos = [[False] * total_samples_per_prompt for _ in prompts]
        response_ids = [[None] * total_samples_per_prompt for _ in prompts]
        response_logprobs = [[None] * total_samples_per_prompt for _ in prompts]
        diayn_option_ids = [[None] * total_samples_per_prompt for _ in prompts]
        canonical_behavior_logprobs = [
            [None] * total_samples_per_prompt for _ in prompts
        ]
        resp_lens = []
        canonical_action_token_ids = getattr(self, "_canonical_action_token_ids", None)
        canonical_action_space = getattr(self, "_canonical_action_space", None)
        canonical_action_count = (
            canonical_action_space.horizon
            if canonical_action_space is not None
            else int(self.args.canonical_graph_action_count)
        )
        canonical_finish_length_count = 0
        canonical_finish_unexpected_count = 0
        canonical_behavior_q_row_count = 0
        canonical_behavior_q_support_min: int | None = None
        canonical_behavior_q_support_max: int | None = None
        canonical_behavior_q_norm_error_max = 0.0
        for i in range(len(outputs)):
            original_prompt_index = (
                generation_prompt_to_original[i]
                if generation_prompt_to_original is not None
                else i
            )
            option_id = (
                generation_prompt_to_option[i]
                if generation_prompt_to_option is not None
                else None
            )
            sample_offset = (
                int(option_id) * samples_per_request if option_id is not None else 0
            )
            for k in range(samples_per_request):
                sample = outputs[i].outputs[k]
                target_sample_index = sample_offset + k
                token_ids = list(sample.token_ids)
                if canonical_action_token_ids is not None:
                    assert canonical_action_space is not None
                    violation = _position_canonical_action_violation(
                        token_ids,
                        action_token_ids_by_position=(
                            canonical_action_space.token_ids_by_position
                        ),
                    )
                    if violation is not None:
                        raise RuntimeError(
                            "canonical rollout violated its fixed action "
                            f"support at prompt={i} sample={k}: {violation}"
                        )
                    # Reaching max_tokens is the intended deterministic
                    # termination event, not a missing-EOS failure.
                    if sample.finish_reason != "length":
                        raise RuntimeError(
                            "canonical rollout terminated unexpectedly at "
                            f"prompt={i} sample={k}: "
                            f"finish_reason={sample.finish_reason!r}"
                        )
                    no_eos[original_prompt_index][target_sample_index] = False
                    canonical_finish_length_count += 1
                else:
                    no_eos[original_prompt_index][target_sample_index] = (
                        sample.finish_reason == "length"
                    )
                candidate = sample.text
                if canonical_action_space is not None:
                    if references is None:
                        raise RuntimeError("canonical rollout is missing its reference")
                    candidate = decode_canonical_action_response(
                        canonical_action_space.task,
                        candidate,
                        references[original_prompt_index],
                    )
                candidates[original_prompt_index][target_sample_index] = candidate
                prompt_token_ids[original_prompt_index][target_sample_index] = outputs[
                    i
                ].prompt_token_ids
                response_ids[original_prompt_index][target_sample_index] = token_ids
                diayn_option_ids[original_prompt_index][target_sample_index] = option_id
                if canonical_action_token_ids is not None:
                    assert canonical_action_space is not None
                    (
                        logps,
                        behavior_rows,
                        behavior_norm_error,
                    ) = _extract_position_canonical_behavior_logprobs(
                        token_ids,
                        sample.logprobs,
                        action_token_ids_by_position=(
                            canonical_action_space.token_ids_by_position
                        ),
                    )
                    canonical_behavior_logprobs[original_prompt_index][
                        target_sample_index
                    ] = behavior_rows
                    canonical_behavior_q_row_count += len(behavior_rows)
                    row_support_min = min(
                        len(support)
                        for support in canonical_action_space.token_ids_by_position
                    )
                    row_support_max = max(
                        len(support)
                        for support in canonical_action_space.token_ids_by_position
                    )
                    canonical_behavior_q_support_min = (
                        row_support_min
                        if canonical_behavior_q_support_min is None
                        else min(canonical_behavior_q_support_min, row_support_min)
                    )
                    canonical_behavior_q_support_max = (
                        row_support_max
                        if canonical_behavior_q_support_max is None
                        else max(canonical_behavior_q_support_max, row_support_max)
                    )
                    canonical_behavior_q_norm_error_max = max(
                        canonical_behavior_q_norm_error_max, behavior_norm_error
                    )
                else:
                    logps = _extract_selected_action_logprobs(
                        token_ids, sample.logprobs
                    )
                response_logprobs[original_prompt_index][target_sample_index] = logps
                resp_lens.append(len(token_ids))

        info["actor/generate_time"] = time.time() - st

        st = time.time()
        rewards, oracle_infos = self.oracle.get_reward(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, total_samples_per_prompt) for x in prompts
                )
            ),
            tree.flatten(candidates),
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, total_samples_per_prompt) for x in references
                )
            ),
        )

        info["actor/verify_time"] = time.time() - st
        logging.info("actor reward %s", rewards.mean())
        info["actor/rewards"] = rewards.mean().item()
        info["actor/num_data"] = rewards.numel()
        info["actor/formatted"] = np.mean([i["formatted"] for i in oracle_infos])
        info["actor/response_tok_len"] = np.mean(resp_lens)
        info["actor/sampling_max_tokens"] = self.sampling_params.max_tokens
        info["actor/sampling_temperature"] = sampling_params.temperature
        if sampling_seed is not None:
            info["actor/sampling_request_seed"] = int(sampling_seed)
        if use_diayn_options:
            info["actor/diayn_num_options"] = float(diayn_option_count)
            info["actor/diayn_samples_per_option"] = float(samples_per_request)
        if canonical_action_token_ids is not None:
            assert canonical_action_space is not None
            info["actor/canonical_graph_actions"] = float(
                canonical_action_space.task == "graph_coloring"
            )
            info["actor/canonical_countdown_actions"] = float(
                canonical_action_space.task == "countdown"
            )
            info["actor/canonical_pantry_support_mask_actions"] = float(
                canonical_action_space.task == "pantry_support_mask"
            )
            info["actor/canonical_action_count"] = canonical_action_count
            info["actor/canonical_action_support_size"] = len(
                canonical_action_token_ids
            )
            info["actor/canonical_sequence_support_size"] = (
                canonical_action_space.sequence_count
            )
            info["actor/canonical_max_sequence_entropy"] = (
                canonical_action_space.max_sequence_entropy
            )
            info["actor/canonical_invalid_count"] = 0
            info["actor/canonical_finish_length_count"] = canonical_finish_length_count
            info["actor/canonical_finish_unexpected_count"] = (
                canonical_finish_unexpected_count
            )
            info["actor/canonical_behavior_q_row_count"] = (
                canonical_behavior_q_row_count
            )
            info["actor/canonical_behavior_q_support_min"] = int(
                canonical_behavior_q_support_min or 0
            )
            info["actor/canonical_behavior_q_support_max"] = int(
                canonical_behavior_q_support_max or 0
            )
            info["actor/canonical_behavior_q_norm_error_max"] = (
                canonical_behavior_q_norm_error_max
            )
            info["actor/canonical_request_seed"] = int(canonical_request_seed)

        rewards = rewards.reshape(len(prompts), -1)
        no_eos_array = np.array(no_eos, dtype=bool).reshape(len(prompts), -1)
        info["actor/no_eos_count"] = no_eos_array.sum()

        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            reference = references[i] if references is not None else None
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                reward = rewards[i][j].item()
                if no_eos_array[i][j]:
                    reward = 0
                if response_ids[i][j] is None or response_logprobs[i][j] is None:
                    raise RuntimeError("actor failed to fill an option-expanded sample")
                dense_rewards = [0] * len(response_ids[i][j])
                dense_rewards[-1] = reward
                trajectory = TrajectoryData(
                    prompt=prompt,
                    prompt_ids=prompt_token_ids[i][j],
                    response=candidates_per_prompt[j],
                    response_ids=response_ids[i][j],
                    response_logprobs=response_logprobs[i][j],
                    rewards=dense_rewards,
                    loss_mask=not no_eos_array[i][j]
                    if self.args.ignore_no_eos
                    else True,
                    info=info,
                )
                setattr(trajectory, "reference", reference)
                setattr(
                    trajectory,
                    "verifier_diagnostic",
                    oracle_infos[i * total_samples_per_prompt + j].get("verifier"),
                )
                if use_diayn_options:
                    option_id = diayn_option_ids[i][j]
                    if option_id is None:
                        option_id = option_id_for_sample(
                            j, diayn_option_count, samples_per_request
                        )
                    setattr(trajectory, "diayn_option_id", int(option_id))
                if canonical_action_token_ids is not None:
                    assert canonical_action_space is not None
                    setattr(
                        trajectory,
                        "canonical_behavior_action_logprobs",
                        canonical_behavior_logprobs[i][j],
                    )
                    setattr(
                        trajectory,
                        "canonical_behavior_action_token_ids",
                        list(canonical_action_token_ids),
                    )
                    setattr(
                        trajectory,
                        "canonical_behavior_action_token_ids_by_position",
                        [
                            list(support)
                            for support in canonical_action_space.token_ids_by_position
                        ],
                    )
                trajectory_data.append(trajectory)
        logging.info("actor finished data_len=%s", len(trajectory_data))
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle
