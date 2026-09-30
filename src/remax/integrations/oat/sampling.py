"""OAT sampling responsibilities, preserving historical state and ordering."""

from __future__ import annotations
import logging
import math
import time
from typing import Any
import numpy as np
import torch
import torch.distributed as dist
from oat.types import TrajectoryData
from ...args import resolve_canonical_action_task
from ...canonical_actions import (
    CanonicalActionSpace,
    _sample_canonical_actions_from_logits,
    decode_canonical_action_response,
)
from .support import _grade_decoded_canonical_response


class OatSamplingMixin:
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
            oracle_info, numeric_reward = _grade_decoded_canonical_response(
                response,
                record["reference"],
                fast=self.args.verifier_version == "fast",
            )
            record["verifier_diagnostic"] = oracle_info.get("verifier")
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
                trajectory, "verifier_diagnostic", record.get("verifier_diagnostic")
            )
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
