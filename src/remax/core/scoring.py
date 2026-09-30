"""Causal response masks and right-padded teacher-forcing inputs."""

from __future__ import annotations

import math
from typing import Protocol, Sequence

import torch

from .replay_types import CanonicalReplayBatch


class ReplayGroup(Protocol):
    """Minimal read-only interface accepted by the teacher-forcing materializer."""

    @property
    def prompt_token_ids(self) -> Sequence[int]: ...

    @property
    def response_token_ids(self) -> Sequence[Sequence[int]]: ...


def materialize_canonical_replay_batch(
    groups: Sequence[ReplayGroup],
    *,
    pad_token_id: int,
    device: torch.device | int | str,
) -> CanonicalReplayBatch:
    """Build causal-LM labels without changing any stored exemplar tokens."""

    if not groups:
        raise ValueError("canonical replay requires at least one group")
    if isinstance(pad_token_id, bool) or int(pad_token_id) < 0:
        raise ValueError("canonical replay pad_token_id must be non-negative")

    sequences: list[tuple[int, ...]] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    group_sizes: list[int] = []
    fresh_observation_counts: list[int] = []
    mass_weights: list[float] = []
    priority_modes = 0
    for group in groups:
        prompt = tuple(int(value) for value in group.prompt_token_ids)
        responses = tuple(
            tuple(int(value) for value in response)
            for response in group.response_token_ids
        )
        if not prompt or not responses or any(not row for row in responses):
            raise ValueError(
                "canonical replay groups require a prompt and at least one "
                "non-empty response"
            )
        if any(value < 0 for value in prompt) or any(
            value < 0 for row in responses for value in row
        ):
            raise ValueError("canonical replay token ids must be non-negative")
        group_sizes.append(len(responses))
        raw_fresh_counts = tuple(
            int(value) for value in getattr(group, "fresh_observation_counts", ())
        )
        if not raw_fresh_counts:
            # Compatibility for external/offline replay groups. Frequency
            # weighting rejects these sentinel zeros in the learner.
            raw_fresh_counts = tuple(0 for _ in responses)
        if len(raw_fresh_counts) != len(responses) or any(
            value < 0 for value in raw_fresh_counts
        ):
            raise ValueError(
                "canonical replay fresh observation counts must be "
                "non-negative and align with response rows"
            )
        raw_mass_weights = tuple(
            float(value) for value in getattr(group, "mass_weights", ())
        )
        if not raw_mass_weights:
            raw_mass_weights = tuple(1.0 for _ in responses)
        if len(raw_mass_weights) != len(responses):
            raise ValueError(
                "canonical replay mass weights must align with response rows"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in raw_mass_weights):
            raise ValueError(
                "canonical replay mass weights must be finite and positive"
            )
        if not math.isclose(
            sum(raw_mass_weights),
            float(len(raw_mass_weights)),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "canonical replay mass weights must preserve the group budget"
            )
        priority_modes += int(getattr(group, "priority_modes", 0))
        for response, fresh_count, mass_weight in zip(
            responses, raw_fresh_counts, raw_mass_weights
        ):
            sequences.append(prompt + response)
            prompt_lengths.append(len(prompt))
            response_lengths.append(len(response))
            fresh_observation_counts.append(fresh_count)
            mass_weights.append(mass_weight)

    max_length = max(len(row) for row in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    response_masks = torch.zeros(
        (len(sequences), max_length - 1),
        dtype=torch.bool,
        device=device,
    )
    for row_index, (sequence, prompt_length, response_length) in enumerate(
        zip(sequences, prompt_lengths, response_lengths)
    ):
        row_length = len(sequence)
        input_ids[row_index, :row_length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row_index, :row_length] = 1
        response_start = prompt_length - 1
        response_masks[
            row_index,
            response_start : response_start + response_length,
        ] = True

    return CanonicalReplayBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_masks=response_masks,
        group_sizes=tuple(group_sizes),
        fresh_observation_counts=torch.tensor(
            fresh_observation_counts,
            dtype=torch.int64,
            device=device,
        ),
        mass_weights=torch.tensor(
            mass_weights,
            dtype=torch.float32,
            device=device,
        ),
        priority_modes=priority_modes,
    )
