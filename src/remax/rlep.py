"""Offline verified-trajectory replay for the RLEP-Dr baseline.

The pool deliberately preserves response frequency: unlike this project's
canonical replay treatments, RLEP samples verified trajectories rather than
one exemplar per semantic or executable mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

import torch


def reference_key(reference: Any) -> str:
    """Return a stable key for the training/evaluation reference payload."""

    if isinstance(reference, str):
        return reference
    return json.dumps(reference, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RLEPReplayGroup:
    """One prompt plus frequency-preserving verified response samples."""

    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RLEPPoolDiagnostics:
    prompts: int
    trajectories: int
    minimum_trajectories_per_prompt: int
    maximum_trajectories_per_prompt: int
    eligible_prompts: int
    ineligible_prompts: int


@dataclass(frozen=True)
class RLEPMixedAdvantages:
    fresh: torch.Tensor
    replay_advantage: float
    mixed_reward_mean: float
    fresh_count: int
    replay_count: int


class RLEPExperiencePool:
    """Verified responses collected by four fixed 16-sample draws."""

    def __init__(
        self,
        responses_by_reference: dict[str, Sequence[str]],
        *,
        allow_sparse: bool = False,
    ) -> None:
        normalized = {
            str(key): tuple(str(response) for response in responses)
            for key, responses in responses_by_reference.items()
        }
        if not normalized:
            raise ValueError("RLEP experience pool is empty")
        short = {key: len(rows) for key, rows in normalized.items() if len(rows) < 2}
        if short and not allow_sparse:
            raise ValueError(
                "RLEP requires at least two verified trajectories per prompt; "
                f"found {len(short)} ineligible prompts"
            )
        if allow_sparse and len(short) == len(normalized):
            raise ValueError("sparse RLEP pool has no replay-eligible prompt")
        self._responses = normalized
        self._allow_sparse = bool(allow_sparse)

    @classmethod
    def from_directory(
        cls, root: str | Path, *, allow_sparse: bool = False
    ) -> "RLEPExperiencePool":
        root = Path(root)
        paths = sorted(root.glob("**/eval_mode_coverage_draws.jsonl"))
        if len(paths) != 1:
            raise ValueError(
                "RLEP collection root must contain exactly one mode-coverage "
                f"sidecar, found {len(paths)} under {root}"
            )

        sampled: dict[int, dict[str, Any]] = {}
        for line in paths[0].read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            draw = record.get("draw_index")
            if draw is None:
                continue
            if (
                record.get("evaluation_kind") != "fixed_seed_sampled_k_neutral"
                or int(record.get("sample_count", -1)) != 16
                or not math.isclose(float(record.get("temperature", -1)), 0.7)
                or not math.isclose(float(record.get("top_p", -1)), 0.95)
            ):
                raise ValueError("RLEP collection sidecar violates 16/T=.7/p=.95")
            sampled[int(draw)] = record
        if set(sampled) != set(range(4)):
            raise ValueError("RLEP collection requires exactly four sampled draws")

        by_reference: dict[str, list[str]] = {}
        prompt_sets: list[set[str]] = []
        for draw in range(4):
            draw_references: set[str] = set()
            prompts = sampled[draw].get("prompts")
            if not isinstance(prompts, list):
                raise ValueError("RLEP collection record has no prompt rows")
            for row in prompts:
                key = reference_key(row.get("reference"))
                if key in draw_references:
                    raise ValueError("RLEP collection repeats a prompt within a draw")
                draw_references.add(key)
                responses = row.get("responses")
                rewards = row.get("rewards")
                if (
                    not isinstance(responses, list)
                    or not isinstance(rewards, list)
                    or len(responses) != 16
                    or len(rewards) != 16
                ):
                    raise ValueError("RLEP collection prompt is not a 16-row grid")
                by_reference.setdefault(key, []).extend(
                    str(response)
                    for response, reward in zip(responses, rewards)
                    if float(reward) > 0.0
                )
            prompt_sets.append(draw_references)
        if any(values != prompt_sets[0] for values in prompt_sets[1:]):
            raise ValueError("RLEP collection draws cover different prompt sets")
        return cls(by_reference, allow_sparse=allow_sparse)

    @property
    def diagnostics(self) -> RLEPPoolDiagnostics:
        sizes = [len(rows) for rows in self._responses.values()]
        return RLEPPoolDiagnostics(
            prompts=len(sizes),
            trajectories=sum(sizes),
            minimum_trajectories_per_prompt=min(sizes),
            maximum_trajectories_per_prompt=max(sizes),
            eligible_prompts=sum(size >= 2 for size in sizes),
            ineligible_prompts=sum(size < 2 for size in sizes),
        )

    def can_sample(self, reference: Any, *, count: int) -> bool:
        """Whether this prompt has the registered prompt-matched replay dose."""

        return len(self._responses.get(reference_key(reference), ())) >= int(count)

    def sample(
        self,
        reference: Any,
        *,
        count: int,
        experiment_seed: int,
        learner_step: int,
    ) -> tuple[str, ...]:
        return _sample_without_replacement(
            self._responses.get(reference_key(reference)),
            key=reference_key(reference),
            count=count,
            experiment_seed=experiment_seed,
            learner_step=learner_step,
        )


def _sample_without_replacement(
    available: Sequence[str] | None,
    *,
    key: str,
    count: int,
    experiment_seed: int,
    learner_step: int,
) -> tuple[str, ...]:
    """Deterministic frequency-preserving draw shared by both pool kinds."""

    if count <= 0:
        raise ValueError("RLEP replay count must be positive")
    if available is None:
        raise KeyError("RLEP pool has no eligible trajectories for this prompt")
    if len(available) < count:
        raise ValueError("RLEP pool has fewer verified trajectories than requested")
    identity = json.dumps(
        [int(experiment_seed), int(learner_step), key],
        separators=(",", ":"),
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
    return tuple(random.Random(seed).sample(list(available), count))


class OnlineRLEPExperiencePool:
    """RLEP's pool, filled from the learner's own verified rollouts as it trains.

    The offline pool above is harvested once, before training, from a policy
    that RL has already trained, so it inherits that policy's collapse. This
    pool keeps RLEP's update, its eligibility rule (at least ``minimum``
    verified trajectories on the prompt) and its frequency-preserving sampling,
    and changes only where the trajectories come from: every validator-positive
    fresh response is appended to its prompt's list as soon as the group that
    produced it has been scored. A prompt therefore becomes replay-eligible on
    the pass after it was first solved at least ``minimum`` times.

    Nothing is deduplicated or balanced by canonical key. Eight copies of one
    mode and eight distinct modes are stored, and drawn, alike.
    """

    schema = "online_rlep_experience_pool_v1"

    def __init__(self, *, minimum: int = 2) -> None:
        if int(minimum) < 1:
            raise ValueError("RLEP online pool needs a positive eligibility minimum")
        self._minimum = int(minimum)
        self._responses: dict[str, list[str]] = {}
        self._observed_groups = 0
        self._observed_rows = 0

    @property
    def minimum(self) -> int:
        return self._minimum

    def observe(
        self,
        reference: Any,
        responses: Sequence[str],
        rewards: Sequence[float],
    ) -> int:
        """Append this group's verified responses; return how many were added."""

        if len(responses) != len(rewards):
            raise ValueError("RLEP online pool needs one reward per response")
        key = reference_key(reference)
        added = [
            str(response)
            for response, reward in zip(responses, rewards)
            if float(reward) > 0.0
        ]
        if added:
            self._responses.setdefault(key, []).extend(added)
        self._observed_groups += 1
        self._observed_rows += len(responses)
        return len(added)

    @property
    def diagnostics(self) -> RLEPPoolDiagnostics:
        sizes = [len(rows) for rows in self._responses.values()]
        return RLEPPoolDiagnostics(
            prompts=len(sizes),
            trajectories=sum(sizes),
            minimum_trajectories_per_prompt=min(sizes) if sizes else 0,
            maximum_trajectories_per_prompt=max(sizes) if sizes else 0,
            eligible_prompts=sum(size >= self._minimum for size in sizes),
            ineligible_prompts=sum(size < self._minimum for size in sizes),
        )

    @property
    def observed_groups(self) -> int:
        return self._observed_groups

    def can_sample(self, reference: Any, *, count: int) -> bool:
        stored = len(self._responses.get(reference_key(reference), ()))
        return stored >= max(int(count), self._minimum)

    def sample(
        self,
        reference: Any,
        *,
        count: int,
        experiment_seed: int,
        learner_step: int,
    ) -> tuple[str, ...]:
        key = reference_key(reference)
        if not self.can_sample(reference, count=count):
            raise ValueError("RLEP online pool is not yet eligible for this prompt")
        return _sample_without_replacement(
            self._responses.get(key),
            key=key,
            count=count,
            experiment_seed=experiment_seed,
            learner_step=learner_step,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "minimum": self._minimum,
            "observed_groups": self._observed_groups,
            "observed_rows": self._observed_rows,
            "responses": {key: list(rows) for key, rows in self._responses.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("schema") != self.schema:
            raise ValueError("invalid RLEP online pool state")
        if int(state["minimum"]) != self._minimum:
            raise ValueError("RLEP online pool eligibility minimum changed on resume")
        responses = state["responses"]
        if not isinstance(responses, dict):
            raise ValueError("RLEP online pool state has no response table")
        self._responses = {
            str(key): [str(value) for value in rows] for key, rows in responses.items()
        }
        self._observed_groups = int(state["observed_groups"])
        self._observed_rows = int(state["observed_rows"])


def rlep_mixed_advantages(
    fresh_rewards: torch.Tensor,
    *,
    replay_count: int,
) -> RLEPMixedAdvantages:
    """Build the common 16+M Dr.GRPO baseline and exact batch weights.

    The ordinary learner averages the fresh rows over G. Multiplying their
    advantages by G/(G+M), then adding each replay row with coefficient
    A_replay/(G+M), is algebraically the same mixed-batch policy gradient.
    """

    flat = fresh_rewards.detach().reshape(-1)
    if flat.numel() <= 1 or replay_count <= 0:
        raise ValueError("RLEP requires at least two fresh and one replay row")
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("RLEP fresh rewards must be finite")
    fresh_count = int(flat.numel())
    total = fresh_count + int(replay_count)
    mean = (float(flat.double().sum().item()) + replay_count) / float(total)
    scale = float(fresh_count) / float(total)
    fresh = (flat - mean) * scale
    return RLEPMixedAdvantages(
        fresh=fresh.reshape(fresh_rewards.shape),
        replay_advantage=1.0 - mean,
        mixed_reward_mean=mean,
        fresh_count=fresh_count,
        replay_count=int(replay_count),
    )
