"""Scheduling for verified replay; retained state semantics are unchanged."""

from __future__ import annotations

from typing import Sequence

from .bank_types import (
    VerifiedCanonicalReplayGroup,
    _normalized_token_tuple,
    _prompt_key,
)


class ReplaySchedulingMixin:
    def replay_groups(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        *,
        min_modes: int = 2,
        consume_priority: bool = False,
    ) -> list[VerifiedCanonicalReplayGroup]:
        """Return deterministic replay banks for the requested prompts."""

        if not self.retain_exemplars:
            raise RuntimeError("canonical replay exemplars are not retained")
        if isinstance(min_modes, bool) or int(min_modes) not in {1, 2}:
            raise ValueError("canonical replay min_modes must be one or two")
        min_modes = int(min_modes)
        groups: list[VerifiedCanonicalReplayGroup] = []
        seen: set[str] = set()
        for raw_prompt_tokens in prompt_token_ids:
            normalized_prompt = _normalized_token_tuple(
                raw_prompt_tokens,
                label="canonical replay prompt token ids",
            )
            prompt_key = _prompt_key(normalized_prompt)
            if prompt_key in seen:
                continue
            seen.add(prompt_key)
            exemplars = self._exemplars.get(prompt_key, {})
            if len(exemplars) < min_modes:
                continue
            groups.append(
                self._replay_group_for_prompt(
                    prompt_key,
                    consume_priority=consume_priority,
                )
            )
        return groups

    def _replay_group_for_prompt(
        self,
        prompt_key: str,
        *,
        consume_priority: bool,
    ) -> VerifiedCanonicalReplayGroup:
        exemplars = self._exemplars[prompt_key]
        outcome_keys = tuple(sorted(exemplars))
        priority = self._proposal_priority_remaining.get(prompt_key, {})
        active_keys = tuple(key for key in outcome_keys if priority.get(key, 0) > 0)
        raw_weights = tuple(
            (self.proposal_replay_priority_multiplier if key in active_keys else 1.0)
            for key in outcome_keys
        )
        normalizer = sum(raw_weights)
        mass_weights = tuple(
            weight * len(raw_weights) / normalizer for weight in raw_weights
        )
        if consume_priority and active_keys:
            for key in active_keys:
                remaining = priority[key] - 1
                if remaining > 0:
                    priority[key] = remaining
                else:
                    del priority[key]
            if priority:
                self._proposal_priority_remaining[prompt_key] = priority
                if prompt_key not in self._proposal_priority_queue:
                    self._proposal_priority_queue.append(prompt_key)
            else:
                self._proposal_priority_remaining.pop(prompt_key, None)
                self._proposal_priority_queue = [
                    key for key in self._proposal_priority_queue if key != prompt_key
                ]
            self._proposal_priority_replay_groups += 1
            self._proposal_priority_replay_modes += len(active_keys)
        return VerifiedCanonicalReplayGroup(
            prompt_token_ids=self._prompt_token_ids[prompt_key],
            outcome_keys=outcome_keys,
            response_token_ids=tuple(exemplars[key] for key in outcome_keys),
            fresh_observation_counts=tuple(
                int(self._counts.get(prompt_key, {}).get(key, 0))
                for key in outcome_keys
            ),
            mass_weights=mass_weights,
            priority_modes=len(active_keys),
        )

    def scheduled_global_replay_groups(
        self,
        *,
        min_modes: int = 1,
    ) -> list[VerifiedCanonicalReplayGroup]:
        """Round-robin a fixed compute budget over all verified prompt banks.

        Selection depends only on the model's accumulated validator-positive
        exemplars. It never consults exhaustive support, evaluation, or a
        desired entropy/mode target. The cursor is checkpointed so resume does
        not silently resample the replay schedule.
        """

        if not self.retain_exemplars:
            raise RuntimeError("canonical replay exemplars are not retained")
        if isinstance(min_modes, bool) or int(min_modes) not in {1, 2}:
            raise ValueError("canonical replay min_modes must be one or two")
        min_modes = int(min_modes)
        if self.global_replay_groups_per_step == 0:
            return []
        if (
            self.global_replay_bootstrap_steps > 0
            and self._global_replay_updates >= self.global_replay_bootstrap_steps
        ):
            return []
        prompt_keys = sorted(
            prompt_key
            for prompt_key, exemplars in self._exemplars.items()
            if len(exemplars) >= min_modes
        )
        if not prompt_keys:
            return []
        count = min(self.global_replay_groups_per_step, len(prompt_keys))
        selected: list[str] = []
        eligible = set(prompt_keys)
        while self._proposal_priority_queue and len(selected) < count:
            prompt_key = self._proposal_priority_queue.pop(0)
            if (
                prompt_key in eligible
                and prompt_key in self._proposal_priority_remaining
                and prompt_key not in selected
            ):
                selected.append(prompt_key)
        if len(selected) < count:
            start = self._global_replay_cursor % len(prompt_keys)
            scanned = 0
            while len(selected) < count and scanned < len(prompt_keys):
                prompt_key = prompt_keys[(start + scanned) % len(prompt_keys)]
                scanned += 1
                if prompt_key not in selected:
                    selected.append(prompt_key)
            self._global_replay_cursor = (start + scanned) % len(prompt_keys)
        if self.global_replay_bootstrap_steps > 0:
            self._global_replay_updates += 1
        return [
            self._replay_group_for_prompt(
                prompt_key,
                consume_priority=True,
            )
            for prompt_key in selected
        ]
