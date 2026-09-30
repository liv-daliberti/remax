"""Bank state for verified replay; retained state semantics are unchanged."""

from __future__ import annotations
import math
from typing import Any

from .bank_types import _normalized_token_tuple


class BankStateMixin:
    def state_dict(self) -> dict[str, Any]:
        state = {
            "schema": "online_growing_support_canonical_maxent_v2",
            "entropy_alpha": self.entropy_alpha,
            "pseudocount": self.pseudocount,
            "surprisal_clip": self.surprisal_clip,
            "groups_scored": self._groups_scored,
            "rows_scored": self._rows_scored,
            "counts": {
                prompt_key: dict(outcome_counts)
                for prompt_key, outcome_counts in self._counts.items()
            },
        }
        if self.retain_exemplars:
            state.update(
                {
                    "schema": (
                        "online_growing_support_canonical_maxent_"
                        "replay_separated_proposal_priority_retention_v8"
                        if self.proposal_retention_tracking
                        else (
                            "online_growing_support_canonical_maxent_"
                            "replay_separated_proposal_priority_fixed_v7"
                            if self.proposal_replay_priority_visits > 0
                            else (
                                "online_growing_support_canonical_maxent_"
                                "replay_separated_proposal_fixed_v6"
                                if self.separate_proposal_objective_support
                                else "online_growing_support_canonical_maxent_replay_fixed_v5"
                            )
                        )
                    ),
                    "retain_exemplars": True,
                    "separate_proposal_objective_support": (
                        self.separate_proposal_objective_support
                    ),
                    "replay_capacity": self.replay_capacity,
                    "global_replay_groups_per_step": (
                        self.global_replay_groups_per_step
                    ),
                    "global_replay_bootstrap_steps": (
                        self.global_replay_bootstrap_steps
                    ),
                    "global_replay_cursor": self._global_replay_cursor,
                    "global_replay_updates": self._global_replay_updates,
                    "proposal_groups": self._proposal_groups,
                    "proposal_rows": self._proposal_rows,
                    "proposal_new_outcomes": (self._proposal_new_outcomes),
                    "prompt_token_ids": {
                        prompt_key: list(token_ids)
                        for prompt_key, token_ids in self._prompt_token_ids.items()
                    },
                    "exemplars": {
                        prompt_key: {
                            outcome_key: list(token_ids)
                            for outcome_key, token_ids in outcome_exemplars.items()
                        }
                        for prompt_key, outcome_exemplars in self._exemplars.items()
                    },
                    "proposal_only_outcomes": {
                        prompt_key: sorted(outcomes)
                        for prompt_key, outcomes in self._proposal_only_outcomes.items()
                        if outcomes
                    },
                }
            )
            if self.proposal_replay_priority_visits > 0:
                state.update(
                    {
                        "proposal_replay_priority_visits": (
                            self.proposal_replay_priority_visits
                        ),
                        "proposal_replay_priority_multiplier": (
                            self.proposal_replay_priority_multiplier
                        ),
                        "proposal_priority_remaining": {
                            prompt_key: dict(outcomes)
                            for prompt_key, outcomes in self._proposal_priority_remaining.items()
                            if outcomes
                        },
                        "proposal_priority_queue": list(self._proposal_priority_queue),
                        "proposal_priority_replay_groups": (
                            self._proposal_priority_replay_groups
                        ),
                        "proposal_priority_replay_modes": (
                            self._proposal_priority_replay_modes
                        ),
                    }
                )
            if self._proposal_retention_tracker is not None:
                state["proposal_admission_retention"] = (
                    self._proposal_retention_tracker.state_dict()
                )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("invalid online canonical bank state")
        expected_schema = (
            (
                "online_growing_support_canonical_maxent_"
                "replay_separated_proposal_priority_retention_v8"
                if self.proposal_retention_tracking
                else (
                    "online_growing_support_canonical_maxent_"
                    "replay_separated_proposal_priority_fixed_v7"
                    if self.proposal_replay_priority_visits > 0
                    else (
                        "online_growing_support_canonical_maxent_"
                        "replay_separated_proposal_fixed_v6"
                        if self.separate_proposal_objective_support
                        else "online_growing_support_canonical_maxent_replay_fixed_v5"
                    )
                )
            )
            if self.retain_exemplars
            else "online_growing_support_canonical_maxent_v2"
        )
        if state.get("schema") != expected_schema:
            raise ValueError(
                "online canonical bank state replay configuration mismatch"
            )
        for name in (
            "entropy_alpha",
            "pseudocount",
            "surprisal_clip",
        ):
            try:
                saved = float(state[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"online canonical bank state is missing valid {name}"
                ) from exc
            configured = float(getattr(self, name))
            if not math.isfinite(saved) or not math.isclose(
                saved, configured, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"online canonical bank resume mismatch for {name}: "
                    f"saved={saved!r} configured={configured!r}"
                )
        raw_counts = state.get("counts")
        if not isinstance(raw_counts, dict):
            raise ValueError("online canonical bank state has invalid counts")
        restored: dict[str, dict[str, int]] = {}
        for prompt_key, outcome_counts in raw_counts.items():
            if not isinstance(prompt_key, str) or not isinstance(outcome_counts, dict):
                raise ValueError("online canonical bank state has invalid bank")
            restored[prompt_key] = {}
            for outcome_key, count in outcome_counts.items():
                if (
                    not isinstance(outcome_key, str)
                    or isinstance(count, bool)
                    or int(count) != count
                    or int(count) <= 0
                ):
                    raise ValueError(
                        "online canonical bank state has invalid outcome count"
                    )
                restored[prompt_key][outcome_key] = int(count)
        self._counts = restored
        if self.retain_exemplars:
            if state.get("retain_exemplars") is not True:
                raise ValueError(
                    "online canonical replay state lacks its configuration"
                )
            if int(state.get("replay_capacity", -1)) != self.replay_capacity:
                raise ValueError(
                    "online canonical bank resume mismatch for replay_capacity"
                )
            if (
                int(state.get("global_replay_groups_per_step", 0))
                != self.global_replay_groups_per_step
            ):
                raise ValueError(
                    "online canonical bank resume mismatch for "
                    "global_replay_groups_per_step"
                )
            if (
                int(state.get("global_replay_bootstrap_steps", 0))
                != self.global_replay_bootstrap_steps
            ):
                raise ValueError(
                    "online canonical bank resume mismatch for "
                    "global_replay_bootstrap_steps"
                )
            raw_prompts = state.get("prompt_token_ids")
            raw_exemplars = state.get("exemplars")
            if not isinstance(raw_prompts, dict) or not isinstance(
                raw_exemplars,
                dict,
            ):
                raise ValueError("online canonical replay state lacks exemplar maps")
            restored_prompts: dict[str, tuple[int, ...]] = {}
            restored_exemplars: dict[
                str,
                dict[str, tuple[int, ...]],
            ] = {}
            if self.separate_proposal_objective_support:
                if state.get("separate_proposal_objective_support") is not True:
                    raise ValueError(
                        "online canonical replay state lacks separated "
                        "proposal support"
                    )
                raw_proposal_only = state.get("proposal_only_outcomes")
                if not isinstance(raw_proposal_only, dict):
                    raise ValueError(
                        "online canonical replay state lacks proposal-only " "support"
                    )
                restored_proposal_only: dict[str, set[str]] = {}
                for prompt_key, outcomes in raw_proposal_only.items():
                    if (
                        not isinstance(prompt_key, str)
                        or not isinstance(outcomes, list)
                        or any(
                            not isinstance(outcome, str) or not outcome
                            for outcome in outcomes
                        )
                        or len(set(outcomes)) != len(outcomes)
                    ):
                        raise ValueError(
                            "online canonical replay state has invalid "
                            "proposal-only support"
                        )
                    restored_proposal_only[prompt_key] = set(outcomes)
                prompt_keys = set(raw_exemplars)
                if (
                    set(raw_prompts) != prompt_keys
                    or not set(restored).issubset(prompt_keys)
                    or not set(restored_proposal_only).issubset(prompt_keys)
                ):
                    raise ValueError(
                        "online canonical replay state contains unknown prompts"
                    )
            else:
                restored_proposal_only = {}
                prompt_keys = set(restored)
            for prompt_key in prompt_keys:
                counts = restored.get(prompt_key, {})
                if prompt_key not in raw_prompts:
                    raise ValueError(
                        "online canonical replay state is missing prompt tokens"
                    )
                restored_prompts[prompt_key] = _normalized_token_tuple(
                    raw_prompts[prompt_key],
                    label="stored canonical replay prompt token ids",
                )
                prompt_exemplars = raw_exemplars.get(prompt_key)
                invalid_exemplars = (
                    not isinstance(prompt_exemplars, dict)
                    or len(prompt_exemplars) > self.replay_capacity
                )
                if not invalid_exemplars:
                    exemplar_keys = set(prompt_exemplars)
                    if self.separate_proposal_objective_support:
                        proposal_keys = restored_proposal_only.get(
                            prompt_key,
                            set(),
                        )
                        invalid_exemplars = (
                            not proposal_keys.issubset(exemplar_keys)
                            or bool(proposal_keys & set(counts))
                            or not exemplar_keys.issubset(set(counts) | proposal_keys)
                        )
                    else:
                        invalid_exemplars = not exemplar_keys.issubset(counts) or (
                            len(counts) <= self.replay_capacity
                            and exemplar_keys != set(counts)
                        )
                if invalid_exemplars:
                    raise ValueError(
                        "online canonical replay exemplars mismatch bank support"
                    )
                restored_exemplars[prompt_key] = {
                    outcome_key: _normalized_token_tuple(
                        prompt_exemplars[outcome_key],
                        label="stored canonical replay response token ids",
                    )
                    for outcome_key in prompt_exemplars
                }
            if not self.separate_proposal_objective_support and (
                set(raw_prompts) != set(restored) or set(raw_exemplars) != set(restored)
            ):
                raise ValueError(
                    "online canonical replay state contains unknown prompts"
                )
            self._prompt_token_ids = restored_prompts
            self._exemplars = restored_exemplars
            self._proposal_only_outcomes = restored_proposal_only
            if self.proposal_replay_priority_visits > 0:
                if int(
                    state.get("proposal_replay_priority_visits", -1)
                ) != self.proposal_replay_priority_visits or not math.isclose(
                    float(
                        state.get(
                            "proposal_replay_priority_multiplier",
                            float("nan"),
                        )
                    ),
                    self.proposal_replay_priority_multiplier,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "online canonical bank resume mismatch for proposal priority"
                    )
                raw_priority = state.get("proposal_priority_remaining")
                raw_queue = state.get("proposal_priority_queue")
                if not isinstance(raw_priority, dict) or not isinstance(
                    raw_queue,
                    list,
                ):
                    raise ValueError(
                        "online canonical replay state lacks proposal priority"
                    )
                restored_priority: dict[str, dict[str, int]] = {}
                for prompt_key, outcomes in raw_priority.items():
                    if prompt_key not in restored_exemplars or not isinstance(
                        outcomes, dict
                    ):
                        raise ValueError(
                            "online canonical replay state has invalid priority prompt"
                        )
                    restored_priority[prompt_key] = {}
                    for outcome_key, visits in outcomes.items():
                        if (
                            outcome_key not in restored_exemplars[prompt_key]
                            or isinstance(visits, bool)
                            or int(visits) != visits
                            or not 1
                            <= int(visits)
                            <= self.proposal_replay_priority_visits
                        ):
                            raise ValueError(
                                "online canonical replay state has invalid priority outcome"
                            )
                        restored_priority[prompt_key][outcome_key] = int(visits)
                if (
                    any(not isinstance(key, str) for key in raw_queue)
                    or len(set(raw_queue)) != len(raw_queue)
                    or set(raw_queue) != set(restored_priority)
                ):
                    raise ValueError(
                        "online canonical replay priority queue is invalid"
                    )
                self._proposal_priority_remaining = restored_priority
                self._proposal_priority_queue = list(raw_queue)
            else:
                self._proposal_priority_remaining = {}
                self._proposal_priority_queue = []
            if self._proposal_retention_tracker is not None:
                restored_tracker = self._proposal_retention_tracker.clone()
                restored_tracker.load_state_dict(
                    state.get("proposal_admission_retention")
                )
                converted_pairs = set(restored_tracker.converted_pairs)
                for prompt_key, outcome_key in restored_tracker.tracked_pairs:
                    pair = (prompt_key, outcome_key)
                    if outcome_key not in restored_exemplars.get(prompt_key, {}):
                        raise ValueError(
                            "proposal retention state refers to an unknown exemplar"
                        )
                    proposal_only = outcome_key in restored_proposal_only.get(
                        prompt_key,
                        set(),
                    )
                    observed_on_policy = outcome_key in restored.get(prompt_key, {})
                    converted = pair in converted_pairs
                    if (converted and (proposal_only or not observed_on_policy)) or (
                        not converted and (not proposal_only or observed_on_policy)
                    ):
                        raise ValueError(
                            "proposal retention state has inconsistent "
                            "admission lifecycle"
                        )
                self._proposal_retention_tracker = restored_tracker
            raw_cursor = state.get("global_replay_cursor", 0)
            if (
                isinstance(raw_cursor, bool)
                or int(raw_cursor) != raw_cursor
                or int(raw_cursor) < 0
            ):
                raise ValueError(
                    "online canonical replay state has invalid global cursor"
                )
            self._global_replay_cursor = int(raw_cursor)
            raw_updates = state.get("global_replay_updates", 0)
            if (
                isinstance(raw_updates, bool)
                or int(raw_updates) != raw_updates
                or int(raw_updates) < 0
                or (
                    self.global_replay_bootstrap_steps > 0
                    and int(raw_updates) > self.global_replay_bootstrap_steps
                )
            ):
                raise ValueError(
                    "online canonical replay state has invalid global " "update count"
                )
            self._global_replay_updates = int(raw_updates)
            for field_name, attr_name in (
                ("proposal_groups", "_proposal_groups"),
                ("proposal_rows", "_proposal_rows"),
                ("proposal_new_outcomes", "_proposal_new_outcomes"),
                (
                    "proposal_priority_replay_groups",
                    "_proposal_priority_replay_groups",
                ),
                (
                    "proposal_priority_replay_modes",
                    "_proposal_priority_replay_modes",
                ),
            ):
                raw_value = state.get(field_name, 0)
                if (
                    isinstance(raw_value, bool)
                    or int(raw_value) != raw_value
                    or int(raw_value) < 0
                ):
                    raise ValueError(
                        "online canonical replay state has invalid " f"{field_name}"
                    )
                setattr(self, attr_name, int(raw_value))
        else:
            self._prompt_token_ids = {}
            self._exemplars = {}
            self._global_replay_cursor = 0
            self._global_replay_updates = 0
            self._proposal_groups = 0
            self._proposal_rows = 0
            self._proposal_new_outcomes = 0
            self._proposal_only_outcomes = {}
            self._proposal_priority_remaining = {}
            self._proposal_priority_queue = []
            self._proposal_priority_replay_groups = 0
            self._proposal_priority_replay_modes = 0
        self._groups_scored = int(state.get("groups_scored", 0))
        self._rows_scored = int(state.get("rows_scored", 0))
