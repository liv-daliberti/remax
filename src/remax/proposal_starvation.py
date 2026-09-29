"""Restart-safe, target-free scheduling for verified proposal search.

The controller observes only whether an *eligible* explorer update admitted a
genuinely new validator-positive bank outcome.  It never observes evaluation,
gold support, a desired mode count, task accuracy, or an entropy target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProposalSearchPlan:
    """The proposal budget selected for one eligible explorer update."""

    max_attempts: int
    fallback_active: bool
    fallback_activation: bool


class ProposalStarvationController:
    """Apply bounded proposal-budget bursts after verified admissions stall."""

    _SCHEMA = "proposal_starvation_controller_v1"

    def __init__(
        self,
        *,
        base_max_attempts: int,
        patience_updates: int,
        fallback_max_attempts: int,
        burst_updates: int,
        cooldown_updates: int,
    ) -> None:
        values = {
            "base_max_attempts": base_max_attempts,
            "patience_updates": patience_updates,
            "fallback_max_attempts": fallback_max_attempts,
            "burst_updates": burst_updates,
            "cooldown_updates": cooldown_updates,
        }
        for name, value in values.items():
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
        if int(base_max_attempts) <= 0:
            raise ValueError("base_max_attempts must be positive")
        if int(patience_updates) <= 0:
            raise ValueError("patience_updates must be positive")
        if int(fallback_max_attempts) <= int(base_max_attempts):
            raise ValueError(
                "fallback_max_attempts must exceed base_max_attempts"
            )
        if int(burst_updates) <= 0:
            raise ValueError("burst_updates must be positive")
        if int(cooldown_updates) < 0:
            raise ValueError("cooldown_updates must be non-negative")

        self.base_max_attempts = int(base_max_attempts)
        self.patience_updates = int(patience_updates)
        self.fallback_max_attempts = int(fallback_max_attempts)
        self.burst_updates = int(burst_updates)
        self.cooldown_updates = int(cooldown_updates)

        self.stalled_eligible_updates = 0
        self.burst_remaining = 0
        self.cooldown_remaining = 0
        self.eligible_updates = 0
        self.fallback_updates = 0
        self.fallback_activations = 0
        self.admitted_new_outcomes = 0

    def plan(self) -> ProposalSearchPlan:
        """Select a budget without mutating state.

        Keeping planning pure lets rank zero generate proposals before the
        admission payload is broadcast, while every learner rank applies the
        same state transition after the bank admission is known.
        """

        activation = (
            self.burst_remaining == 0
            and self.cooldown_remaining == 0
            and self.stalled_eligible_updates >= self.patience_updates
        )
        active = self.burst_remaining > 0 or activation
        return ProposalSearchPlan(
            max_attempts=(
                self.fallback_max_attempts if active else self.base_max_attempts
            ),
            fallback_active=active,
            fallback_activation=activation,
        )

    def observe(
        self,
        plan: ProposalSearchPlan,
        *,
        admitted_new_outcomes: int,
    ) -> None:
        """Commit one eligible update after verified bank admission."""

        if plan != self.plan():
            raise ValueError("proposal starvation plan is stale or inconsistent")
        if (
            isinstance(admitted_new_outcomes, bool)
            or int(admitted_new_outcomes) != admitted_new_outcomes
            or int(admitted_new_outcomes) < 0
        ):
            raise ValueError("admitted_new_outcomes must be a non-negative integer")

        admitted = int(admitted_new_outcomes)
        self.eligible_updates += 1
        if plan.fallback_activation:
            self.fallback_activations += 1
            self.burst_remaining = self.burst_updates
        if plan.fallback_active:
            self.fallback_updates += 1

        if admitted > 0:
            self.admitted_new_outcomes += admitted
            self.stalled_eligible_updates = 0
            self.burst_remaining = 0
            self.cooldown_remaining = 0
            return

        self.stalled_eligible_updates += 1
        if plan.fallback_active:
            if self.burst_remaining <= 0:
                raise RuntimeError("fallback burst has no remaining update")
            self.burst_remaining -= 1
            if self.burst_remaining == 0:
                self.cooldown_remaining = self.cooldown_updates
        elif self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def observe_admission_without_opportunity(
        self,
        *,
        admitted_new_outcomes: int,
    ) -> None:
        """Reset starvation after a verified admission from a non-sampling path."""

        if (
            isinstance(admitted_new_outcomes, bool)
            or int(admitted_new_outcomes) != admitted_new_outcomes
            or int(admitted_new_outcomes) <= 0
        ):
            raise ValueError("admitted_new_outcomes must be a positive integer")
        self.admitted_new_outcomes += int(admitted_new_outcomes)
        self.stalled_eligible_updates = 0
        self.burst_remaining = 0
        self.cooldown_remaining = 0

    def diagnostics(self) -> dict[str, int]:
        return {
            "stalled_eligible_updates": self.stalled_eligible_updates,
            "burst_remaining": self.burst_remaining,
            "cooldown_remaining": self.cooldown_remaining,
            "eligible_updates": self.eligible_updates,
            "fallback_updates": self.fallback_updates,
            "fallback_activations": self.fallback_activations,
            "admitted_new_outcomes": self.admitted_new_outcomes,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self._SCHEMA,
            "config": {
                "base_max_attempts": self.base_max_attempts,
                "patience_updates": self.patience_updates,
                "fallback_max_attempts": self.fallback_max_attempts,
                "burst_updates": self.burst_updates,
                "cooldown_updates": self.cooldown_updates,
            },
            "state": self.diagnostics(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict) or payload.get("schema") != self._SCHEMA:
            raise ValueError("invalid proposal starvation controller state")
        expected_config = self.state_dict()["config"]
        if payload.get("config") != expected_config:
            raise ValueError("proposal starvation controller resume config mismatch")
        raw_state = payload.get("state")
        if not isinstance(raw_state, dict) or set(raw_state) != set(
            self.diagnostics()
        ):
            raise ValueError("proposal starvation controller state is incomplete")
        restored: dict[str, int] = {}
        for name, value in raw_state.items():
            if (
                isinstance(value, bool)
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError(
                    f"proposal starvation controller has invalid {name}"
                )
            restored[name] = int(value)
        if restored["burst_remaining"] > self.burst_updates:
            raise ValueError("proposal starvation burst state exceeds its bound")
        if restored["cooldown_remaining"] > self.cooldown_updates:
            raise ValueError("proposal starvation cooldown state exceeds its bound")
        if restored["fallback_updates"] > restored["eligible_updates"]:
            raise ValueError("proposal starvation update counters are inconsistent")
        if restored["fallback_activations"] > restored["fallback_updates"]:
            raise ValueError("proposal starvation activation counters are inconsistent")

        for name, value in restored.items():
            setattr(self, name, value)
