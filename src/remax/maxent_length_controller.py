"""Projected dual control for a completion expected-length constraint."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class MaxEntLengthController:
    r"""Control the non-negative multiplier on an expected-length constraint.

    The controller smooths raw generated-response lengths with an EMA that is
    initialized at the configured target.  It then takes a projected dual step

    ``lambda <- clip(lambda + dual_lr * (ema / target - 1), 0, max_lambda)``.

    Consequently, a response-length excess increases the multiplier used by
    the next actor update, while a deficit decreases it.
    """

    target_length: float
    init_lambda: float
    max_lambda: float
    dual_lr: float
    ema_decay: float
    current_lambda: float | None = None
    length_ema: float | None = None
    observation_count: int = 0

    observation_metric_key = "maxent_expected_length"

    _CONTROLLER_KIND = "maxent_length_dual"
    _CONTROLLER_RULE = "projected_relative_ema_v1"
    _LENGTH_UNITS = "generated_response_tokens_v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("target_length", self.target_length),
            ("init_lambda", self.init_lambda),
            ("max_lambda", self.max_lambda),
            ("dual_lr", self.dual_lr),
            ("ema_decay", self.ema_decay),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.target_length <= 0:
            raise ValueError("target_length must be positive")
        if self.init_lambda < 0:
            raise ValueError("init_lambda must be non-negative")
        if self.max_lambda <= 0 or self.max_lambda < self.init_lambda:
            raise ValueError(
                "max_lambda must be positive and at least init_lambda"
            )
        if self.dual_lr <= 0:
            raise ValueError("dual_lr must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")

        if self.current_lambda is None:
            self.current_lambda = float(self.init_lambda)
        self.current_lambda = self._validated_lambda(
            self.current_lambda, name="current_lambda"
        )
        if self.length_ema is None:
            self.length_ema = float(self.target_length)
        self.length_ema = self._validated_length(
            self.length_ema, name="length_ema"
        )
        self.observation_count = self._validated_observation_count(
            self.observation_count
        )

    def _validated_lambda(self, value: Any, *, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or not 0 <= result <= self.max_lambda:
            raise ValueError(
                f"{name} must be finite and in [0, max_lambda]"
            )
        return result

    @staticmethod
    def _validated_length(value: Any, *, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    @staticmethod
    def _validated_observation_count(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("observation_count must be a non-negative integer")
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "observation_count must be a non-negative integer"
            ) from exc
        if result < 0 or result != value:
            raise ValueError("observation_count must be a non-negative integer")
        return result

    def observe(self, observed_length: float) -> dict[str, float]:
        """Advance the controller and return diagnostics for this observation."""

        value = self._validated_length(
            observed_length, name="observed_length"
        )
        self.observation_count += 1
        self.length_ema = (
            self.ema_decay * self.length_ema
            + (1.0 - self.ema_decay) * value
        )
        relative_violation = self.length_ema / self.target_length - 1.0
        self.current_lambda = min(
            self.max_lambda,
            max(
                0.0,
                self.current_lambda + self.dual_lr * relative_violation,
            ),
        )
        return {
            "maxent_length_target": float(self.target_length),
            "maxent_length_lambda_next": float(self.current_lambda),
            "maxent_length_lambda_max": float(self.max_lambda),
            "maxent_length_ema": float(self.length_ema),
            "maxent_length_ema_decay": float(self.ema_decay),
            "maxent_length_relative_violation": float(relative_violation),
            "maxent_length_dual_lr": float(self.dual_lr),
            "maxent_length_observed_length": value,
            "maxent_length_observations": float(self.observation_count),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint state with explicit rule and unit tags."""

        return {
            "controller_kind": self._CONTROLLER_KIND,
            "controller_rule": self._CONTROLLER_RULE,
            "length_units": self._LENGTH_UNITS,
            "target_length": float(self.target_length),
            "init_lambda": float(self.init_lambda),
            "max_lambda": float(self.max_lambda),
            "dual_lr": float(self.dual_lr),
            "ema_decay": float(self.ema_decay),
            "current_lambda": float(self.current_lambda),
            "length_ema": float(self.length_ema),
            "observation_count": int(self.observation_count),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore compatible state, failing closed on rule or config drift."""

        if not isinstance(state, dict):
            raise ValueError("controller state must be a dictionary")
        if state.get("controller_kind") != self._CONTROLLER_KIND:
            raise ValueError("checkpoint contains a different length controller")
        if state.get("controller_rule") != self._CONTROLLER_RULE:
            raise ValueError("checkpoint uses an incompatible length-control rule")
        if state.get("length_units") != self._LENGTH_UNITS:
            raise ValueError("checkpoint uses incompatible response-length units")

        for name in (
            "target_length",
            "init_lambda",
            "max_lambda",
            "dual_lr",
            "ema_decay",
        ):
            try:
                saved = float(state[name])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"checkpoint has invalid or missing {name}"
                ) from exc
            configured = float(getattr(self, name))
            if not math.isfinite(saved) or not math.isclose(
                saved, configured, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(f"checkpoint uses a different {name}")

        try:
            current_lambda = state["current_lambda"]
            length_ema = state["length_ema"]
            observation_count = state["observation_count"]
        except KeyError as exc:
            raise ValueError(
                f"checkpoint is missing {exc.args[0]}"
            ) from exc
        self.current_lambda = self._validated_lambda(
            current_lambda, name="checkpoint current_lambda"
        )
        self.length_ema = self._validated_length(
            length_ema, name="checkpoint length_ema"
        )
        self.observation_count = self._validated_observation_count(
            observation_count
        )
