"""One-sided token-entropy feedback for xDr's aggregation temperature.

The controller observes globally averaged training-token entropy after each
update and chooses the temperature for the *next* update. It never adds a
token-entropy reward. Instead, it changes candidate-level credit assignment by
lowering xDr's temperature when policy entropy falls below a scale-calibrated
target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class XdrTauController:
    """Bounded proportional controller with an EMA observation.

    The first ``warmup_steps`` observations run at ``base_tau`` and define
    ``target_entropy = target_ratio * mean(warmup_entropy)``. Afterwards,

    ``tau = clip(base_tau * exp(-gain * max(target - entropy_ema, 0)),
                  min_tau, base_tau)``.

    The one-sided rule leaves the known fixed-tau treatment unchanged while
    entropy is healthy and only sharpens candidate credit when it falls below
    target. It is an experimental controller, not a guarantee of stabilization.
    """

    base_tau: float
    min_tau: float
    target_ratio: float
    warmup_steps: int
    ema_decay: float
    gain: float
    current_tau: float | None = None
    entropy_ema: float | None = None
    target_entropy: float | None = None
    observation_count: int = 0
    _warmup_entropy_sum: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("base_tau", self.base_tau),
            ("min_tau", self.min_tau),
            ("target_ratio", self.target_ratio),
            ("ema_decay", self.ema_decay),
            ("gain", self.gain),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.base_tau <= 0:
            raise ValueError("base_tau must be positive")
        if self.min_tau <= 0 or self.min_tau > self.base_tau:
            raise ValueError("min_tau must be in (0, base_tau]")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in (0, 1]")
        if int(self.warmup_steps) <= 0:
            raise ValueError("warmup_steps must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.gain <= 0:
            raise ValueError("gain must be positive")
        if self.current_tau is None:
            self.current_tau = float(self.base_tau)

    def observe(self, entropy: float) -> dict[str, float]:
        """Update from one globally averaged entropy observation."""

        value = float(entropy)
        if not math.isfinite(value) or value < 0:
            raise ValueError("entropy observation must be finite and non-negative")

        self.observation_count += 1
        if self.entropy_ema is None:
            self.entropy_ema = value
        else:
            self.entropy_ema = (
                self.ema_decay * self.entropy_ema + (1.0 - self.ema_decay) * value
            )

        if self.observation_count <= self.warmup_steps:
            self._warmup_entropy_sum += value
            if self.observation_count == self.warmup_steps:
                warmup_mean = self._warmup_entropy_sum / float(self.warmup_steps)
                self.target_entropy = self.target_ratio * warmup_mean

        deficit = 0.0
        if self.target_entropy is not None:
            deficit = max(self.target_entropy - self.entropy_ema, 0.0)
            proposed = self.base_tau * math.exp(-self.gain * deficit)
            self.current_tau = min(self.base_tau, max(self.min_tau, proposed))
        else:
            self.current_tau = float(self.base_tau)

        diagnostics = {
            "xdr_tau_control_entropy_ema": float(self.entropy_ema),
            "xdr_tau_control_deficit": float(deficit),
            "xdr_tau_control_next_tau": float(self.current_tau),
            "xdr_tau_control_observations": float(self.observation_count),
        }
        if self.target_entropy is not None:
            diagnostics["xdr_tau_control_target_entropy"] = float(
                self.target_entropy
            )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_tau": float(self.current_tau),
            "entropy_ema": self.entropy_ema,
            "target_entropy": self.target_entropy,
            "observation_count": int(self.observation_count),
            "warmup_entropy_sum": float(self._warmup_entropy_sum),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("controller state must be a dictionary")
        current_tau = float(state.get("current_tau", self.base_tau))
        self.current_tau = min(self.base_tau, max(self.min_tau, current_tau))
        entropy_ema = state.get("entropy_ema")
        target_entropy = state.get("target_entropy")
        self.entropy_ema = None if entropy_ema is None else float(entropy_ema)
        self.target_entropy = (
            None if target_entropy is None else float(target_entropy)
        )
        self.observation_count = max(int(state.get("observation_count", 0)), 0)
        self._warmup_entropy_sum = float(state.get("warmup_entropy_sum", 0.0))
