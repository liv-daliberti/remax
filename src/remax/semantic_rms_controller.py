"""Adapt the semantic MaxEnt coefficient to a registered pressure ratio.

At a fixed coefficient the *realized* semantic pressure is not fixed. Measured
over E81's five domains at eta = .10, the ratio of semantic-advantage RMS to
task-advantage RMS ran 1.0%, 1.6%, 1.9%, and 3.7%, and within one domain the
per-seed RMS varied eightfold. The cause is structural rather than incidental:
the centered surprisal of Equation (semantic-advantage) depends on how many
verified modes the prompt's bank happens to hold, so the same nominal
coefficient buys an order of magnitude more pressure in one domain than another.

This controller does not chase an outcome. It observes only the two quantities
that define the applied dose and moves the coefficient so that

    RMS(A_sem) / RMS(A_task) -> rho,

for a rho registered before execution. It never sees evaluation results, gold
support, a desired mode count, or a target diversity.

The dangerous failure mode for any such rule is the singleton trap. When a
prompt has one verified mode the centered semantic signal is exactly zero, so a
naive ratio controller divides by ~0, drives the coefficient to its ceiling, and
then delivers an oversized update the moment a second mode appears. Earlier
controllers in this program failed in exactly that shape. The defence here is
that an observation is *refused* rather than extrapolated: when the eligible
fraction is too low, when the task RMS is degenerate, or when the measured
semantic RMS is below a floor, the controller freezes the coefficient and does
not update its EMAs at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class SemanticRmsController:
    """Projection-free semantic coefficient targeting a fixed pressure ratio."""

    base_coefficient: float
    target_ratio: float
    min_coefficient: float = 0.02
    max_coefficient: float = 0.40
    ema_decay: float = 0.98
    gain: float = 0.5
    max_step_ratio: float = 1.1
    warmup_steps: int = 64
    min_eligible_fraction: float = 0.05
    min_task_rms: float = 1e-3
    min_semantic_rms: float = 1e-6

    current_coefficient: float | None = None
    semantic_rms_ema: float | None = None
    task_rms_ema: float | None = None
    observations: int = 0
    frozen_observations: int = 0
    updates_applied: int = 0
    bound_hits: int = 0
    _last_ratio: float = field(default=0.0)

    units: str = "semantic_task_advantage_rms_ratio_v1"

    def __post_init__(self) -> None:
        for name in ("base_coefficient", "target_ratio", "ema_decay", "gain",
                     "max_step_ratio", "min_coefficient", "max_coefficient"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.base_coefficient <= 0:
            raise ValueError("base_coefficient must be positive")
        if self.target_ratio <= 0:
            raise ValueError("target_ratio must be positive")
        if not 0 < self.min_coefficient <= self.max_coefficient:
            raise ValueError("coefficient bounds must satisfy 0 < min <= max")
        if not self.min_coefficient <= self.base_coefficient <= self.max_coefficient:
            raise ValueError("base_coefficient must lie inside its bounds")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.gain <= 0:
            raise ValueError("gain must be positive")
        if self.max_step_ratio <= 1:
            raise ValueError("max_step_ratio must exceed 1")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.current_coefficient is None:
            self.current_coefficient = float(self.base_coefficient)

    # -- observation ------------------------------------------------------

    def admissible(
        self,
        *,
        semantic_rms: float,
        task_rms: float,
        eligible_fraction: float,
    ) -> bool:
        """Whether this update carries a usable measurement of the dose.

        A refused observation is not scored as a zero ratio. Treating it as zero
        is precisely the singleton trap: it would push the coefficient upward
        while the signal it multiplies is identically zero.
        """

        for value in (semantic_rms, task_rms, eligible_fraction):
            if not math.isfinite(float(value)):
                return False
        if float(eligible_fraction) < self.min_eligible_fraction:
            return False
        if float(task_rms) < self.min_task_rms:
            return False
        if float(semantic_rms) < self.min_semantic_rms:
            return False
        return True

    def observe(
        self,
        *,
        semantic_rms: float,
        task_rms: float,
        eligible_fraction: float,
    ) -> float:
        """Fold one update into the controller and return the coefficient."""

        if not self.admissible(
            semantic_rms=semantic_rms,
            task_rms=task_rms,
            eligible_fraction=eligible_fraction,
        ):
            self.frozen_observations += 1
            return float(self.current_coefficient)

        decay = float(self.ema_decay)
        semantic = float(semantic_rms)
        task = float(task_rms)
        self.semantic_rms_ema = (
            semantic
            if self.semantic_rms_ema is None
            else decay * self.semantic_rms_ema + (1 - decay) * semantic
        )
        self.task_rms_ema = (
            task
            if self.task_rms_ema is None
            else decay * self.task_rms_ema + (1 - decay) * task
        )
        self.observations += 1

        # The warmup window only fills the EMAs. Acting on one or two noisy
        # batches is how a controller ends up chasing its own transient.
        if self.observations <= int(self.warmup_steps):
            return float(self.current_coefficient)

        realized = self.semantic_rms_ema / self.task_rms_ema
        self._last_ratio = float(realized)
        if realized <= 0:
            return float(self.current_coefficient)

        proposal = float(self.current_coefficient) * (
            (float(self.target_ratio) / realized) ** float(self.gain)
        )
        ceiling = float(self.current_coefficient) * float(self.max_step_ratio)
        floor = float(self.current_coefficient) / float(self.max_step_ratio)
        proposal = min(max(proposal, floor), ceiling)
        clipped = min(max(proposal, self.min_coefficient), self.max_coefficient)
        if clipped in (self.min_coefficient, self.max_coefficient) and (
            not math.isclose(clipped, proposal, rel_tol=1e-12)
        ):
            self.bound_hits += 1
        self.current_coefficient = float(clipped)
        self.updates_applied += 1
        return float(self.current_coefficient)

    # -- telemetry and resume --------------------------------------------

    def diagnostics(self) -> dict[str, float]:
        total = self.observations + self.frozen_observations
        return {
            "semantic_rms_controller_coefficient": float(self.current_coefficient),
            "semantic_rms_controller_realized_ratio": float(self._last_ratio),
            "semantic_rms_controller_target_ratio": float(self.target_ratio),
            "semantic_rms_controller_semantic_rms_ema": float(
                self.semantic_rms_ema if self.semantic_rms_ema is not None else 0.0
            ),
            "semantic_rms_controller_task_rms_ema": float(
                self.task_rms_ema if self.task_rms_ema is not None else 0.0
            ),
            "semantic_rms_controller_observations": float(self.observations),
            "semantic_rms_controller_frozen_observations": float(
                self.frozen_observations
            ),
            "semantic_rms_controller_frozen_fraction": float(
                self.frozen_observations / total if total else 0.0
            ),
            "semantic_rms_controller_updates_applied": float(self.updates_applied),
            "semantic_rms_controller_bound_hits": float(self.bound_hits),
            "semantic_rms_controller_bound_hit_fraction": float(
                self.bound_hits / self.updates_applied if self.updates_applied else 0.0
            ),
            "semantic_rms_controller_at_min": float(
                math.isclose(float(self.current_coefficient), self.min_coefficient)
            ),
            "semantic_rms_controller_at_max": float(
                math.isclose(float(self.current_coefficient), self.max_coefficient)
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.units,
            "current_coefficient": float(self.current_coefficient),
            "semantic_rms_ema": self.semantic_rms_ema,
            "task_rms_ema": self.task_rms_ema,
            "observations": int(self.observations),
            "frozen_observations": int(self.frozen_observations),
            "updates_applied": int(self.updates_applied),
            "bound_hits": int(self.bound_hits),
            "last_ratio": float(self._last_ratio),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("schema")) != self.units:
            raise ValueError("semantic RMS controller state schema mismatch")
        coefficient = float(state["current_coefficient"])
        if not self.min_coefficient <= coefficient <= self.max_coefficient:
            raise ValueError("restored coefficient lies outside its registered bounds")
        self.current_coefficient = coefficient
        self.semantic_rms_ema = (
            None if state["semantic_rms_ema"] is None else float(state["semantic_rms_ema"])
        )
        self.task_rms_ema = (
            None if state["task_rms_ema"] is None else float(state["task_rms_ema"])
        )
        self.observations = int(state["observations"])
        self.frozen_observations = int(state["frozen_observations"])
        self.updates_applied = int(state["updates_applied"])
        self.bound_hits = int(state["bound_hits"])
        self._last_ratio = float(state.get("last_ratio", 0.0))
