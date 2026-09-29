"""Group-local semantic outcome-collision reward shaping.

The mechanism operates on canonical answer keys extracted by the task grader.
It does not require a catalogue of valid answers: every observed key, including
the shared invalid key used for parse failures, participates in the collision
count.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence


INVALID_OUTCOME_KEY = "__OUTCOME_COLLISION_INVALID__"


def add_outcome_collision_outside_centering_advantage(
    centered_advantages: Any,
    collision_advantages: Any,
) -> Any:
    """Add the detached collision term to a precomputed sequence advantage.

    The function deliberately does no centering. ``collision_advantages`` is
    already the coefficient-scaled leave-one-out penalty returned by
    :func:`compute_outcome_collision_bonuses`. The learner calls this exactly
    once after computing ordinary Dr.GRPO advantages and before reusing the
    frozen result across PPO epochs.

    Array-like inputs must have identical shapes. Keeping the helper
    array-library agnostic makes the essential E40 composition testable
    without importing the training runtime; production passes torch tensors.
    """

    centered_shape = getattr(centered_advantages, "shape", None)
    collision_shape = getattr(collision_advantages, "shape", None)
    if (
        centered_shape is not None
        and collision_shape is not None
        and centered_shape != collision_shape
    ):
        raise ValueError(
            "centered and outcome-collision advantages must have identical shapes"
        )
    return centered_advantages + collision_advantages


@dataclass(frozen=True)
class OutcomeCollisionDiagnostics:
    """Batch diagnostics in the same units as the shaping reward."""

    collision_rate: float
    bonus_mean: float
    bonus_min: float
    bonus_max: float
    distinct_outcomes_mean: float
    distinct_fraction: float
    invalid_fraction: float
    parseable_fraction: float


def compute_outcome_collision_bonuses(
    answer_keys: Sequence[str | None],
    *,
    num_samples: int,
    coefficient: float,
) -> tuple[list[float], OutcomeCollisionDiagnostics]:
    """Return the negative within-group duplicate penalty for every row.

    For a group of size ``G``, row ``i`` receives

    ``coefficient * (-(1 / G) * sum_{j != i} 1[key_i == key_j])``.

    ``None`` is deliberately mapped to one shared invalid key. Consequently,
    repeated unparsable outputs collide rather than being treated as diverse.
    Groups never interact with one another.
    """

    if num_samples <= 1:
        raise ValueError("num_samples must be greater than one")
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and non-negative")
    if len(answer_keys) % num_samples != 0:
        raise ValueError("answer_keys must contain complete candidate groups")

    normalized = [
        INVALID_OUTCOME_KEY if key is None else str(key) for key in answer_keys
    ]
    bonuses: list[float] = []
    collision_fractions: list[float] = []
    distinct_counts: list[int] = []

    for start in range(0, len(normalized), num_samples):
        group = normalized[start : start + num_samples]
        counts = Counter(group)
        distinct_counts.append(len(counts))
        for key in group:
            collision_fraction = (counts[key] - 1) / num_samples
            collision_fractions.append(collision_fraction)
            bonuses.append(-coefficient * collision_fraction)

    if not bonuses:
        diagnostics = OutcomeCollisionDiagnostics(
            collision_rate=0.0,
            bonus_mean=0.0,
            bonus_min=0.0,
            bonus_max=0.0,
            distinct_outcomes_mean=0.0,
            distinct_fraction=0.0,
            invalid_fraction=0.0,
            parseable_fraction=0.0,
        )
        return bonuses, diagnostics

    invalid_fraction = (
        sum(key == INVALID_OUTCOME_KEY for key in normalized) / len(normalized)
    )
    diagnostics = OutcomeCollisionDiagnostics(
        collision_rate=sum(collision_fractions) / len(collision_fractions),
        bonus_mean=sum(bonuses) / len(bonuses),
        bonus_min=min(bonuses),
        bonus_max=max(bonuses),
        distinct_outcomes_mean=sum(distinct_counts) / len(distinct_counts),
        distinct_fraction=(
            sum(distinct_counts) / (len(distinct_counts) * num_samples)
        ),
        invalid_fraction=invalid_fraction,
        parseable_fraction=1.0 - invalid_fraction,
    )
    return bonuses, diagnostics
