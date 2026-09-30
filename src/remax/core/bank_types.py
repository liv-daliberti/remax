"""Verified keys, exemplars and bank diagnostics; no trainer dependencies."""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Sequence


def _prompt_key(prompt_token_ids: Sequence[int]) -> str:
    normalized: list[int] = []
    for value in prompt_token_ids:
        if isinstance(value, bool):
            raise ValueError("prompt token ids must be integers, not booleans")
        try:
            token_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("prompt token ids must be integers") from exc
        if token_id < 0 or token_id != value:
            raise ValueError("prompt token ids must be non-negative")
        normalized.append(token_id)
    if not normalized:
        raise ValueError("prompt token ids must not be empty")
    encoded = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OnlineCanonicalBankDiagnostics:
    entropy_estimate_mean: float
    normalized_entropy_mean: float
    normalized_entropy_ratio_mean: float
    normalized_entropy_ratio_eligible_fraction: float
    log_support_mean: float
    entropy_alpha_used: float
    entropy_advantage_mean: float
    entropy_advantage_rms: float
    combined_advantage_mean: float
    combined_advantage_rms: float
    eligible_fraction: float
    reward_positive_fraction: float
    canonicalizable_correct_fraction: float
    new_outcome_count: float
    new_outcome_row_fraction: float
    bank_size_before_mean: float
    bank_size_after_mean: float
    tracked_prompts: float
    tracked_outcomes: float
    support_at_least_two_prompt_fraction: float


@dataclass(frozen=True)
class VerifiedCanonicalReplayGroup:
    """One prompt and its deterministic validator-positive mode exemplars."""

    prompt_token_ids: tuple[int, ...]
    outcome_keys: tuple[str, ...]
    response_token_ids: tuple[tuple[int, ...], ...]
    # Cumulative validator-positive observations from fresh on-policy rows.
    # Replay and separated proposal admissions never increment these counts.
    fresh_observation_counts: tuple[int, ...] = ()
    mass_weights: tuple[float, ...] = ()
    priority_modes: int = 0


@dataclass(frozen=True)
class VerifiedProposalAdmissionDiagnostics:
    """Atomic support-only admission from conditioned proposal rollouts."""

    proposal_groups: int
    proposal_rows: int
    new_outcomes: int
    stored_exemplars: int
    tracked_prompts: int
    tracked_outcomes: int


def _normalized_token_tuple(
    values: Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{label} must contain integers, not booleans")
        try:
            token_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must contain integers") from exc
        if token_id < 0 or token_id != value:
            raise ValueError(f"{label} must contain non-negative integers")
        normalized.append(token_id)
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return tuple(normalized)
