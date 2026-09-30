"""Persistent verified bank configuration, storage and diagnostics.

Admission, selection, and checkpoint schema live in their own modules.
The maintained method uses entropy_alpha=0 and retains verified exemplars.
"""

from __future__ import annotations

import math
from typing import Sequence

from remax.admission_retention import (
    AdmissionRetentionTracker,
    RetentionPriorityRequest,
)

from .admission import BankAdmissionMixin
from .bank_state import BankStateMixin
from .bank_types import VerifiedCanonicalReplayGroup, _prompt_key
from .scheduling import ReplaySchedulingMixin


class OnlineCanonicalBank(BankAdmissionMixin, ReplaySchedulingMixin, BankStateMixin):
    """Prompt-local verified state shared by Re:Dr, Re:Max and their controls."""

    def __init__(
        self,
        *,
        entropy_alpha: float,
        pseudocount: float = 1.0,
        surprisal_clip: float = 5.0,
        retain_exemplars: bool = False,
        replay_capacity: int = 16,
        global_replay_groups_per_step: int = 0,
        global_replay_bootstrap_steps: int = 0,
        separate_proposal_objective_support: bool = False,
        proposal_replay_priority_visits: int = 0,
        proposal_replay_priority_multiplier: float = 1.0,
        proposal_retention_tracking: bool = False,
        proposal_adaptive_retention_priority: bool = False,
        proposal_retention_max_missed_rollout_opportunities: int = 2,
        proposal_retention_max_mean_logprob_drop: float = 0.5,
        proposal_retention_refresh_visits: int = 4,
        proposal_retention_score_cooldown_observations: int = 2,
    ) -> None:
        for name, value in (
            ("entropy_alpha", entropy_alpha),
            ("pseudocount", pseudocount),
            ("surprisal_clip", surprisal_clip),
        ):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in {"pseudocount", "surprisal_clip"} and value <= 0:
                raise ValueError(f"{name} must be positive")
            if name == "entropy_alpha" and value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.entropy_alpha = float(entropy_alpha)
        self.pseudocount = float(pseudocount)
        self.surprisal_clip = float(surprisal_clip)
        self.retain_exemplars = bool(retain_exemplars)
        self.separate_proposal_objective_support = bool(
            separate_proposal_objective_support
        )
        if self.separate_proposal_objective_support and not self.retain_exemplars:
            raise ValueError(
                "separate proposal objective support requires replay exemplars"
            )
        if (
            isinstance(proposal_replay_priority_visits, bool)
            or int(proposal_replay_priority_visits) != proposal_replay_priority_visits
            or int(proposal_replay_priority_visits) < 0
        ):
            raise ValueError(
                "proposal replay priority visits must be a non-negative integer"
            )
        self.proposal_replay_priority_visits = int(proposal_replay_priority_visits)
        self.proposal_replay_priority_multiplier = float(
            proposal_replay_priority_multiplier
        )
        if (
            not math.isfinite(self.proposal_replay_priority_multiplier)
            or self.proposal_replay_priority_multiplier < 1.0
        ):
            raise ValueError(
                "proposal replay priority multiplier must be finite and at least one"
            )
        if self.proposal_replay_priority_visits > 0:
            if not self.separate_proposal_objective_support:
                raise ValueError(
                    "proposal replay priority requires separate proposal support"
                )
            if self.proposal_replay_priority_multiplier <= 1.0:
                raise ValueError(
                    "positive priority visits require multiplier greater than one"
                )
        elif not math.isclose(
            self.proposal_replay_priority_multiplier,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "priority multiplier must be one when priority is disabled"
            )
        self.proposal_retention_tracking = bool(proposal_retention_tracking)
        self.proposal_adaptive_retention_priority = bool(
            proposal_adaptive_retention_priority
        )
        if (
            self.proposal_adaptive_retention_priority
            and not self.proposal_retention_tracking
        ):
            raise ValueError(
                "adaptive proposal retention priority requires retention tracking"
            )
        if (
            self.proposal_retention_tracking
            and not self.separate_proposal_objective_support
        ):
            raise ValueError(
                "proposal retention tracking requires separated proposal support"
            )
        if self.proposal_adaptive_retention_priority:
            if self.proposal_replay_priority_visits <= 0:
                raise ValueError(
                    "adaptive proposal retention priority requires proposal priority"
                )
            if int(global_replay_groups_per_step) <= 0:
                raise ValueError(
                    "adaptive proposal retention priority requires global replay"
                )
            if (
                int(proposal_retention_refresh_visits)
                > self.proposal_replay_priority_visits
            ):
                raise ValueError(
                    "retention refresh visits may not exceed initial priority visits"
                )
        self._proposal_retention_tracker = (
            AdmissionRetentionTracker(
                adaptive_priority=self.proposal_adaptive_retention_priority,
                max_missed_rollout_opportunities=(
                    proposal_retention_max_missed_rollout_opportunities
                ),
                max_mean_logprob_drop=(proposal_retention_max_mean_logprob_drop),
                refresh_visits=proposal_retention_refresh_visits,
                score_cooldown_observations=(
                    proposal_retention_score_cooldown_observations
                ),
            )
            if self.proposal_retention_tracking
            else None
        )
        # One slot is a legal bank: it retains each prompt's first discovered
        # mode and admits no other, which is the E130 mode-agnostic ablation.
        # The bank does not know which replay objective will score it, and one
        # of them -- ``bank_balance`` -- cannot score a one-mode group, so the
        # objective-dependent floor is enforced in ``validate_zero_math_args``
        # rather than duplicated here with only half the information.
        if (
            isinstance(replay_capacity, bool)
            or int(replay_capacity) != replay_capacity
            or int(replay_capacity) < 1
        ):
            raise ValueError("replay_capacity must be an integer at least one")
        self.replay_capacity = int(replay_capacity)
        if (
            isinstance(global_replay_groups_per_step, bool)
            or int(global_replay_groups_per_step) != global_replay_groups_per_step
            or int(global_replay_groups_per_step) < 0
        ):
            raise ValueError(
                "global_replay_groups_per_step must be a non-negative integer"
            )
        self.global_replay_groups_per_step = int(global_replay_groups_per_step)
        if (
            isinstance(global_replay_bootstrap_steps, bool)
            or int(global_replay_bootstrap_steps) != global_replay_bootstrap_steps
            or int(global_replay_bootstrap_steps) < 0
        ):
            raise ValueError(
                "global_replay_bootstrap_steps must be a non-negative integer"
            )
        self.global_replay_bootstrap_steps = int(global_replay_bootstrap_steps)
        if (
            self.global_replay_bootstrap_steps > 0
            and self.global_replay_groups_per_step == 0
        ):
            raise ValueError(
                "global replay bootstrap requires positive global groups per step"
            )
        self._counts: dict[str, dict[str, int]] = {}
        self._prompt_token_ids: dict[str, tuple[int, ...]] = {}
        self._exemplars: dict[
            str,
            dict[str, tuple[int, ...]],
        ] = {}
        self._groups_scored = 0
        self._rows_scored = 0
        self._global_replay_cursor = 0
        self._global_replay_updates = 0
        self._proposal_groups = 0
        self._proposal_rows = 0
        self._proposal_new_outcomes = 0
        self._proposal_only_outcomes: dict[str, set[str]] = {}
        self._proposal_priority_remaining: dict[str, dict[str, int]] = {}
        self._proposal_priority_queue: list[str] = []
        self._proposal_priority_replay_groups = 0
        self._proposal_priority_replay_modes = 0

    @property
    def proposal_retention_tracking_enabled(self) -> bool:
        return self._proposal_retention_tracker is not None

    def proposal_retention_diagnostics(self) -> dict[str, float]:
        if self._proposal_retention_tracker is None:
            return {
                "tracking_enabled": 0.0,
                "adaptive_priority_enabled": 0.0,
            }
        return self._proposal_retention_tracker.diagnostics()

    def _apply_retention_priority_requests(
        self,
        requests: Sequence[RetentionPriorityRequest],
    ) -> None:
        tracker = self._proposal_retention_tracker
        if tracker is None and requests:
            raise RuntimeError("retention priority request without a tracker")
        for request in requests:
            if request.outcome_key not in self._exemplars.get(
                request.prompt_key,
                {},
            ):
                raise RuntimeError(
                    "retention priority request refers to a missing exemplar"
                )
            priority = self._proposal_priority_remaining.setdefault(
                request.prompt_key,
                {},
            )
            previous = int(priority.get(request.outcome_key, 0))
            updated = max(previous, int(request.visits))
            priority[request.outcome_key] = updated
            if request.prompt_key not in self._proposal_priority_queue:
                self._proposal_priority_queue.append(request.prompt_key)
            assert tracker is not None
            tracker.record_priority_application(
                request,
                visits_added=updated - previous,
            )

    def observe_replay_retention_scores(
        self,
        *,
        groups: Sequence[VerifiedCanonicalReplayGroup],
        mean_logprobs: Sequence[float],
        sequence_logprobs: Sequence[float],
    ) -> dict[str, float]:
        """Record training-only replay likelihoods and refresh priority."""

        tracker = self._proposal_retention_tracker
        if tracker is None:
            return self.proposal_retention_diagnostics()
        expected_rows = sum(len(group.outcome_keys) for group in groups)
        if not (len(mean_logprobs) == len(sequence_logprobs) == expected_rows):
            raise ValueError(
                "proposal retention replay score rows do not match replay groups"
            )
        observations: list[tuple[str, str, float, float]] = []
        cursor = 0
        for group in groups:
            prompt_key = _prompt_key(group.prompt_token_ids)
            for outcome_key in group.outcome_keys:
                observations.append(
                    (
                        prompt_key,
                        outcome_key,
                        float(mean_logprobs[cursor]),
                        float(sequence_logprobs[cursor]),
                    )
                )
                cursor += 1
        requests = tracker.observe_scores(observations)
        self._apply_retention_priority_requests(requests)
        return tracker.diagnostics()

    def resource_counts(self) -> dict[str, int]:
        """Logical storage, distinct from process RSS and allocator reservation.

        Capacity bounds exemplars per prompt, not the discovery ledger or the
        number of prompts. Exact cumulative identities are deliberately retained.
        Ledger observations are fresh-only in maintained recipes; historical
        proposal admission may also add pseudo-observations.
        """
        return {
            "discovered_prompts": len(self._counts),
            "discovered_modes": sum(len(v) for v in self._counts.values()),
            "ledger_observations": sum(sum(v.values()) for v in self._counts.values()),
            "retained_exemplars": sum(len(v) for v in self._exemplars.values()),
            "retained_response_tokens": sum(
                len(tokens) for v in self._exemplars.values() for tokens in v.values()
            ),
            "retained_prompt_tokens": sum(
                len(v) for v in self._prompt_token_ids.values()
            ),
            "capacity_per_prompt": self.replay_capacity,
        }

    @property
    def tracked_prompt_count(self) -> int:
        return len(self._counts)

    @property
    def tracked_outcome_count(self) -> int:
        return sum(len(counts) for counts in self._counts.values())

    @property
    def objective_active(self) -> bool:
        """Whether this bank contributes a nonzero exploration objective."""

        return self.entropy_alpha > 0.0

    @property
    def mean_support_per_prompt(self) -> float:
        """Mean on-policy support used by the entropy advantage."""

        if self.tracked_prompt_count == 0:
            return 0.0
        return self.tracked_outcome_count / self.tracked_prompt_count

    @property
    def replay_mean_support_per_prompt(self) -> float:
        """Mean replay-exemplar support, including proposal-only outcomes."""

        nonempty = [exemplars for exemplars in self._exemplars.values() if exemplars]
        if not nonempty:
            return 0.0
        return sum(len(exemplars) for exemplars in nonempty) / len(nonempty)

    def verified_replay_support(
        self, prompt_token_ids: Sequence[int]
    ) -> tuple[str, ...]:
        """Return validator-positive replay support for one prompt.

        This exposes membership only. Proposal rows never become on-policy
        frequency observations through this interface.
        """

        prompt_key = _prompt_key(prompt_token_ids)
        return tuple(sorted(self._exemplars.get(prompt_key, {})))

    @property
    def support_at_least_two_prompt_fraction(self) -> float:
        """Fraction of discovered prompts with at least two verified routes."""

        if self.tracked_prompt_count == 0:
            return 0.0
        return (
            sum(int(len(counts) >= 2) for counts in self._counts.values())
            / self.tracked_prompt_count
        )

    @property
    def global_replay_updates(self) -> int:
        """Number of non-empty finite-bootstrap global replay updates."""

        return self._global_replay_updates

    @property
    def proposal_groups(self) -> int:
        return self._proposal_groups

    @property
    def proposal_rows(self) -> int:
        return self._proposal_rows

    @property
    def proposal_new_outcomes(self) -> int:
        return self._proposal_new_outcomes

    @property
    def proposal_priority_remaining_visits(self) -> int:
        return sum(
            visits
            for outcomes in self._proposal_priority_remaining.values()
            for visits in outcomes.values()
        )

    @property
    def proposal_priority_replay_groups(self) -> int:
        return self._proposal_priority_replay_groups

    @property
    def proposal_priority_replay_modes(self) -> int:
        return self._proposal_priority_replay_modes

    @property
    def global_replay_bootstrap_active(self) -> bool:
        """Whether the finite global cold-start phase may still schedule."""

        return (
            self.global_replay_groups_per_step > 0
            and self.global_replay_bootstrap_steps > 0
            and self._global_replay_updates < self.global_replay_bootstrap_steps
        )
