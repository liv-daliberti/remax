"""Training-only retention tracking for proposal-admitted replay exemplars.

The tracker never sees evaluation output, exhaustive support, or a desired
mode count.  It follows each newly admitted exemplar with two policy-derived
signals:

* verifier-positive reappearance in later on-policy groups for the same prompt;
* teacher-forced mean-token and full-sequence log likelihood on later replay.

An optional bounded controller emits requests to refresh replay priority after
repeated on-policy misses or a drop from the exemplar's own first measured
post-admission score.  Applying a request remains the bank's responsibility so
the tracker has no access to loss coefficients or PPO rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import copy
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RetentionPriorityRequest:
    prompt_key: str
    outcome_key: str
    reason: str
    visits: int


@dataclass
class _RetentionRecord:
    admission_index: int
    skip_pre_admission_rollout: bool = True
    rollout_opportunities: int = 0
    rollout_rows: int = 0
    rollout_hits: int = 0
    rollout_groups_with_hit: int = 0
    rollout_miss_streak: int = 0
    converted_on_policy: bool = False
    score_observations: int = 0
    baseline_mean_logprob: float | None = None
    latest_mean_logprob: float | None = None
    minimum_mean_logprob: float | None = None
    baseline_sequence_logprob: float | None = None
    latest_sequence_logprob: float | None = None
    minimum_sequence_logprob: float | None = None
    last_score_refresh_observation: int = 0
    rollout_refresh_requests: int = 0
    score_refresh_requests: int = 0
    priority_visits_added: int = 0


def _positive_int(name: str, value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or normalized != value or normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class AdmissionRetentionTracker:
    """Checkpointed proposal-admission survival sensor and priority controller."""

    SCHEMA = "proposal_admission_retention_tracker_v1"

    def __init__(
        self,
        *,
        adaptive_priority: bool = False,
        max_missed_rollout_opportunities: int = 2,
        max_mean_logprob_drop: float = 0.5,
        refresh_visits: int = 4,
        score_cooldown_observations: int = 2,
    ) -> None:
        self.adaptive_priority = bool(adaptive_priority)
        self.max_missed_rollout_opportunities = _positive_int(
            "max_missed_rollout_opportunities",
            max_missed_rollout_opportunities,
        )
        self.max_mean_logprob_drop = _finite_positive(
            "max_mean_logprob_drop",
            max_mean_logprob_drop,
        )
        self.refresh_visits = _positive_int("refresh_visits", refresh_visits)
        self.score_cooldown_observations = _positive_int(
            "score_cooldown_observations",
            score_cooldown_observations,
        )
        self._records: dict[str, dict[str, _RetentionRecord]] = {}
        self._admissions = 0
        self._rollout_refresh_requests = 0
        self._score_refresh_requests = 0
        self._priority_visits_added = 0

    def clone(self) -> "AdmissionRetentionTracker":
        return copy.deepcopy(self)

    def admit(self, prompt_key: str, outcome_key: str) -> None:
        if not isinstance(prompt_key, str) or not prompt_key:
            raise ValueError("retention prompt key must be a non-empty string")
        if not isinstance(outcome_key, str) or not outcome_key:
            raise ValueError("retention outcome key must be a non-empty string")
        prompt_records = self._records.setdefault(prompt_key, {})
        if outcome_key in prompt_records:
            raise ValueError("retention tracker received a duplicate admission")
        self._admissions += 1
        prompt_records[outcome_key] = _RetentionRecord(
            admission_index=self._admissions,
        )

    def observe_rollout(
        self,
        *,
        prompt_key: str,
        verified_outcome_counts: Mapping[str, int],
        row_count: int,
    ) -> tuple[RetentionPriorityRequest, ...]:
        if isinstance(row_count, bool) or int(row_count) != row_count or row_count <= 0:
            raise ValueError("retention rollout row_count must be positive")
        normalized_counts: Counter[str] = Counter()
        for outcome_key, count in verified_outcome_counts.items():
            if (
                not isinstance(outcome_key, str)
                or not outcome_key
                or isinstance(count, bool)
                or int(count) != count
                or int(count) < 0
            ):
                raise ValueError("retention rollout counts are invalid")
            if count:
                normalized_counts[outcome_key] = int(count)
        if sum(normalized_counts.values()) > int(row_count):
            raise ValueError("retention rollout hits exceed generated rows")

        requests: list[RetentionPriorityRequest] = []
        for outcome_key, record in self._records.get(prompt_key, {}).items():
            # Proposal admission happens after the neutral group was sampled
            # but before that group reaches score_and_update.  Excluding this
            # first observation makes every counted opportunity post-admission.
            if record.skip_pre_admission_rollout:
                record.skip_pre_admission_rollout = False
                continue
            hits = int(normalized_counts.get(outcome_key, 0))
            record.rollout_opportunities += 1
            record.rollout_rows += int(row_count)
            record.rollout_hits += hits
            if hits > 0:
                record.rollout_groups_with_hit += 1
                record.rollout_miss_streak = 0
                record.converted_on_policy = True
                continue
            record.rollout_miss_streak += 1
            if (
                self.adaptive_priority
                and record.rollout_miss_streak >= self.max_missed_rollout_opportunities
            ):
                record.rollout_miss_streak = 0
                record.rollout_refresh_requests += 1
                self._rollout_refresh_requests += 1
                requests.append(
                    RetentionPriorityRequest(
                        prompt_key=prompt_key,
                        outcome_key=outcome_key,
                        reason="rollout_absence",
                        visits=self.refresh_visits,
                    )
                )
        return tuple(requests)

    def observe_scores(
        self,
        observations: Sequence[tuple[str, str, float, float]],
    ) -> tuple[RetentionPriorityRequest, ...]:
        seen: set[tuple[str, str]] = set()
        requests: list[RetentionPriorityRequest] = []
        for prompt_key, outcome_key, mean_logprob, sequence_logprob in observations:
            if not isinstance(prompt_key, str) or not prompt_key:
                raise ValueError("retention score prompt key must be non-empty")
            if not isinstance(outcome_key, str) or not outcome_key:
                raise ValueError("retention score outcome key must be non-empty")
            pair = (prompt_key, outcome_key)
            if pair in seen:
                raise ValueError("retention score observations contain a duplicate")
            seen.add(pair)
            mean_logprob = _finite("mean_logprob", mean_logprob)
            sequence_logprob = _finite("sequence_logprob", sequence_logprob)
            record = self._records.get(prompt_key, {}).get(outcome_key)
            if record is None:
                continue

            record.score_observations += 1
            if record.baseline_mean_logprob is None:
                record.baseline_mean_logprob = mean_logprob
                record.baseline_sequence_logprob = sequence_logprob
            record.latest_mean_logprob = mean_logprob
            record.latest_sequence_logprob = sequence_logprob
            record.minimum_mean_logprob = (
                mean_logprob
                if record.minimum_mean_logprob is None
                else min(record.minimum_mean_logprob, mean_logprob)
            )
            record.minimum_sequence_logprob = (
                sequence_logprob
                if record.minimum_sequence_logprob is None
                else min(record.minimum_sequence_logprob, sequence_logprob)
            )

            assert record.baseline_mean_logprob is not None
            mean_drop = record.baseline_mean_logprob - mean_logprob
            cooled_down = (
                record.score_observations - record.last_score_refresh_observation
                >= self.score_cooldown_observations
            )
            if (
                self.adaptive_priority
                and mean_drop > self.max_mean_logprob_drop
                and cooled_down
            ):
                record.last_score_refresh_observation = record.score_observations
                record.score_refresh_requests += 1
                self._score_refresh_requests += 1
                requests.append(
                    RetentionPriorityRequest(
                        prompt_key=prompt_key,
                        outcome_key=outcome_key,
                        reason="score_drop",
                        visits=self.refresh_visits,
                    )
                )
        return tuple(requests)

    def record_priority_application(
        self,
        request: RetentionPriorityRequest,
        *,
        visits_added: int,
    ) -> None:
        if (
            isinstance(visits_added, bool)
            or int(visits_added) != visits_added
            or int(visits_added) < 0
        ):
            raise ValueError("retention priority visits_added must be non-negative")
        record = self._records.get(request.prompt_key, {}).get(request.outcome_key)
        if record is None:
            raise ValueError(
                "retention priority request refers to an unknown admission"
            )
        record.priority_visits_added += int(visits_added)
        self._priority_visits_added += int(visits_added)

    @property
    def tracked_admissions(self) -> int:
        return sum(len(records) for records in self._records.values())

    @property
    def tracked_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (prompt_key, outcome_key)
            for prompt_key, prompt_records in sorted(self._records.items())
            for outcome_key in sorted(prompt_records)
        )

    @property
    def converted_pairs(self) -> tuple[tuple[str, str], ...]:
        """Proposal admissions later observed by the neutral policy."""

        return tuple(
            (prompt_key, outcome_key)
            for prompt_key, prompt_records in sorted(self._records.items())
            for outcome_key, record in sorted(prompt_records.items())
            if record.converted_on_policy
        )

    def diagnostics(self) -> dict[str, float]:
        records = [
            record
            for prompt_records in self._records.values()
            for record in prompt_records.values()
        ]
        rollout_eligible = [
            record for record in records if record.rollout_opportunities > 0
        ]
        score_followup = [
            record for record in records if record.score_observations >= 2
        ]
        joint_eligible = [
            record
            for record in records
            if record.rollout_opportunities > 0 and record.score_observations >= 2
        ]
        mean_drops = [
            float(record.baseline_mean_logprob - record.latest_mean_logprob)
            for record in score_followup
            if record.baseline_mean_logprob is not None
            and record.latest_mean_logprob is not None
        ]
        sequence_drops = [
            float(record.baseline_sequence_logprob - record.latest_sequence_logprob)
            for record in score_followup
            if record.baseline_sequence_logprob is not None
            and record.latest_sequence_logprob is not None
        ]
        score_retained = [
            record
            for record in score_followup
            if record.baseline_mean_logprob is not None
            and record.latest_mean_logprob is not None
            and (
                record.baseline_mean_logprob - record.latest_mean_logprob
                <= self.max_mean_logprob_drop
            )
        ]
        joint_retained = [
            record
            for record in joint_eligible
            if record.converted_on_policy
            and record.baseline_mean_logprob is not None
            and record.latest_mean_logprob is not None
            and (
                record.baseline_mean_logprob - record.latest_mean_logprob
                <= self.max_mean_logprob_drop
            )
        ]
        total_rollout_rows = sum(record.rollout_rows for record in records)
        total_rollout_hits = sum(record.rollout_hits for record in records)

        def fraction(numerator: int, denominator: int) -> float:
            return float(numerator / denominator) if denominator else 0.0

        def mean(values: Sequence[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        return {
            "tracking_enabled": 1.0,
            "adaptive_priority_enabled": float(self.adaptive_priority),
            "tracked_admissions": float(len(records)),
            "rollout_eligible_admissions": float(len(rollout_eligible)),
            "rollout_converted_admissions": float(
                sum(record.converted_on_policy for record in rollout_eligible)
            ),
            "rollout_conversion_fraction": fraction(
                sum(record.converted_on_policy for record in rollout_eligible),
                len(rollout_eligible),
            ),
            "rollout_row_frequency": fraction(
                total_rollout_hits,
                total_rollout_rows,
            ),
            "score_observed_admissions": float(
                sum(record.score_observations > 0 for record in records)
            ),
            "score_followup_admissions": float(len(score_followup)),
            "score_retained_admissions": float(len(score_retained)),
            "score_retained_fraction": fraction(
                len(score_retained),
                len(score_followup),
            ),
            "joint_eligible_admissions": float(len(joint_eligible)),
            "joint_retained_admissions": float(len(joint_retained)),
            "joint_retained_fraction": fraction(
                len(joint_retained),
                len(joint_eligible),
            ),
            "mean_logprob_drop_mean": mean(mean_drops),
            "mean_logprob_drop_max": max(mean_drops, default=0.0),
            "sequence_logprob_drop_mean": mean(sequence_drops),
            "sequence_logprob_drop_max": max(sequence_drops, default=0.0),
            "rollout_refresh_requests_cumulative": float(
                self._rollout_refresh_requests
            ),
            "score_refresh_requests_cumulative": float(self._score_refresh_requests),
            "refresh_requests_cumulative": float(
                self._rollout_refresh_requests + self._score_refresh_requests
            ),
            "priority_visits_added_cumulative": float(self._priority_visits_added),
            "gold_support_feedback": 0.0,
            "desired_mode_count_feedback": 0.0,
            "eval_feedback": 0.0,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "adaptive_priority": self.adaptive_priority,
            "max_missed_rollout_opportunities": (self.max_missed_rollout_opportunities),
            "max_mean_logprob_drop": self.max_mean_logprob_drop,
            "refresh_visits": self.refresh_visits,
            "score_cooldown_observations": self.score_cooldown_observations,
            "admissions": self._admissions,
            "rollout_refresh_requests": self._rollout_refresh_requests,
            "score_refresh_requests": self._score_refresh_requests,
            "priority_visits_added": self._priority_visits_added,
            "records": {
                prompt_key: {
                    outcome_key: {
                        field: getattr(record, field)
                        for field in _RetentionRecord.__dataclass_fields__
                    }
                    for outcome_key, record in sorted(prompt_records.items())
                }
                for prompt_key, prompt_records in sorted(self._records.items())
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != self.SCHEMA:
            raise ValueError("invalid proposal admission retention state")
        expected = {
            "adaptive_priority": self.adaptive_priority,
            "max_missed_rollout_opportunities": (self.max_missed_rollout_opportunities),
            "max_mean_logprob_drop": self.max_mean_logprob_drop,
            "refresh_visits": self.refresh_visits,
            "score_cooldown_observations": self.score_cooldown_observations,
        }
        for name, configured in expected.items():
            saved = state.get(name)
            if isinstance(configured, float):
                try:
                    matches = math.isclose(
                        float(saved),
                        configured,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = saved == configured
            if not matches:
                raise ValueError(f"proposal retention resume mismatch for {name}")

        raw_records = state.get("records")
        if not isinstance(raw_records, Mapping):
            raise ValueError("proposal retention state has invalid records")
        restored: dict[str, dict[str, _RetentionRecord]] = {}
        seen_admission_indices: set[int] = set()
        for prompt_key, prompt_records in raw_records.items():
            if not isinstance(prompt_key, str) or not prompt_key:
                raise ValueError("proposal retention state has invalid prompt")
            if not isinstance(prompt_records, Mapping):
                raise ValueError("proposal retention state has invalid prompt records")
            restored[prompt_key] = {}
            for outcome_key, raw_record in prompt_records.items():
                if (
                    not isinstance(outcome_key, str)
                    or not outcome_key
                    or not isinstance(raw_record, Mapping)
                    or set(raw_record) != set(_RetentionRecord.__dataclass_fields__)
                ):
                    raise ValueError("proposal retention state has invalid record")
                try:
                    record = _RetentionRecord(**dict(raw_record))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "proposal retention state has malformed record"
                    ) from exc
                if not isinstance(
                    record.skip_pre_admission_rollout, bool
                ) or not isinstance(record.converted_on_policy, bool):
                    raise ValueError(
                        "proposal retention state has invalid boolean flags"
                    )
                integer_fields = (
                    "admission_index",
                    "rollout_opportunities",
                    "rollout_rows",
                    "rollout_hits",
                    "rollout_groups_with_hit",
                    "rollout_miss_streak",
                    "score_observations",
                    "last_score_refresh_observation",
                    "rollout_refresh_requests",
                    "score_refresh_requests",
                    "priority_visits_added",
                )
                if any(
                    isinstance(getattr(record, field), bool)
                    or int(getattr(record, field)) != getattr(record, field)
                    or int(getattr(record, field))
                    < (1 if field == "admission_index" else 0)
                    for field in integer_fields
                ):
                    raise ValueError("proposal retention state has invalid counters")
                for field in (
                    "baseline_mean_logprob",
                    "latest_mean_logprob",
                    "minimum_mean_logprob",
                    "baseline_sequence_logprob",
                    "latest_sequence_logprob",
                    "minimum_sequence_logprob",
                ):
                    value = getattr(record, field)
                    if value is not None and not math.isfinite(float(value)):
                        raise ValueError(
                            "proposal retention state has non-finite score"
                        )
                if record.rollout_hits > record.rollout_rows:
                    raise ValueError("proposal retention state has impossible hits")
                if record.score_observations == 0 and any(
                    getattr(record, field) is not None
                    for field in (
                        "baseline_mean_logprob",
                        "latest_mean_logprob",
                        "minimum_mean_logprob",
                        "baseline_sequence_logprob",
                        "latest_sequence_logprob",
                        "minimum_sequence_logprob",
                    )
                ):
                    raise ValueError(
                        "proposal retention state has scores without observations"
                    )
                if record.score_observations > 0 and any(
                    getattr(record, field) is None
                    for field in (
                        "baseline_mean_logprob",
                        "latest_mean_logprob",
                        "minimum_mean_logprob",
                        "baseline_sequence_logprob",
                        "latest_sequence_logprob",
                        "minimum_sequence_logprob",
                    )
                ):
                    raise ValueError(
                        "proposal retention state is missing observed scores"
                    )
                if record.admission_index in seen_admission_indices:
                    raise ValueError(
                        "proposal retention state has duplicate admission index"
                    )
                seen_admission_indices.add(record.admission_index)
                restored[prompt_key][outcome_key] = record

        counter_names = (
            "admissions",
            "rollout_refresh_requests",
            "score_refresh_requests",
            "priority_visits_added",
        )
        counters: dict[str, int] = {}
        for name in counter_names:
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"proposal retention state has invalid {name}")
            counters[name] = value
        if counters["admissions"] != len(seen_admission_indices):
            raise ValueError("proposal retention admission count mismatch")
        if seen_admission_indices and (
            max(seen_admission_indices) != counters["admissions"]
        ):
            raise ValueError("proposal retention admission indices are not dense")
        if counters["priority_visits_added"] != sum(
            record.priority_visits_added
            for prompt_records in restored.values()
            for record in prompt_records.values()
        ):
            raise ValueError("proposal retention priority visit count mismatch")
        if counters["rollout_refresh_requests"] != sum(
            record.rollout_refresh_requests
            for prompt_records in restored.values()
            for record in prompt_records.values()
        ):
            raise ValueError("proposal retention rollout request count mismatch")
        if counters["score_refresh_requests"] != sum(
            record.score_refresh_requests
            for prompt_records in restored.values()
            for record in prompt_records.values()
        ):
            raise ValueError("proposal retention score request count mismatch")

        self._records = restored
        self._admissions = counters["admissions"]
        self._rollout_refresh_requests = counters["rollout_refresh_requests"]
        self._score_refresh_requests = counters["score_refresh_requests"]
        self._priority_visits_added = counters["priority_visits_added"]
