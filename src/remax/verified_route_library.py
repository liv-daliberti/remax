"""Checkpointed cross-prompt library of independently verified routes.

The existing :mod:`online_canonical_bank` remains the prompt-local endpoint
store. This module is deliberately separate: route identity may recur across
different prompts, while endpoints are meaningful only within a prompt.

Only ordinary task-positive, loss-active rollouts enter neutral counts.
Explorer proposals are support-only records and become neutral records only
after an unconditioned rollout reproduces the same verified route on the same
prompt. Replay is fixed-budget, deterministic, and restricted to a source
prompt other than the current prompt batch.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from .online_canonical_bank import VerifiedCanonicalReplayGroup


def _token_tuple(values: Sequence[int], *, label: str) -> tuple[int, ...]:
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


def _prompt_key(prompt_token_ids: Sequence[int]) -> str:
    normalized = _token_tuple(
        prompt_token_ids,
        label="verified route prompt token ids",
    )
    payload = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _route_key(verifier: str, route_signature: str) -> str:
    if not isinstance(verifier, str) or not verifier:
        raise ValueError("verified route verifier must be a non-empty string")
    if not isinstance(route_signature, str) or not route_signature:
        raise ValueError("verified route signature must be a non-empty string")
    return json.dumps(
        [verifier, route_signature],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _finite_logprob(value: float, *, label: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(normalized) or normalized > 1e-8:
        raise ValueError(f"{label} must be finite and non-positive")
    return normalized


@dataclass(frozen=True)
class VerifiedRouteLibraryDiagnostics:
    neutral_rows_observed: int
    neutral_routes_observed: int
    proposal_rows_admitted: int
    proposal_rows_rejected_trust: int
    proposal_graduations: int
    distinct_routes: int
    recurring_routes: int
    distinct_source_prompts: int
    cross_prompt_neutral_reproductions: int
    post_replay_cross_prompt_neutral_reproductions: int
    cross_prompt_replay_updates: int
    cross_prompt_replay_groups: int
    cross_prompt_replay_rows: int


@dataclass(frozen=True)
class VerifiedRouteProposalAdmission:
    admitted: bool
    rejected_trust: bool
    already_known: bool
    route_signature: str
    source_prompt_key: str


@dataclass(frozen=True)
class VerifiedRouteExemplar:
    verifier: str
    endpoint_key: str
    route_signature: str
    response_token_ids: tuple[int, ...]
    model_mean_logprob: float
    neutral_reproduced: bool
    proposal_observed: bool


class VerifiedRouteLibrary:
    """Global verified-route records plus deterministic cross-prompt replay."""

    def __init__(
        self,
        *,
        replay_groups_per_step: int,
        replay_capacity_per_route: int = 16,
        recurring_min_neutral_prompts: int = 2,
        proposal_max_mean_logprob_drop: float = 2.0,
    ) -> None:
        for name, value, minimum in (
            ("replay_groups_per_step", replay_groups_per_step, 0),
            ("replay_capacity_per_route", replay_capacity_per_route, 1),
            (
                "recurring_min_neutral_prompts",
                recurring_min_neutral_prompts,
                2,
            ),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
        drop = float(proposal_max_mean_logprob_drop)
        if not math.isfinite(drop) or drop < 0:
            raise ValueError(
                "proposal_max_mean_logprob_drop must be finite and non-negative"
            )
        self.replay_groups_per_step = int(replay_groups_per_step)
        self.replay_capacity_per_route = int(replay_capacity_per_route)
        self.recurring_min_neutral_prompts = int(recurring_min_neutral_prompts)
        self.proposal_max_mean_logprob_drop = drop
        self._prompt_token_ids: dict[str, tuple[int, ...]] = {}
        self._records: dict[str, dict[str, dict[str, Any]]] = {}
        self._replay_route_cursor = 0
        self._replay_prompt_cursors: dict[str, int] = {}
        self._neutral_rows_observed = 0
        self._neutral_routes_observed = 0
        self._proposal_rows_admitted = 0
        self._proposal_rows_rejected_trust = 0
        self._proposal_graduations = 0
        self._cross_prompt_neutral_reproductions = 0
        self._post_replay_cross_prompt_neutral_reproductions = 0
        self._replayed_route_targets: set[tuple[str, str]] = set()
        self._cross_prompt_replay_updates = 0
        self._cross_prompt_replay_groups = 0
        self._cross_prompt_replay_rows = 0

    def _neutral_prompt_keys(self, route_key: str) -> set[str]:
        return {
            prompt_key
            for prompt_key, record in self._records.get(route_key, {}).items()
            if int(record["neutral_count"]) > 0
        }

    @property
    def distinct_route_count(self) -> int:
        return len(self._records)

    @property
    def recurring_route_count(self) -> int:
        return sum(
            len(self._neutral_prompt_keys(route_key))
            >= self.recurring_min_neutral_prompts
            for route_key in self._records
        )

    @property
    def distinct_source_prompt_count(self) -> int:
        return len(
            {prompt_key for records in self._records.values() for prompt_key in records}
        )

    def diagnostics(self) -> VerifiedRouteLibraryDiagnostics:
        return VerifiedRouteLibraryDiagnostics(
            neutral_rows_observed=self._neutral_rows_observed,
            neutral_routes_observed=self._neutral_routes_observed,
            proposal_rows_admitted=self._proposal_rows_admitted,
            proposal_rows_rejected_trust=(self._proposal_rows_rejected_trust),
            proposal_graduations=self._proposal_graduations,
            distinct_routes=self.distinct_route_count,
            recurring_routes=self.recurring_route_count,
            distinct_source_prompts=self.distinct_source_prompt_count,
            cross_prompt_neutral_reproductions=(
                self._cross_prompt_neutral_reproductions
            ),
            post_replay_cross_prompt_neutral_reproductions=(
                self._post_replay_cross_prompt_neutral_reproductions
            ),
            cross_prompt_replay_updates=self._cross_prompt_replay_updates,
            cross_prompt_replay_groups=self._cross_prompt_replay_groups,
            cross_prompt_replay_rows=self._cross_prompt_replay_rows,
        )

    def prompt_exemplars(
        self,
        prompt_token_ids: Sequence[int],
    ) -> tuple[VerifiedRouteExemplar, ...]:
        """Return immutable route records for one prompt in signature order."""

        prompt_key = _prompt_key(prompt_token_ids)
        exemplars: list[VerifiedRouteExemplar] = []
        for records in self._records.values():
            record = records.get(prompt_key)
            if record is None:
                continue
            mean_logprob = (
                record["neutral_mean_logprob"]
                if record["neutral_mean_logprob"] is not None
                else record["proposal_mean_logprob"]
            )
            if mean_logprob is None:
                raise RuntimeError("verified route exemplar lacks a model likelihood")
            exemplars.append(
                VerifiedRouteExemplar(
                    verifier=str(record["verifier"]),
                    endpoint_key=str(record["endpoint_key"]),
                    route_signature=str(record["route_signature"]),
                    response_token_ids=tuple(record["response_token_ids"]),
                    model_mean_logprob=float(mean_logprob),
                    neutral_reproduced=int(record["neutral_count"]) > 0,
                    proposal_observed=int(record["proposal_count"]) > 0,
                )
            )
        return tuple(sorted(exemplars, key=lambda value: value.route_signature))

    @staticmethod
    def _new_record(
        *,
        verifier: str,
        route_signature: str,
        endpoint_key: str,
        response_token_ids: tuple[int, ...],
        provenance: str,
        mean_logprob: float,
    ) -> dict[str, Any]:
        return {
            "verifier": verifier,
            "route_signature": route_signature,
            "endpoint_key": endpoint_key,
            "response_token_ids": response_token_ids,
            "first_provenance": provenance,
            "neutral_count": int(provenance == "neutral"),
            "proposal_count": int(provenance == "proposal"),
            "neutral_mean_logprob": (mean_logprob if provenance == "neutral" else None),
            "proposal_mean_logprob": (
                mean_logprob if provenance == "proposal" else None
            ),
            "proposal_anchor_mean_logprob": None,
        }

    def observe_neutral(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        verifier_ids: Sequence[str | None],
        endpoint_keys: Sequence[str | None],
        route_signatures: Sequence[str | None],
        response_token_ids: Sequence[Sequence[int]],
        model_mean_logprobs: Sequence[float],
        task_verified: Sequence[bool],
        active_mask: Sequence[bool | float],
    ) -> None:
        """Commit task-positive route observations from the neutral policy."""

        row_count = len(route_signatures)
        if not all(
            len(values) == row_count
            for values in (
                prompt_token_ids,
                verifier_ids,
                endpoint_keys,
                response_token_ids,
                model_mean_logprobs,
                task_verified,
                active_mask,
            )
        ):
            raise ValueError(
                "verified route neutral observation inputs must have equal lengths"
            )
        self._neutral_rows_observed += row_count
        for (
            raw_prompt,
            verifier,
            endpoint_key,
            route_signature,
            raw_response,
            raw_logprob,
            verified,
            active,
        ) in zip(
            prompt_token_ids,
            verifier_ids,
            endpoint_keys,
            route_signatures,
            response_token_ids,
            model_mean_logprobs,
            task_verified,
            active_mask,
        ):
            if not (
                bool(active)
                and bool(verified)
                and isinstance(verifier, str)
                and verifier
                and isinstance(endpoint_key, str)
                and endpoint_key
                and isinstance(route_signature, str)
                and route_signature
            ):
                continue
            prompt_tokens = _token_tuple(
                raw_prompt,
                label="verified route neutral prompt token ids",
            )
            response_tokens = _token_tuple(
                raw_response,
                label="verified route neutral response token ids",
            )
            mean_logprob = _finite_logprob(
                raw_logprob,
                label="verified route neutral mean log probability",
            )
            prompt_key = _prompt_key(prompt_tokens)
            route_key = _route_key(verifier, route_signature)
            stored_prompt = self._prompt_token_ids.get(prompt_key)
            if stored_prompt is not None and stored_prompt != prompt_tokens:
                raise ValueError("verified route prompt hash collision")
            self._prompt_token_ids[prompt_key] = prompt_tokens
            prior_neutral_prompts = self._neutral_prompt_keys(route_key)
            records = self._records.setdefault(route_key, {})
            record = records.get(prompt_key)
            if record is None:
                if len(records) >= self.replay_capacity_per_route:
                    # Proposal-only rows may never crowd independently
                    # reproduced neutral evidence out of a bounded route
                    # library. Evict one deterministically; if every retained
                    # source is already neutral, preserve the earlier fixed
                    # sample and drop this later observation.
                    proposal_only_keys = sorted(
                        key
                        for key, value in records.items()
                        if int(value["neutral_count"]) == 0
                    )
                    if not proposal_only_keys:
                        continue
                    evicted_prompt_key = proposal_only_keys[-1]
                    del records[evicted_prompt_key]
                    if not any(
                        evicted_prompt_key in other_records
                        for other_records in self._records.values()
                    ):
                        self._prompt_token_ids.pop(
                            evicted_prompt_key,
                            None,
                        )
                record = self._new_record(
                    verifier=verifier,
                    route_signature=route_signature,
                    endpoint_key=endpoint_key,
                    response_token_ids=response_tokens,
                    provenance="neutral",
                    mean_logprob=mean_logprob,
                )
                records[prompt_key] = record
                if prior_neutral_prompts:
                    self._cross_prompt_neutral_reproductions += 1
                    if (route_key, prompt_key) in self._replayed_route_targets:
                        self._post_replay_cross_prompt_neutral_reproductions += 1
            else:
                if (
                    record["verifier"] != verifier
                    or record["route_signature"] != route_signature
                ):
                    raise ValueError(
                        "verified route identity changed for one source prompt"
                    )
                stored_response_tokens = tuple(record["response_token_ids"])
                if (
                    stored_response_tokens == response_tokens
                    and record["endpoint_key"] != endpoint_key
                ):
                    raise ValueError(
                        "verified route endpoint changed for one response"
                    )
                if int(record["neutral_count"]) == 0:
                    self._proposal_graduations += 1
                    if prior_neutral_prompts - {prompt_key}:
                        self._cross_prompt_neutral_reproductions += 1
                        if (route_key, prompt_key) in (
                            self._replayed_route_targets
                        ):
                            self._post_replay_cross_prompt_neutral_reproductions += 1
                record["neutral_count"] = int(record["neutral_count"]) + 1
                record["neutral_mean_logprob"] = mean_logprob
                if response_tokens < stored_response_tokens:
                    record["response_token_ids"] = response_tokens
                    record["endpoint_key"] = endpoint_key
            self._neutral_routes_observed += 1

    def admit_proposal(
        self,
        *,
        prompt_token_ids: Sequence[int],
        verifier: str,
        endpoint_key: str,
        route_signature: str,
        response_token_ids: Sequence[int],
        proposal_mean_logprob: float,
        anchor_mean_logprob: float,
    ) -> VerifiedRouteProposalAdmission:
        """Admit one support-only proposal after an anchor-relative trust check."""

        prompt_tokens = _token_tuple(
            prompt_token_ids,
            label="verified route proposal prompt token ids",
        )
        response_tokens = _token_tuple(
            response_token_ids,
            label="verified route proposal response token ids",
        )
        if not isinstance(endpoint_key, str) or not endpoint_key:
            raise ValueError(
                "verified route proposal endpoint must be a non-empty string"
            )
        proposal_logprob = _finite_logprob(
            proposal_mean_logprob,
            label="verified route proposal mean log probability",
        )
        anchor_logprob = _finite_logprob(
            anchor_mean_logprob,
            label="verified route anchor mean log probability",
        )
        prompt_key = _prompt_key(prompt_tokens)
        route_key = _route_key(verifier, route_signature)
        if proposal_logprob < anchor_logprob - self.proposal_max_mean_logprob_drop:
            self._proposal_rows_rejected_trust += 1
            return VerifiedRouteProposalAdmission(
                admitted=False,
                rejected_trust=True,
                already_known=False,
                route_signature=route_signature,
                source_prompt_key=prompt_key,
            )
        records = self._records.setdefault(route_key, {})
        if prompt_key in records:
            return VerifiedRouteProposalAdmission(
                admitted=False,
                rejected_trust=False,
                already_known=True,
                route_signature=route_signature,
                source_prompt_key=prompt_key,
            )
        if len(records) >= self.replay_capacity_per_route:
            return VerifiedRouteProposalAdmission(
                admitted=False,
                rejected_trust=False,
                already_known=False,
                route_signature=route_signature,
                source_prompt_key=prompt_key,
            )
        stored_prompt = self._prompt_token_ids.get(prompt_key)
        if stored_prompt is not None and stored_prompt != prompt_tokens:
            raise ValueError("verified route prompt hash collision")
        self._prompt_token_ids[prompt_key] = prompt_tokens
        record = self._new_record(
            verifier=verifier,
            route_signature=route_signature,
            endpoint_key=endpoint_key,
            response_token_ids=response_tokens,
            provenance="proposal",
            mean_logprob=proposal_logprob,
        )
        record["proposal_anchor_mean_logprob"] = anchor_logprob
        records[prompt_key] = record
        self._proposal_rows_admitted += 1
        return VerifiedRouteProposalAdmission(
            admitted=True,
            rejected_trust=False,
            already_known=False,
            route_signature=route_signature,
            source_prompt_key=prompt_key,
        )

    def scheduled_cross_prompt_replay_groups(
        self,
        current_prompt_token_ids: Sequence[Sequence[int]],
    ) -> list[VerifiedCanonicalReplayGroup]:
        """Return fixed-budget recurring-route exemplars from other prompts."""

        if self.replay_groups_per_step == 0:
            return []
        current_prompt_keys = {
            _prompt_key(prompt) for prompt in current_prompt_token_ids
        }
        eligible_routes = sorted(
            route_key
            for route_key in self._records
            if len(self._neutral_prompt_keys(route_key))
            >= self.recurring_min_neutral_prompts
            and any(
                prompt_key not in current_prompt_keys
                and int(record["neutral_count"]) > 0
                for prompt_key, record in self._records[route_key].items()
            )
        )
        if not eligible_routes:
            return []
        count = min(self.replay_groups_per_step, len(eligible_routes))
        start = self._replay_route_cursor % len(eligible_routes)
        selected_routes = [
            eligible_routes[(start + offset) % len(eligible_routes)]
            for offset in range(count)
        ]
        self._replay_route_cursor = (start + count) % len(eligible_routes)
        groups: list[VerifiedCanonicalReplayGroup] = []
        for route_key in selected_routes:
            records = self._records[route_key]
            source_prompts = sorted(
                prompt_key
                for prompt_key, record in records.items()
                if prompt_key not in current_prompt_keys
                and int(record["neutral_count"]) > 0
            )
            if not source_prompts:
                continue
            prompt_cursor = self._replay_prompt_cursors.get(route_key, 0)
            prompt_key = source_prompts[prompt_cursor % len(source_prompts)]
            self._replay_prompt_cursors[route_key] = (prompt_cursor + 1) % len(
                source_prompts
            )
            record = records[prompt_key]
            groups.append(
                VerifiedCanonicalReplayGroup(
                    prompt_token_ids=self._prompt_token_ids[prompt_key],
                    outcome_keys=(str(record["route_signature"]),),
                    response_token_ids=(tuple(record["response_token_ids"]),),
                )
            )
        if groups:
            for route_key in selected_routes:
                for target_prompt_key in current_prompt_keys:
                    self._replayed_route_targets.add(
                        (route_key, target_prompt_key)
                    )
            self._cross_prompt_replay_updates += 1
            self._cross_prompt_replay_groups += len(groups)
            self._cross_prompt_replay_rows += sum(
                len(group.response_token_ids) for group in groups
            )
        return groups

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "verified_cross_prompt_route_library_v1",
            "replay_groups_per_step": self.replay_groups_per_step,
            "replay_capacity_per_route": self.replay_capacity_per_route,
            "recurring_min_neutral_prompts": (self.recurring_min_neutral_prompts),
            "proposal_max_mean_logprob_drop": (self.proposal_max_mean_logprob_drop),
            "prompt_token_ids": {
                key: list(value) for key, value in self._prompt_token_ids.items()
            },
            "records": {
                route_key: {
                    prompt_key: {
                        key: (list(value) if key == "response_token_ids" else value)
                        for key, value in record.items()
                    }
                    for prompt_key, record in records.items()
                }
                for route_key, records in self._records.items()
            },
            "replay_route_cursor": self._replay_route_cursor,
            "replay_prompt_cursors": dict(self._replay_prompt_cursors),
            "replayed_route_targets": [
                [route_key, prompt_key]
                for route_key, prompt_key in sorted(
                    self._replayed_route_targets
                )
            ],
            "counters": {
                "neutral_rows_observed": self._neutral_rows_observed,
                "neutral_routes_observed": self._neutral_routes_observed,
                "proposal_rows_admitted": self._proposal_rows_admitted,
                "proposal_rows_rejected_trust": (self._proposal_rows_rejected_trust),
                "proposal_graduations": self._proposal_graduations,
                "cross_prompt_neutral_reproductions": (
                    self._cross_prompt_neutral_reproductions
                ),
                "post_replay_cross_prompt_neutral_reproductions": (
                    self._post_replay_cross_prompt_neutral_reproductions
                ),
                "cross_prompt_replay_updates": (self._cross_prompt_replay_updates),
                "cross_prompt_replay_groups": (self._cross_prompt_replay_groups),
                "cross_prompt_replay_rows": self._cross_prompt_replay_rows,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore an exact route library and reject configuration drift."""

        if (
            not isinstance(state, dict)
            or state.get("schema") != "verified_cross_prompt_route_library_v1"
        ):
            raise ValueError("invalid verified route library state schema")
        for name in (
            "replay_groups_per_step",
            "replay_capacity_per_route",
            "recurring_min_neutral_prompts",
        ):
            if int(state.get(name, -1)) != int(getattr(self, name)):
                raise ValueError(f"verified route library resume mismatch for {name}")
        saved_drop = float(state.get("proposal_max_mean_logprob_drop", math.nan))
        if not math.isclose(
            saved_drop,
            self.proposal_max_mean_logprob_drop,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "verified route library resume mismatch for "
                "proposal_max_mean_logprob_drop"
            )
        raw_prompts = state.get("prompt_token_ids")
        raw_records = state.get("records")
        if not isinstance(raw_prompts, dict) or not isinstance(
            raw_records,
            dict,
        ):
            raise ValueError("verified route library state lacks record maps")
        restored_prompts = {
            str(prompt_key): _token_tuple(
                token_ids,
                label="restored verified route prompt token ids",
            )
            for prompt_key, token_ids in raw_prompts.items()
        }
        restored_records: dict[str, dict[str, dict[str, Any]]] = {}
        allowed_record_fields = {
            "verifier",
            "route_signature",
            "endpoint_key",
            "response_token_ids",
            "first_provenance",
            "neutral_count",
            "proposal_count",
            "neutral_mean_logprob",
            "proposal_mean_logprob",
            "proposal_anchor_mean_logprob",
        }
        for route_key, records in raw_records.items():
            if not isinstance(route_key, str) or not isinstance(records, dict):
                raise ValueError("verified route library state has invalid routes")
            if len(records) > self.replay_capacity_per_route:
                raise ValueError("verified route library state exceeds route capacity")
            restored_records[route_key] = {}
            for prompt_key, raw_record in records.items():
                if (
                    prompt_key not in restored_prompts
                    or not isinstance(raw_record, dict)
                    or set(raw_record) != allowed_record_fields
                ):
                    raise ValueError("verified route library state has invalid records")
                verifier = str(raw_record["verifier"])
                route_signature = str(raw_record["route_signature"])
                if _route_key(verifier, route_signature) != route_key:
                    raise ValueError("verified route library state route key mismatch")
                endpoint_key = str(raw_record["endpoint_key"])
                provenance = str(raw_record["first_provenance"])
                neutral_count = int(raw_record["neutral_count"])
                proposal_count = int(raw_record["proposal_count"])
                if (
                    not endpoint_key
                    or provenance not in {"neutral", "proposal"}
                    or neutral_count < 0
                    or proposal_count < 0
                    or neutral_count + proposal_count <= 0
                ):
                    raise ValueError(
                        "verified route library state has invalid provenance"
                    )
                normalized_record = dict(raw_record)
                normalized_record["response_token_ids"] = _token_tuple(
                    raw_record["response_token_ids"],
                    label="restored verified route response token ids",
                )
                normalized_record["neutral_count"] = neutral_count
                normalized_record["proposal_count"] = proposal_count
                for name in (
                    "neutral_mean_logprob",
                    "proposal_mean_logprob",
                    "proposal_anchor_mean_logprob",
                ):
                    value = raw_record[name]
                    normalized_record[name] = (
                        None
                        if value is None
                        else _finite_logprob(
                            value,
                            label=f"restored verified route {name}",
                        )
                    )
                restored_records[route_key][prompt_key] = normalized_record
        cursor = state.get("replay_route_cursor", 0)
        prompt_cursors = state.get("replay_prompt_cursors", {})
        raw_replayed_route_targets = state.get("replayed_route_targets", [])
        counters = state.get("counters", {})
        if (
            isinstance(cursor, bool)
            or int(cursor) != cursor
            or int(cursor) < 0
            or not isinstance(prompt_cursors, dict)
            or not isinstance(raw_replayed_route_targets, list)
            or not isinstance(counters, dict)
        ):
            raise ValueError("verified route library state has invalid scheduler state")
        restored_prompt_cursors: dict[str, int] = {}
        for route_key, value in prompt_cursors.items():
            if (
                route_key not in restored_records
                or isinstance(value, bool)
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError(
                    "verified route library state has invalid prompt cursor"
                )
            restored_prompt_cursors[route_key] = int(value)
        restored_replayed_route_targets: set[tuple[str, str]] = set()
        for raw_pair in raw_replayed_route_targets:
            if (
                not isinstance(raw_pair, list)
                or len(raw_pair) != 2
                or not all(isinstance(value, str) for value in raw_pair)
            ):
                raise ValueError(
                    "verified route library state has invalid replay targets"
                )
            route_key, prompt_key = raw_pair
            if (
                route_key not in restored_records
                or prompt_key not in restored_prompts
            ):
                raise ValueError(
                    "verified route library state replay target is unknown"
                )
            restored_replayed_route_targets.add((route_key, prompt_key))
        counter_names = (
            "neutral_rows_observed",
            "neutral_routes_observed",
            "proposal_rows_admitted",
            "proposal_rows_rejected_trust",
            "proposal_graduations",
            "cross_prompt_neutral_reproductions",
            "post_replay_cross_prompt_neutral_reproductions",
            "cross_prompt_replay_updates",
            "cross_prompt_replay_groups",
            "cross_prompt_replay_rows",
        )
        restored_counters: dict[str, int] = {}
        for name in counter_names:
            value = counters.get(name, 0)
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"verified route library state has invalid {name}")
            restored_counters[name] = int(value)
        self._prompt_token_ids = restored_prompts
        self._records = restored_records
        self._replay_route_cursor = int(cursor)
        self._replay_prompt_cursors = restored_prompt_cursors
        self._replayed_route_targets = restored_replayed_route_targets
        for name, value in restored_counters.items():
            setattr(self, f"_{name}", value)
