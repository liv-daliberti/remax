"""Conservative online strategy canonicalization for answer-matched MATH.

The task validator establishes final-answer equivalence, not proof validity.
The 72B judge therefore first rejects responses whose written reasoning does
not establish that answer, then partitions the remaining responses by
essential mathematical route. Two independently permuted, temperature-zero
partitions propose boundaries. Before any proposed boundary can create
novelty, two further reduced-set pairwise audits compare only one
representative per proposed component and retain a decisive rationale. A
same-route decision in either audit coarsens the boundary. Candidate
components touching one existing
representative reuse that key; components touching multiple representatives
are rejected. Disagreement therefore cannot manufacture novelty, and
ambiguity receives no key.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Sequence

from .math_strategy_menu import (
    MathStrategyMenu,
    parse_strategy_execution,
    parse_strategy_menu,
)


JUDGE_SYSTEM_PROMPT = (
    "You are a conservative mathematical reasoning-integrity and "
    "strategy-boundary auditor. "
    "Output valid JSON only."
)


class JudgeResponseFormatError(RuntimeError):
    """A judge response arrived but could not satisfy its strict schema."""


def _prompt_key(prompt_token_ids: Sequence[int]) -> str:
    normalized = [int(value) for value in prompt_token_ids]
    if not normalized or any(value < 0 for value in normalized):
        raise ValueError("prompt token ids must be non-empty and non-negative")
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _compact(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 4 // 5
    tail = max_chars - head
    return text[:head] + "\n[...middle elided...]\n" + text[-tail:]


def _extract_partition(
    content: str,
    expected_ids: set[str],
    *,
    missing_ids_are_ambiguous: bool = False,
) -> tuple[dict[str, str], set[str], dict[str, str]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    payload = json.loads(text)
    clusters = payload.get("clusters")
    ambiguous = payload.get("ambiguous_ids", [])
    if not isinstance(clusters, list) or not isinstance(ambiguous, list):
        raise ValueError("judge JSON lacks clusters/ambiguous_ids lists")
    assignment: dict[str, str] = {}
    strategies: dict[str, str] = {}
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict) or not isinstance(
            cluster.get("member_ids"), list
        ):
            raise ValueError("judge cluster lacks member_ids list")
        cluster_id = str(cluster.get("cluster_id", f"c{index + 1}"))
        strategies[cluster_id] = str(cluster.get("strategy", "")).strip()
        for raw_member in cluster["member_ids"]:
            member = str(raw_member)
            if member in assignment:
                raise ValueError("judge assigned an ID more than once")
            assignment[member] = cluster_id
    ambiguous_ids = {str(value) for value in ambiguous}
    if set(assignment) & ambiguous_ids:
        raise ValueError("judge both assigned and marked an ID ambiguous")
    observed = set(assignment) | ambiguous_ids
    if observed != expected_ids:
        extra = observed - expected_ids
        if extra:
            raise ValueError(
                f"judge returned unexpected IDs {sorted(extra)}"
            )
        missing = expected_ids - observed
        if missing_ids_are_ambiguous:
            ambiguous_ids.update(missing)
        else:
            raise ValueError(
                f"judge assignment mismatch missing={sorted(missing)} extra=[]"
            )
    return assignment, ambiguous_ids, strategies


def _extract_integrity(
    content: str,
    expected_ids: set[str],
) -> dict[str, str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    payload = json.loads(text)
    assessments = payload.get("assessments")
    if not isinstance(assessments, list):
        raise ValueError("integrity JSON lacks assessments list")
    statuses: dict[str, str] = {}
    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("integrity assessment is not an object")
        item_id = str(assessment.get("item_id", ""))
        status = str(assessment.get("status", ""))
        brief_check = assessment.get("brief_check")
        if item_id in statuses:
            raise ValueError("integrity judge assessed an ID more than once")
        if status not in {"valid", "invalid", "ambiguous"}:
            raise ValueError("integrity judge returned an invalid status")
        if not isinstance(brief_check, str) or not brief_check.strip():
            raise ValueError("integrity judge omitted its decisive check")
        statuses[item_id] = status
    if set(statuses) != expected_ids:
        raise ValueError(
            "integrity assessment mismatch "
            f"missing={sorted(expected_ids - set(statuses))} "
            f"extra={sorted(set(statuses) - expected_ids)}"
        )
    return statuses


def _extract_relations(
    content: str,
    expected_pair_ids: set[str],
) -> dict[str, str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    payload = json.loads(text)
    assessments = payload.get("assessments")
    if not isinstance(assessments, list):
        raise ValueError("relation JSON lacks assessments list")
    relations: dict[str, str] = {}
    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("relation assessment is not an object")
        pair_id = str(assessment.get("pair_id", ""))
        relation = str(assessment.get("relation", ""))
        brief_check = assessment.get("brief_check")
        if pair_id in relations:
            raise ValueError("relation judge assessed a pair twice")
        if relation not in {"same", "different", "ambiguous"}:
            raise ValueError("relation judge returned an invalid relation")
        if not isinstance(brief_check, str) or not brief_check.strip():
            raise ValueError("relation judge omitted its decisive comparison")
        relations[pair_id] = relation
    if set(relations) != expected_pair_ids:
        raise ValueError(
            "relation assessment mismatch "
            f"missing={sorted(expected_pair_ids - set(relations))} "
            f"extra={sorted(set(relations) - expected_pair_ids)}"
        )
    return relations


def _same(assignment: dict[str, str], left: str, right: str) -> bool:
    return assignment[left] == assignment[right]


@dataclass(frozen=True)
class MathStrategyDiagnostics:
    judge_calls: float = 0.0
    validator_positive_rows: float = 0.0
    accepted_rows: float = 0.0
    rejected_integrity_rows: float = 0.0
    rejected_ambiguous_rows: float = 0.0
    rejected_disagreement_rows: float = 0.0
    matched_existing_rows: float = 0.0
    new_strategy_rows: float = 0.0
    new_strategy_count: float = 0.0
    judge_format_failure_rows: float = 0.0
    rejected_contract_rows: float = 0.0
    inferred_unstructured_rows: float = 0.0
    rejected_strategy_inference_rows: float = 0.0


class MathStrategyCanonicalizer:
    """Persistent prompt-local 72B strategy representative bank."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str = "qwen2.5-72b",
        timeout_seconds: int = 600,
        max_workers: int = 4,
        permutation_seeds: tuple[int, int] = (470721, 470722),
        max_item_chars: int = 4000,
        missing_ids_are_ambiguous: bool = True,
        allow_unstructured_menu_inference: bool = False,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        endpoint = endpoint.rstrip("/")
        if not endpoint:
            raise ValueError("math strategy endpoint must not be empty")
        if len(permutation_seeds) != 2 or permutation_seeds[0] == permutation_seeds[1]:
            raise ValueError("exactly two distinct permutation seeds are required")
        if timeout_seconds <= 0 or max_workers <= 0 or max_item_chars < 600:
            raise ValueError("invalid canonicalizer resource bound")
        self.endpoint = endpoint
        self.model = str(model)
        self.timeout_seconds = int(timeout_seconds)
        self.max_workers = int(max_workers)
        self.permutation_seeds = tuple(int(value) for value in permutation_seeds)
        self.max_item_chars = int(max_item_chars)
        self.missing_ids_are_ambiguous = bool(missing_ids_are_ambiguous)
        self.allow_unstructured_menu_inference = bool(
            allow_unstructured_menu_inference
        )
        self._transport = transport
        self._representatives: dict[str, dict[str, dict[str, str]]] = {}
        self._next_strategy_id: dict[str, int] = {}
        self._menu_seen: dict[str, set[str]] = {}
        self._prompt_locks: dict[str, threading.Lock] = {}
        self._prompt_locks_guard = threading.Lock()
        self._judge_calls = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "math_strategy_canonicalizer_menu_bound_v18_trace",
            "model": self.model,
            "permutation_seeds": list(self.permutation_seeds),
            "max_item_chars": self.max_item_chars,
            "missing_ids_are_ambiguous": self.missing_ids_are_ambiguous,
            "allow_unstructured_menu_inference": (
                self.allow_unstructured_menu_inference
            ),
            "representatives": self._representatives,
            "next_strategy_id": self._next_strategy_id,
            "menu_seen": {
                prompt_key: sorted(strategy_keys)
                for prompt_key, strategy_keys in self._menu_seen.items()
            },
            "judge_calls": self._judge_calls,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            not isinstance(state, dict)
            or state.get("schema")
            != "math_strategy_canonicalizer_menu_bound_v18_trace"
        ):
            raise ValueError("invalid math strategy canonicalizer state")
        for name, configured in (
            ("model", self.model),
            ("permutation_seeds", list(self.permutation_seeds)),
            ("max_item_chars", self.max_item_chars),
            ("missing_ids_are_ambiguous", self.missing_ids_are_ambiguous),
            (
                "allow_unstructured_menu_inference",
                self.allow_unstructured_menu_inference,
            ),
        ):
            restored_value = state.get(
                name,
                False if name == "allow_unstructured_menu_inference" else None,
            )
            if restored_value != configured:
                raise ValueError(
                    f"math strategy canonicalizer resume mismatch for {name}"
                )
        representatives = state.get("representatives")
        next_ids = state.get("next_strategy_id")
        menu_seen = state.get("menu_seen", {})
        if (
            not isinstance(representatives, dict)
            or not isinstance(next_ids, dict)
            or not isinstance(menu_seen, dict)
        ):
            raise ValueError("invalid math strategy representative state")
        restored: dict[str, dict[str, dict[str, str]]] = {}
        for prompt_key, prompt_representatives in representatives.items():
            if not isinstance(prompt_key, str) or not isinstance(
                prompt_representatives, dict
            ):
                raise ValueError("invalid prompt-local representative bank")
            restored[prompt_key] = {}
            for strategy_key, record in prompt_representatives.items():
                if (
                    not isinstance(strategy_key, str)
                    or not isinstance(record, dict)
                    or not isinstance(record.get("text"), str)
                    or not isinstance(record.get("description"), str)
                ):
                    raise ValueError("invalid strategy representative")
                restored[prompt_key][strategy_key] = dict(record)
        self._representatives = restored
        self._next_strategy_id = {
            str(key): int(value) for key, value in next_ids.items()
        }
        restored_menu_seen: dict[str, set[str]] = {}
        for prompt_key, strategy_keys in menu_seen.items():
            if (
                not isinstance(prompt_key, str)
                or not isinstance(strategy_keys, list)
                or any(not isinstance(value, str) for value in strategy_keys)
            ):
                raise ValueError("invalid menu-bound strategy state")
            restored_menu_seen[prompt_key] = set(strategy_keys)
        self._menu_seen = restored_menu_seen
        self._judge_calls = int(state.get("judge_calls", 0))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(payload)
        request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Cluster-local judge traffic must not inherit login-node HTTP
        # proxies; node302 reaches node105 directly on the private fabric.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        error: Exception | None = None
        for attempt in range(3):
            try:
                with opener.open(
                    request, timeout=self.timeout_seconds
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"math strategy judge request failed after 3 attempts: {error}"
        ) from error

    def _integrity_messages(
        self,
        *,
        problem: str,
        items: list[tuple[str, str]],
    ) -> list[dict[str, str]]:
        rendered = "\n\n".join(
            f"### ID {item_id}\n{_compact(text, self.max_item_chars)}"
            for item_id, text in items
        )
        menu = parse_strategy_menu(problem)
        if menu is None:
            contract_rules = ""
        else:
            rendered_menu = "\n".join(
                f"- {strategy.strategy_id} = {strategy.action_combo}: "
                f"{strategy.plan}"
                for strategy in menu.strategies
            )
            action_definitions = "\n".join(
                f"- {action.action_id}: {action.operation}"
                for action in menu.actions
            )
            contract_rules = f"""

This problem also has a finite execution contract. The response header and
ordered <action_step> tags have already passed an exact machine parser, but
those tags alone prove nothing. For every block, compare its written
mathematics against that action's definition. Mark a response valid only if
every block materially and correctly executes its named action, the outputs
connect in order, and those blocks suffice to derive the boxed answer without
importing an omitted action or another strategy. Merely naming actions,
switching to another listed or unlisted route, skipping an essential
operation, silently doing it in a different block, or appending the declared
route as an irrelevant afterthought is invalid. Routine arithmetic internal
to a declared action is allowed.

ACTION DEFINITIONS:
{action_definitions}

VALID STRATEGIES:
{rendered_menu}
"""
        prompt = f"""Independently audit whether each answer-matched response
contains a mathematically valid written derivation of its final answer.

An answer-only checker matched the final answer. It did NOT verify the
reasoning. For every response, check the actual algebra, arithmetic, logical
inferences, domain restrictions, and necessary cases. Do not assume a path is
valid because its answer matches. Do not repair or replace a flawed path.

Status rules:
- valid: the written route, allowing only omitted routine arithmetic, actually
  establishes the answer;
- invalid: a material false statement, invalid inference, contradiction,
  missing necessary case, or computation error breaks the derivation;
- ambiguous: truncation or insufficient text prevents a reliable decision.

For every opaque ID, first perform the mathematical check and return one
brief_check naming either a decisive verification or the first material
error, then return the status. A longer case proof is valid when its
calculations and case restrictions are correct even if a shorter proof exists.
Be conservative: uncertainty is ambiguous, never valid.
{contract_rules}

PROBLEM:
{problem}

ANSWER-MATCHED RESPONSES (opaque IDs, randomly permuted):
{rendered}
"""
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def _menu_strategy_key(
        menu: MathStrategyMenu,
        strategy_id: str,
    ) -> str:
        strategy = menu.strategy(strategy_id)
        if strategy is None:
            raise ValueError("unknown menu strategy")
        return (
            f"math_menu_{menu.sha256[:16]}_"
            f"{strategy.strategy_id}_{strategy.action_combo}"
        )

    def _canonicalize_menu_group_locked(
        self,
        *,
        prompt_key: str,
        problem: str,
        menu: MathStrategyMenu,
        candidate_rows: list[tuple[int, str]],
    ) -> tuple[dict[int, str], MathStrategyDiagnostics]:
        candidate_ids = {
            f"CAND_{index:04d}": row_index
            for index, (row_index, _) in enumerate(candidate_rows)
        }
        declarations = {}
        candidate_text = {}
        rejected_contract: set[str] = set()
        for index, candidate_id in enumerate(candidate_ids):
            text = candidate_rows[index][1]
            declaration = parse_strategy_execution(text, menu)
            if declaration is None:
                rejected_contract.add(candidate_id)
                continue
            declarations[candidate_id] = declaration
            candidate_text[candidate_id] = text

        inferred_text = {
            candidate_id: candidate_rows[index][1]
            for index, candidate_id in enumerate(candidate_ids)
            if candidate_id in rejected_contract
        }
        inferred_declarations: dict[str, str] = {}
        inferred_diagnostics = MathStrategyDiagnostics()
        if self.allow_unstructured_menu_inference and inferred_text:
            inferred_declarations, inferred_diagnostics = (
                self._infer_unstructured_menu_routes(
                    problem=problem,
                    menu=menu,
                    candidate_text=inferred_text,
                )
            )
            rejected_contract -= set(inferred_declarations)

        if not candidate_text and not inferred_declarations:
            return {}, MathStrategyDiagnostics(
                validator_positive_rows=float(len(candidate_rows)),
                rejected_contract_rows=float(len(rejected_contract)),
                judge_calls=inferred_diagnostics.judge_calls,
                rejected_ambiguous_rows=(
                    inferred_diagnostics.rejected_ambiguous_rows
                ),
                judge_format_failure_rows=(
                    inferred_diagnostics.judge_format_failure_rows
                ),
                rejected_strategy_inference_rows=(
                    inferred_diagnostics.rejected_strategy_inference_rows
                ),
            )

        rejected_integrity: set[str] = set()
        rejected_ambiguous: set[str] = set()
        if candidate_text:
            candidate_items = list(candidate_text.items())
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        self._judge_integrity,
                        problem=problem,
                        base_items=candidate_items,
                        seed=seed,
                    )
                    for seed in self.permutation_seeds
                ]
                try:
                    integrity_passes = [future.result() for future in futures]
                except JudgeResponseFormatError:
                    return {}, MathStrategyDiagnostics(
                        judge_calls=(
                            2.0 + inferred_diagnostics.judge_calls
                        ),
                        validator_positive_rows=float(len(candidate_rows)),
                        rejected_ambiguous_rows=float(len(candidate_text))
                        + inferred_diagnostics.rejected_ambiguous_rows,
                        judge_format_failure_rows=float(len(candidate_text))
                        + inferred_diagnostics.judge_format_failure_rows,
                        rejected_contract_rows=float(len(rejected_contract)),
                        inferred_unstructured_rows=(
                            inferred_diagnostics.inferred_unstructured_rows
                        ),
                        rejected_strategy_inference_rows=(
                            inferred_diagnostics.rejected_strategy_inference_rows
                        ),
                    )

            first, second = integrity_passes
            rejected_integrity = {
                candidate_id
                for candidate_id in candidate_text
                if "invalid" in {first[candidate_id], second[candidate_id]}
            }
            rejected_ambiguous = {
                candidate_id
                for candidate_id in candidate_text
                if (
                    candidate_id not in rejected_integrity
                    and (
                        first[candidate_id] != "valid"
                        or second[candidate_id] != "valid"
                    )
                )
            }
        accepted_ids = (
            set(candidate_text) - rejected_integrity - rejected_ambiguous
        )
        seen = self._menu_seen.setdefault(prompt_key, set())
        accepted: dict[int, str] = {}
        new_rows = 0
        matched_existing = 0
        new_keys: set[str] = set()
        for candidate_id in sorted(accepted_ids):
            declaration = declarations[candidate_id]
            key = self._menu_strategy_key(menu, declaration.strategy_id)
            accepted[candidate_ids[candidate_id]] = key
            if key in seen:
                matched_existing += 1
            else:
                new_rows += 1
                new_keys.add(key)
        for candidate_id in sorted(inferred_declarations):
            strategy_id = inferred_declarations[candidate_id]
            key = self._menu_strategy_key(menu, strategy_id)
            accepted[candidate_ids[candidate_id]] = key
            if key in seen:
                matched_existing += 1
            else:
                new_rows += 1
                new_keys.add(key)
        seen.update(new_keys)
        return accepted, MathStrategyDiagnostics(
            judge_calls=(2.0 if candidate_text else 0.0)
            + inferred_diagnostics.judge_calls,
            validator_positive_rows=float(len(candidate_rows)),
            accepted_rows=float(len(accepted)),
            rejected_integrity_rows=float(len(rejected_integrity)),
            rejected_ambiguous_rows=float(len(rejected_ambiguous))
            + inferred_diagnostics.rejected_ambiguous_rows,
            matched_existing_rows=float(matched_existing),
            new_strategy_rows=float(new_rows),
            new_strategy_count=float(len(new_keys)),
            rejected_contract_rows=float(len(rejected_contract)),
            inferred_unstructured_rows=(
                inferred_diagnostics.inferred_unstructured_rows
            ),
            rejected_strategy_inference_rows=(
                inferred_diagnostics.rejected_strategy_inference_rows
            ),
        )

    def _infer_unstructured_menu_routes(
        self,
        *,
        problem: str,
        menu: MathStrategyMenu,
        candidate_text: dict[str, str],
    ) -> tuple[dict[str, str], MathStrategyDiagnostics]:
        items = list(candidate_text.items())
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    self._judge_menu_inference,
                    problem=problem,
                    menu=menu,
                    base_items=items,
                    seed=seed,
                )
                for seed in self.permutation_seeds
            ]
            try:
                passes = [future.result() for future in futures]
            except JudgeResponseFormatError:
                return {}, MathStrategyDiagnostics(
                    judge_calls=2.0,
                    rejected_ambiguous_rows=float(len(candidate_text)),
                    judge_format_failure_rows=float(len(candidate_text)),
                    rejected_strategy_inference_rows=float(len(candidate_text)),
                )
        first, second = passes
        inferred = {}
        rejected = 0
        for candidate_id in candidate_text:
            first_status, first_strategy = first[candidate_id]
            second_status, second_strategy = second[candidate_id]
            if (
                first_status == "valid"
                and second_status == "valid"
                and first_strategy == second_strategy
                and menu.strategy(first_strategy) is not None
            ):
                inferred[candidate_id] = first_strategy
            else:
                rejected += 1
        return inferred, MathStrategyDiagnostics(
            judge_calls=2.0,
            rejected_ambiguous_rows=float(rejected),
            inferred_unstructured_rows=float(len(inferred)),
            rejected_strategy_inference_rows=float(rejected),
        )

    def _judge_menu_inference(
        self,
        *,
        problem: str,
        menu: MathStrategyMenu,
        base_items: list[tuple[str, str]],
        seed: int,
    ) -> dict[str, tuple[str, str]]:
        items = list(base_items)
        random.Random(seed).shuffle(items)
        expected_ids = sorted(item_id for item_id, _ in base_items)
        strategy_ids = [strategy.strategy_id for strategy in menu.strategies]
        action_definitions = "\n".join(
            f"- {action.action_id}: {action.operation}"
            for action in menu.actions
        )
        strategies = "\n".join(
            f"- {strategy.strategy_id} = {strategy.action_combo}: "
            f"{strategy.plan}"
            for strategy in menu.strategies
        )
        rendered = "\n\n".join(
            f"### ID {item_id}\n{_compact(text, self.max_item_chars)}"
            for item_id, text in items
        )
        prompt = f"""Map each answer-matched mathematical derivation to one
frozen finite-menu route, or reject it. The response may omit XML tags, so
judge its actual written mathematics rather than its formatting.

The answer-only checker matched the final answer but did not verify reasoning.
Return status=valid and one strategy_id only when the derivation materially,
correctly, and sufficiently executes every action in that strategy's exact
combo, in a logically compatible order, without switching to another route.
Routine arithmetic internal to an action is allowed. Merely mentioning an
operation, giving only the answer, leaving an essential operation undone,
using an unlisted route, combining multiple routes ambiguously, or containing
a material mathematical error is not valid. If the response explicitly
claims a strategy ID or action combo, that declaration must match the route
its mathematics actually executes; a mismatched declaration is invalid even
when the mathematics would otherwise solve the problem. Use invalid for a
decisive flaw and ambiguous when the text is insufficient. For every
non-valid item return strategy_id=NONE. Do not repair a response.

ACTION DEFINITIONS:
{action_definitions}

VALID STRATEGIES:
{strategies}

PROBLEM:
{problem}

ANSWER-MATCHED RESPONSES:
{rendered}
"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096,
            "seed": seed,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "math_menu_route_inference",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["assessments"],
                        "properties": {
                            "assessments": {
                                "type": "array",
                                "minItems": len(expected_ids),
                                "maxItems": len(expected_ids),
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "item_id",
                                        "brief_check",
                                        "status",
                                        "strategy_id",
                                    ],
                                    "properties": {
                                        "item_id": {
                                            "type": "string",
                                            "enum": expected_ids,
                                        },
                                        "brief_check": {"type": "string"},
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "valid",
                                                "invalid",
                                                "ambiguous",
                                            ],
                                        },
                                        "strategy_id": {
                                            "type": "string",
                                            "enum": [*strategy_ids, "NONE"],
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        response = self._post(payload)
        choices = response.get("choices") or []
        if not choices:
            raise JudgeResponseFormatError(
                "menu inference judge returned no choices"
            )
        content = str((choices[0].get("message") or {}).get("content") or "")
        self._judge_calls += 1
        try:
            parsed = json.loads(content)
            assessments = parsed.get("assessments")
            if not isinstance(assessments, list):
                raise ValueError("missing assessments")
            results = {}
            for assessment in assessments:
                item_id = assessment.get("item_id")
                status = assessment.get("status")
                strategy_id = assessment.get("strategy_id")
                brief_check = assessment.get("brief_check")
                if (
                    item_id in results
                    or item_id not in expected_ids
                    or status not in {"valid", "invalid", "ambiguous"}
                    or strategy_id not in {*strategy_ids, "NONE"}
                    or not isinstance(brief_check, str)
                    or not brief_check.strip()
                    or (
                        status == "valid"
                        and strategy_id == "NONE"
                    )
                    or (
                        status != "valid"
                        and strategy_id != "NONE"
                    )
                ):
                    raise ValueError("invalid menu inference assessment")
                results[item_id] = (status, strategy_id)
            if set(results) != set(expected_ids):
                raise ValueError("menu inference coverage mismatch")
            return results
        except (json.JSONDecodeError, ValueError) as exc:
            raise JudgeResponseFormatError(
                "menu inference judge returned malformed content"
            ) from exc

    def _judge_integrity(
        self,
        *,
        problem: str,
        base_items: list[tuple[str, str]],
        seed: int,
    ) -> dict[str, str]:
        items = list(base_items)
        random.Random(seed).shuffle(items)
        expected_ids = sorted(item_id for item_id, _ in base_items)
        payload = {
            "model": self.model,
            "messages": self._integrity_messages(
                problem=problem,
                items=items,
            ),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096,
            "seed": seed,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "math_derivation_integrity",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["assessments"],
                        "properties": {
                            "assessments": {
                                "type": "array",
                                "minItems": len(expected_ids),
                                "maxItems": len(expected_ids),
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "item_id",
                                        "brief_check",
                                        "status",
                                    ],
                                    "properties": {
                                        "item_id": {
                                            "type": "string",
                                            "enum": expected_ids,
                                        },
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "valid",
                                                "invalid",
                                                "ambiguous",
                                            ],
                                        },
                                        "brief_check": {
                                            "type": "string",
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        response = self._post(payload)
        choices = response.get("choices") or []
        if not choices:
            raise JudgeResponseFormatError(
                "math integrity judge returned no choices"
            )
        content = str((choices[0].get("message") or {}).get("content") or "")
        self._judge_calls += 1
        try:
            return _extract_integrity(content, set(expected_ids))
        except (json.JSONDecodeError, ValueError) as exc:
            raise JudgeResponseFormatError(
                "math integrity judge returned malformed content"
            ) from exc

    def _messages(
        self,
        *,
        problem: str,
        items: list[tuple[str, str]],
        equivalence_veto: bool = False,
    ) -> list[dict[str, str]]:
        rendered = "\n\n".join(
            f"### ID {item_id}\n{_compact(text, self.max_item_chars)}"
            for item_id, text in items
        )
        if equivalence_veto:
            task = """This is a focused NOVELTY-VETO pass over a reduced
set containing only one representative of each provisionally distinct
component. Decide whether each proposed distinction is only a routine
reparameterization or is a substantively different decisive method. Merge
only when you can identify the shared decisive route; matching answers or
algebraically equal final quantities are not enough.

Keep two responses together when they apply the same decisive identity,
theorem, invariant, or construction to the same quantities and differ only
in prose, routine algebra, arithmetic detail, or irrelevant afterthoughts.
Different regroupings of one rate-times-time,
distance-per-event-times-count, or unit-conversion computation are the same
route. A verbal description versus its corresponding formula is the same
route.

Hard separation rule: explicitly expanding a polynomial's coefficients and
avoiding expansion by evaluating the polynomial at a strategically chosen
point are different routes, even though both compute the same requested
quantity. Likewise keep direct contradiction separate from a
case/discriminant proof, synthetic from coordinate geometry, and direct from
complementary counting. Do not erase a genuinely different central operation
merely because both proofs can be rewritten into a common final equality."""
        else:
            task = """Use the finest defensible partition under the rules
below. This pass proposes strategy boundaries, which a later reduced-set
novelty veto may conservatively coarsen."""
        if equivalence_veto:
            output_example = (
                '{"clusters":[{"cluster_id":"c1",'
                '"member_ids":["opaque id"],'
                '"strategy":"brief decisive shared route"}],'
                '"ambiguous_ids":[]}'
            )
            output_rule = (
                "For every veto cluster, strategy must briefly name the "
                "decisive route shared by every member; this rationale is "
                "required before merging IDs."
            )
        else:
            output_example = (
                '{"clusters":[{"cluster_id":"c1",'
                '"member_ids":["opaque id"]}],"ambiguous_ids":[]}'
            )
            output_rule = (
                "Do not add strategy descriptions, mathematical notation, "
                "or extra fields."
            )
        prompt = f"""Audit and partition the answer-matched candidate responses
below by their ESSENTIAL MATHEMATICAL STRATEGY for solving the stated
problem.

{task}

IMPORTANT: an answer-only checker matched each final answer. It did NOT verify
the written derivation. Before clustering, put an ID in ambiguous_ids if its
reasoning has a material false inference, an unhandled case, a contradiction,
or otherwise does not actually establish its answer. Do not repair, excuse,
or infer a valid proof that the response did not write. A shared final answer
is never evidence that two routes are equivalent.

For the remaining coherent responses:

Same cluster: rewording, formatting or notation changes, reordered routine
algebra, or expansion/omission of routine steps while retaining the same
central mathematical route. Algebraically equivalent parameterizations of the
same computation are also the same route: for example, per-event distance
times event count versus per-time speed times elapsed time, regrouping the
same factors, or substituting an equivalent closed formula.

Different cluster: a genuinely different central construction, theorem,
substitution, invariant, case decomposition, counting argument, or proof
route. In particular, a direct sign/bound/invariant contradiction is
different from expanding an expression and resolving cases, roots, or
discriminants; synthetic and coordinate constructions are different; direct
and complementary counting arguments are different; explicitly expanding
coefficients is different from avoiding expansion by evaluating a function
at a strategically chosen point. These are examples of a general rule, not
problem-specific hints. A routine change of units or algebraic reassociation
is not a different proof route. Do not split merely because prose, LaTeX,
variable names, arithmetic detail, or the order of routine steps differs.

If the text is insufficient to decide either integrity or the essential
route, put its ID in ambiguous_ids. Ambiguity earns no strategy key.

Return one JSON object only:
{output_example}
Every non-ambiguous ID must occur exactly once.
{output_rule}

PROBLEM:
{problem}

SOLUTIONS (opaque IDs, randomly permuted):
{rendered}
"""
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def _judge_partition(
        self,
        *,
        problem: str,
        base_items: list[tuple[str, str]],
        seed: int,
        equivalence_veto: bool = False,
    ) -> tuple[dict[str, str], set[str], dict[str, str]]:
        items = list(base_items)
        random.Random(seed).shuffle(items)
        expected_ids = sorted(item_id for item_id, _ in base_items)
        cluster_required = ["cluster_id", "member_ids"]
        cluster_properties: dict[str, Any] = {
            "cluster_id": {
                "type": "string",
                "enum": [
                    f"c{index + 1}"
                    for index in range(len(expected_ids))
                ],
            },
            "member_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(expected_ids),
                "items": {
                    "type": "string",
                    "enum": expected_ids,
                },
            },
        }
        if equivalence_veto:
            cluster_required.append("strategy")
            cluster_properties["strategy"] = {
                "type": "string",
                "minLength": 1,
            }
        payload = {
            "model": self.model,
            "messages": self._messages(
                problem=problem,
                items=items,
                equivalence_veto=equivalence_veto,
            ),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096,
            "seed": seed,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "math_strategy_partition",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["clusters", "ambiguous_ids"],
                        "properties": {
                            "clusters": {
                                "type": "array",
                                "maxItems": len(expected_ids),
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": cluster_required,
                                    "properties": cluster_properties,
                                },
                            },
                            "ambiguous_ids": {
                                "type": "array",
                                "maxItems": len(expected_ids),
                                "items": {
                                    "type": "string",
                                    "enum": expected_ids,
                                },
                            },
                        },
                    },
                },
            },
        }
        response = self._post(payload)
        choices = response.get("choices") or []
        if not choices:
            raise JudgeResponseFormatError(
                "math strategy judge returned no choices"
            )
        content = str((choices[0].get("message") or {}).get("content") or "")
        self._judge_calls += 1
        try:
            return _extract_partition(
                content,
                set(expected_ids),
                missing_ids_are_ambiguous=self.missing_ids_are_ambiguous,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise JudgeResponseFormatError(
                "math strategy judge returned malformed content"
            ) from exc

    def _relation_messages(
        self,
        *,
        problem: str,
        items: list[tuple[str, str]],
        pairs: list[tuple[str, str, str]],
    ) -> list[dict[str, str]]:
        rendered_items = "\n\n".join(
            f"### ID {item_id}\n{_compact(text, self.max_item_chars)}"
            for item_id, text in items
        )
        rendered_pairs = "\n".join(
            f"{pair_id}: {left_id} versus {right_id}"
            for pair_id, left_id, right_id in pairs
        )
        prompt = f"""Act as a conservative pairwise novelty-boundary auditor.
Every item below has already passed two derivation-integrity checks. For each
listed pair, decide whether the two responses use the SAME essential
mathematical route or DIFFERENT decisive routes.

Relation rules:
- same: the same decisive identity, theorem, invariant, construction, or
  computation applied to the same quantities, differing only in prose,
  routine algebra, arithmetic detail, irrelevant afterthoughts, unit
  conversion, or regrouping such as distance-per-event times count versus
  speed times elapsed time;
- different: a genuinely different central operation or proof route;
- ambiguous: the texts do not permit a reliable boundary decision.

Matching answers or a common final equality are not evidence of sameness.
Explicit coefficient expansion and avoiding expansion by strategic point
evaluation are different. Direct contradiction and case/discriminant
analysis are different. Synthetic and coordinate geometry are different.
Direct and complementary counting are different.

For each opaque pair ID, first compare the decisive operations in a nonempty
brief_check, then output relation. If the brief_check itself names different
central operations (for example, expansion versus evaluation), relation must
be different. If it names only routine reparameterizations of one
calculation (for example, revolutions versus speed and time), relation must be
same. Do not create, omit, or duplicate pair IDs.

PROBLEM:
{problem}

ITEMS:
{rendered_items}

REQUIRED PAIRS:
{rendered_pairs}
"""
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def _judge_relations(
        self,
        *,
        problem: str,
        base_items: list[tuple[str, str]],
        pairs: list[tuple[str, str, str]],
        seed: int,
    ) -> dict[str, str]:
        items = list(base_items)
        rendered_pairs = list(pairs)
        rng = random.Random(seed)
        rng.shuffle(items)
        rng.shuffle(rendered_pairs)
        expected_pair_ids = sorted(pair_id for pair_id, _, _ in pairs)
        payload = {
            "model": self.model,
            "messages": self._relation_messages(
                problem=problem,
                items=items,
                pairs=rendered_pairs,
            ),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096,
            "seed": seed,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "math_strategy_pair_relations",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["assessments"],
                        "properties": {
                            "assessments": {
                                "type": "array",
                                "minItems": len(expected_pair_ids),
                                "maxItems": len(expected_pair_ids),
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "pair_id",
                                        "brief_check",
                                        "relation",
                                    ],
                                    "properties": {
                                        "pair_id": {
                                            "type": "string",
                                            "enum": expected_pair_ids,
                                        },
                                        "brief_check": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                        "relation": {
                                            "type": "string",
                                            "enum": [
                                                "same",
                                                "different",
                                                "ambiguous",
                                            ],
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        response = self._post(payload)
        choices = response.get("choices") or []
        if not choices:
            raise JudgeResponseFormatError(
                "math relation judge returned no choices"
            )
        content = str((choices[0].get("message") or {}).get("content") or "")
        self._judge_calls += 1
        try:
            return _extract_relations(content, set(expected_pair_ids))
        except (json.JSONDecodeError, ValueError) as exc:
            raise JudgeResponseFormatError(
                "math relation judge returned malformed content"
            ) from exc

    def _canonicalize_group(
        self,
        *,
        prompt_tokens: Sequence[int],
        problem: str,
        candidate_rows: list[tuple[int, str]],
    ) -> tuple[dict[int, str], MathStrategyDiagnostics]:
        if not candidate_rows:
            return {}, MathStrategyDiagnostics()
        prompt_key = _prompt_key(prompt_tokens)
        with self._prompt_locks_guard:
            prompt_lock = self._prompt_locks.setdefault(
                prompt_key, threading.Lock()
            )
        with prompt_lock:
            return self._canonicalize_group_locked(
                prompt_key=prompt_key,
                problem=problem,
                candidate_rows=candidate_rows,
            )

    def _canonicalize_group_locked(
        self,
        *,
        prompt_key: str,
        problem: str,
        candidate_rows: list[tuple[int, str]],
    ) -> tuple[dict[int, str], MathStrategyDiagnostics]:
        menu = parse_strategy_menu(problem)
        if menu is not None:
            return self._canonicalize_menu_group_locked(
                prompt_key=prompt_key,
                problem=problem,
                menu=menu,
                candidate_rows=candidate_rows,
            )
        representatives = self._representatives.setdefault(prompt_key, {})
        rep_ids = {
            f"REP_{index:04d}": strategy_key
            for index, strategy_key in enumerate(sorted(representatives))
        }
        candidate_ids = {
            f"CAND_{index:04d}": row_index
            for index, (row_index, _) in enumerate(candidate_rows)
        }
        candidate_text = {
            candidate_id: candidate_rows[index][1]
            for index, candidate_id in enumerate(candidate_ids)
        }

        candidate_items = list(candidate_text.items())
        with ThreadPoolExecutor(max_workers=2) as pool:
            integrity_futures = [
                pool.submit(
                    self._judge_integrity,
                    problem=problem,
                    base_items=candidate_items,
                    seed=seed,
                )
                for seed in self.permutation_seeds
            ]
            try:
                integrity_passes = [
                    future.result() for future in integrity_futures
                ]
            except JudgeResponseFormatError:
                return {}, MathStrategyDiagnostics(
                    judge_calls=2.0,
                    validator_positive_rows=float(len(candidate_rows)),
                    rejected_ambiguous_rows=float(len(candidate_rows)),
                    judge_format_failure_rows=float(len(candidate_rows)),
                )
        first_integrity, second_integrity = integrity_passes
        rejected_integrity = {
            candidate_id
            for candidate_id in candidate_ids
            if (
                first_integrity[candidate_id] != "valid"
                or second_integrity[candidate_id] != "valid"
            )
        }
        eligible_candidate_text = {
            candidate_id: text
            for candidate_id, text in candidate_text.items()
            if candidate_id not in rejected_integrity
        }
        if not eligible_candidate_text:
            return {}, MathStrategyDiagnostics(
                judge_calls=2.0,
                validator_positive_rows=float(len(candidate_rows)),
                rejected_integrity_rows=float(len(rejected_integrity)),
            )

        items = [
            (rep_id, representatives[strategy_key]["text"])
            for rep_id, strategy_key in rep_ids.items()
        ] + list(eligible_candidate_text.items())

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    self._judge_partition,
                    problem=problem,
                    base_items=items,
                    seed=seed,
                )
                for seed in self.permutation_seeds
            ]
            try:
                partitions = [future.result() for future in futures]
            except JudgeResponseFormatError:
                return {}, MathStrategyDiagnostics(
                    judge_calls=4.0,
                    validator_positive_rows=float(len(candidate_rows)),
                    rejected_integrity_rows=float(len(rejected_integrity)),
                    rejected_ambiguous_rows=float(
                        len(eligible_candidate_text)
                    ),
                    judge_format_failure_rows=float(
                        len(eligible_candidate_text)
                    ),
                )
        (first, first_ambiguous, first_descriptions), (
            second,
            second_ambiguous,
            _,
        ) = partitions

        ambiguous_reps = {
            rep_id
            for rep_id in rep_ids
            if (
                rep_id in first_ambiguous
                or rep_id in second_ambiguous
            )
        }

        rejected_ambiguous: set[str] = set()
        rejected_disagreement: set[str] = set()
        accepted: dict[int, str] = {}
        matched_existing = 0
        active_candidates = []
        for candidate_id in sorted(candidate_ids):
            if candidate_id in rejected_integrity:
                continue
            if (
                candidate_id in first_ambiguous
                or candidate_id in second_ambiguous
            ):
                rejected_ambiguous.add(candidate_id)
                continue
            active_candidates.append(candidate_id)

        # The full-context partitions are good at proposing fine boundaries,
        # but calibration showed that surrounding solutions can make them
        # split two routine parameterizations that merge when directly
        # compared. Collapse each proposed candidate component to one anchor
        # and run two explicitly reasoned pairwise novelty-veto audits. Any
        # same-route edge in either reduced pass coarsens the final component.
        primary_anchor: dict[str, str] = {}
        primary_pending = set(active_candidates)
        while primary_pending:
            anchor = min(primary_pending)
            component = {anchor}
            frontier = [anchor]
            while frontier:
                current = frontier.pop()
                neighbors = {
                    node
                    for node in primary_pending - component
                    if _same(first, current, node)
                    or _same(second, current, node)
                }
                component.update(neighbors)
                frontier.extend(sorted(neighbors))
            primary_pending -= component
            for candidate_id in component:
                primary_anchor[candidate_id] = anchor

        reduced_candidate_anchors = sorted(set(primary_anchor.values()))
        reduced_items = [
            (rep_id, representatives[strategy_key]["text"])
            for rep_id, strategy_key in rep_ids.items()
        ] + [
            (candidate_id, candidate_text[candidate_id])
            for candidate_id in reduced_candidate_anchors
        ]
        relation_node_pairs = list(
            combinations(reduced_candidate_anchors, 2)
        ) + [
            (candidate_id, rep_id)
            for candidate_id in reduced_candidate_anchors
            for rep_id in sorted(rep_ids)
        ]
        relation_pairs = [
            (f"PAIR_{index:04d}", left, right)
            for index, (left, right) in enumerate(relation_node_pairs)
        ]
        pair_id_for_nodes = {
            tuple(sorted((left, right))): pair_id
            for pair_id, left, right in relation_pairs
        }
        relation_first: dict[str, str] = {}
        relation_second: dict[str, str] = {}
        relation_uncertain_anchors: set[str] = set()
        judge_calls = 4.0
        if relation_pairs and len(relation_pairs) <= 128:
            with ThreadPoolExecutor(max_workers=2) as pool:
                relation_futures = [
                    pool.submit(
                        self._judge_relations,
                        problem=problem,
                        base_items=reduced_items,
                        pairs=relation_pairs,
                        seed=seed,
                    )
                    for seed in self.permutation_seeds
                ]
                try:
                    relation_passes = [
                        future.result() for future in relation_futures
                    ]
                except JudgeResponseFormatError:
                    return {}, MathStrategyDiagnostics(
                        judge_calls=6.0,
                        validator_positive_rows=float(len(candidate_rows)),
                        rejected_integrity_rows=float(
                            len(rejected_integrity)
                        ),
                        rejected_ambiguous_rows=float(
                            len(eligible_candidate_text)
                        ),
                        judge_format_failure_rows=float(
                            len(eligible_candidate_text)
                        ),
                    )
            relation_first, relation_second = relation_passes
            judge_calls = 6.0
            for pair_id, left, right in relation_pairs:
                first_relation = relation_first[pair_id]
                second_relation = relation_second[pair_id]
                if (
                    "same" not in {first_relation, second_relation}
                    and "ambiguous" in {first_relation, second_relation}
                ):
                    if left in primary_anchor:
                        relation_uncertain_anchors.add(left)
                    if right in primary_anchor:
                        relation_uncertain_anchors.add(right)
        elif len(relation_pairs) > 128:
            # A quadratic comparison explosion cannot silently authorize
            # novelty. Existing matches from the full partitions remain
            # reusable, but every proposed new component fails closed.
            relation_uncertain_anchors.update(reduced_candidate_anchors)

        def veto_node(candidate_id: str) -> str:
            return primary_anchor[candidate_id]

        def veto_same(left: str, right: str) -> bool:
            left_node = veto_node(left) if left in primary_anchor else left
            right_node = veto_node(right) if right in primary_anchor else right
            if left_node == right_node:
                return True
            pair_id = pair_id_for_nodes.get(
                tuple(sorted((left_node, right_node)))
            )
            return bool(
                pair_id is not None
                and (
                    relation_first.get(pair_id) == "same"
                    or relation_second.get(pair_id) == "same"
                )
            )

        new_rows = 0
        new_count = 0
        active_candidate_set = set(active_candidates)
        pending_nodes = set(active_candidate_set)
        while pending_nodes:
            anchor = min(pending_nodes)
            component = {anchor}
            frontier = [anchor]
            while frontier:
                current = frontier.pop()
                neighbors = {
                    node
                    for node in pending_nodes - component
                    if _same(first, current, node)
                    or _same(second, current, node)
                    or veto_same(current, node)
                }
                component.update(neighbors)
                frontier.extend(sorted(neighbors))
            pending_nodes -= component
            component_candidates = sorted(component)
            possible_reps = {
                rep_id
                for rep_id in rep_ids
                if any(
                    (
                        rep_id not in first_ambiguous
                        and _same(first, candidate_id, rep_id)
                    )
                    or (
                        rep_id not in second_ambiguous
                        and _same(second, candidate_id, rep_id)
                    )
                    or veto_same(candidate_id, rep_id)
                    for candidate_id in component_candidates
                )
            }
            if len(possible_reps) > 1:
                rejected_disagreement.update(component_candidates)
                continue
            if len(possible_reps) == 1:
                strategy_key = rep_ids[next(iter(possible_reps))]
                for candidate_id in component_candidates:
                    accepted[candidate_ids[candidate_id]] = strategy_key
                matched_existing += len(component_candidates)
                continue
            if ambiguous_reps:
                # A missing representative could be the component's existing
                # key. Rejection prevents an unobserved relationship from
                # becoming rewarded novelty, while components with one known
                # representative above can still safely reuse it.
                rejected_disagreement.update(component_candidates)
                continue
            if any(
                veto_node(candidate_id) in relation_uncertain_anchors
                for candidate_id in component_candidates
            ):
                # An uncertain reduced comparison cannot authorize a new
                # rewarded strategy boundary.
                rejected_disagreement.update(component_candidates)
                continue

            # Candidate-only union components coarsen both full-context and
            # reduced pairwise equivalence-veto audits.
            # Components with no representative incidence are separated from
            # every observed bank representative and from every other new
            # component in every novelty-authorizing pass.
            candidate_anchor = component_candidates[0]
            next_id = self._next_strategy_id.get(prompt_key, 0)
            strategy_key = f"math_strategy_{next_id:06d}"
            self._next_strategy_id[prompt_key] = next_id + 1
            description = first_descriptions.get(first[candidate_anchor], "")
            representatives[strategy_key] = {
                "text": candidate_text[candidate_anchor],
                "description": description,
            }
            for candidate_id in component_candidates:
                accepted[candidate_ids[candidate_id]] = strategy_key
            new_rows += len(component_candidates)
            new_count += 1

        return accepted, MathStrategyDiagnostics(
            judge_calls=judge_calls,
            validator_positive_rows=float(len(candidate_rows)),
            accepted_rows=float(len(accepted)),
            rejected_integrity_rows=float(len(rejected_integrity)),
            rejected_ambiguous_rows=float(len(rejected_ambiguous)),
            rejected_disagreement_rows=float(len(rejected_disagreement)),
            matched_existing_rows=float(matched_existing),
            new_strategy_rows=float(new_rows),
            new_strategy_count=float(new_count),
        )

    def canonicalize(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        prompt_texts: Sequence[str],
        response_texts: Sequence[str],
        task_reward_positive: Sequence[bool],
        active_mask: Sequence[float | bool],
        num_samples: int,
    ) -> tuple[list[str | None], MathStrategyDiagnostics]:
        row_count = len(response_texts)
        if row_count == 0 or row_count % num_samples:
            raise ValueError("math strategy inputs must contain complete groups")
        if not (
            len(prompt_token_ids)
            == len(prompt_texts)
            == len(task_reward_positive)
            == len(active_mask)
            == row_count
        ):
            raise ValueError("math strategy canonicalizer inputs differ in length")
        outcome_keys: list[str | None] = [None] * row_count
        tasks = []
        for start in range(0, row_count, num_samples):
            stop = start + num_samples
            group_prompt_keys = {
                _prompt_key(tokens) for tokens in prompt_token_ids[start:stop]
            }
            if len(group_prompt_keys) != 1:
                raise ValueError("canonicalization group mixes prompts")
            candidates = [
                (index, response_texts[index])
                for index in range(start, stop)
                if bool(active_mask[index]) and bool(task_reward_positive[index])
            ]
            if candidates:
                tasks.append(
                    (
                        start,
                        prompt_token_ids[start],
                        prompt_texts[start],
                        candidates,
                    )
                )

        aggregate = {
            field: 0.0 for field in MathStrategyDiagnostics.__dataclass_fields__
        }
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._canonicalize_group,
                    prompt_tokens=prompt_tokens,
                    problem=problem,
                    candidate_rows=candidates,
                ): start
                for start, prompt_tokens, problem, candidates in tasks
            }
            for future in as_completed(futures):
                accepted, diagnostics = future.result()
                for row_index, strategy_key in accepted.items():
                    outcome_keys[row_index] = strategy_key
                for field in aggregate:
                    aggregate[field] += float(getattr(diagnostics, field))
        return outcome_keys, MathStrategyDiagnostics(**aggregate)
