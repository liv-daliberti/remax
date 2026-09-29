"""Finite, machine-readable strategy menus for validator-bound MATH.

Each problem carries a frozen prompt-local action vocabulary and a small set
of distinct action sequences.  A completion must declare both a strategy ID
and its exact action sequence before giving the derivation.  Deterministic
parsing prevents missing/unknown/mismatched declarations from reaching the
semantic judge; the judge then checks that the written mathematics actually
executes the declared sequence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


MENU_SCHEMA = "math_strategy_action_menu_v1"
MENU_START = "<strategy_menu_json>"
MENU_END = "</strategy_menu_json>"
ACTION_TRACE_START = "<action_trace>"
ACTION_TRACE_END = "</action_trace>"
STRATEGY_PREFIX_RE = re.compile(
    r"\A<strategy_id>(S[1-9][0-9]*)</strategy_id>\n"
    r"<action_combo>(A[1-9][0-9]*(?:>A[1-9][0-9]*)*)</action_combo>\n"
)
ACTION_STEP_RE = re.compile(
    r'<action_step id="(A[1-9][0-9]*)">\n'
    r"(.+?)\n"
    r"</action_step>",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class MathStrategyAction:
    action_id: str
    operation: str


@dataclass(frozen=True)
class MathStrategyOption:
    strategy_id: str
    action_ids: tuple[str, ...]
    plan: str

    @property
    def action_combo(self) -> str:
        return ">".join(self.action_ids)


@dataclass(frozen=True)
class MathStrategyMenu:
    actions: tuple[MathStrategyAction, ...]
    strategies: tuple[MathStrategyOption, ...]
    sha256: str

    def strategy(self, strategy_id: str) -> MathStrategyOption | None:
        return next(
            (
                strategy
                for strategy in self.strategies
                if strategy.strategy_id == strategy_id
            ),
            None,
        )

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {
                "schema": MENU_SCHEMA,
                "actions": [
                    {
                        "action_id": action.action_id,
                        "operation": action.operation,
                    }
                    for action in self.actions
                ],
                "strategies": [
                    {
                        "strategy_id": strategy.strategy_id,
                        "action_ids": list(strategy.action_ids),
                        "plan": strategy.plan,
                    }
                    for strategy in self.strategies
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class MathStrategyDeclaration:
    strategy_id: str
    action_combo: str
    derivation: str


@dataclass(frozen=True)
class MathStrategyExecutionDeclaration:
    strategy_id: str
    action_combo: str
    action_steps: tuple[tuple[str, str], ...]
    conclusion: str


def _strict_object(
    value: Any,
    *,
    required: set[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"{context} must contain exactly {sorted(required)}"
        )
    return value


def parse_strategy_menu(problem: str) -> MathStrategyMenu | None:
    """Parse one exact menu block, returning ``None`` when none is present."""

    if MENU_START not in problem and MENU_END not in problem:
        return None
    if problem.count(MENU_START) != 1 or problem.count(MENU_END) != 1:
        raise ValueError("problem must contain exactly one strategy menu")
    start = problem.index(MENU_START) + len(MENU_START)
    stop = problem.index(MENU_END, start)
    if stop <= start:
        raise ValueError("strategy menu is empty or misordered")
    raw = problem[start:stop].strip()
    payload = _strict_object(
        json.loads(raw),
        required={"schema", "actions", "strategies"},
        context="strategy menu",
    )
    if payload["schema"] != MENU_SCHEMA:
        raise ValueError("unexpected strategy menu schema")
    if not isinstance(payload["actions"], list) or not (
        1 <= len(payload["actions"]) <= 12
    ):
        raise ValueError("strategy menu must contain 1..12 actions")
    if not isinstance(payload["strategies"], list) or not (
        1 <= len(payload["strategies"]) <= 6
    ):
        raise ValueError("strategy menu must contain 1..6 strategies")

    actions: list[MathStrategyAction] = []
    for index, raw_action in enumerate(payload["actions"], start=1):
        action = _strict_object(
            raw_action,
            required={"action_id", "operation"},
            context="strategy action",
        )
        expected_id = f"A{index}"
        operation = action["operation"]
        if action["action_id"] != expected_id:
            raise ValueError("strategy action IDs must be A1..An in order")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("strategy action operation must be nonempty")
        if len(operation) > 320:
            raise ValueError("strategy action operation is too long")
        actions.append(
            MathStrategyAction(
                action_id=expected_id,
                operation=operation.strip(),
            )
        )
    action_ids = {action.action_id for action in actions}

    strategies: list[MathStrategyOption] = []
    observed_combos: set[tuple[str, ...]] = set()
    for index, raw_strategy in enumerate(payload["strategies"], start=1):
        strategy = _strict_object(
            raw_strategy,
            required={"strategy_id", "action_ids", "plan"},
            context="strategy option",
        )
        expected_id = f"S{index}"
        raw_ids = strategy["action_ids"]
        plan = strategy["plan"]
        if strategy["strategy_id"] != expected_id:
            raise ValueError("strategy IDs must be S1..Sn in order")
        if (
            not isinstance(raw_ids, list)
            or not 1 <= len(raw_ids) <= 8
            or any(not isinstance(value, str) for value in raw_ids)
        ):
            raise ValueError("strategy action_ids must contain 1..8 IDs")
        combo = tuple(raw_ids)
        if len(set(combo)) != len(combo) or not set(combo) <= action_ids:
            raise ValueError("strategy action combo is invalid")
        if combo in observed_combos:
            raise ValueError("strategy action combos must be distinct")
        observed_combos.add(combo)
        if not isinstance(plan, str) or not plan.strip() or len(plan) > 640:
            raise ValueError("strategy plan must be concise and nonempty")
        strategies.append(
            MathStrategyOption(
                strategy_id=expected_id,
                action_ids=combo,
                plan=plan.strip(),
            )
        )

    canonical = json.dumps(
        {
            "schema": MENU_SCHEMA,
            "actions": [
                {
                    "action_id": action.action_id,
                    "operation": action.operation,
                }
                for action in actions
            ],
            "strategies": [
                {
                    "strategy_id": strategy.strategy_id,
                    "action_ids": list(strategy.action_ids),
                    "plan": strategy.plan,
                }
                for strategy in strategies
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return MathStrategyMenu(
        actions=tuple(actions),
        strategies=tuple(strategies),
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def parse_strategy_declaration(
    response: str,
    menu: MathStrategyMenu,
) -> MathStrategyDeclaration | None:
    """Return an exact, menu-consistent prefix declaration or ``None``."""

    match = STRATEGY_PREFIX_RE.match(response)
    if match is None:
        return None
    strategy_id, action_combo = match.groups()
    strategy = menu.strategy(strategy_id)
    if strategy is None or action_combo != strategy.action_combo:
        return None
    derivation = response[match.end() :].strip()
    if not derivation:
        return None
    return MathStrategyDeclaration(
        strategy_id=strategy_id,
        action_combo=action_combo,
        derivation=derivation,
    )


def parse_strategy_execution(
    response: str,
    menu: MathStrategyMenu,
) -> MathStrategyExecutionDeclaration | None:
    """Parse an exact, ordered action trace for one declared menu strategy."""

    declaration = parse_strategy_declaration(response, menu)
    if declaration is None:
        return None
    strategy = menu.strategy(declaration.strategy_id)
    if strategy is None:
        return None
    trace_prefix = ACTION_TRACE_START + "\n"
    trace_suffix = "\n" + ACTION_TRACE_END + "\n"
    if (
        not declaration.derivation.startswith(trace_prefix)
        or declaration.derivation.count(ACTION_TRACE_START) != 1
        or declaration.derivation.count(ACTION_TRACE_END) != 1
        or trace_suffix not in declaration.derivation
    ):
        return None
    body, conclusion = declaration.derivation[
        len(trace_prefix) :
    ].split(trace_suffix, maxsplit=1)
    if not body or not conclusion.strip() or "\\boxed{" not in conclusion:
        return None

    steps: list[tuple[str, str]] = []
    position = 0
    for match in ACTION_STEP_RE.finditer(body):
        if match.start() != position:
            return None
        action_id = match.group(1)
        content = match.group(2).strip()
        if (
            not content
            or "<action_step" in content
            or "</action_step>" in content
        ):
            return None
        steps.append((action_id, content))
        position = match.end()
        if position < len(body):
            if body[position] != "\n":
                return None
            position += 1
    if position != len(body):
        return None
    if tuple(action_id for action_id, _ in steps) != strategy.action_ids:
        return None
    return MathStrategyExecutionDeclaration(
        strategy_id=declaration.strategy_id,
        action_combo=declaration.action_combo,
        action_steps=tuple(steps),
        conclusion=conclusion.strip(),
    )


def strategy_menu_response_instructions(menu: MathStrategyMenu) -> str:
    headers = "\n\n".join(
        f"<strategy_id>{strategy.strategy_id}</strategy_id>\n"
        f"<action_combo>{strategy.action_combo}</action_combo>"
        for strategy in menu.strategies
    )
    return (
        "\n\nChoose exactly one listed strategy and execute its actions in the "
        "declared order. Your response MUST begin at its first character with "
        "exactly one of these two-line headers:\n\n"
        f"{headers}\n"
        "Immediately after the header, write <action_trace>, then exactly one "
        "<action_step id=\"Aj\"> block for every declared action, in the exact "
        "combo order, then </action_trace> and a boxed final answer. Each "
        "action-step block must contain the actual mathematics that executes "
        "that action. For example:\n"
        "<action_trace>\n"
        "<action_step id=\"A1\">\n"
        "[mathematics executing A1]\n"
        "</action_step>\n"
        "</action_trace>\n"
        "Therefore \\boxed{[answer]}.\n"
        "Missing, extra, empty, duplicated, or reordered action blocks fail "
        "before the semantic judge. Merely naming a strategy, using a "
        "different route, skipping an essential declared action, or adding "
        "an undeclared substitute route fails the execution contract and "
        "earns zero reward."
    )


def strategy_menu_natural_response_instructions(
    menu: MathStrategyMenu,
) -> str:
    """Render E49T's format-tolerant but execution-strict menu contract."""

    choices = "\n".join(
        f"- {strategy.strategy_id}: {strategy.action_combo}"
        for strategy in menu.strategies
    )
    return (
        "\n\nChoose exactly one listed strategy:\n"
        f"{choices}\n"
        "State the chosen strategy ID and action combo, then give a natural "
        "mathematical derivation that materially executes every action in "
        "that exact combo, in order. A declaration by itself proves nothing: "
        "the reward auditor checks the mathematics actually written. The "
        "derivation must be sufficient to establish the boxed answer without "
        "silently switching to another listed or unlisted route, mixing "
        "routes, omitting an essential action, or merely naming an operation. "
        "Routine arithmetic internal to a declared action is allowed. Finish "
        "with a boxed final answer. A wrong answer, flawed derivation, "
        "answer-only response, incomplete combo, or route mismatch earns "
        "zero reward."
    )
