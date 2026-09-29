"""Validator-preserving transformations of model-generated solutions.

These transformations use one executable response produced by the current
policy and the public task constraints.  They never enumerate a gold support,
read a desired mode count, or consult evaluation feedback.  Every derived
surface is independently revalidated by the same executable admission
boundary before it can enter a replay bank.
"""

from __future__ import annotations

import ast
import copy
import itertools
import json
from collections.abc import Iterable, Mapping
from typing import Any

from .legacy_modebench import (
    _canonical_countdown_expression_key,
    _extract_modebench_candidate,
    _graph_coloring_from_candidate,
    _verify_graph_coloring_colors,
)
from .benchmark import outcome_key as validated_modebench_outcome_key
from .canonical_actions import (
    decode_countdown_action_code,
    enumerate_countdown_action_codes,
)
from modebench.mathir import (
    Command,
    Expr,
    MATHIR_VERIFIER,
    MATHIR_MENU_VERIFIER,
    parse_mathir_action_program,
    parse_mathir_program,
    validate_mathir_algebra,
    validate_mathir_action_menu,
)
from modebench.python_modebench import (
    PYTHON_FACTOR_VERIFIER,
    parse_python_factor_candidate,
)
from modebench.python_modebench_process import validate_python_factor_function_external


def _parse_spec(reference: Any) -> dict[str, Any] | None:
    try:
        parsed = json.loads(reference) if isinstance(reference, str) else reference
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _boxed(candidate: str) -> str:
    return f"\\boxed{{{str(candidate).strip()}}}"


def _validated_surfaces(
    candidates: Iterable[str],
    reference: Any,
    *,
    original_key: str,
) -> tuple[str, ...]:
    """Deduplicate surfaces and outcomes after exact executable validation."""

    accepted: dict[str, str] = {}
    for candidate in candidates:
        surface = _boxed(candidate)
        key = validated_modebench_outcome_key(surface, reference)
        if key is None or key == original_key:
            continue
        previous = accepted.get(key)
        if previous is None or surface < previous:
            accepted[key] = surface
    return tuple(accepted[key] for key in sorted(accepted))


def _python_factor_candidates(
    candidate: str,
    spec: dict[str, Any],
) -> tuple[str, ...]:
    validation = validate_python_factor_function_external(candidate, spec)
    if validation is None:
        return ()
    try:
        parse_python_factor_candidate(candidate)
        cases = tuple(int(value) for value in spec["cases"])
        outputs = tuple(int(value) for value in validation.outputs)
    except Exception:
        return ()

    def lookup_lambda(values: tuple[int, ...]) -> str:
        # Fully case-local serialization prevents a prompt-specific algebraic
        # transform such as ``n // 2`` from becoming an unsafe shortcut on a
        # different prompt. Every constant comes from the model's own verified
        # execution or its mathematically forced cofactor.
        body: ast.expr = ast.Constant(value=int(values[-1]))
        for case, output in reversed(tuple(zip(cases, values))[:-1]):
            body = ast.IfExp(
                test=ast.Compare(
                    left=ast.Name(id="n", ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=int(case))],
                ),
                body=ast.Constant(value=int(output)),
                orelse=body,
            )
        expression = ast.Expression(
            body=ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="n")],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=body,
            )
        )
        return ast.unparse(ast.fix_missing_locations(expression))

    candidates: list[str] = []
    cofactors = tuple(
        int(case) // int(output)
        for case, output in zip(cases, outputs)
    )
    candidates.append(lookup_lambda(cofactors))
    for index, (output, cofactor) in enumerate(zip(outputs, cofactors)):
        if output == cofactor:
            continue
        variant = list(outputs)
        variant[index] = cofactor
        candidates.append(
            lookup_lambda(tuple(variant))
        )
    return tuple(candidates)


def _countdown_binop_paths(node: ast.AST) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = []

    def visit(current: ast.AST, path: tuple[str, ...]) -> None:
        if isinstance(current, ast.BinOp):
            paths.append(path)
            visit(current.left, path + ("left",))
            visit(current.right, path + ("right",))
        elif isinstance(current, ast.UnaryOp):
            visit(current.operand, path + ("operand",))

    visit(node, ())
    return paths


def _node_at_path(root: ast.AST, path: tuple[str, ...]) -> ast.AST:
    node = root
    for field in path:
        node = getattr(node, field)
    return node


def _replace_at_path(
    root: ast.AST,
    path: tuple[str, ...],
    replacement: ast.AST,
) -> ast.AST:
    if not path:
        return replacement
    parent = _node_at_path(root, path[:-1])
    setattr(parent, path[-1], replacement)
    return root


def _countdown_equivalent_binop(node: ast.BinOp) -> ast.BinOp:
    left = copy.deepcopy(node.left)
    right = copy.deepcopy(node.right)
    negative_right = ast.UnaryOp(op=ast.USub(), operand=right)
    if isinstance(node.op, ast.Add):
        return ast.BinOp(left=left, op=ast.Sub(), right=negative_right)
    if isinstance(node.op, ast.Sub):
        return ast.BinOp(left=left, op=ast.Add(), right=negative_right)
    negative_left = ast.UnaryOp(op=ast.USub(), operand=left)
    if isinstance(node.op, ast.Mult):
        return ast.BinOp(
            left=negative_left,
            op=ast.Mult(),
            right=negative_right,
        )
    if isinstance(node.op, ast.Div):
        return ast.BinOp(
            left=negative_left,
            op=ast.Div(),
            right=negative_right,
        )
    raise ValueError("unsupported Countdown binary operator")


def _countdown_candidates(candidate: str) -> tuple[str, ...]:
    text = str(candidate).strip()
    parts = [part.strip() for part in text.split("=") if part.strip()] or [text]
    parsed: ast.Expression | None = None
    for part in parts:
        try:
            expression = ast.parse(part, mode="eval")
        except (SyntaxError, ValueError):
            continue
        if isinstance(expression, ast.Expression):
            parsed = expression
            break
    if parsed is None:
        return ()
    candidates: list[str] = []
    for path in _countdown_binop_paths(parsed.body):
        variant = copy.deepcopy(parsed.body)
        selected = _node_at_path(variant, path)
        if not isinstance(selected, ast.BinOp):
            continue
        variant = _replace_at_path(
            variant,
            path,
            _countdown_equivalent_binop(selected),
        )
        candidates.append(ast.unparse(ast.fix_missing_locations(variant)))
    return tuple(candidates)


def _countdown_action_neighborhood_candidates(
    candidate: str,
    spec: dict[str, Any],
    *,
    radius: int,
) -> tuple[str, ...]:
    """Return a fixed local neighborhood in the public easy3 grammar.

    The 108 action codes describe syntax, not valid answers. We invert the
    verified anchor to all of its canonical code aliases without consulting
    the target, then mutate at most ``radius`` code positions. The ordinary
    validator later decides which local mutations solve the public instance.
    Every decoded candidate therefore belongs to the exact binary-expression
    grammar: positive input leaves, two binary operations, and no unary-sign
    extension accepted only by the more permissive runtime validator.
    """

    if radius not in (1, 2):
        raise ValueError("Countdown action-neighborhood radius must be 1 or 2")
    try:
        numbers = tuple(int(value) for value in spec["numbers"])
    except Exception:
        return ()
    if len(numbers) != 3 or len(set(numbers)) != 3:
        return ()
    original_key = _canonical_countdown_expression_key(candidate, spec)
    if original_key is None:
        return ()

    code_expressions = {
        code: decode_countdown_action_code(code, spec)
        for code in enumerate_countdown_action_codes()
    }
    anchor_codes = tuple(
        code
        for code, expression in code_expressions.items()
        if _canonical_countdown_expression_key(expression, spec) == original_key
    )
    if not anchor_codes:
        return ()

    def distance(left: str, right: str) -> int:
        return sum(a != b for a, b in zip(left, right))

    return tuple(
        expression
        for code, expression in code_expressions.items()
        if any(0 < distance(anchor_code, code) <= radius for anchor_code in anchor_codes)
    )


def _graph_candidates(
    candidate: str,
    spec: dict[str, Any],
) -> tuple[str, ...]:
    colors = _graph_coloring_from_candidate(candidate, spec)
    if not _verify_graph_coloring_colors(colors, spec):
        return ()
    assert colors is not None
    n = len(colors)
    adjacency = [set() for _ in range(n)]
    try:
        for raw_u, raw_v in spec["edges"]:
            u, v = int(raw_u) - 1, int(raw_v) - 1
            adjacency[u].add(v)
            adjacency[v].add(u)
    except Exception:
        return ()
    partial = spec.get("partial_colors")
    fixed = (
        [value is not None for value in partial]
        if isinstance(partial, list) and len(partial) == n
        else [False] * n
    )
    candidates: list[str] = []

    # First try the smallest local recoloring that preserves every edge and
    # every public fixed-color constraint.
    for vertex in range(n):
        if fixed[vertex]:
            continue
        for replacement in (1, 2, 3):
            if replacement == colors[vertex]:
                continue
            if any(colors[neighbor] == replacement for neighbor in adjacency[vertex]):
                continue
            variant = list(colors)
            variant[vertex] = replacement
            candidates.append("".join(str(value) for value in variant))

    # Kempe-component swaps preserve proper coloring by construction.  A swap
    # is allowed only when its component contains no fixed-color vertex.
    for first, second in itertools.combinations((1, 2, 3), 2):
        eligible = {
            index
            for index, color in enumerate(colors)
            if color in {first, second}
        }
        unseen = set(eligible)
        while unseen:
            start = min(unseen)
            component = {start}
            frontier = [start]
            unseen.remove(start)
            while frontier:
                current = frontier.pop()
                for neighbor in sorted(adjacency[current] & unseen):
                    component.add(neighbor)
                    unseen.remove(neighbor)
                    frontier.append(neighbor)
            if any(fixed[index] for index in component):
                continue
            variant = list(colors)
            for index in component:
                variant[index] = second if colors[index] == first else first
            candidates.append("".join(str(value) for value in variant))
    return tuple(candidates)


def _mathir_expr_text(expression: Expr) -> str:
    if expression.op == "symbol":
        return str(expression.value)
    if expression.op == "neg":
        return f"neg({_mathir_expr_text(expression.args[0])})"
    if expression.op in {"add", "sub", "mul", "div"}:
        return (
            f"{expression.op}("
            f"{_mathir_expr_text(expression.args[0])},"
            f"{_mathir_expr_text(expression.args[1])})"
        )
    raise ValueError("model-authored MathIR expressions cannot contain constants")


def _mathir_program_text(commands: Iterable[Command]) -> str:
    return ";".join(
        f"{command.op}({_mathir_expr_text(command.argument)})"
        for command in commands
    )


def _mathir_candidates(
    candidate: str,
    spec: dict[str, Any],
) -> tuple[str, ...]:
    validation = validate_mathir_algebra(candidate, spec)
    if validation is None:
        return ()
    try:
        bindings = spec["bindings"]
        allowed_symbols = frozenset(str(name) for name in bindings) | {"x"}
        commands = list(
            parse_mathir_program(
                candidate,
                allowed_symbols=allowed_symbols,
                max_steps=int(spec["max_steps"]),
            )
        )
    except Exception:
        return ()
    candidates: list[str] = []
    additive = {"add", "sub"}
    scaling = {"mul", "div"}
    for index in range(len(commands) - 1):
        first, second = commands[index], commands[index + 1]
        replacement: tuple[Command, Command] | None = None
        if first.op in additive and second.op in scaling:
            adjusted = Expr(
                second.op,
                (
                    copy.deepcopy(first.argument),
                    copy.deepcopy(second.argument),
                ),
            )
            replacement = (
                copy.deepcopy(second),
                Command(first.op, adjusted),
            )
        elif first.op in scaling and second.op in additive:
            inverse = "div" if first.op == "mul" else "mul"
            adjusted = Expr(
                inverse,
                (
                    copy.deepcopy(second.argument),
                    copy.deepcopy(first.argument),
                ),
            )
            replacement = (
                Command(second.op, adjusted),
                copy.deepcopy(first),
            )
        if replacement is None:
            continue
        variant = list(copy.deepcopy(commands))
        variant[index : index + 2] = replacement
        candidates.append(_mathir_program_text(variant))
    return tuple(candidates)


def _mathir_menu_candidates(
    candidate: str,
    spec: dict[str, Any],
) -> tuple[str, ...]:
    """Return a fixed local neighborhood of the model's verified menu path."""

    if validate_mathir_action_menu(candidate, spec) is None:
        return ()
    raw_actions = spec.get("actions")
    if not isinstance(raw_actions, Mapping):
        return ()
    action_ids = tuple(str(action_id) for action_id in raw_actions)
    try:
        max_steps = int(spec["max_steps"])
        selected = list(
            parse_mathir_action_program(
                candidate,
                action_ids=action_ids,
                max_steps=max_steps,
            )
        )
    except Exception:
        return ()
    candidates: list[str] = []

    # Commuting independent adjacent operations is the common MathIR route
    # alternative and changes the executed state path without changing the
    # endpoint. The ordinary interpreter decides whether a particular swap is
    # actually valid.
    for index in range(len(selected) - 1):
        variant = list(selected)
        variant[index], variant[index + 1] = (
            variant[index + 1],
            variant[index],
        )
        candidates.append(";".join(variant))

    # A fixed edit-distance-one neighborhood also covers a menu action that
    # combines two model-selected operations or splits one operation into two.
    # This is not an exhaustive sequence search and never reads the declared
    # support size; every resulting path must independently execute to a valid
    # terminal equation.
    for index in range(len(selected)):
        candidates.append(
            ";".join(selected[:index] + selected[index + 1 :])
        )
        for action_id in action_ids:
            if action_id == selected[index]:
                continue
            variant = list(selected)
            variant[index] = action_id
            candidates.append(";".join(variant))
    if len(selected) < max_steps:
        for index in range(len(selected) + 1):
            for action_id in action_ids:
                candidates.append(
                    ";".join(
                        selected[:index]
                        + [action_id]
                        + selected[index:]
                    )
                )
    for start in range(len(selected) - 1):
        for action_id in action_ids:
            candidates.append(
                ";".join(
                    selected[:start]
                    + [action_id]
                    + selected[start + 2 :]
                )
            )
    return tuple(candidates)


def derive_validator_preserving_counterfactuals(
    model_response: str,
    reference: Any,
) -> tuple[str, ...]:
    """Return ordered, independently verified alternatives to one response."""

    spec = _parse_spec(reference)
    if spec is None:
        return ()
    original_key = validated_modebench_outcome_key(model_response, reference)
    if original_key is None:
        return ()
    candidate = _extract_modebench_candidate(model_response, reference)
    if candidate is None:
        return ()
    verifier = str(spec.get("verifier", ""))
    if verifier == PYTHON_FACTOR_VERIFIER:
        raw_candidates = _python_factor_candidates(candidate, spec)
    elif verifier == "countdown":
        raw_candidates = _countdown_candidates(candidate)
    elif verifier == "graph_coloring":
        raw_candidates = _graph_candidates(candidate, spec)
    elif verifier == MATHIR_VERIFIER:
        raw_candidates = _mathir_candidates(candidate, spec)
    elif verifier == MATHIR_MENU_VERIFIER:
        raw_candidates = _mathir_menu_candidates(candidate, spec)
    else:
        raw_candidates = ()
    return _validated_surfaces(
        raw_candidates,
        reference,
        original_key=original_key,
    )


def derive_countdown_action_neighborhood_counterfactuals(
    model_response: str,
    reference: Any,
    *,
    radius: int = 2,
) -> tuple[str, ...]:
    """Return verified local alternatives inside Countdown's exact grammar."""

    spec = _parse_spec(reference)
    if spec is None or str(spec.get("verifier", "")) != "countdown":
        return ()
    original_key = validated_modebench_outcome_key(model_response, reference)
    if original_key is None:
        return ()
    candidate = _extract_modebench_candidate(model_response, reference)
    if candidate is None:
        return ()
    return _validated_surfaces(
        _countdown_action_neighborhood_candidates(candidate, spec, radius=radius),
        reference,
        original_key=original_key,
    )
