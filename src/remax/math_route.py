"""Fail-closed executable route traces for answer-verified free-form math.

The trace is deliberately separate from the natural-language derivation and
from the boxed answer. It contains only references and operations; it never
contains model-declared intermediate results. Numeric leaves must be grounded
in the problem text, every operation is executed exactly, and every trace node
must contribute to the terminal value.

This validator establishes a checkable computational route, not a proof that
the route's modeling assumptions follow from the prose problem. Ordinary MATH
answer verification remains a separate mandatory admission condition.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import re
from typing import Any, Mapping

import sympy


MATH_ROUTE_VERSION = "math-route-v1"
MATH_ROUTE_RPN_VERSION = "math-route-rpn-v2"
_MAX_TRACE_CHARS = 4_096
_MAX_STEPS = 20
_MAX_ARGS = 6
_MAX_INTEGER_BITS = 512
_ID_RE = re.compile(r"s[1-9][0-9]?")
_NUMBER_RE = re.compile(
    r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)(?!\.\d)"
)
_SLASH_FRACTION_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:,\d{3})*)\s*/\s*(\d+(?:,\d{3})*)"
    r"(?!\w)(?!\.\d)"
)
_LATEX_FRACTION_RE = re.compile(
    r"\\(?:d?frac)\s*\{\s*(-?\d+(?:,\d{3})*)\s*\}"
    r"\s*\{\s*(\d+(?:,\d{3})*)\s*\}"
)
_ROUTE_BLOCK_RE = re.compile(r"<route>(.*?)</route>", re.DOTALL)
_COMMUTATIVE_OPS = frozenset({"add", "mul", "gcd", "lcm", "min", "max"})
_UNARY_OPS = frozenset(
    {"neg", "abs", "square", "cube", "sqrt", "factorial", "percent"}
)
_BINARY_OPS = frozenset(
    {"sub", "div", "pow", "mod", "choose", "permute"}
)
_NARY_OPS = frozenset({"add", "mul", "gcd", "lcm", "min", "max", "average"})
_ALLOWED_TOP_LEVEL = frozenset({"version", "steps", "final"})
_ALLOWED_STEP_KEYS = frozenset({"id", "op", "args", "value"})
_RPN_UNARY_OPS = {
    "neg": "neg",
    "abs": "abs",
    "square": "square",
    "cube": "cube",
    "sqrt": "sqrt",
    "factorial": "factorial",
    "percent": "percent",
}
_RPN_BINARY_OPS = {
    "add": "add",
    "mul": "mul",
    "sub": "sub",
    "div": "div",
    "pow": "pow",
    "mod": "mod",
    "choose": "choose",
    "permute": "permute",
    "gcd": "gcd",
    "lcm": "lcm",
    "min": "min",
    "max": "max",
    "average": "average",
}


@dataclass(frozen=True)
class MathRouteValidation:
    route_signature: str
    terminal_value: sympy.Expr
    operations: tuple[str, ...]
    step_count: int
    source_count: int


class MathRouteError(ValueError):
    """Raised for a malformed, ungrounded, or non-executable trace."""


def _fraction(text: str) -> Fraction:
    compact = str(text).replace(",", "").strip()
    if re.fullmatch(r"-?\d+", compact):
        return Fraction(int(compact), 1)
    if re.fullmatch(r"-?\d+/\d+", compact):
        numerator, denominator = compact.split("/", 1)
        if int(denominator) == 0:
            raise MathRouteError("zero denominator")
        return Fraction(int(numerator), int(denominator))
    if re.fullmatch(r"-?\d+\.\d+", compact):
        return Fraction(compact)
    raise MathRouteError("source values must be exact integers, decimals, or fractions")


def problem_number_inventory(problem: str) -> Counter[Fraction]:
    """Return exact numeric leaves present in problem text, with multiplicity."""

    text = str(problem)
    inventory: Counter[Fraction] = Counter()

    def consume_latex(match: re.Match[str]) -> str:
        numerator, denominator = match.groups()
        inventory[Fraction(int(numerator.replace(",", "")), int(denominator.replace(",", "")))] += 1
        return " "

    def consume_slash(match: re.Match[str]) -> str:
        numerator, denominator = match.groups()
        inventory[Fraction(int(numerator.replace(",", "")), int(denominator.replace(",", "")))] += 1
        return " "

    text = _LATEX_FRACTION_RE.sub(consume_latex, text)
    text = _SLASH_FRACTION_RE.sub(consume_slash, text)
    for match in _NUMBER_RE.finditer(text):
        inventory[_fraction(match.group(0))] += 1
    return inventory


def extract_math_route_block(model_response: str) -> str | None:
    matches = _ROUTE_BLOCK_RE.findall(str(model_response))
    if len(matches) != 1:
        return None
    block = matches[0].strip()
    if not block or len(block) > _MAX_TRACE_CHARS:
        return None
    return block


def _exact_integer(value: sympy.Expr, *, name: str) -> int:
    simplified = sympy.simplify(value)
    if not simplified.is_Integer:
        raise MathRouteError(f"{name} requires an exact integer")
    integer = int(simplified)
    if abs(integer).bit_length() > _MAX_INTEGER_BITS:
        raise MathRouteError(f"{name} integer is too large")
    return integer


def _bounded(value: sympy.Expr) -> sympy.Expr:
    simplified = sympy.simplify(value)
    if simplified.has(
        sympy.nan,
        sympy.zoo,
        sympy.oo,
        -sympy.oo,
        sympy.I,
    ):
        raise MathRouteError("operation produced a non-finite or complex value")
    if len(str(simplified)) > 1_024 or int(sympy.count_ops(simplified)) > 128:
        raise MathRouteError("operation result is too large")
    for atom in simplified.atoms(sympy.Integer):
        if abs(int(atom)).bit_length() > _MAX_INTEGER_BITS:
            raise MathRouteError("operation result integer is too large")
    return simplified


def _execute_operation(op: str, arguments: tuple[sympy.Expr, ...]) -> sympy.Expr:
    if op in _UNARY_OPS and len(arguments) != 1:
        raise MathRouteError(f"{op} requires one argument")
    if op in _BINARY_OPS and len(arguments) != 2:
        raise MathRouteError(f"{op} requires two arguments")
    if op in _NARY_OPS and not 2 <= len(arguments) <= _MAX_ARGS:
        raise MathRouteError(f"{op} requires two to {_MAX_ARGS} arguments")
    if op == "neg":
        result = -arguments[0]
    elif op == "abs":
        result = sympy.Abs(arguments[0])
    elif op == "square":
        result = arguments[0] ** 2
    elif op == "cube":
        result = arguments[0] ** 3
    elif op == "sqrt":
        if arguments[0].is_nonnegative is not True:
            raise MathRouteError("sqrt requires a provably nonnegative argument")
        result = sympy.sqrt(arguments[0])
    elif op == "factorial":
        integer = _exact_integer(arguments[0], name=op)
        if not 0 <= integer <= 100:
            raise MathRouteError("factorial input is outside [0,100]")
        result = sympy.factorial(integer)
    elif op == "percent":
        result = arguments[0] / 100
    elif op == "add":
        result = sum(arguments, sympy.Integer(0))
    elif op == "mul":
        result = math.prod(arguments, start=sympy.Integer(1))
    elif op == "sub":
        result = arguments[0] - arguments[1]
    elif op == "div":
        if sympy.simplify(arguments[1]) == 0:
            raise MathRouteError("division by zero")
        result = arguments[0] / arguments[1]
    elif op == "pow":
        exponent = _exact_integer(arguments[1], name=op)
        if not -12 <= exponent <= 12:
            raise MathRouteError("power exponent is outside [-12,12]")
        if arguments[0] == 0 and exponent < 0:
            raise MathRouteError("zero to a negative power")
        result = arguments[0] ** exponent
    elif op == "mod":
        left = _exact_integer(arguments[0], name=op)
        right = _exact_integer(arguments[1], name=op)
        if right == 0:
            raise MathRouteError("modulo by zero")
        result = sympy.Integer(left % right)
    elif op == "choose":
        n = _exact_integer(arguments[0], name=op)
        k = _exact_integer(arguments[1], name=op)
        if not 0 <= k <= n <= 10_000:
            raise MathRouteError("choose inputs are outside 0 <= k <= n <= 10000")
        result = sympy.binomial(n, k)
    elif op == "permute":
        n = _exact_integer(arguments[0], name=op)
        k = _exact_integer(arguments[1], name=op)
        if not 0 <= k <= n <= 1_000:
            raise MathRouteError("permute inputs are outside 0 <= k <= n <= 1000")
        result = sympy.factorial(n) / sympy.factorial(n - k)
    elif op in {"gcd", "lcm"}:
        integers = [_exact_integer(argument, name=op) for argument in arguments]
        function = math.gcd if op == "gcd" else math.lcm
        result = sympy.Integer(function(*integers))
    elif op == "min":
        if not all(argument.is_real is True for argument in arguments):
            raise MathRouteError("min requires real arguments")
        result = sympy.Min(*arguments)
    elif op == "max":
        if not all(argument.is_real is True for argument in arguments):
            raise MathRouteError("max requires real arguments")
        result = sympy.Max(*arguments)
    elif op == "average":
        result = sum(arguments, sympy.Integer(0)) / len(arguments)
    else:
        raise MathRouteError(f"unsupported route operation {op!r}")
    return _bounded(result)


def _signature(op: str, argument_signatures: tuple[str, ...]) -> str:
    children = (
        tuple(sorted(argument_signatures))
        if op in _COMMUTATIVE_OPS
        else argument_signatures
    )
    return f"{op}({','.join(children)})"


def validate_math_route_trace(
    trace: str | Mapping[str, Any],
    problem: str,
) -> MathRouteValidation | None:
    """Parse and execute a grounded route trace; return ``None`` on any failure."""

    try:
        parsed = json.loads(trace) if isinstance(trace, str) else dict(trace)
        if not isinstance(parsed, dict) or set(parsed) != _ALLOWED_TOP_LEVEL:
            raise MathRouteError("route object has missing or extra fields")
        if parsed["version"] != MATH_ROUTE_VERSION:
            raise MathRouteError("unsupported route version")
        raw_steps = parsed["steps"]
        if not isinstance(raw_steps, list) or not 2 <= len(raw_steps) <= _MAX_STEPS:
            raise MathRouteError("route has an invalid number of steps")
        final_id = str(parsed["final"])
        if _ID_RE.fullmatch(final_id) is None:
            raise MathRouteError("invalid final node ID")

        inventory = problem_number_inventory(problem)
        values: dict[str, sympy.Expr] = {}
        signatures: dict[str, str] = {}
        dependencies: dict[str, tuple[str, ...]] = {}
        operations: list[str] = []
        source_count = 0
        non_source_count = 0
        for expected_index, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, dict):
                raise MathRouteError("route step must be an object")
            if not set(raw_step).issubset(_ALLOWED_STEP_KEYS):
                raise MathRouteError("route step has extra fields")
            step_id = str(raw_step.get("id", ""))
            if step_id != f"s{expected_index}" or step_id in values:
                raise MathRouteError("route node IDs must be contiguous and ordered")
            op = str(raw_step.get("op", ""))
            if op == "source":
                if set(raw_step) != {"id", "op", "value"}:
                    raise MathRouteError("source step schema is invalid")
                value = _fraction(str(raw_step["value"]))
                if inventory[value] <= 0:
                    raise MathRouteError("source value is not available in the problem")
                inventory[value] -= 1
                values[step_id] = sympy.Rational(value.numerator, value.denominator)
                signatures[step_id] = "input"
                dependencies[step_id] = ()
                source_count += 1
                operations.append(op)
                continue

            if set(raw_step) != {"id", "op", "args"}:
                raise MathRouteError("operation step schema is invalid")
            raw_args = raw_step["args"]
            if not isinstance(raw_args, list):
                raise MathRouteError("operation args must be a list")
            argument_ids = tuple(str(argument) for argument in raw_args)
            if (
                not argument_ids
                or len(argument_ids) > _MAX_ARGS
                or len(set(argument_ids)) != len(argument_ids)
                or any(argument_id not in values for argument_id in argument_ids)
            ):
                raise MathRouteError("operation dependencies are invalid")
            arguments = tuple(values[argument_id] for argument_id in argument_ids)
            values[step_id] = _execute_operation(op, arguments)
            signatures[step_id] = _signature(
                op,
                tuple(signatures[argument_id] for argument_id in argument_ids),
            )
            dependencies[step_id] = argument_ids
            operations.append(op)
            non_source_count += 1

        if final_id not in values or non_source_count < 1 or source_count < 1:
            raise MathRouteError("route lacks a computed terminal value")
        reachable = {final_id}
        frontier = [final_id]
        while frontier:
            node = frontier.pop()
            for dependency in dependencies[node]:
                if dependency not in reachable:
                    reachable.add(dependency)
                    frontier.append(dependency)
        if reachable != set(values):
            raise MathRouteError("every route node must contribute to the final value")
        return MathRouteValidation(
            route_signature=(
                f"math-route:{MATH_ROUTE_VERSION}:{signatures[final_id]}"
            ),
            terminal_value=values[final_id],
            operations=tuple(operations),
            step_count=len(values),
            source_count=source_count,
        )
    except Exception:
        return None


def validate_math_route_rpn(
    trace: str,
    problem: str,
) -> MathRouteValidation | None:
    """Execute the compact v2 reverse-Polish route language.

    Numeric tokens are grounded problem leaves. Unary and binary operator
    tokens consume the stack, so a successful one-item terminal stack also
    proves that every emitted token contributes to the final value. The
    language has no result literals, variable names, prose, or code execution.
    """

    try:
        compact = str(trace).strip()
        if not compact or len(compact) > 1_024:
            raise MathRouteError("RPN route is empty or too long")
        tokens = compact.split()
        if not tokens or tokens[0] not in {
            "v2",
            MATH_ROUTE_RPN_VERSION,
        }:
            raise MathRouteError("unsupported RPN route version")
        tokens = tokens[1:]
        if not 2 <= len(tokens) <= 40:
            raise MathRouteError("RPN route has an invalid token count")

        inventory = problem_number_inventory(problem)
        stack: list[tuple[sympy.Expr, str]] = []
        operations: list[str] = []
        source_count = 0
        operation_count = 0
        for token in tokens:
            if token in _RPN_UNARY_OPS:
                if len(stack) < 1:
                    raise MathRouteError("RPN unary operation underflow")
                argument_value, argument_signature = stack.pop()
                op = _RPN_UNARY_OPS[token]
                stack.append(
                    (
                        _execute_operation(op, (argument_value,)),
                        _signature(op, (argument_signature,)),
                    )
                )
                operations.append(op)
                operation_count += 1
                continue
            if token in _RPN_BINARY_OPS:
                if len(stack) < 2:
                    raise MathRouteError("RPN binary operation underflow")
                right_value, right_signature = stack.pop()
                left_value, left_signature = stack.pop()
                op = _RPN_BINARY_OPS[token]
                stack.append(
                    (
                        _execute_operation(op, (left_value, right_value)),
                        _signature(op, (left_signature, right_signature)),
                    )
                )
                operations.append(op)
                operation_count += 1
                continue

            value = _fraction(token)
            if inventory[value] <= 0:
                raise MathRouteError("RPN source value is not available in the problem")
            inventory[value] -= 1
            stack.append(
                (
                    sympy.Rational(value.numerator, value.denominator),
                    "input",
                )
            )
            operations.append("source")
            source_count += 1

        if len(stack) != 1 or source_count < 1 or operation_count < 1:
            raise MathRouteError("RPN route lacks one computed terminal value")
        terminal_value, terminal_signature = stack[0]
        return MathRouteValidation(
            route_signature=(
                f"math-route:{MATH_ROUTE_RPN_VERSION}:{terminal_signature}"
            ),
            terminal_value=terminal_value,
            operations=tuple(operations),
            step_count=len(tokens),
            source_count=source_count,
        )
    except Exception:
        return None


def validate_math_route_response(
    model_response: str,
    problem: str,
) -> MathRouteValidation | None:
    """Extract the response's unique route block and execute it."""

    block = extract_math_route_block(model_response)
    if block is None:
        return None
    if block.lstrip().startswith("{"):
        return validate_math_route_trace(block, problem)
    return validate_math_route_rpn(block, problem)
