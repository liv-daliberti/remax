"""Supported ModeBench boundary and fail-stop reward diagnostics.

Maintained recipes pass historical decoded responses. In particular, Pantry
support masks have already been projected to allocations by the actor. Use the
explicit allocation contract there; callers with raw benchmark responses should
pass a public Task to grade_task instead. Never infer a surface from an answer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from modebench import api
from modebench.api_types import GradeResult, Task

SCORABLE = frozenset({"correct", "incorrect", "malformed"})
UNSCORABLE = frozenset(
    {"timeout", "invalid_reference", "worker_failure", "resource_limit"}
)
VERIFIER_DOMAINS = {
    "countdown": "countdown",
    "graph_coloring": "graph_coloring",
    "python_factor_function": "python_factors",
    "mathir_algebra": "mathir",
    "mathir_action_menu": "mathir",
    "pantry_plan": "pantry_plan",
}


class EvaluationFailure(api.VerifierExecutionError):
    """Serializable fatal diagnostic; never represent this as a model reward."""

    def __init__(self, diagnostic):
        self.diagnostic = dict(diagnostic)
        # Keep the constructor argument structured for pickle/RPC round trips.
        RuntimeError.__init__(self, self.diagnostic)

    def __str__(self):
        return f"evaluation {self.diagnostic.get('status')}: {self.diagnostic.get('detail')}"


def failure(status, detail, **context):
    return EvaluationFailure(
        dict(
            status=status,
            verified=False,
            canonical_key=None,
            graded_text="",
            detail=detail,
            **context,
        )
    )


def require_scorable(diagnostic: GradeResult) -> GradeResult:
    """Validate public results defensively before reward, identity or metrics."""
    if (
        not isinstance(diagnostic, dict)
        or diagnostic.get("status") not in SCORABLE | UNSCORABLE
    ):
        raise failure("worker_failure", "missing or unknown verifier status")
    if diagnostic["status"] in UNSCORABLE:
        raise EvaluationFailure(diagnostic)
    correct = diagnostic["status"] == "correct"
    key = diagnostic.get("canonical_key")
    if (
        type(diagnostic.get("verified")) is not bool
        or diagnostic["verified"] != correct
        or (correct and (not isinstance(key, str) or not key))
        or (not correct and key is not None)
        or not isinstance(diagnostic.get("graded_text"), str)
    ):
        raise failure("worker_failure", "inconsistent verifier result")
    return diagnostic


def parse_reference(reference: Any) -> dict | None:
    """None means ordinary MATH, not a malformed/unknown benchmark reference."""
    if isinstance(reference, Mapping):
        spec = dict(reference)
    elif isinstance(reference, str) and reference.lstrip().startswith("{"):
        try:
            spec = json.loads(reference)
        except (ValueError, RecursionError) as error:
            body = reference.lstrip()[1:].lstrip()
            if "verifier" not in reference and not body.startswith(('"', "'")):
                return None  # ordinary MATH may use a set such as {1, 2}
            raise failure("invalid_reference", "malformed benchmark JSON") from error
    else:
        return None
    if (
        not isinstance(spec, dict)
        or not isinstance(spec.get("verifier"), str)
        or spec["verifier"] not in VERIFIER_DOMAINS
    ):
        raise failure("invalid_reference", "missing or unsupported benchmark verifier")
    return spec


def grade_task(task: Task, response: str) -> GradeResult:
    """Grade raw registered responses with their explicit level/domain."""
    try:
        diagnostic = api.grade(task, response)
    except EvaluationFailure:
        raise
    except api.VerifierExecutionError as error:
        raise EvaluationFailure(error.diagnostic) from error
    except Exception as error:
        raise failure("worker_failure", f"{type(error).__name__}: {error}") from error
    return require_scorable(diagnostic)


def grade_reference_response(response: str, reference: Any) -> GradeResult | None:
    """Compatibility adapter for reference-only, decoded training trajectories.

    All non-Pantry validators have level-independent response semantics. Pantry
    allocations use the public level-2 allocation contract, including decoded
    level-1 responses; this does not change the dataset's reported level.
    """
    spec = parse_reference(reference)
    if spec is None:
        return None
    domain = VERIFIER_DOMAINS[spec["verifier"]]
    level = 2 if domain == "pantry_plan" else 1
    task = Task(
        id="remax-decoded-reference",
        level=level,
        domain=domain,
        problem="",
        answer=spec,
    )
    return grade_task(task, response)


def outcome_key(response: str, reference: Any) -> str | None:
    diagnostic = grade_reference_response(response, reference)
    return diagnostic["canonical_key"] if diagnostic is not None else None


def reward_from_diagnostic(diagnostic):
    diagnostic = require_scorable(diagnostic)
    return {
        "formatted": diagnostic["status"] != "malformed",
        "verifier": diagnostic,
    }, float(diagnostic["verified"])


def validate_reward_batch(rewards, infos, *, count, context, references=None):
    """Reject missing rows, failure flags, and fabricated numeric fallbacks."""
    if len(rewards) != count or len(infos) != count:
        raise failure("worker_failure", "incomplete reward batch", context=context)
    if references is not None and len(references) != count:
        raise failure(
            "invalid_reference", "incomplete batch references", context=context
        )
    for index, (reward, info) in enumerate(zip(rewards, infos)):
        if not isinstance(info, dict):
            raise failure(
                "worker_failure",
                "invalid reward info",
                context=context,
                row_index=index,
            )
        diagnostic = info.get("verifier")
        try:
            if info.get("verifier_timeout") or info.get("verifier_worker_error"):
                raise failure(
                    "timeout" if info.get("verifier_timeout") else "worker_failure",
                    "legacy verifier failure flag",
                )
            if diagnostic is not None:
                require_scorable(diagnostic)
            elif (
                references is not None
                and parse_reference(references[index]) is not None
            ):
                raise failure("worker_failure", "missing benchmark diagnostic")
            if (
                not isinstance(reward, (int, float))
                or not math.isfinite(reward)
                or reward not in (0.0, 1.0)
            ):
                raise failure("worker_failure", "reward must be finite and binary")
            if diagnostic is not None and reward != float(diagnostic["verified"]):
                raise failure("worker_failure", "reward disagrees with verifier status")
        except EvaluationFailure as error:
            raise EvaluationFailure(
                dict(error.diagnostic, context=context, row_index=index)
            ) from error
