"""Quarantined compatibility for historical proposal/route experiments.

These are not used by the 20 maintained recipes. Endpoint/reward decisions
must use benchmark.py; do not add legacy helpers to that path.
"""

from modebench.grading import (
    _canonical_countdown_expression_key,
    _extract_modebench_candidate,
    _graph_coloring_from_candidate,
    _verify_graph_coloring_colors,
)
from .benchmark import EvaluationFailure, api, grade_reference_response


def exploration_identity(response, reference):
    diagnostic = grade_reference_response(response, reference)
    if diagnostic is None or not diagnostic["verified"]:
        return None
    from modebench.grading import validated_modebench_exploration_identity

    try:
        return validated_modebench_exploration_identity(response, reference)
    except api.VerifierExecutionError as error:
        raise EvaluationFailure(error.diagnostic) from error
