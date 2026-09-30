"""Evaluation metrics, verifier transport, and deterministic request identities."""

from __future__ import annotations
import hashlib
import json
import math
import numpy as np
from oat.types import TrajectoryData
from ...benchmark import validate_reward_batch
from ...math_grader import VerifiedExplorationIdentity, boxed_reward_fn


def _grade_decoded_canonical_response(response, reference, *, fast):
    """Grade the environment transition; a legal action can be infeasible.

    For example, a well-formed Pantry mask may decode to the invalid-support
    sentinel. That is a scorable rejection, not an evaluator failure. The
    caller has already checked the policy action's serialization separately.
    """
    info, reward = boxed_reward_fn(response, reference, fast=fast)
    validate_reward_batch(
        [reward],
        [info],
        count=1,
        references=[reference],
        context="canonical learner verification",
    )
    return info, float(reward)


def _derive_freeform_request_seed(
    *,
    base_seed: int,
    prompt_batch_index: int,
    stream: str,
) -> int:
    """Derive an isolated deterministic vLLM request stream.

    Extra proposal requests must not advance or otherwise perturb subsequent
    neutral rollouts. Every replicated free-form neutral and proposal request
    therefore has an explicit seed derived from disjoint named streams.
    """

    if isinstance(base_seed, bool) or int(base_seed) != base_seed:
        raise ValueError("free-form request base seed must be an integer")
    if (
        isinstance(prompt_batch_index, bool)
        or int(prompt_batch_index) != prompt_batch_index
        or int(prompt_batch_index) < 0
    ):
        raise ValueError(
            "free-form request prompt-batch index must be a non-negative integer"
        )
    if not isinstance(stream, str) or not stream:
        raise ValueError("free-form request stream must be a non-empty string")
    payload = (
        f"replicated-freeform-v1|{int(base_seed)}|{int(prompt_batch_index)}|{stream}"
    ).encode("utf-8")
    # Keep the value exactly representable in IEEE-754 telemetry while leaving
    # a collision-resistant namespace for long multi-domain campaigns.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 52) - 1)


def _parse_answer_mode_count(ref: str) -> int:
    """Extract the number of valid answer modes from a modebench reference JSON."""
    try:
        spec = json.loads(ref) if isinstance(ref, str) else ref
        if bool(spec.get("support_is_open", False)):
            return 0
        return max(int(spec.get("num_completions", 1)), 1)
    except Exception:
        return 1


def _parse_public_seed_key(ref: str) -> str | None:
    """Return a public MathIR seed strategy key when the prompt declares one."""

    try:
        spec = json.loads(ref) if isinstance(ref, str) else ref
        key = spec.get("public_seed_key")
        return str(key) if key else None
    except Exception:
        return None


def _trajectory_mean_logprob(trajectory: TrajectoryData) -> float:
    values = [float(value) for value in trajectory.response_logprobs]
    if not values or any(not math.isfinite(value) or value > 1e-8 for value in values):
        raise RuntimeError(
            "proposal trust checking requires finite non-positive response "
            "log probabilities"
        )
    return float(sum(values) / len(values))


def _exploration_support_key(
    identity: VerifiedExplorationIdentity,
) -> str:
    """Prefer transferable route identity, with endpoint fallback for Graph."""

    return str(identity.route_signature or identity.endpoint_key)


def _compute_mode_coverage_metrics(
    rewards: list[float],
    answer_keys: list,
    answer_mode_count: int,
    public_seed_key: str | None = None,
) -> dict[str, float]:
    k = len(rewards)
    correct_keys = {
        str(key)
        for reward, key in zip(rewards, answer_keys)
        if float(reward) > 0.0 and key is not None
    }
    distinct = len(correct_keys)
    nonseed_keys = (
        correct_keys - {str(public_seed_key)} if public_seed_key is not None else set()
    )
    total = int(answer_mode_count)
    return {
        "any_correct_at_k": float(any(float(r) > 0.0 for r in rewards)),
        "mean_at_k": float(sum(float(r) for r in rewards) / k),
        "distinct_correct_modes_at_k": float(distinct),
        "distinct_nonseed_correct_modes_at_k": float(len(nonseed_keys)),
        "any_nonseed_correct_at_k": float(bool(nonseed_keys)),
        # A growing-support task has no known exhaustive denominator.  Preserve
        # distinct validated outcomes while refusing to report fake coverage.
        "mode_coverage_at_k": (
            float(distinct) / float(total) if total > 0 else float("nan")
        ),
    }


MODE_COVERAGE_METRICS = (
    "mode_coverage_at_k",
    "any_correct_at_k",
    "mean_at_k",
    "distinct_correct_modes_at_k",
    "distinct_nonseed_correct_modes_at_k",
    "any_nonseed_correct_at_k",
)


OPTION_BINDING_METRICS = (
    "option_answer_mi_lower_bound_nats",
    "option_classifier_accuracy",
    "option_eligible_fraction",
    "option_correct_rate_range",
)


def _mode_coverage_log_key(
    benchmark_name: str,
    metric: str,
    k: int,
    *,
    metric_namespace: str = "neutral",
) -> str:
    names = {
        "mode_coverage_at_k": "sampled_mode_coverage",
        "any_correct_at_k": "sampled_any_correct",
        "mean_at_k": "sampled_mean",
        "distinct_correct_modes_at_k": "sampled_distinct_correct",
        "distinct_nonseed_correct_modes_at_k": "sampled_distinct_nonseed_correct",
        "any_nonseed_correct_at_k": "sampled_any_nonseed_correct",
        "option_answer_mi_lower_bound_nats": (
            "sampled_option_answer_mi_lower_bound_nats"
        ),
        "option_classifier_accuracy": "sampled_option_classifier_accuracy",
        "option_eligible_fraction": "sampled_option_eligible_fraction",
        "option_correct_rate_range": "sampled_option_correct_rate_range",
    }
    name = names[metric]
    if metric_namespace == "latent":
        if metric not in MODE_COVERAGE_METRICS:
            raise ValueError("only quality metrics support the latent namespace")
        name = name.replace("sampled_", "sampled_latent_", 1)
    elif metric_namespace != "neutral":
        raise ValueError(f"unknown mode-coverage metric namespace: {metric_namespace}")
    return f"eval/{benchmark_name}/{name}_at_{k}"


def _summarize_mode_coverage_draws(
    draw_means: list[dict[str, float]],
    benchmark_name: str,
    k: int,
    *,
    metric_namespace: str = "neutral",
) -> dict[str, float]:
    """Flatten raw draw means and their Monte Carlo spread into log metrics."""

    if not draw_means:
        return {}
    summary: dict[str, float] = {}
    for metric in draw_means[0]:
        key = _mode_coverage_log_key(
            benchmark_name,
            metric,
            k,
            metric_namespace=metric_namespace,
        )
        values = np.asarray([draw[metric] for draw in draw_means], dtype=float)
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary[key] = float(np.mean(values))
        summary[f"{key}_draw_count"] = float(len(values))
        summary[f"{key}_draw_std"] = standard_deviation
        summary[f"{key}_draw_se"] = standard_deviation / math.sqrt(len(values))
        summary[f"{key}_draw_min"] = float(np.min(values))
        summary[f"{key}_draw_max"] = float(np.max(values))
        for draw_index, value in enumerate(values):
            summary[f"{key}_draw_{draw_index}"] = float(value)
    return summary
