"""JSON-lines worker for full MATH grading in a Python main thread."""

from __future__ import annotations

import json
import sys

from .benchmark import EvaluationFailure, failure, validate_reward_batch
from .math_grader import answer_tag_reward_fn, boxed_reward_fn


def main() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            reward_fn = (
                answer_tag_reward_fn
                if request["reward_kind"] == "r1"
                else boxed_reward_fn
            )
            info, reward = reward_fn(
                request["response"], request["reference"], fast=False
            )
            validate_reward_batch(
                [reward], [info], count=1, context="full_verifier_worker"
            )
            payload = {"info": info, "reward": float(reward)}
        except EvaluationFailure as error:
            payload = {
                "info": {"formatted": False, "verifier": error.diagnostic},
                "reward": None,
            }
        except Exception as error:
            diagnostic = failure(
                "worker_failure", f"{type(error).__name__}: {error}"
            ).diagnostic
            payload = {
                "info": {"formatted": False, "verifier": diagnostic},
                "reward": None,
            }
        sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
