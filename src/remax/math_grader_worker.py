"""JSON-lines worker for full MATH grading in a Python main thread."""

from __future__ import annotations

import json
import sys

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
            payload = {"info": info, "reward": float(reward)}
        except Exception as error:
            payload = {
                "info": {
                    "formatted": False,
                    "verifier_worker_error": type(error).__name__,
                },
                "reward": 0.0,
            }
        sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
