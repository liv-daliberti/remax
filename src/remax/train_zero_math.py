"""Entry point for Dr.GRPO and xDr.GRPO training on exact-answer tasks."""

from . import fused_adam_shim as _fused_adam_shim

_FUSED_ADAM_SHIM_INSTALLED = _fused_adam_shim.install()

from oat.algorithms.ppo import PPOLearner  # noqa: E402
from oat.args import default_args_validation, get_default_args  # noqa: E402

from .args import ZeroMathArgs, validate_zero_math_args  # noqa: E402
from .learner.base import ZeroMathLearnerBaseMixin  # noqa: E402
from .learner.grpo import ZeroMathGrpoMixin  # noqa: E402
from .learner.init import ZeroMathInitMixin  # noqa: E402
from .learner.run import ZeroMathRunMixin  # noqa: E402
from .runtime import run_zero_math_rl  # noqa: E402


class ZeroMathLearner(
    ZeroMathGrpoMixin,
    ZeroMathLearnerBaseMixin,
    ZeroMathRunMixin,
    ZeroMathInitMixin,
    PPOLearner,
):
    """OAT learner with exact-answer evaluation and candidate aggregation."""


if __name__ == "__main__":
    args: ZeroMathArgs = get_default_args(ZeroMathArgs)
    # OAT's generic validator clamps max_queries to max_train. In this
    # campaign max_train is deliberately the number of unique prompt rows,
    # while num_prompt_epoch controls repeated passes, so that clamp would
    # terminate a five-pass run after only one pool's worth of rollout queries.
    # Preserve an explicit positive query budget; retain upstream behavior only
    # when the caller leaves it unset/non-positive.
    requested_max_queries = args.max_queries
    args.algo = "PPO"
    args.online_evaluation = True
    args = default_args_validation(args)
    if requested_max_queries > 0:
        args.max_queries = requested_max_queries
    run_zero_math_rl(validate_zero_math_args(args))
