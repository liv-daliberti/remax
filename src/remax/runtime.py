"""Runtime helpers for zero-math training launches."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from oat.interface import get_program, lp


def resolve_fixed_oat_exp_suffix() -> str | None:
    """Return a per-job save-path suffix shared by every learner rank."""

    value = os.environ.get("OAT_ZERO_FIXED_EXP_SUFFIX")
    if value is None:
        return None
    value = value.strip()
    return value or None


@contextmanager
def patch_oat_learner_datetime(fixed_suffix: str | None):
    """Force OAT's learner workspace suffix to stay identical across ranks."""

    if not fixed_suffix:
        yield
        return

    import oat.learners.base as oat_learner_base

    original_datetime = oat_learner_base.datetime

    class _FixedNow:
        def strftime(self, _fmt: str) -> str:
            return fixed_suffix

    class _FixedDateTime:
        @staticmethod
        def now() -> _FixedNow:
            return _FixedNow()

    oat_learner_base.datetime = _FixedDateTime
    try:
        yield
    finally:
        oat_learner_base.datetime = original_datetime


def launch_zero_math_program(
    args: Any,
    *,
    learner_cls: type,
    actor_cls: type,
) -> None:
    """Compose and launch the local OAT actor/learner program."""

    program, local_resources = get_program(
        args,
        learner_cls=learner_cls,
        actor_cls=actor_cls,
    )
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


def run_zero_math_rl(args: Any) -> None:
    """Launch the default zero-math actor/learner program."""

    from .actor import ZeroMathActor
    from .train_zero_math import ZeroMathLearner

    launch_zero_math_program(
        args,
        learner_cls=ZeroMathLearner,
        actor_cls=ZeroMathActor,
    )
