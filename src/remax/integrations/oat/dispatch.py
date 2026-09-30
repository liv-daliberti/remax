"""Compatibility entry point; maintained code lives in integrations.oat."""

from .grpo import ZeroMathGrpoMixin as MaintainedGrpoMixin
from .selection import historical_reasons


def _implementation(learner):
    reasons = historical_reasons(learner)
    learner._remax_adapter = "historical" if reasons else "maintained"
    learner._remax_adapter_reasons = reasons
    if reasons:
        if getattr(learner.args, "_remax_resume_identity", None) is not None:
            raise RuntimeError(
                f"maintained recipe requires historical adapter: {reasons}"
            )
        from ...experiments.oat.grpo import HistoricalGrpoMixin

        return HistoricalGrpoMixin
    return MaintainedGrpoMixin


class ZeroMathGrpoMixin(MaintainedGrpoMixin):
    """Route historical settings without importing their optional integrations."""

    def _seed_answer_keys_grouped(self, *args, **kwargs):
        from ...experiments.oat.grpo import HistoricalGrpoMixin

        return HistoricalGrpoMixin._seed_answer_keys_grouped(self, *args, **kwargs)

    def _gapo_support_index(self, *args, **kwargs):
        from ...experiments.oat.grpo import HistoricalGrpoMixin

        return HistoricalGrpoMixin._gapo_support_index(self, *args, **kwargs)

    def _setpo_embedder(self, *args, **kwargs):
        from ...experiments.oat.grpo import HistoricalGrpoMixin

        return HistoricalGrpoMixin._setpo_embedder(self, *args, **kwargs)

    def _setpo_response_surfaces(self, *args, **kwargs):
        from ...experiments.oat.grpo import HistoricalGrpoMixin

        return HistoricalGrpoMixin._setpo_response_surfaces(self, *args, **kwargs)

    def _score_rlep_rows(self, *args, **kwargs):
        from ...experiments.oat.grpo import HistoricalGrpoMixin

        return HistoricalGrpoMixin._score_rlep_rows(self, *args, **kwargs)

    def _baseline_update_with_precomputed_advantages(self, *args, **kwargs):
        return _implementation(self)._baseline_update_with_precomputed_advantages(
            self, *args, **kwargs
        )

    def _grpo_learning_step_with_progress(self, *args, **kwargs):
        return _implementation(self)._grpo_learning_step_with_progress(
            self, *args, **kwargs
        )


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    from ...experiments.oat import grpo

    try:
        return getattr(grpo, name)
    except AttributeError:
        raise AttributeError(name) from None
