"""Compatibility entry point for the composed OAT run adapter."""

import torch.distributed as dist
from oat.interface import lp
from tqdm import tqdm
from .support import (
    _grade_decoded_canonical_response,
    _derive_freeform_request_seed,
)
from .sync import OatSyncMixin
from .sampling import OatSamplingMixin
from .progress import OatProgressMixin
from .checkpoints import OatCheckpointsMixin
from .evaluation import OatEvaluationMixin
from .data import OatDataMixin
from .lifecycle import OatLifecycleMixin


class ZeroMathRunMixin(
    OatSyncMixin,
    OatSamplingMixin,
    OatProgressMixin,
    OatCheckpointsMixin,
    OatEvaluationMixin,
    OatDataMixin,
    OatLifecycleMixin,
):

    def _generate_verified_counterfactual_proposals(self, *args, **kwargs):
        from ...experiments.oat.run import HistoricalRunMixin

        return HistoricalRunMixin._generate_verified_counterfactual_proposals(
            self, *args, **kwargs
        )

    def _generate_counterfactual_fixed_control_groups(self, *args, **kwargs):
        from ...experiments.oat.run import HistoricalRunMixin

        return HistoricalRunMixin._generate_counterfactual_fixed_control_groups(
            self, *args, **kwargs
        )

    def _sample_replicated_freeform_feedback(self, *args, **kwargs):
        from ...experiments.oat.run import HistoricalRunMixin

        return HistoricalRunMixin._sample_replicated_freeform_feedback(
            self, *args, **kwargs
        )

    def _collect_dapo_dynamic_feedback(self, *args, **kwargs):
        from ...experiments.oat.run import HistoricalRunMixin

        return HistoricalRunMixin._collect_dapo_dynamic_feedback(self, *args, **kwargs)

    def _record_counterfactual_starvation_outcome(self, *args, **kwargs):
        from ...experiments.oat.run import HistoricalRunMixin

        return HistoricalRunMixin._record_counterfactual_starvation_outcome(
            self, *args, **kwargs
        )

    def _sample_and_admit_canonical_counterfactual_proposals(self, *args, **kwargs):
        from ...experiments.oat.run import HistoricalRunMixin

        return HistoricalRunMixin._sample_and_admit_canonical_counterfactual_proposals(
            self, *args, **kwargs
        )

    def _update_xdr_tau_controller(self, *args, **kwargs):
        if getattr(self, "_xdr_tau_controller", None) is None:
            return
        from ...experiments.oat.run import HistoricalRunMixin

        return HistoricalRunMixin._update_xdr_tau_controller(self, *args, **kwargs)

    def _update_maxent_alpha_controller(self, *args, **kwargs):
        if getattr(self, "_maxent_alpha_controller", None) is None:
            return
        from ...experiments.oat.run import HistoricalRunMixin

        return HistoricalRunMixin._update_maxent_alpha_controller(self, *args, **kwargs)

    def _update_maxent_length_controller(self, *args, **kwargs):
        if getattr(self, "_maxent_length_controller", None) is None:
            return
        from ...experiments.oat.run import HistoricalRunMixin

        return HistoricalRunMixin._update_maxent_length_controller(
            self, *args, **kwargs
        )

    def _update_online_canonical_alpha_controller(self, *args, **kwargs):
        if getattr(self, "_online_canonical_alpha_controller", None) is None:
            return
        from ...experiments.oat.run import HistoricalRunMixin

        return HistoricalRunMixin._update_online_canonical_alpha_controller(
            self, *args, **kwargs
        )


def __getattr__(name):
    from . import support

    try:
        return getattr(support, name)
    except AttributeError:
        raise AttributeError(name) from None
