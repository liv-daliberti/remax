"""Compatibility imports; maintained losses and masks live in remax.core."""

from .core.replay_types import CanonicalReplayLoss, CanonicalReplayBatch, CanonicalReplaySplitLoss
from .core.scoring import materialize_canonical_replay_batch
from .core.objectives import canonical_replay_uniform_verified_likelihood_loss

_EXPERIMENTAL = ('canonical_replay_key_target_weights', 'canonical_replay_uniform_loss', 'canonical_replay_split_mass_balance_loss', 'project_retention_safe_score_gradients', 'cap_retention_safe_balance_score_gradients')
__all__ = ('CanonicalReplayLoss', 'CanonicalReplayBatch', 'CanonicalReplaySplitLoss', 'materialize_canonical_replay_batch', 'canonical_replay_key_target_weights', 'canonical_replay_uniform_loss', 'canonical_replay_uniform_verified_likelihood_loss', 'canonical_replay_split_mass_balance_loss', 'project_retention_safe_score_gradients', 'cap_retention_safe_balance_score_gradients')

def __getattr__(name):
    if name in _EXPERIMENTAL:
        from .experiments import replay_objectives
        return getattr(replay_objectives, name)
    raise AttributeError(name)
