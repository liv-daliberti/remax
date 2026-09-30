"""Framework-independent verified replay primitives.

Start with OnlineCanonicalBank.score_and_update(), schedule a replay group,
materialize its causal masks, then score the exemplars with your model and
apply canonical_replay_uniform_verified_likelihood_loss(). See docs/method.md
for the registered coefficient and optimizer-accumulation contract.
"""

from .bank import OnlineCanonicalBank
from .bank_types import OnlineCanonicalBankDiagnostics, VerifiedCanonicalReplayGroup
from .objectives import (
    binary_maxrl_advantages,
    canonical_replay_uniform_verified_likelihood_loss,
)
from .replay_types import CanonicalReplayBatch, CanonicalReplayLoss
from .scoring import materialize_canonical_replay_batch

__all__ = [
    "OnlineCanonicalBank",
    "OnlineCanonicalBankDiagnostics",
    "VerifiedCanonicalReplayGroup",
    "CanonicalReplayBatch",
    "CanonicalReplayLoss",
    "binary_maxrl_advantages",
    "canonical_replay_uniform_verified_likelihood_loss",
    "materialize_canonical_replay_batch",
]
