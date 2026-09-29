"""Exact logical-group layout for replicated free-form optimization."""

from __future__ import annotations

from dataclasses import dataclass


NUMPY_RANDOMSTATE_SEED_MODULUS = 2**32


@dataclass(frozen=True)
class ReplicatedGroupLayout:
    local_candidate_count: int
    micro_batches_per_rank: int


def replicated_group_permutation_seed(
    *,
    experiment_seed: int,
    learner_step: int,
    ppo_epoch: int,
) -> int:
    """Return the deterministic update seed accepted by NumPy RandomState.

    The historical schedule is preserved exactly until it reaches NumPy's
    unsigned 32-bit boundary. Thereafter it wraps deterministically instead of
    crashing long resumed runs.
    """

    raw_seed = (
        int(experiment_seed)
        + 1_000_003 * int(learner_step)
        + int(ppo_epoch)
    )
    return raw_seed % NUMPY_RANDOMSTATE_SEED_MODULUS


def validate_replicated_group_layout(
    *,
    num_samples: int,
    learner_world_size: int,
    train_batch_size: int,
    train_batch_size_per_device: int,
) -> ReplicatedGroupLayout:
    """Validate one logical candidate group, allowing gradient accumulation.

    Every learner rank receives the complete group so it can compute exact
    group statistics, then optimizes only its deterministic rank shard. A
    shard may contain multiple physical microbatches; DeepSpeed accumulates
    those microbatches into the one logical group update.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if learner_world_size <= 0:
        raise ValueError("learner_world_size must be positive")
    if train_batch_size_per_device <= 0:
        raise ValueError("train_batch_size_per_device must be positive")
    if train_batch_size != num_samples:
        raise ValueError(
            "replicated free-form train_batch_size must equal num_samples"
        )
    if num_samples % learner_world_size:
        raise ValueError(
            "the replicated candidate group must divide across learner ranks"
        )
    local_candidate_count = num_samples // learner_world_size
    if local_candidate_count % train_batch_size_per_device:
        raise ValueError(
            "each replicated rank shard must divide into complete physical "
            "microbatches"
        )
    return ReplicatedGroupLayout(
        local_candidate_count=local_candidate_count,
        micro_batches_per_rank=(
            local_candidate_count // train_batch_size_per_device
        ),
    )
