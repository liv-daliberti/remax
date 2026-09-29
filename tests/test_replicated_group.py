import numpy as np
import pytest

from remax.replicated_group import (
    NUMPY_RANDOMSTATE_SEED_MODULUS,
    replicated_group_permutation_seed,
    validate_replicated_group_layout,
)


def test_four_rank_group_uses_one_microbatch_per_rank():
    layout = validate_replicated_group_layout(
        num_samples=16,
        learner_world_size=4,
        train_batch_size=16,
        train_batch_size_per_device=4,
    )

    assert layout.local_candidate_count == 4
    assert layout.micro_batches_per_rank == 1


def test_one_rank_group_uses_memory_safe_gradient_accumulation():
    layout = validate_replicated_group_layout(
        num_samples=16,
        learner_world_size=1,
        train_batch_size=16,
        train_batch_size_per_device=4,
    )

    assert layout.local_candidate_count == 16
    assert layout.micro_batches_per_rank == 4


@pytest.mark.parametrize(
    ("train_batch_size", "microbatch", "message"),
    [
        (8, 4, "train_batch_size must equal num_samples"),
        (16, 3, "rank shard must divide"),
    ],
)
def test_invalid_logical_or_physical_layout_is_rejected(
    train_batch_size,
    microbatch,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_replicated_group_layout(
            num_samples=16,
            learner_world_size=1,
            train_batch_size=train_batch_size,
            train_batch_size_per_device=microbatch,
        )


def test_permutation_seed_preserves_historical_schedule_before_boundary():
    raw_seed = 43 + 1_000_003 * 4294

    assert raw_seed < NUMPY_RANDOMSTATE_SEED_MODULUS
    assert (
        replicated_group_permutation_seed(
            experiment_seed=43,
            learner_step=4294,
            ppo_epoch=0,
        )
        == raw_seed
    )


def test_permutation_seed_wraps_at_first_failed_mathir_step():
    raw_seed = 43 + 1_000_003 * 4295
    wrapped_seed = replicated_group_permutation_seed(
        experiment_seed=43,
        learner_step=4295,
        ppo_epoch=0,
    )

    assert raw_seed >= NUMPY_RANDOMSTATE_SEED_MODULUS
    assert wrapped_seed == raw_seed % NUMPY_RANDOMSTATE_SEED_MODULUS
    assert 0 <= wrapped_seed < NUMPY_RANDOMSTATE_SEED_MODULUS
    np.random.RandomState(wrapped_seed)


def test_wrapped_permutation_is_resume_stable_and_partitions_all_candidates():
    seed_kwargs = {
        "experiment_seed": 45,
        "learner_step": 4295,
        "ppo_epoch": 0,
    }
    first = np.random.RandomState(
        replicated_group_permutation_seed(**seed_kwargs)
    ).permutation(16)
    resumed = np.random.RandomState(
        replicated_group_permutation_seed(**seed_kwargs)
    ).permutation(16)
    shards = [first[rank * 4 : (rank + 1) * 4] for rank in range(4)]

    assert np.array_equal(first, resumed)
    assert sorted(item for shard in shards for item in shard) == list(range(16))
    assert all(len(shard) == 4 for shard in shards)
