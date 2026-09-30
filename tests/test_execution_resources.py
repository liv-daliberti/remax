import json
from pathlib import Path

import pytest
import torch.multiprocessing as mp
from remax.core.bank import OnlineCanonicalBank
from remax.core.execution import validate_update_partition
from tests.distributed_harness import worker


@pytest.mark.parametrize("world", [1, 2, 4])
def test_distributed_objective_matches_independent_oracle(world, tmp_path):
    output = tmp_path / "result.json"
    mp.spawn(
        worker,
        args=(world, str(tmp_path / "rendezvous"), str(output)),
        nprocs=world,
        join=True,
    )
    cases = json.loads(output.read_text())
    assert len(cases) == 12
    assert all(c["updates"] == 3 for c in cases)


@pytest.mark.parametrize("rows,micro,acc", [(5, 2, 2), (4, 3, 1), (4, 1, 3), (4, 1, 0)])
def test_partial_accumulation_is_rejected(rows, micro, acc):
    with pytest.raises(ValueError):
        validate_update_partition(local_rows=rows, microbatch=micro, accumulation=acc)


def test_discovery_ledger_is_not_the_bounded_exemplar_bank():
    bank = OnlineCanonicalBank(
        entropy_alpha=0, retain_exemplars=True, replay_capacity=2
    )
    for i in range(20):
        bank.score_and_update(
            prompt_token_ids=[[1], [1]],
            outcome_keys=[f"mode:{i}"] * 2,
            task_rewards=[1, 1],
            active_mask=[1, 1],
            num_samples=2,
            response_token_ids=[[2, 3]] * 2,
        )
    counts = bank.resource_counts()
    assert counts["discovered_modes"] == 20
    assert counts["ledger_observations"] == 40
    assert counts["retained_exemplars"] == 2
    assert counts["retained_response_tokens"] == 4
    saved = bank.state_dict()
    for _ in range(20):
        bank.scheduled_global_replay_groups(min_modes=1)
    assert bank.resource_counts() == counts
    restored = OnlineCanonicalBank(
        entropy_alpha=0, retain_exemplars=True, replay_capacity=2
    )
    restored.load_state_dict(saved)
    assert restored.resource_counts() == counts


def test_changing_accumulation_cannot_silently_change_logical_batch():
    with pytest.raises(ValueError, match="configured global batch"):
        validate_update_partition(
            local_rows=16, microbatch=4, accumulation=1, global_batch=16, world_size=1
        )
