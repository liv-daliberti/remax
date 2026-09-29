"""Frozen admission-to-optimizer behavior for all four maintained methods."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from training_harness import learner_types, make_learner, run_step, snapshot
from training_oracle import reference_update

FIXTURES = Path(__file__).parent / "fixtures" / "training_v1"
METHODS = ("drgrpo", "redr", "maxrl", "remax")


@pytest.fixture(scope="module")
def types():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with learner_types() as classes:
            yield classes
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture
def spec():
    return json.loads((FIXTURES / "inputs.json").read_text())


def assert_close(actual, expected):
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=torch.float64),
        torch.as_tensor(expected, dtype=torch.float64),
        rtol=2e-6,
        atol=2e-7,
    )


def assert_reference(actual, expected):
    # Identity, membership, counts, scheduler state, token labels and masks are
    # exact. Float32 forward/backward/SGD observables have tight tolerances.
    assert actual["bank"] == expected["bank"]
    assert json.loads(json.dumps(actual["replay"])) == json.loads(
        json.dumps(expected["replay"])
    )
    for key in ("scores", "gradient", "parameters"):
        assert_close(actual[key], expected[key])
    assert actual["infos"].keys() == expected["infos"].keys()
    for key in actual["infos"]:
        assert_close(actual["infos"][key], expected["infos"][key])


def check_oracle(learner, before, spec, step, method, *, alpha=None):
    groups = learner.replay_batches[0]["groups"] if learner.replay_batches else []
    expected = reference_update(before, spec, step, groups, method=method, alpha=alpha)
    assert_close(learner.strategy.updates[-1], expected["gradient"])
    assert_close(learner.model.table.detach(), expected["parameters"])
    for phase in ("fresh", "replay"):
        calls = [
            c["gradient"]
            for c in learner.strategy.backward_calls
            if c["phase"] == phase
        ]
        observed = sum(calls, torch.zeros_like(learner.model.table))
        assert_close(observed, expected[phase])
    for grad in (False, True):
        scores = [
            v for c in learner.replay_scores if c["grad"] == grad for v in c["scores"]
        ]
        assert_close(scores, expected["scores"])
    return expected


def test_reference_identity():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert manifest["schema"] == "remax-training-conformance-v1"
    assert len(manifest["baseline_commit"]) == 40
    assert all(c in "0123456789abcdef" for c in manifest["baseline_commit"])
    assert set(manifest["files"]) == {"inputs.json", "expected.json"}
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_frozen_admission_through_optimizer(types, spec, method, microbatch):
    expected = json.loads((FIXTURES / "expected.json").read_text())[method]
    learner = make_learner(types, spec, method, microbatch)
    for step, frozen in zip(spec["steps"], expected):
        before = learner.model.table.detach().tolist()
        updates = len(learner.strategy.updates)
        infos = run_step(learner, spec, step)
        assert_reference(snapshot(learner, infos), frozen)
        oracle = check_oracle(learner, before, spec, step, method)
        assert len(learner.strategy.updates) == updates + 1
        assert learner.model.training  # replay's eval scope must be restored
        assert len(learner.replay_batches) == 1
        rows = len(oracle["scores"])
        chunks = (rows + microbatch - 1) // microbatch
        fresh_calls = spec["num_samples"] // microbatch
        assert [c["phase"] for c in learner.strategy.backward_calls] == [
            "fresh"
        ] * fresh_calls + ["replay"] * chunks
        calls = [c for c in learner.model.calls if c["phase"] == "replay"]
        assert [(c["grad"], c["training"]) for c in calls] == [
            (False, False)
        ] * chunks + [(True, False)] * chunks
        assert sum(c["rows"] for c in calls) == 2 * rows
        assert float(infos["canonical_replay_raw_weighted_loss"]) == pytest.approx(
            oracle["raw_weighted_loss"], rel=2e-6
        )
        if method in ("maxrl", "drgrpo"):
            assert float(infos["canonical_replay_weighted_loss"]) == 0.0
            assert float(infos["canonical_replay_applied_score_gradient_l2"]) == 0.0
            for call in learner.strategy.backward_calls:
                if call["phase"] == "replay":
                    assert call["loss"] == 0.0
                    assert torch.count_nonzero(call["gradient"]) == 0


@pytest.mark.parametrize("method", ["redr", "remax"])
@pytest.mark.parametrize(
    "width,microbatch,temperature,alpha",
    [
        (4, 1, 1.0, 0.0),
        (4, 2, 0.7, 0.3),
        (16, 1, 1.0, 0.1),
        (16, 4, 0.7, 0.3),
    ],
)
def test_coefficients_lengths_and_accumulation_against_scalar_oracle(
    types, spec, method, width, microbatch, temperature, alpha
):
    spec["num_samples"] = width
    spec["temperature"] = temperature
    step = copy.deepcopy(spec["steps"][0])
    for key in ("responses", "rewards", "active"):
        step[key] *= width // 4
    learner = make_learner(types, spec, method, microbatch, alpha=alpha)
    before = learner.model.table.detach().tolist()
    infos = run_step(learner, spec, step)
    check_oracle(learner, before, spec, step, method, alpha=alpha)
    assert float(infos["canonical_replay_objective_scale"]) == 1 / width
    assert (
        float(infos["canonical_replay_reward_estimator_scale"]) == (width - 1) / width
    )
    assert float(infos["canonical_replay_backward_scale"]) == width / microbatch


@pytest.mark.parametrize("control", ["drgrpo", "maxrl"])
def test_compute_matched_control_is_bitwise_fresh_update(types, spec, control):
    control_learner = make_learner(types, spec, control)
    fresh_learner = make_learner(types, spec, control)
    fresh_learner._online_canonical_bank = None
    for step in spec["steps"]:
        run_step(control_learner, spec, step)
        run_step(fresh_learner, spec, step)
        assert torch.equal(control_learner.model.table, fresh_learner.model.table)
        assert torch.equal(
            control_learner.strategy.updates[-1], fresh_learner.strategy.updates[-1]
        )
        assert any(
            c["phase"] == "replay" and c["grad"] for c in control_learner.model.calls
        )
        assert not any(c["phase"] == "replay" for c in fresh_learner.model.calls)


@pytest.mark.parametrize("method", METHODS)
def test_empty_bank_and_singleton_replay(types, spec, method):
    learner = make_learner(types, spec, method)
    failure = spec["steps"][2]
    initial = learner.model.table.detach().clone()
    run_step(learner, spec, failure)
    assert not learner.replay_batches
    assert torch.equal(initial, learner.model.table)
    assert learner._online_canonical_bank.tracked_outcome_count == 0
    # This admission batch has only one eligible mode: actor-negative,
    # verifier-negative and inactive rows must all be excluded.
    run_step(learner, spec, spec["steps"][1])
    before = learner.model.table.detach().tolist()
    infos = run_step(learner, spec, failure)
    oracle = check_oracle(learner, before, spec, failure, method)
    assert len(oracle["scores"]) == 1
    assert float(infos["canonical_replay_eligible_groups"]) == 0.0
    assert float(infos["canonical_replay_actuator_groups"]) == 1.0
    assert torch.count_nonzero(torch.tensor(oracle["fresh"])) == 0
    if method in ("redr", "remax"):
        assert torch.count_nonzero(learner.strategy.updates[-1]) > 0
    else:
        assert torch.equal(learner.model.table, torch.tensor(before))


@pytest.mark.parametrize("method", ["redr", "remax"])
def test_admission_order_does_not_change_update(types, spec, method):
    # Reverse candidate order, including two canonical aliases of unequal length.
    step = spec["steps"][0]
    reverse = copy.deepcopy(step)
    for key in ("responses", "rewards", "active"):
        reverse[key].reverse()
    first, second = [make_learner(types, spec, method) for _ in range(2)]
    a = run_step(first, spec, step)
    b = run_step(second, spec, reverse)
    assert_reference(snapshot(first, a), snapshot(second, b))


@pytest.mark.parametrize("method", METHODS)
def test_bank_and_scheduler_restore_preserve_next_update(types, spec, method):
    learner = make_learner(types, spec, method)
    for step in spec["steps"][:2]:
        run_step(learner, spec, step)
    restored = make_learner(types, spec, method)
    restored.model.load_state_dict(learner.model.state_dict())
    restored.optimizer.load_state_dict(learner.optimizer.state_dict())
    restored._online_canonical_bank.load_state_dict(
        copy.deepcopy(learner._online_canonical_bank.state_dict())
    )
    restored.steps = learner.steps
    a = run_step(learner, spec, spec["steps"][2])
    b = run_step(restored, spec, spec["steps"][2])
    assert_reference(snapshot(learner, a), snapshot(restored, b))


def test_capacity_limits_replay_without_losing_discovery_counts(types, spec):
    spec["capacity"] = 1
    learner = make_learner(types, spec, "remax")
    before = learner.model.table.detach().tolist()
    run_step(learner, spec, spec["steps"][0])
    check_oracle(learner, before, spec, spec["steps"][0], "remax")
    assert learner._online_canonical_bank.tracked_outcome_count == 2
    assert len(learner.replay_batches[0]["groups"][0]["response_token_ids"]) == 1


def test_replay_response_mask_covers_eos_but_not_prompt_or_padding(types, spec):
    learner = make_learner(types, spec, "remax", microbatch=4)
    run_step(learner, spec, spec["steps"][0])
    batch = learner.replay_batches[0]["batch"]
    assert batch.input_ids.tolist() == [[1, 2, 3, 4, 7, 0], [1, 2, 4, 5, 6, 7]]
    assert batch.response_masks.tolist() == [
        [False, True, True, True, False],
        [False, True, True, True, True],
    ]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]
    for c in learner.strategy.backward_calls:
        if c["phase"] == "replay":
            # No labels originate at pad, the interior prompt token, or EOS.
            assert torch.count_nonzero(c["gradient"][[0, 1, 7]]) == 0


@pytest.mark.parametrize("method", METHODS)
def test_replicated_single_rank_layout_matches_reference(types, spec, method, tmp_path):
    # Exercise the production deterministic sharding/layout branch with a real
    # one-rank Gloo group. This does not claim multi-rank or ZeRO validation.
    import torch.distributed as dist

    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1
    )
    try:
        learner = make_learner(types, spec, method)
        learner.args.replicated_freeform_sampling = True
        expected = json.loads((FIXTURES / "expected.json").read_text())[method]
        for step, frozen in zip(spec["steps"], expected):
            infos = run_step(learner, spec, step)
            assert_reference(snapshot(learner, infos), frozen)
    finally:
        dist.destroy_process_group()
