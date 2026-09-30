"""Supported API, historical recipe compatibility, and fatal diagnostics."""

import ast
import copy
import json
from pathlib import Path
import pickle
from multiprocessing.pool import ThreadPool
from unittest.mock import patch

import pytest
import torch
import remax
from modebench.api import Task

from remax import benchmark, math_grader
from tests.training_harness import learner_types, make_learner, trajectory

ROOT = Path(__file__).resolve().parents[1]
FROZEN = json.loads((ROOT / "tests/fixtures/benchmark_boundary_v1.json").read_text())
REFERENCE = {"verifier": "countdown", "numbers": [1, 2, 3], "target": 6}


@pytest.mark.parametrize("case", FROZEN["cases"], ids=lambda c: c["id"])
def test_historical_rewards_and_keys(case):
    info, reward = math_grader.boxed_reward_fn(
        case["response"], case["reference"], fast=True
    )
    key = math_grader.validated_modebench_outcome_key(
        case["response"], case["reference"]
    )
    assert dict(reward=reward, canonical_key=key) == case["expected"]
    assert info["verifier"]["status"] in benchmark.SCORABLE
    assert info["verifier"]["canonical_key"] == key
    assert (
        math_grader.extract_normalized_final_answer(
            case["response"], gt_answer=case["reference"]
        )
        == key
    )


def test_historical_fixture_covers_all_registered_cells():
    assert {(c["level"], c["domain"]) for c in FROZEN["cases"]} == {
        (level, domain)
        for level in range(1, 6)
        for domain in (
            "countdown",
            "graph_coloring",
            "python_factors",
            "mathir",
            "pantry_plan",
        )
    }


def test_raw_pantry_mask_and_historical_decoded_allocation_agree():
    case = next(
        c for c in FROZEN["cases"] if c["id"] == "level1_pantry_plan/accepted_a"
    )
    task = Task(
        id="pantry", level=1, domain="pantry_plan", problem="", answer=case["reference"]
    )
    raw = benchmark.grade_task(task, case["raw_response"])
    decoded = benchmark.grade_reference_response(case["response"], case["reference"])
    assert raw["verified"] and decoded["verified"]
    assert (
        raw["canonical_key"]
        == decoded["canonical_key"]
        == case["expected"]["canonical_key"]
    )


def test_maintained_grading_imports_no_private_helpers():
    for name in ("benchmark.py", "math_grader.py"):
        tree = ast.parse((Path(remax.__file__).parent / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "modebench"
            ):
                assert all(not alias.name.startswith("_") for alias in node.names)


@pytest.mark.parametrize("status", sorted(benchmark.UNSCORABLE))
@pytest.mark.parametrize("entry", ["reward", "key", "display", "r1", "orz", "route"])
def test_verifier_failures_never_become_rewards_or_missing_keys(
    monkeypatch, status, entry
):
    diagnostic = benchmark.failure(
        status, "injected failure", worker_pid=123
    ).diagnostic
    monkeypatch.setattr(benchmark.api, "grade", lambda *a: diagnostic)
    functions = {
        "reward": lambda: math_grader.boxed_reward_fn("1+2+3", REFERENCE),
        "key": lambda: math_grader.validated_modebench_outcome_key("1+2+3", REFERENCE),
        "display": lambda: math_grader.extract_normalized_final_answer(
            "1+2+3", gt_answer=REFERENCE
        ),
        "r1": lambda: math_grader.answer_tag_reward_fn("bad formatting", REFERENCE),
        "orz": lambda: math_grader.answer_tag_reward_fn_for_orz(
            "bad formatting", REFERENCE
        ),
        "route": lambda: math_grader.validated_exploration_identity(
            "1+2+3", "", REFERENCE
        ),
    }
    with pytest.raises(benchmark.EvaluationFailure) as caught:
        functions[entry]()
    assert caught.value.diagnostic == diagnostic
    assert pickle.loads(pickle.dumps(caught.value)).diagnostic == diagnostic


@pytest.mark.parametrize(
    "reference",
    [
        dict(REFERENCE, target=None),
        {"verifier": "unknown"},
        {"verifier": []},
        '{"numbers": [1, 2, 3]',
        "{}",
        '{"verifier":',
    ],
)
def test_invalid_references_fail_even_for_malformed_responses(reference):
    with pytest.raises(benchmark.EvaluationFailure) as caught:
        math_grader.boxed_reward_fn("garbage", reference)
    assert caught.value.diagnostic["status"] == "invalid_reference"


@pytest.mark.parametrize(
    "status", ["timeout", "worker_failure", "invalid_reference", "resource_limit"]
)
def test_learner_failure_precedes_bank_mutation_and_optimizer(monkeypatch, status):
    spec = json.loads((ROOT / "tests/fixtures/training_v1/inputs.json").read_text())
    with learner_types() as types:
        learner = make_learner(types, spec, "remax")
        before_bank = copy.deepcopy(learner._online_canonical_bank.state_dict())
        before_params = learner.model.table.detach().clone()
        count = 0
        real = benchmark.api.grade

        def injected(task, response):
            nonlocal count
            count += 1
            return (
                real(task, response)
                if count < 3
                else benchmark.failure(status, "row 2 failed").diagnostic
            )

        monkeypatch.setattr(benchmark.api, "grade", injected)
        with patch("torch.cuda.current_device", return_value=torch.device("cpu")):
            with pytest.raises(benchmark.EvaluationFailure):
                learner._grpo_learning_step_with_progress(
                    trajectory(spec, spec["steps"][0])
                )
        assert learner._online_canonical_bank.state_dict() == before_bank
        assert torch.equal(learner.model.table, before_params)
        assert not learner.strategy.backward_calls and not learner.strategy.updates


def test_transport_diagnostic_cannot_be_hidden_by_zero_reward():
    spec = json.loads((ROOT / "tests/fixtures/training_v1/inputs.json").read_text())
    with learner_types() as types:
        learner = make_learner(types, spec, "remax")
        batch = trajectory(spec, spec["steps"][2])
        batch["verifier_diagnostics"] = [
            benchmark.failure("timeout", "actor failed").diagnostic
        ] + [None] * 3
        with pytest.raises(benchmark.EvaluationFailure, match="timeout"):
            learner._grpo_learning_step_with_progress(batch)
        assert learner._online_canonical_bank.tracked_outcome_count == 0
        assert not learner.model.calls


@pytest.mark.parametrize(
    "kind", ["timeout", "crash", "legacy_flag", "missing_row", "missing_diagnostic"]
)
def test_reward_batch_failure_policy(kind):
    from multiprocessing import TimeoutError

    class Pending:
        def get(self, timeout):
            if kind == "timeout":
                raise TimeoutError()
            if kind == "crash":
                raise RuntimeError("broken worker")
            return {"formatted": False, "verifier_worker_error": True}, 0.0

    class Pool:
        def apply_async(self, *a):
            return Pending()

    if kind in ("timeout", "crash", "legacy_flag"):
        with pytest.raises(benchmark.EvaluationFailure):
            math_grader.collect_threaded_math_rewards(
                Pool(), None, ["x"], ["1"], timeout_seconds=0.01
            )
    else:
        with pytest.raises(benchmark.EvaluationFailure):
            benchmark.validate_reward_batch(
                [0.0],
                [{}],
                count=2 if kind == "missing_row" else 1,
                context="eval",
                references=[REFERENCE],
            )


def test_modebench_uses_its_own_deadline_not_outer_thread_timeout():
    class NoThreads:
        def apply_async(self, *a):
            raise AssertionError("ModeBench must own its deadline")

    rewards, infos = math_grader.collect_threaded_math_rewards(
        NoThreads(),
        math_grader.boxed_reward_fn,
        ["1+2+3"],
        [REFERENCE],
        timeout_seconds=0.000001,
    )
    assert rewards == [1.0] and infos[0]["verifier"]["status"] == "correct"


@pytest.mark.parametrize(
    "fn", [math_grader.answer_tag_reward_fn, math_grader.answer_tag_reward_fn_for_orz]
)
def test_historical_format_rejection_retains_nonfatal_diagnostic(fn):
    info, reward = fn("1+2+3", REFERENCE)
    assert reward == 0.0 and info["verifier"]["status"] == "malformed"
    benchmark.validate_reward_batch(
        [reward], [info], count=1, context="test", references=[REFERENCE]
    )
    info, reward = fn("</think> <answer>1+2+3</answer>", REFERENCE)
    assert reward == 1.0 and info["verifier"]["status"] == "correct"
