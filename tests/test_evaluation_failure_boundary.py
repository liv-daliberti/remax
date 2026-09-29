"""Actual oracle and evaluator refuse failed/partial actor results."""

import copy
import json
from types import SimpleNamespace

import pytest
from remax import benchmark
from boundary_runtime_harness import runtime_modules

REFERENCE = {
    "verifier": "countdown",
    "numbers": [1, 2, 3],
    "target": 6,
    "num_completions": 2,
}


@pytest.fixture(scope="module")
def runtime():
    with runtime_modules() as modules:
        yield modules


@pytest.mark.parametrize("version", ["fast", "math_verify"])
@pytest.mark.parametrize("status", sorted(benchmark.UNSCORABLE))
def test_real_actor_oracle_propagates_failure(runtime, monkeypatch, version, status):
    actor, _ = runtime
    oracle = actor.MATHOracle("qwen_boxed", version)
    monkeypatch.setattr(
        benchmark.api,
        "grade",
        lambda *a: benchmark.failure(status, "injected").diagnostic,
    )
    try:
        with pytest.raises(benchmark.EvaluationFailure, match=status):
            oracle.get_reward(["prompt"], ["1+2+3"], [REFERENCE])
    finally:
        if oracle.mp_pool is not None:
            oracle.mp_pool.close()
            oracle.mp_pool.join()
        if oracle.full_math_verifier is not None:
            oracle.full_math_verifier.close()


def evaluator(runtime, payload, tmp_path):
    _, run = runtime
    learner = run.ZeroMathRunMixin()
    learner.args = SimpleNamespace(eval_batch_size=1)
    learner.save_path = str(tmp_path)
    learner.strategy = SimpleNamespace(is_rank_0=lambda: True)
    learner.actors = [
        SimpleNamespace(
            futures=SimpleNamespace(
                generate_for_mode_coverage=lambda *a: SimpleNamespace(
                    result=lambda: payload
                )
            )
        )
    ]
    learner.eval_dataloader_collate_fn = lambda items: (
        ["formatted"],
        ["raw"],
        [REFERENCE],
    )

    def evaluate_verified(*a):
        return learner._run_sampled_mode_coverage_draw(
            [0], k=1, temperature=1.0, seed=7
        )

    learner._evaluate_verified = evaluate_verified
    return learner


@pytest.mark.parametrize(
    "error",
    [
        "timeout",
        "worker_failure",
        "invalid_reference",
        "resource_limit",
        "missing_diagnostics",
        "missing_rows",
        "short_response",
        "legacy_flag",
        "wrong_key",
        "short_options",
    ],
)
def test_evaluation_writes_failure_marker_and_returns_no_metrics(
    runtime, tmp_path, error
):
    good = benchmark.grade_reference_response("1+2+3", REFERENCE)
    info, reward = benchmark.reward_from_diagnostic(good)
    payload = dict(
        rewards=[[reward]],
        answer_keys=[[good["canonical_key"]]],
        responses=[["1+2+3"]],
        verifier_infos=[[info]],
    )
    if error in benchmark.UNSCORABLE:
        payload["verifier_infos"][0][0]["verifier"] = benchmark.failure(
            error, "injected"
        ).diagnostic
        payload["rewards"] = [[0.0]]  # tempting but invalid fallback
    elif error == "missing_diagnostics":
        del payload["verifier_infos"]
    elif error == "missing_rows":
        payload["rewards"] = []
    elif error == "short_response":
        payload["responses"] = [[]]
    elif error == "wrong_key":
        payload["answer_keys"] = [["fabricated"]]
    elif error == "short_options":
        payload["option_ids"] = []
    elif error == "legacy_flag":
        payload["verifier_infos"] = [[{"verifier_timeout": True}]]
    learner = evaluator(runtime, payload, tmp_path)
    with pytest.raises(benchmark.EvaluationFailure):
        learner.evaluate(None, 12)
    record = json.loads((tmp_path / "evaluation_failures.jsonl").read_text())
    assert (
        record["status"] == "failed"
        and record["metrics"] is None
        and record["step"] == 12
    )
    assert not (tmp_path / "eval_mode_coverage_draws.jsonl").exists()


def test_valid_evaluation_keeps_structured_status_and_historical_score(
    runtime, tmp_path
):
    info, reward = benchmark.reward_from_diagnostic(
        benchmark.grade_reference_response("1+2+3", REFERENCE)
    )
    payload = dict(
        rewards=[[reward]],
        answer_keys=[[info["verifier"]["canonical_key"]]],
        responses=[["1+2+3"]],
        verifier_infos=[[info]],
    )
    metrics, rows = evaluator(runtime, payload, tmp_path).evaluate(None, 12)
    assert metrics["mean_at_k"] == 1.0 and metrics["distinct_correct_modes_at_k"] == 1.0
    assert rows[0]["verifier_infos"][0]["verifier"]["status"] == "correct"
    assert not (tmp_path / "evaluation_failures.jsonl").exists()


def test_remote_rank_failure_precedes_score_broadcast(runtime, monkeypatch):
    _, run = runtime
    learner = run.ZeroMathRunMixin()
    learner.args = SimpleNamespace()
    calls = []
    learner._pre_evaluate = lambda: calls.append("pre")
    learner._post_evaluate = lambda: calls.append("post")
    learner._run_sampled_mode_coverage = lambda *a, **kw: {}
    monkeypatch.setattr(run.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(run.dist, "get_world_size", lambda: 2)

    def gather(out, diagnostic):
        assert diagnostic is None
        out[:] = [
            benchmark.failure("timeout", "rank 0 verifier failed").diagnostic,
            None,
        ]

    monkeypatch.setattr(run.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        run.dist,
        "broadcast",
        lambda *a, **kw: pytest.fail("no score broadcast on failure"),
    )
    with pytest.raises(benchmark.EvaluationFailure, match="timeout"):
        learner.evaluate_mode_coverage([], "eval", 1, k=1, temperature=1.0)
    assert calls == ["pre", "post"]


def test_trajectory_dataset_keeps_verifier_diagnostics(runtime):
    from remax.trajectory_dataset import ZeroMathTrajectoryDataset

    diagnostic = benchmark.failure("timeout", "transported failure").diagnostic
    item = SimpleNamespace(
        prompt="p",
        response="r",
        prompt_ids=[1],
        response_ids=[2],
        rewards=[0.0],
        loss_mask=True,
        response_logprobs=[-0.5],
        reference=REFERENCE,
        verifier_diagnostic=diagnostic,
    )
    dataset = ZeroMathTrajectoryDataset(
        [item],
        SimpleNamespace(pad_token_id=0),
        SimpleNamespace(is_rank_0=lambda: False),
    )
    batch = dataset.collate_fn([dataset[0]])
    assert batch["verifier_diagnostics"] == [diagnostic]


def test_oracle_rejects_legacy_worker_zero_fallback(runtime, monkeypatch):
    actor, _ = runtime
    oracle = actor.MATHOracle("qwen_boxed", "math_verify")
    monkeypatch.setattr(
        oracle.full_math_verifier,
        "grade_batch",
        lambda *a: ([0.0], [{"formatted": False, "verifier_timeout": True}]),
    )
    try:
        with pytest.raises(benchmark.EvaluationFailure):
            oracle.get_reward(["p"], ["response"], [REFERENCE])
    finally:
        oracle.full_math_verifier.close()


def test_all_canonical_pantry_masks_keep_reference_reward_and_identity(runtime):
    from pathlib import Path
    from modebench.api_types import Task
    from remax.canonical_actions import decode_canonical_action_response

    _, run = runtime
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/benchmark_boundary_v1.json").read_text()
    )
    reference = next(
        c["reference"]
        for c in fixture["cases"]
        if c["id"] == "level1_pantry_plan/accepted_a"
    )
    task = Task(
        id="all-pantry-actions",
        level=1,
        domain="pantry_plan",
        problem="",
        answer=reference,
    )
    rejected = accepted = 0
    for number in range(64):
        mask = f"{number:06b}"
        decoded = decode_canonical_action_response(
            "pantry_support_mask", mask, reference
        )
        info, reward = run._grade_decoded_canonical_response(
            decoded, reference, fast=True
        )
        expected = benchmark.grade_task(task, mask)
        assert reward == float(expected["verified"])
        assert info["verifier"]["canonical_key"] == expected["canonical_key"]
        if reward:
            accepted += 1
        else:
            rejected += 1
    assert accepted and rejected


@pytest.mark.parametrize("status", sorted(benchmark.UNSCORABLE))
def test_canonical_learner_never_accepts_fatal_grading_status(
    runtime, monkeypatch, status
):
    _, run = runtime
    monkeypatch.setattr(
        benchmark.api,
        "grade",
        lambda *a: benchmark.failure(status, "injected").diagnostic,
    )
    with pytest.raises(benchmark.EvaluationFailure, match=status):
        run._grade_decoded_canonical_response("1+2+3", REFERENCE, fast=True)
