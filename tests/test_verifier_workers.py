"""Real process lifecycle tests: failed work is fatal, next work can restart."""

import json
import os
import subprocess
import sys
import time

import pytest

from remax import benchmark, math_grader
from remax.math_grader_process import FullMathVerifierProcess

REFERENCE = {"verifier": "countdown", "numbers": [1, 2, 3], "target": 6}
BAD_WORKERS = [
    ("timeout", "import time; time.sleep(30)"),
    (
        "timeout",
        "import sys,time; sys.stdout.write('{');sys.stdout.flush();time.sleep(30)",
    ),
    ("worker_failure", "import os; os._exit(7)"),
    ("worker_failure", "print('not json',flush=True)"),
    (
        "worker_failure",
        'print(\'{"info":{"formatted":false},"reward":null}\',flush=True)',
    ),
]


@pytest.mark.parametrize("status,code", BAD_WORKERS)
def test_full_verifier_failure_reaps_process_and_next_request_restarts(
    monkeypatch, status, code
):
    worker = FullMathVerifierProcess(reward_kind="boxed", timeout_seconds=0.2)
    real_popen = subprocess.Popen
    children = []

    def bad_popen(command, **kw):
        process = real_popen([sys.executable, "-c", code], **kw)
        children.append(process)
        return process

    try:
        with monkeypatch.context() as m:
            m.setattr(subprocess, "Popen", bad_popen)
            started = time.monotonic()
            with pytest.raises(benchmark.EvaluationFailure) as caught:
                worker.grade_batch(["\\boxed{1}"], ["1"])
            assert caught.value.diagnostic["status"] == status
            assert caught.value.diagnostic["row_index"] == 0
            assert time.monotonic() - started < 5
            assert worker._process is None
            assert children and all(p.poll() is not None for p in children)
        worker.timeout_seconds = 10
        rewards, _ = worker.grade_batch(["\\boxed{1}"], ["1"])
        assert rewards == [1.0]
    finally:
        worker.close()
    assert worker._process is None


def test_full_worker_structured_failure_payload_is_not_zero(monkeypatch):
    worker = FullMathVerifierProcess(reward_kind="boxed", timeout_seconds=2)
    real_popen = subprocess.Popen
    diagnostic = benchmark.failure(
        "invalid_reference", "upstream reference error"
    ).diagnostic
    payload = {"info": {"formatted": False, "verifier": diagnostic}, "reward": None}

    def spawn(command, **kw):
        return real_popen(
            [
                sys.executable,
                "-c",
                f"import sys;sys.stdin.readline();print({json.dumps(payload)!r},flush=True)",
            ],
            **kw,
        )

    monkeypatch.setattr(subprocess, "Popen", spawn)
    try:
        with pytest.raises(benchmark.EvaluationFailure) as caught:
            worker.grade_batch(["x"], ["1"])
        assert caught.value.diagnostic["status"] == "invalid_reference"
        assert caught.value.diagnostic["detail"] == "upstream reference error"
        assert worker._process is None
    finally:
        worker.close()


def test_full_worker_blocked_write_is_bounded(monkeypatch):
    worker = FullMathVerifierProcess(reward_kind="boxed", timeout_seconds=0.2)
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, **kw: real_popen(
            [sys.executable, "-c", "import time;time.sleep(30)"], **kw
        ),
    )
    try:
        with pytest.raises(benchmark.EvaluationFailure, match="timeout"):
            worker.grade_batch(["x" * 900_000], ["1"])
        assert worker._process is None
    finally:
        worker.close()


@pytest.mark.parametrize("status,code", BAD_WORKERS[:4])
def test_actual_modebench_process_failure_reaches_reward_boundary(
    monkeypatch, status, code
):
    # Test transport internals may inject a process. Production uses only api.grade.
    from modebench import verifier
    from modebench.worker_process import BoundedWorker

    worker = BoundedWorker(timeout_seconds=0.2)
    monkeypatch.setattr(verifier, "_SHARED", worker)
    try:
        with monkeypatch.context() as m:
            m.setattr(worker, "worker_command", lambda: [sys.executable, "-c", code])
            with pytest.raises(benchmark.EvaluationFailure) as caught:
                math_grader.boxed_reward_fn("1+2+3", REFERENCE)
            assert caught.value.diagnostic["status"] == status
            assert worker._process is None
        worker.timeout_seconds = 5
        assert math_grader.boxed_reward_fn("1+2+3", REFERENCE)[1] == 1.0
        dead = worker._process
        dead.kill()
        dead.wait(timeout=2)
        with pytest.raises(benchmark.EvaluationFailure, match="worker_failure"):
            math_grader.boxed_reward_fn("1+2+3", REFERENCE)
        assert math_grader.boxed_reward_fn("1+2+3", REFERENCE)[1] == 1.0
    finally:
        worker.close()


def test_full_worker_reports_exception_without_fabricating_reward():
    # Send an invalid reward request to the real worker's JSON protocol.
    process = subprocess.run(
        [sys.executable, "-m", "remax.math_grader_worker"],
        input="{}\n",
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert payload["reward"] is None
    assert payload["info"]["verifier"]["status"] == "worker_failure"


def test_math_verify_cannot_suppress_its_own_timeout(monkeypatch):
    def slow(*a, **kw):
        raise math_grader.MathVerifyTimeout("deadline")

    wrapped = math_grader._thread_compatible_math_verify_timeout(1)(slow)

    def suppress(*a, **kw):
        try:
            wrapped()
        except math_grader.MathVerifyTimeout:
            return []

    monkeypatch.setattr(math_grader, "parse", suppress)
    monkeypatch.setattr(math_grader, "verify", lambda *a, **kw: False)
    with pytest.raises(benchmark.EvaluationFailure, match="timeout"):
        math_grader.is_latex_equal("2", "1")


def test_legacy_signal_deadline_is_not_an_incorrect_answer(monkeypatch):
    def timeout(*a, **kw):
        raise TimeoutError("signal deadline")

    monkeypatch.setattr(math_grader, "_normalize", timeout)
    with pytest.raises(benchmark.EvaluationFailure, match="timeout"):
        math_grader.is_latex_equal("2", "1")
