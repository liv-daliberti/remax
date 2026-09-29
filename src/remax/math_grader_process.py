"""Killable subprocess boundary for the full MATH verifier."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
from typing import Any


class FullMathVerifierProcess:
    """Grade on a clean Python main thread outside the CUDA actor process."""

    def __init__(self, *, reward_kind: str, timeout_seconds: float = 5.0) -> None:
        if reward_kind not in {"boxed", "r1"}:
            raise ValueError(f"unsupported MATH reward kind: {reward_kind}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.reward_kind = reward_kind
        self.timeout_seconds = float(timeout_seconds)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen[str]:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        worker_env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1])
        inherited_pythonpath = worker_env.get("PYTHONPATH")
        worker_env["PYTHONPATH"] = (
            source_root
            if not inherited_pythonpath
            else source_root + os.pathsep + inherited_pythonpath
        )
        self._process = subprocess.Popen(
            [sys.executable, "-m", "remax.math_grader_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
            env=worker_env,
        )
        return self._process

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()

    def close(self) -> None:
        with self._lock:
            self._stop()

    def _grade_one(self, response: str, reference: Any) -> tuple[dict, float]:
        request = json.dumps(
            {
                "reward_kind": self.reward_kind,
                "response": response,
                "reference": reference,
            },
            allow_nan=False,
        )
        for attempt in range(2):
            process = self._start()
            try:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(request + "\n")
                process.stdin.flush()
                readable, _, _ = select.select(
                    [process.stdout], [], [], self.timeout_seconds
                )
                if not readable:
                    self._stop()
                    return {"formatted": False, "verifier_timeout": True}, 0.0
                line = process.stdout.readline()
                if not line:
                    raise BrokenPipeError("MATH verifier worker closed stdout")
                payload = json.loads(line)
                return dict(payload["info"]), float(payload["reward"])
            except (BrokenPipeError, OSError, ValueError, json.JSONDecodeError):
                self._stop()
                if attempt:
                    return {"formatted": False, "verifier_worker_error": True}, 0.0
        raise AssertionError("unreachable")

    def grade_batch(
        self, responses: list[str], references: list[Any]
    ) -> tuple[list[float], list[dict]]:
        rewards: list[float] = []
        infos: list[dict] = []
        with self._lock:
            for response, reference in zip(responses, references):
                info, reward = self._grade_one(response, reference)
                infos.append(info)
                rewards.append(reward)
        return rewards, infos

    def __del__(self) -> None:
        try:
            self._stop()
        except Exception:
            pass
