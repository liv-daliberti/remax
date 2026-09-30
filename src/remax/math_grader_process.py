"""Killable full-MATH subprocess; failures are never numeric model rewards."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from .benchmark import (
    EvaluationFailure,
    failure,
    parse_reference,
    validate_reward_batch,
)

MAX_REQUEST_BYTES = 1_048_576
MAX_REPLY_BYTES = 262_144


class FullMathVerifierProcess:
    def __init__(self, *, reward_kind: str, timeout_seconds: float = 5.0) -> None:
        if reward_kind not in {"boxed", "r1"}:
            raise ValueError(f"unsupported MATH reward kind: {reward_kind}")
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        self.reward_kind = reward_kind
        self.timeout_seconds = float(timeout_seconds)
        self._process = None
        self._lock = threading.Lock()

    def _start(self):
        if self._process is not None:
            if self._process.poll() is None:
                return self._process
            self._stop()
            raise OSError("MATH verifier worker exited between requests")
        worker_env = os.environ.copy()
        self._process = subprocess.Popen(
            [sys.executable, "-I", "-m", "remax.math_grader_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
            start_new_session=True,
            env=worker_env,
        )
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
        return self._process

    def _stop(self):
        process, self._process = self._process, None
        if process is None:
            return
        # Reap this worker's process group even when its leader has died.
        # ModeBench requests bypass this process and own their own lifecycle.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=0.5)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    def close(self):
        with self._lock:
            self._stop()

    def _grade_one(self, response: str, reference: Any):
        # ModeBench owns an isolated, bounded worker already. Avoid wrapping it
        # in a second process/deadline (and losing its structured diagnostic).
        if parse_reference(reference) is not None:
            from .math_grader import answer_tag_reward_fn, boxed_reward_fn

            grade = (
                answer_tag_reward_fn if self.reward_kind == "r1" else boxed_reward_fn
            )
            return grade(response, reference, fast=False)
        try:
            request = (
                json.dumps(
                    dict(
                        reward_kind=self.reward_kind,
                        response=response,
                        reference=reference,
                    ),
                    allow_nan=False,
                )
                + "\n"
            ).encode()
        except (ValueError, TypeError, RecursionError) as error:
            raise failure(
                "invalid_reference", "reward request is not finite JSON"
            ) from error
        if len(request) > MAX_REQUEST_BYTES:
            raise failure("resource_limit", "reward request byte limit exceeded")
        try:
            deadline = time.monotonic() + self.timeout_seconds
            process = self._start()
            received, sent = bytearray(), 0
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdin, selectors.EVENT_WRITE)
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("full verifier request deadline exceeded")
                    for event, _ in selector.select(remaining):
                        try:
                            if event.fileobj is process.stdin:
                                sent += os.write(
                                    process.stdin.fileno(), request[sent : sent + 4096]
                                )
                                if sent == len(request):
                                    selector.unregister(process.stdin)
                            else:
                                chunk = os.read(process.stdout.fileno(), 65536)
                                if not chunk:
                                    raise OSError("full verifier worker closed stdout")
                                received.extend(chunk)
                        except BlockingIOError:
                            continue
                    if len(received) > MAX_REPLY_BYTES:
                        raise ValueError("full verifier reply byte limit exceeded")
                    if b"\n" not in received:
                        continue
                    line, extra = received.split(b"\n", 1)
                    if extra.strip():
                        raise ValueError("unsolicited full verifier output")
                    payload = json.loads(line)
                    info, reward = payload["info"], payload["reward"]
                    validate_reward_batch(
                        [reward], [info], count=1, context="full_verifier"
                    )
                    return info, float(reward)
        except EvaluationFailure:
            self._stop()
            raise
        except TimeoutError as error:
            self._stop()
            raise failure("timeout", str(error), context="full_verifier") from error
        except (OSError, ValueError, TypeError, KeyError) as error:
            self._stop()
            raise failure(
                "worker_failure",
                f"{type(error).__name__}: {error}",
                context="full_verifier",
            ) from error
        except BaseException:
            self._stop()
            raise

    def grade_batch(self, responses, references):
        if len(responses) != len(references):
            raise failure(
                "invalid_reference", "response/reference batch lengths differ"
            )
        rewards, infos = [], []
        with self._lock:
            for index, (response, reference) in enumerate(zip(responses, references)):
                try:
                    info, reward = self._grade_one(response, reference)
                except EvaluationFailure as error:
                    raise EvaluationFailure(
                        dict(error.diagnostic, row_index=index)
                    ) from error
                rewards.append(reward)
                infos.append(info)
        validate_reward_batch(
            rewards,
            infos,
            count=len(responses),
            context="full_verifier_batch",
            references=references,
        )
        return rewards, infos

    def __del__(self):
        try:
            self._stop()
        except Exception:
            pass
