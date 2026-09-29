"""Logging policy for the compact public experiment surface."""

from __future__ import annotations

import os
from typing import Any

WANDB_DEBUG_METRIC_ENV = "OAT_ZERO_WANDB_LOG_DEBUG_METRICS"

_DEBUG_KEYS = {
    "train/get_grad_norm_time",
    "train/logprobs_diff_max",
    "train/logprobs_diff_min",
    "train/zero_pg_loss_count",
}


def filter_wandb_logs(logs: dict[str, Any]) -> dict[str, Any]:
    """Hide low-level PPO diagnostics unless explicitly requested."""

    debug = os.environ.get(WANDB_DEBUG_METRIC_ENV, "").strip().lower()
    if debug in {"1", "true", "yes", "on"}:
        return logs
    return {key: value for key, value in logs.items() if key not in _DEBUG_KEYS}
