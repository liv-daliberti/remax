"""The GPU artifact audit must compare values and reject meaningful divergence."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "resume_gpu_audit", Path(__file__).parents[1] / "ops/resume_gpu.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("difference", ["bank", "rng", "scheduler", "shape", "dtype"])
def test_exact_state_comparison_rejects_drift(difference):
    left = {
        "bank": {"cursor": 2},
        "rng": np.array([1, 2]),
        "scheduler": 4,
        "tensor": torch.tensor([1.0, 2.0]),
    }
    right = {**left}
    if difference == "bank":
        right["bank"] = {"cursor": 3}
    elif difference == "rng":
        right["rng"] = np.array([2, 1])
    elif difference == "scheduler":
        right["scheduler"] = 5
    elif difference == "shape":
        right["tensor"] = torch.tensor([[1.0, 2.0]])
    else:
        right["tensor"] = left["tensor"].double()
    with pytest.raises(ValueError):
        audit.compare_tree(left, right)


def test_numerical_tolerances_are_explicit_and_nan_is_never_equal():
    original = torch.tensor([0.0], dtype=torch.float64)
    audit.compare_tree(original, original + 5e-7, **audit.TOLERANCES["model"])
    with pytest.raises(ValueError, match="beyond tolerance"):
        audit.compare_tree(original, original + 2e-6, **audit.TOLERANCES["model"])
    with pytest.raises(ValueError, match="nonfinite"):
        audit.compare_tree(torch.tensor([float("nan")]), torch.tensor([float("nan")]))


def test_deepspeed_loss_scaler_compares_serialized_state():
    # Match the pickle object's protocol without requiring DeepSpeed in CPU CI.
    scaler_type = type(
        "LossScaler", (), {"__module__": "deepspeed.runtime.fp16.loss_scaler"}
    )
    left, right = scaler_type(), scaler_type()
    left.cur_scale = right.cur_scale = 1.0
    left.dynamic = right.dynamic = False
    audit.compare_tree({"loss_scaler": left}, {"loss_scaler": right})
    right.cur_scale = 2.0
    with pytest.raises(ValueError, match="cur_scale"):
        audit.compare_tree({"loss_scaler": left}, {"loss_scaler": right})


@pytest.mark.parametrize("failure", ["empty", "missing_resume", "failed", "tolerances"])
def test_audit_cannot_pass_incomplete_experiments(failure):
    experiment = {
        "schema": "remax-resume-gpu-v1",
        "tolerances": audit.TOLERANCES,
        "runs": {"remax-whole": {"returncode": 0}, "remax-resumed": {"returncode": 0}},
    }
    if failure == "empty":
        experiment["runs"] = {}
    elif failure == "missing_resume":
        del experiment["runs"]["remax-resumed"]
    elif failure == "failed":
        experiment["runs"]["remax-resumed"]["returncode"] = 1
    else:
        experiment["tolerances"] = {"model": {"atol": 1}}
    with pytest.raises(ValueError):
        audit.completed_methods(experiment)


def test_large_optimizer_comparison_checks_every_chunk_and_rejects_infinity():
    left = torch.zeros(1_000_003)
    right = left.clone()
    audit.compare_tree(left, right)
    right[-1] = 1
    with pytest.raises(ValueError, match="beyond tolerance"):
        audit.compare_tree(left, right)
    with pytest.raises(ValueError, match="nonfinite"):
        audit.compare_tree(torch.tensor([float("inf")]), torch.tensor([float("inf")]))
