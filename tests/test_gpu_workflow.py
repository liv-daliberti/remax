"""CPU checks for the GPU runner's immutable inputs and evidence acceptance gates."""

import copy
import json
from pathlib import Path

import pytest

from ops.audit_gpu_smoke import audit_draws, audit_metrics
from ops.compare_gpu_recipe import REFERENCE, compare
from ops.gpu_smoke import (
    METHODS,
    OVERRIDES,
    digest,
    recipe,
    smoke_environment,
    validate_inputs,
)


def test_smoke_preserves_all_method_settings(tmp_path):
    inputs = {"model": str(tmp_path / "model")}
    for method in METHODS:
        env = smoke_environment(method, tmp_path, inputs)
        for key, value in recipe(method)["environment"].items():
            if key not in OVERRIDES:
                assert env[key] == value
        assert env["OAT_ZERO_SAVE_CKPT"] == "1"
        assert env["OAT_ZERO_PRUNE_RESUME_ON_SUCCESS"] == "0"
        assert env["OAT_ZERO_AUTO_RESUME"] == "0"


def test_prepared_inputs_reject_changed_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "model.safetensors"
    weight.write_bytes(b"frozen model")
    data = tmp_path / "data.arrow"
    data.write_bytes(b"frozen data")
    payload = {
        "model": str(model),
        "model_sha256": {weight.name: digest(weight)},
        "data_sha256": {data.name: digest(data)},
    }
    (tmp_path / "inputs.json").write_text(json.dumps(payload))
    assert validate_inputs(tmp_path) == payload
    data.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="Dataset changed"):
        validate_inputs(tmp_path)
    weight.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="Model changed"):
        validate_inputs(tmp_path)


def test_historical_comparison_detects_method_drift():
    baseline = json.loads(REFERENCE.read_text())
    result = compare(recipe("remax"), baseline)
    assert result["registered_environment_fields_matched"] == 85
    assert result["runtime_differences"]["eval_steps"] == {
        "historical": 96,
        "current": "192",
    }
    changed = recipe("remax")
    changed["environment"]["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] = "0.2"
    with pytest.raises(ValueError, match="method settings differ"):
        compare(changed, baseline)


def metric_row(control):
    return {
        "trainer/policy_sgd_step": 4,
        "train/canonical_replay_score_passes": 2,
        "train/canonical_replay_compute_only": float(control),
        "train/canonical_replay_raw_weighted_loss": 0.04,
        "train/canonical_replay_weighted_loss": 0 if control else 0.04,
        "train/canonical_replay_applied_score_gradient_l2": 0 if control else 0.001,
    }


@pytest.mark.parametrize("method", METHODS)
def test_gpu_evidence_rejects_missing_replay_or_updates(method):
    row = metric_row(method in ("drgrpo", "maxrl"))
    assert audit_metrics([row], method)["optimizer_updates"] == 4
    for key, value in [
        ("train/canonical_replay_score_passes", 0),
        ("trainer/policy_sgd_step", 0),
        ("train/canonical_replay_raw_weighted_loss", 0),
        ("train/canonical_replay_weighted_loss", float("nan")),
    ]:
        changed = {**row, key: value}
        with pytest.raises(ValueError):
            audit_metrics([changed], method)


def test_gpu_evidence_rejects_control_gradient_and_inactive_treatment():
    with pytest.raises(ValueError, match="control applies"):
        audit_metrics(
            [
                {
                    **metric_row(True),
                    "train/canonical_replay_applied_score_gradient_l2": 0.1,
                }
            ],
            "drgrpo",
        )
    with pytest.raises(ValueError, match="never applied"):
        audit_metrics(
            [
                {
                    **metric_row(False),
                    "train/canonical_replay_applied_score_gradient_l2": 0,
                }
            ],
            "remax",
        )


def test_gpu_evidence_rejects_partial_and_failed_evaluation():
    draw = {
        "sample_count": 1,
        "prompts": [
            {"verifier_infos": [{"verifier": {"status": "correct"}}]} for _ in range(2)
        ],
    }
    assert audit_draws([draw]) == 2
    failed = copy.deepcopy(draw)
    failed["prompts"][0]["verifier_infos"][0]["verifier"]["status"] = "timeout"
    with pytest.raises(ValueError, match="Fatal"):
        audit_draws([failed])
    with pytest.raises(ValueError, match="Incomplete"):
        audit_draws([{**draw, "sample_count": 2}])
