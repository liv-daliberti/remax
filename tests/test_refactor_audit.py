"""A comparison must reject incomplete or scientifically different runs first."""

import json
import sys

import pytest

from tests import compare_training_runs as comparison


@pytest.mark.parametrize(
    "fault,match",
    [
        ("failed_run", "complete successful"),
        ("data_identity", "non-source run identity"),
        ("same_source", "distinct source revisions"),
        ("ambiguous_run", "exactly one training run"),
        ("partial_trace", "incomplete six-update"),
        ("different_evaluation", "eval_mode_coverage_draws.jsonl differs"),
    ],
)
def test_invalid_comparison_stops_before_loading_model(
    tmp_path, monkeypatch, fault, match
):
    roots = [tmp_path / name for name in ("before", "after")]
    for side, root in enumerate(roots):
        run = root / "drgrpo-whole"
        (run / "attempt/checkpoints").mkdir(parents=True)
        manifest = {
            "schema": "remax-resume-gpu-v1",
            "tolerances": comparison.audit.TOLERANCES,
            "runs": {
                "drgrpo-whole": {"returncode": 0},
                "drgrpo-resumed": {
                    "returncode": int(fault == "failed_run" and side == 1)
                },
            },
        }
        (root / "experiment.json").write_text(json.dumps(manifest))
        identity = {
            "source_sha256": {"method.py": str(0 if fault == "same_source" else side)},
            "data": "different" if side == 1 and fault == "data_identity" else "frozen",
        }
        (run / "effective_config.json").write_text(
            json.dumps({"resume_identity": identity})
        )
        if side == 1 and fault == "ambiguous_run":
            (run / "second_attempt/checkpoints").mkdir(parents=True)
        steps = range(1, 6 if fault == "partial_trace" else 7)
        (run / "attempt/resume_decisions.jsonl").write_text(
            "".join(json.dumps({"step": i}) + "\n" for i in steps)
        )
        value = 1 if side == 1 and fault == "different_evaluation" else 0
        (run / "attempt/eval_mode_coverage_draws.jsonl").write_text(
            json.dumps({"score": value}) + "\n"
        )

    def unexpected(*args, **kwargs):
        pytest.fail("invalid comparison reached checkpoint/model loading")

    monkeypatch.setattr(comparison, "validate_checkpoint", unexpected)
    monkeypatch.setattr(comparison.torch, "load", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_training_runs.py", *map(str, roots), "--methods", "drgrpo"],
    )
    with pytest.raises(ValueError, match=match):
        comparison.main()
    assert not (roots[1] / "pre_refactor_comparison.json").exists()
