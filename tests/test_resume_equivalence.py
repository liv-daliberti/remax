"""Full production-loop interruption/restoration, including RNG and decision traces."""

import copy
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from remax.checkpointing import (
    capture_rng,
    commit_checkpoint,
    restore_rng,
    validate_checkpoint,
)
from tests.resume_harness import Interrupted, run_case


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("method", ["drgrpo", "redr", "maxrl", "remax"])
@pytest.mark.parametrize("cut", [1, 2, 3, 6, 9])
@pytest.mark.parametrize("clear_every", [1, float("inf")])
def test_full_run_resume_is_exact(tmp_path, method, cut, clear_every):
    seed(123)
    with run_case(tmp_path / "whole", method, clear_every=clear_every) as full:
        full.run()
        assert len(full.strategy.updates) == 9
        assert full.scheduler.last_epoch == 9
        assert full.scheduler.get_last_lr()[0] == pytest.approx(0.003)
        assert all(
            float(v["step"]) == 9 for v in full.optimizer.state_dict()["state"].values()
        )
        assert full._online_canonical_bank.tracked_outcome_count > 0
        assert full.replay_scores
        expected = {
            "model": copy.deepcopy(full.model.state_dict()),
            "optimizer": copy.deepcopy(full.optimizer.state_dict()),
            "scheduler": copy.deepcopy(full.scheduler.state_dict()),
            "rng": capture_rng(),
            "bank": full._online_canonical_bank.state_dict(),
            "decisions": full.decisions,
            "eval": full.eval_history,
            "micro_steps": full.model.micro_steps,
        }
    seed(123)
    with run_case(
        tmp_path / "prefix", method, stop=cut, clear_every=clear_every
    ) as prefix:
        with pytest.raises(Interrupted):
            prefix.run()
        decisions = prefix.decisions
        evals = prefix.eval_history
    checkpoint = tmp_path / "prefix/checkpoints" / f"step_{cut:05d}"
    validate_checkpoint(checkpoint)
    seed(999)  # fresh process construction must not influence continuation
    with run_case(
        tmp_path / "suffix", method, resume=checkpoint, clear_every=clear_every
    ) as resumed:
        resumed.run()
        assert_state_equal(expected["model"], resumed.model.state_dict())
        assert_state_equal(expected["optimizer"], resumed.optimizer.state_dict())
        assert_state_equal(expected["scheduler"], resumed.scheduler.state_dict())
        assert_state_equal(expected["rng"], capture_rng())
        assert_state_equal(
            expected["bank"], resumed._online_canonical_bank.state_dict()
        )
        assert expected["micro_steps"] == resumed.model.micro_steps
        assert expected["decisions"] == decisions + resumed.decisions
        assert expected["eval"] == evals + resumed.eval_history


def commit(root, step=1, identity=None):
    return commit_checkpoint(
        root,
        step=step,
        identity=identity or {"run": "a"},
        keep=2,
        writer=lambda p: (p / "state.bin").write_bytes(b"complete state"),
    )


def test_interrupted_write_preserves_last_committed_state(tmp_path):
    old = commit(tmp_path)

    def broken(folder):
        (folder / "state.bin").write_bytes(b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        commit_checkpoint(
            tmp_path, step=2, identity={"run": "a"}, writer=broken, keep=1
        )
    assert (tmp_path / "latest").read_text().strip() == old.name
    assert validate_checkpoint(old)["step"] == 1
    assert not (tmp_path / "step_00002").exists()


def test_process_death_during_write_leaves_only_uncommitted_staging(tmp_path):
    old = commit(tmp_path)
    script = """
import os,sys
from pathlib import Path
from remax.checkpointing import commit_checkpoint
root=Path(sys.argv[1])
def writer(folder):
    (folder/'state.bin').write_bytes(b'partial')
    os._exit(9)
commit_checkpoint(root,step=2,identity={'run':'a'},writer=writer,keep=1)
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path)])
    assert result.returncode == 9
    assert validate_checkpoint(old)["step"] == 1
    assert (tmp_path / "latest").read_text().strip() == old.name
    with pytest.raises(ValueError, match="explicit committed"):
        validate_checkpoint(next(tmp_path.glob(".pending-*")))


@pytest.mark.parametrize(
    "change",
    ["missing", "corrupt", "extra", "identity", "step", "no_manifest", "symlink"],
)
def test_invalid_checkpoint_rejected_before_loading(tmp_path, change):
    folder = commit(tmp_path)
    if change == "missing":
        (folder / "state.bin").unlink()
    elif change == "corrupt":
        (folder / "state.bin").write_bytes(b"corrupt")
    elif change == "extra":
        (folder / "other.bin").write_bytes(b"extra")
    elif change == "identity":
        with pytest.raises(ValueError, match="incompatible"):
            validate_checkpoint(folder, {"run": "b"})
        return
    elif change == "step":
        p = folder / "resume_manifest.json"
        r = json.loads(p.read_text())
        r["step"] = 2
        p.write_text(json.dumps(r))
    elif change == "no_manifest":
        (folder / "resume_manifest.json").unlink()
    elif change == "symlink":
        (folder / "alias").symlink_to(folder / "state.bin")
    with pytest.raises(ValueError):
        validate_checkpoint(folder)


def test_commit_refuses_overwrite_and_prunes_only_after_commit(tmp_path):
    old = commit(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        commit(tmp_path)
    latest = commit_checkpoint(
        tmp_path,
        step=2,
        identity={"run": "a"},
        keep=1,
        writer=lambda p: (p / "state.bin").write_bytes(b"new"),
    )
    assert not old.exists()
    assert validate_checkpoint(latest)["step"] == 2


def identity_record():
    return {
        "effective_arguments": {
            "learning_rate": 0.01,
            "seed": 43,
            "num_prompt_epoch": 3,
            "prompt_template": "qwen_boxed",
            "maxrl_task_objective": True,
            "eval_steps": 2,
            "save_path": "old/output",
            "resume_dir": None,
            "resume_tag": None,
            "pretrain": "old/model",
            "prompt_data": "old/data/train",
            "eval_data": "old/data/eval",
        },
        "inputs": {
            "model": {
                "path": "old/model",
                "id": "model",
                "revision": "fixed",
                "sha256": {"weights": "hash"},
            },
            "data": {
                "path": "old/data",
                "splits": {"train": {"rows": 3, "sha256": "trainhash"}},
            },
            "prompt": {"template": "qwen_boxed"},
            "registry_sha256": "registry",
        },
        "source_sha256": {"learner.py": "source"},
        "dependencies": {"torch": "2.6.0"},
        "hardware": {"gpu_names": ["same GPU"]},
    }


@pytest.mark.parametrize(
    "field",
    [
        "learning_rate",
        "seed",
        "num_prompt_epoch",
        "prompt_template",
        "maxrl_task_objective",
        "eval_steps",
        "model",
        "dataset",
        "code",
        "dependency",
        "hardware",
    ],
)
def test_resume_identity_binds_all_scientific_inputs(tmp_path, field):
    from remax.checkpointing import run_identity

    record = identity_record()
    expected = run_identity(record)
    folder = commit(tmp_path, identity=expected)
    if field in record["effective_arguments"]:
        record["effective_arguments"][field] = "changed"
    elif field == "model":
        record["inputs"]["model"]["sha256"]["weights"] = "changed"
    elif field == "dataset":
        record["inputs"]["data"]["splits"]["train"]["sha256"] = "changed"
    elif field == "code":
        record["source_sha256"]["learner.py"] = "changed"
    elif field == "dependency":
        record["dependencies"]["torch"] = "changed"
    elif field == "hardware":
        record["hardware"]["gpu_names"] = ["another GPU"]
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint(folder, run_identity(record))


def test_identity_allows_relocation_and_explicit_resume_selectors():
    from remax.checkpointing import run_identity

    record = identity_record()
    expected = run_identity(record)
    record["effective_arguments"].update(
        save_path="new/output",
        pretrain="new/model",
        prompt_data="new/data/train",
        eval_data="new/data/eval",
        resume_dir="checkpoints",
        resume_tag="step_00002",
    )
    record["inputs"]["model"]["path"] = "new/model"
    record["inputs"]["data"]["path"] = "new/data"
    assert run_identity(record) == expected


def test_interruption_after_commit_before_latest_preserves_both_valid_states(
    tmp_path, monkeypatch
):
    import remax.checkpointing as checkpoints

    old = commit(tmp_path)
    replace = checkpoints.os.replace

    def fail_latest(source, destination):
        if Path(destination).name == "latest":
            raise OSError("interrupted latest update")
        return replace(source, destination)

    monkeypatch.setattr(checkpoints.os, "replace", fail_latest)
    with pytest.raises(OSError, match="interrupted latest"):
        commit(tmp_path, step=2)
    assert validate_checkpoint(old)["step"] == 1
    assert validate_checkpoint(tmp_path / "step_00002")["step"] == 2
    assert (tmp_path / "latest").read_text().strip() == old.name


def test_incompatible_restore_fails_before_any_optimizer_step(tmp_path):
    seed(123)
    with run_case(tmp_path / "prefix", "remax", stop=2) as prefix:
        with pytest.raises(Interrupted):
            prefix.run()
    checkpoint = tmp_path / "prefix/checkpoints/step_00002"
    with run_case(
        tmp_path / "suffix",
        "remax",
        resume=checkpoint,
        identity={"different": "method"},
    ) as resumed:
        with pytest.raises(ValueError, match="incompatible"):
            resumed.run()
        assert resumed.strategy.updates == []
        assert resumed.decisions == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("prompt_batches_consumed_total", 0),
        ("update_interval", 2),
        ("prompt_epoch", 7),
        ("steps", True),
        ("rng_state", {}),
        ("last_evaluated_global_step", 99),
    ],
)
def test_malformed_full_run_state_is_not_coerced(tmp_path, key, value):
    from remax.checkpointing import validate_client_state

    seed(123)
    with run_case(tmp_path / "prefix", "remax", stop=1) as prefix:
        with pytest.raises(Interrupted):
            prefix.run()
        identity = prefix.args._remax_resume_identity
    state = torch.load(
        tmp_path / "prefix/checkpoints/step_00001/state.pt", weights_only=False
    )["client"]
    state[key] = value
    with pytest.raises(ValueError):
        validate_client_state(
            state, identity=identity, step=1, batches_per_epoch=3, epochs=3
        )
