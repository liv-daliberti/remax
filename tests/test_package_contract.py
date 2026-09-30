"""Contracts that must hold for built, installed distributions."""

from importlib import metadata
import json
from pathlib import Path
import platform
import sys

import pytest
import torch

from remax import launcher
from remax import launch_record


def test_bundled_assets_match_release_sources():
    root = Path(__file__).parents[1]
    assert len(launcher.recipe_names()) == 20
    for relative in [
        p.relative_to(root) for p in sorted((root / "configs").glob("*.json"))
    ]:
        # The outside-checkout test bundle has the same canonical release assets.
        assert (launcher.ASSETS / relative).read_bytes() == (
            root / relative
        ).read_bytes()
    for name in ("train.sh", "repo_env.sh"):
        assert (launcher.ASSETS / "ops" / name).read_bytes() == (
            root / "ops" / name
        ).read_bytes()
    assert "PYTHONPATH=" not in (launcher.ASSETS / "ops/repo_env.sh").read_text()


def test_distribution_separates_core_from_training():
    from packaging.requirements import Requirement

    requirements = [Requirement(r) for r in metadata.requires("remax-rl")]
    core = {
        r.name
        for r in requirements
        if r.marker is None or r.marker.evaluate({"extra": ""})
    }
    assert core == {"torch", "numpy", "modebench"}
    training = {
        r.name
        for r in requirements
        if r.marker is None or r.marker.evaluate({"extra": "train"})
    }
    assert {
        "oat-llm",
        "vllm",
        "deepspeed",
        "datasets",
        "transformers",
        "pyarrow",
    } <= training
    assert metadata.metadata("remax-rl")["Requires-Python"] == "<3.13,>=3.10"


@pytest.mark.parametrize("cuda", [None, "11.8", "12.1", "12.4"])
def test_training_runtime_requires_qualified_cuda(monkeypatch, cuda):
    versions = {
        "torch": "2.6.0",
        "transformers": "4.51.3",
        "vllm": "0.8.4",
        "oat-llm": "0.1.3.post1",
        "deepspeed": "0.16.8",
        "datasets": "2.16.1",
        "numpy": "1.26.4",
        "pyarrow": "11.0.0",
        "modebench": "0.4.0",
    }
    monkeypatch.setattr(sys, "version_info", (3, 10))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(metadata, "version", versions.__getitem__)

    monkeypatch.setattr(torch.version, "cuda", cuda)
    if cuda == "12.4":
        assert launch_record.validate_runtime() == versions
    else:
        with pytest.raises(ValueError, match="CUDA 12.4"):
            launch_record.validate_runtime()


def test_launcher_preserves_external_cache_location(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    raw = json.loads(launcher.resolve_recipe("remax_countdown_05b").read_text())
    env = launcher.environment(
        raw,
        data_root=tmp_path,
        model=tmp_path / "model",
        output=tmp_path / "run",
        seed=43,
    )
    assert env["XDG_CACHE_HOME"] == str(tmp_path / "cache")
    assert "PYTHONPATH" not in env
