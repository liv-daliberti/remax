"""A version label alone must not authenticate a changed benchmark installation."""

from importlib import metadata
from pathlib import Path
import shutil

import pytest

from remax.benchmark_identity import validate_modebench_release


@pytest.fixture
def benchmark_copy(tmp_path):
    root = Path(metadata.distribution("modebench").locate_file("modebench"))
    shutil.copytree(
        root, tmp_path / "modebench", ignore=shutil.ignore_patterns("__pycache__")
    )
    return tmp_path / "modebench"


def test_regular_package_needs_no_git_install_metadata():
    assert validate_modebench_release()["version"] == "0.4.0"


@pytest.mark.parametrize("change", ["modify", "delete", "add"])
def test_changed_benchmark_bytes_are_rejected(benchmark_copy, change):
    path = benchmark_copy / "api.py"
    if change == "modify":
        path.write_text(path.read_text() + "\n# changed after qualification\n")
    elif change == "delete":
        path.unlink()
    else:
        (benchmark_copy / "extra.py").write_text("")
    with pytest.raises(ValueError, match="package content differs"):
        validate_modebench_release(package_root=benchmark_copy)


def test_other_benchmark_version_is_rejected(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda _: "0.4.1")
    with pytest.raises(ValueError, match="version differs"):
        validate_modebench_release()
