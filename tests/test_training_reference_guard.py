"""The PR guard must reject edits even when local hashes were refreshed."""

import hashlib
import json
import subprocess

import pytest
from check_training_reference import check


def command(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


@pytest.mark.parametrize(
    "change", ["rewrite_and_rehash", "delete", "unchanged", "new_version"]
)
def test_reference_guard_compares_base_bytes(tmp_path, change):
    folder = tmp_path / "tests/fixtures/training_v1"
    folder.mkdir(parents=True)
    for name in ("inputs.json", "expected.json"):
        (folder / name).write_text("{}\n")

    def manifest():
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "files": {
                        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                        for name in ("inputs.json", "expected.json")
                    }
                }
            )
        )

    manifest()
    command(tmp_path, "init", "-q")
    command(tmp_path, "add", ".")
    command(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Freeze",
    )
    base = command(tmp_path, "rev-parse", "HEAD").decode().strip()
    if change == "rewrite_and_rehash":
        (folder / "expected.json").write_text('{"changed":true}\n')
        manifest()
    elif change == "delete":
        (folder / "expected.json").unlink()
    elif change == "new_version":
        import shutil

        shutil.copytree(folder, folder.with_name("training_v2"))
    if change in ("rewrite_and_rehash", "delete"):
        with pytest.raises(
            ValueError, match="(changes|changed) frozen training reference"
        ):
            check(tmp_path, base)
    else:
        assert check(tmp_path, base)["preserved_base_files"] == 3
    if change != "delete":
        with pytest.raises(subprocess.CalledProcessError):
            check(tmp_path, "missing-ref")


@pytest.mark.parametrize(
    "name", ["benchmark_boundary_v1.json", "gpu_historical_pantry_v1.json"]
)
def test_boundary_fixture_cannot_be_rewritten_and_rehashed(tmp_path, name):
    import shutil
    from pathlib import Path

    root = Path(__file__).parent / "fixtures"
    shutil.copytree(root, tmp_path / "tests/fixtures")
    command(tmp_path, "init", "-q")
    command(tmp_path, "add", ".")
    command(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Freeze boundary",
    )
    base = command(tmp_path, "rev-parse", "HEAD").decode().strip()
    assert check(tmp_path, base)["preserved_base_files"] == 7
    path = tmp_path / "tests/fixtures" / name
    path.write_text("{}\n")
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n"
    )
    with pytest.raises(ValueError, match="PR changes frozen training reference"):
        check(tmp_path, base)
