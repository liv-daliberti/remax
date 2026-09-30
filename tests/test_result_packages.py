"""Result claims must be bound, exhaustive and weaker than their available evidence."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from remax import results
from ops.export_experiment_packages import (
    export_environment,
    model_identity,
    runtime_audit,
)


@pytest.fixture
def copy_assets(tmp_path):
    # Hardlink immutable inputs; replace the catalog, never mutate a linked file.
    shutil.copytree(
        results.ASSETS, tmp_path / "assets", copy_function=__import__("os").link
    )
    return tmp_path / "assets"


def rewrite(root, mutate):
    path = root / results.CATALOG
    data = json.loads(path.read_text())
    mutate(data)
    path.unlink()
    path.write_text(json.dumps(data))


def test_all_results_and_exclusions_have_bindings():
    catalog = results.load_catalog()
    packages = catalog["packages"]
    assert len(packages) == 101
    assert sum(p["status"] == "saved_key_reproducible" for p in packages) == 95
    bindings = results.read(results.ASSETS, results.BINDINGS)
    assert len(bindings["cells"]) == 475
    assert (
        sum(c["included_in_saved_key_analysis"] for c in bindings["cells"].values())
        == 473
    )
    assert len(bindings["runtime_audits"]) == 8
    assert {r["status"] for r in bindings["runtime_audits"].values()} == {
        "missing_snapshot_identity",
        "snapshot_tree_changed",
    }
    assert catalog["training_export_audit"]["promoted_training_recipes"] == []
    assert catalog["training_export_audit"]["level3_qwen3b_plan"]["launched_runs"] == 0


def test_installed_level2_numerical_package():
    report = results.reproduce("level2/qwen05b/countdown/replay_maxrl")
    assert report["status"] == "saved_key_reproduced"
    assert report["whole_archive_coverage_checked"]["cells"] == 473
    assert [s["seed"] for s in report["seeds"]] == [43, 44, 45, 46, 47]
    assert report["training_reproduction_verified"] is False


def test_larger_model_records_and_prompt_are_explicit():
    package = results.inspect("level1/qwen3b/countdown/replay_maxrl")
    assert package["recipe"]["path"] is None
    assert [c["seed"] for c in package["cells"]] == [70, 71, 72, 73, 74]
    for cell in package["cells"]:
        assert cell["launch"]["model"]["id"] == "Qwen/Qwen2.5-3B-Instruct"
        assert (
            cell["launch"]["model"]["revision"]
            == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
        )
        assert cell["launch"]["prompt_condition"]["PROMPT_TEMPLATE"] == "qwen_boxed"
        assert cell["launch"]["dependency_lock"] is None


def test_level3_is_the_actual_campaign_and_refuses_numerical_claim():
    identity = "snapshot/level3_comparison_20260917"
    package = results.inspect(identity)
    assert package["snapshot"]["campaign"] == "e122"
    assert package["snapshot"]["model"] == "Qwen2.5-0.5B-Instruct"
    assert len(package["snapshot"]["unadmitted_cells"]) == 1
    assert len(package["registered_launches"]["runs"]) == 100
    with pytest.raises(ValueError, match="summary snapshot only"):
        results.reproduce(identity)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: c["packages"].append(c["packages"][0]), "duplicate experiment"),
        (
            lambda c: c["packages"][0].update(training_reproduction_verified=True),
            "training reproduction",
        ),
        (lambda c: c["packages"][0]["result"].update(pmd_after=99), "frozen arms"),
        (
            lambda c: c["packages"][0]["cell_ids"].pop(),
            "frozen arms|unaccounted historical cell",
        ),
        (lambda c: c["artifacts"].pop(results.EXPECTED), "missing required artifact"),
        (
            lambda c: c["artifacts"].update({"../../escape.json": "0" * 64}),
            "nonlocal result",
        ),
        (
            lambda c: c["artifacts"].update({results.EXPECTED: "0" * 64}),
            "artifact hash mismatch",
        ),
        (
            lambda c: c["evaluation_protocol"].update(min_defined_prompts=0),
            "evaluation protocol",
        ),
        (
            lambda c: c["packages"][-1].update(reproduce_command="pretend"),
            "summary snapshot",
        ),
    ],
)
def test_claims_fail_closed(copy_assets, mutation, match):
    rewrite(copy_assets, mutation)
    with pytest.raises(ValueError, match=match):
        results.load_catalog(copy_assets)


def test_unrecognized_result_is_an_error():
    with pytest.raises(ValueError, match="unknown experiment"):
        results.inspect("level3/qwen3b/countdown/remax")


def test_historical_environment_is_data_not_executed(tmp_path):
    sentinel = tmp_path / "must-not-exist"
    record = (
        f'SubmitLine=sbatch --export=ALL,NAME="$(touch {sentinel})",SEED=43 script.sh'
    )
    assert export_environment(record) == {"NAME": f"$(touch {sentinel})", "SEED": "43"}
    assert not sentinel.exists()
    with pytest.raises(ValueError, match="duplicate historical"):
        export_environment("sbatch --export=ALL,SEED=43,SEED=44")
    assert model_identity("/unknown/path")["revision"] is None


def test_original_runtime_hash_detects_post_snapshot_changes(tmp_path):
    import hashlib

    base = tmp_path / "src/oat_drgrpo"
    base.mkdir(parents=True)
    (base / "run.py").write_bytes(b"original\n")
    (tmp_path / "ops").mkdir()
    expected = hashlib.sha256(b"oat_drgrpo/run.py\0original\n\0").hexdigest()
    (tmp_path / "SNAPSHOT_IDENTITY.json").write_text(
        json.dumps({"schema": "e76_runtime_snapshot_v1", "sha256": expected})
    )
    assert runtime_audit(tmp_path)["status"] == "matches_recorded_tree"
    (base / "run.py").write_bytes(b"recovery\n")
    assert runtime_audit(tmp_path)["status"] == "snapshot_tree_changed"
