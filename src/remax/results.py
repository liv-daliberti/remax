"""Inspect and reproduce release-bound results without a research checkout or GPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .recipes import strict_json

ASSETS = Path(__file__).resolve().parent / "_assets"
CATALOG = "evidence/experiments/catalog_v1.json"
BINDINGS = "evidence/experiments/historical_bindings_v1.json"
EXPECTED = "evidence/mode_diversity_training.json"
KEYS = ("level", "scale", "domain", "method")


def read(root: Path, relative: str):
    return strict_json(safe_path(root, relative).read_text())


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"nonlocal result artifact: {relative}")
    return path


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_catalog(root: Path = ASSETS) -> dict:
    """Validate structure and identities; numerical recomputation is explicit."""
    catalog = read(root, CATALOG)
    if catalog.get("schema") != "remax-result-packages-v1":
        raise ValueError("unsupported result catalog schema")
    packages = catalog["packages"]
    ids = [p["id"] for p in packages]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate experiment identifier")
    required = {
        EXPECTED,
        BINDINGS,
        "evidence/verified_samples_completed_cohort.jsonl.gz",
        "ops/build_mode_diversity_training.py",
        "ops/followup_metrics.py",
        "ops/export_experiment_packages.py",
    }
    if not required.issubset(catalog["artifacts"]):
        raise ValueError("missing required artifact identity")
    for path, sha in catalog["artifacts"].items():
        if checksum(safe_path(root, path)) != sha:
            raise ValueError(f"result artifact hash mismatch: {path}")
    expected = read(root, EXPECTED)
    bindings = read(root, BINDINGS)
    expected_arms = {"/".join(r[k] for k in KEYS): r for r in expected["arms"]}
    expected_seeds = {
        "/".join(r[k] for k in KEYS) + f"/s{r['seed']}": r for r in expected["seeds"]
    }
    actual_arms, actual_seeds, referenced_cells = {}, {}, set()
    for package in packages:
        if package.get("training_reproduction_verified") is not False:
            raise ValueError(
                "training reproduction cannot be inferred from saved keys or summaries"
            )
        if package["status"] == "saved_key_reproducible":
            identity = package["id"]
            actual_arms[identity] = package["result"]
            if not package["training_gaps"] or not package["cell_ids"]:
                raise ValueError("missing training gaps or seed bindings")
            recipe = package["recipe"]
            if recipe["path"] is not None:
                if (
                    recipe["path"] not in catalog["artifacts"]
                    or recipe["relationship"]
                    != "related_maintained_method_not_historical_equivalence"
                ):
                    raise ValueError("unbound or overstated recipe relationship")
            for cell_id in package["cell_ids"]:
                if cell_id in referenced_cells or not cell_id.startswith(
                    identity + "/s"
                ):
                    raise ValueError("duplicate or mismatched cell binding")
                referenced_cells.add(cell_id)
                cell = bindings["cells"][cell_id]
                if cell["id"] != cell_id:
                    raise ValueError("cell identifier mismatch")
                if cell["included_in_saved_key_analysis"]:
                    actual_seeds[cell_id] = cell["analysis_result"]
                elif cell["analysis_result"] is not None:
                    raise ValueError("excluded cell carries an included result")
        elif package["status"] == "summary_snapshot":
            if (
                package["reproduce_command"] is not None
                or package["recipe"] is not None
            ):
                raise ValueError("a summary snapshot cannot claim a runnable result")
            if package["path"] not in catalog["artifacts"]:
                raise ValueError("unbound summary snapshot")
        else:
            raise ValueError("unknown result qualification")
    if actual_arms != expected_arms or actual_seeds != expected_seeds:
        raise ValueError("result catalog differs from frozen arms or seed decisions")
    if referenced_cells != set(bindings["cells"]):
        raise ValueError("unaccounted historical cell")
    if catalog["evaluation_protocol"] != expected["definition"]:
        raise ValueError("evaluation protocol differs from frozen analysis")
    return catalog


def inspect(identifier: str, root: Path = ASSETS) -> dict:
    catalog = load_catalog(root)
    matches = [p for p in catalog["packages"] if p["id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"unknown experiment: {identifier}")
    package = dict(matches[0])
    package["evaluation_protocol"] = (
        catalog["evaluation_protocol"]
        if package["status"] == "saved_key_reproducible"
        else None
    )
    if package["status"] == "saved_key_reproducible":
        bindings = read(root, BINDINGS)
        package["cells"] = [bindings["cells"][cell] for cell in package["cell_ids"]]
        paths = {
            c["launch"]["runtime_snapshot"]
            for c in package["cells"]
            if c["launch"] and c["launch"]["runtime_snapshot"]
        }
        package["runtime_audits"] = {
            p: bindings["runtime_audits"][p] for p in sorted(paths)
        }
    else:
        package["snapshot"] = read(root, package["path"])
        if package["snapshot"].get("campaign") == "e122":
            package["registered_launches"] = read(root, BINDINGS)[
                "level3_registered_launches"
            ]
    return package


def reproduce(identifier: str, root: Path = ASSETS) -> dict:
    package = inspect(identifier, root)
    if package["status"] != "saved_key_reproducible":
        raise ValueError(
            "summary snapshot only: numerical reproduction is not available"
        )
    # Ship the original analysis code unchanged; its only scientific dependency
    # is the same pinned ModeBench metrics API used by repository reproduction.
    from ._assets.ops.build_mode_diversity_training import build

    actual = build(root / "evidence/verified_samples_completed_cohort.jsonl.gz")
    expected = read(root, EXPECTED)
    for field in (
        "schema",
        "archive_manifest",
        "definition",
        "arms",
        "seeds",
        "coverage",
    ):
        if actual[field] != expected[field]:
            raise ValueError(f"frozen numerical reproduction differs: {field}")
    if actual["archive"]["sha256"] != expected["archive"]["sha256"]:
        raise ValueError("frozen archive identity differs")
    return {
        "id": identifier,
        "status": "saved_key_reproduced",
        "training_reproduction_verified": False,
        "result": package["result"],
        "seeds": [
            c["analysis_result"]
            for c in package["cells"]
            if c["included_in_saved_key_analysis"]
        ],
        "excluded_cells": [
            c for c in package["cells"] if not c["included_in_saved_key_analysis"]
        ],
        "whole_archive_coverage_checked": actual["coverage"],
        "scope": "Recomputed saved canonical keys. No training, sampling or response regrading.",
    }
