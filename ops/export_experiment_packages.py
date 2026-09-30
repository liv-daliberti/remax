"""Export result bindings from a retained research checkout; never execute its code.

This is a release-author tool. Installed readers use ``remax results`` instead.
The export records missing evidence rather than synthesizing training recipes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("level", "scale", "domain", "method")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def export_environment(record):
    """Read sbatch's comma-delimited export argument as data, never shell code."""
    tokens = shlex.split(record.split("SubmitLine=", 1)[-1])
    export = next((t.split("=", 1)[1] for t in tokens if t.startswith("--export=")), "")
    result = {}
    for item in export.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            if key in result:
                raise ValueError(f"duplicate historical environment field: {key}")
            result[key] = value
    return result


def model_identity(path):
    if not path:
        return None
    parts = Path(path).parts
    caches = [p for p in parts if p.startswith("models--")]
    return {
        "id": (
            caches[0].removeprefix("models--").replace("--", "/")
            if len(caches) == 1
            else None
        ),
        "revision": parts[-1] if len(parts) >= 2 and parts[-2] == "snapshots" else None,
        "authentication": "Recorded path only; model bytes not authenticated by this receipt.",
    }


def runtime_audit(path):
    """Recompute the original E76 tree-hash algorithm, including its path framing."""
    identity = path / "SNAPSHOT_IDENTITY.json"
    if not identity.is_file():
        return {"status": "missing_snapshot_identity", "path": str(path)}
    recorded = json.loads(identity.read_text())
    if recorded.get("schema") != "e76_runtime_snapshot_v1":
        return {
            "status": "unsupported_snapshot_identity",
            "path": str(path),
            "identity": recorded,
        }
    h = hashlib.sha256()
    count = 0
    for base in (path / "src/oat_drgrpo", path / "ops"):
        for file in sorted(p for p in base.rglob("*") if p.is_file()):
            if "__pycache__" in file.parts or file.suffix == ".pyc":
                continue
            h.update(str(file.relative_to(base.parent)).encode())
            h.update(b"\0")
            h.update(file.read_bytes())
            h.update(b"\0")
            count += 1
    return {
        "path": str(path),
        "recorded_sha256": recorded["sha256"],
        "observed_sha256": h.hexdigest(),
        "files": count,
        "status": (
            "matches_recorded_tree"
            if h.hexdigest() == recorded["sha256"]
            else "snapshot_tree_changed"
        ),
    }


def export(source, destination=ROOT):
    evidence = destination / "evidence"
    directory = evidence / "experiments"
    expected = json.loads((evidence / "mode_diversity_training.json").read_text())
    with gzip.open(
        evidence / "verified_samples_completed_cohort.jsonl.gz", "rt"
    ) as stream:
        cells = [
            r for line in stream if (r := json.loads(line)).get("record_kind") == "cell"
        ]
    selected = {tuple(r[k] for k in KEYS) + (r["seed"],): r for r in expected["seeds"]}
    ledgers, launches, runtimes, groups = {}, {}, {}, defaultdict(list)
    for cell in cells:
        identity = tuple(cell[k] for k in KEYS)
        cell_id = "/".join(identity) + f"/s{cell['seed']}"
        ledger_name = cell.get("ledger")
        launch = None
        if ledger_name:
            if ledger_name not in ledgers:
                p = source / ledger_name
                data = json.loads(p.read_text())
                ledgers[ledger_name] = {
                    "path": ledger_name,
                    "sha256": sha(p),
                    "data": data,
                    "registered_metadata": {
                        k: data[k]
                        for k in (
                            "protocol",
                            "protocol_sha256",
                            "identity",
                            "identity_sha256",
                            "model",
                            "model_revision",
                            "optimizer",
                            "sampling",
                            "train_rows",
                            "eval_rows",
                            "passes",
                            "target_steps",
                            "seeds",
                            "checkpoint_interval_steps",
                            "sources",
                        )
                        if k in data
                    },
                }
            data = ledgers[ledger_name]["data"]
            matches = [r for r in data["runs"] if r["run_dir"] == cell.get("run_dir")]
            if len(matches) != 1:
                raise ValueError(f"expected unique registered launch for {cell_id}")
            run = matches[0]
            env = export_environment(run.get("held_scheduler_record", ""))
            runtime_path = env.get("OAT_ZERO_OPS_SNAPSHOT_ROOT")
            if runtime_path:
                path = Path(runtime_path).parent
                if str(path) not in runtimes:
                    runtimes[str(path)] = runtime_audit(path)
            launch = {
                "ledger": ledger_name,
                "run_stamp": run.get("run_stamp"),
                "registered_job_id": run["job_id"],
                "registered_environment": env,
                "runtime_snapshot": (
                    str(Path(runtime_path).parent) if runtime_path else None
                ),
                "model_path": env.get("OAT_ZERO_PRETRAIN"),
                "model": model_identity(env.get("OAT_ZERO_PRETRAIN")),
                "dataset_paths": {
                    k: env.get("OAT_ZERO_" + k) for k in ("PROMPT_DATA", "EVAL_DATA")
                },
                "prompt_condition": {
                    k: env.get("OAT_ZERO_" + k)
                    for k in (
                        "PROMPT_TEMPLATE",
                        "MODEBENCH_DOMAIN",
                        "MODEBENCH_SYNTAX_PROFILE",
                        "CANONICAL_ACTION_TASK",
                    )
                },
                "effective_configuration_status": "registered_exports_only_wrapper_defaults_and_recoveries_not_authenticated",
                "dependency_lock": None,
            }
        seed_result = selected.get(identity + (cell["seed"],))
        record = {
            "id": cell_id,
            "seed": cell["seed"],
            "launch": launch,
            "included_in_saved_key_analysis": seed_result is not None,
            "analysis_result": seed_result,
            "sample_issues": cell.get("sample_issues", []),
            "approved_exclusion": cell.get("approved_exclusion"),
            "before_after_available": cell.get("before_after_available"),
            "in_terminal_paired_cohort": cell.get("in_terminal_paired_cohort"),
            "source_checks": cell.get("source_checks", []),
            "evaluation": {
                step: {
                    "sampling_certificate": checkpoint.get("sampling_certificate"),
                    "origins": checkpoint.get("origins", []),
                }
                for step, checkpoint in cell.get("checkpoints", {}).items()
                if checkpoint
            },
        }
        launches[cell_id] = record
        groups[identity].append(cell_id)
    bindings = {
        "schema": "remax-historical-bindings-v1",
        "scope": "Recorded launch intent and saved-key provenance, not an authenticated full effective training environment.",
        "source_ledgers": {
            name: {k: v for k, v in receipt.items() if k != "data"}
            for name, receipt in ledgers.items()
        },
        "runtime_audits": runtimes,
        "cells": launches,
    }
    # Retain E122 rather than confusing the unlaunched Qwen-3B E123 plan with it.
    source_snapshot = source / "paper/results/modebench_level3_comparison_snapshot.json"
    snapshot_path = evidence / "snapshots/level3_comparison_20260917.json"
    snapshot_path.write_bytes(source_snapshot.read_bytes())
    plan_path = source / "var/artifacts/e123_level3_factorial_jobs.json"
    plan = json.loads(plan_path.read_text())
    level3_ledger_path = source / "var/artifacts/e122_level3_factorial_jobs.json"
    level3_ledger = json.loads(level3_ledger_path.read_text())
    bindings["level3_registered_launches"] = {
        "ledger": str(level3_ledger_path.relative_to(source)),
        "sha256": sha(level3_ledger_path),
        "matches_snapshot_source": sha(level3_ledger_path)
        == json.loads(source_snapshot.read_text())["sources"].get(
            "var/artifacts/e122_level3_factorial_jobs.json"
        ),
        "model": {k: level3_ledger.get(k) for k in ("model", "model_revision")},
        "scope": "Current retained ledger; source hash comparison does not authenticate every recovery attempt.",
        "runs": [
            {
                k: r.get(k)
                for k in ("domain", "arm", "seed", "run_stamp", "job_id", "run_dir")
            }
            | {
                "registered_environment": export_environment(
                    r.get("held_scheduler_record", "")
                )
            }
            for r in level3_ledger["runs"]
        ],
    }
    write(directory / "historical_bindings_v1.json", bindings)
    packages = []
    methods = {
        "replay_drgrpo": "redr",
        "replay_maxrl": "remax",
        "drgrpo": "drgrpo",
        "maxrl": "maxrl",
    }
    for arm in expected["arms"]:
        identity = tuple(arm[k] for k in KEYS)
        related = None
        if identity[:2] == ("level1", "qwen05b") and arm["method"] in methods:
            related = f"configs/{methods[arm['method']]}_{arm['domain']}_05b.json"
        packages.append(
            {
                "id": "/".join(identity),
                "status": "saved_key_reproducible",
                "result": arm,
                "cell_ids": sorted(groups[identity]),
                "recipe": {
                    "path": related,
                    "relationship": (
                        "related_maintained_method_not_historical_equivalence"
                        if related
                        else "no_qualified_training_recipe"
                    ),
                },
                "training_reproduction_verified": False,
                "training_gaps": [
                    "No per-attempt authenticated effective configuration and dependency lock in this export.",
                    "Registered launch settings are not resolved wrapper/recovery settings.",
                    "Historical model/dataset paths do not authenticate their bytes for a fresh launch.",
                    "Saved keys do not contain original responses or trained weights.",
                ],
                "reproduce_command": "remax results reproduce " + "/".join(identity),
            }
        )
    snapshots = []
    for path in sorted((evidence / "snapshots").glob("*.json")):
        data = json.loads(path.read_text())
        snapshots.append(
            {
                "id": "snapshot/" + path.stem,
                "status": "summary_snapshot",
                "path": str(path.relative_to(destination)),
                "schema": data.get("schema"),
                "sources": data.get("sources"),
                "recipe": None,
                "training_reproduction_verified": False,
                "reproduce_command": None,
                "limitations": [
                    "Integrity checking preserves this summary; it does not recompute its numerical results.",
                    "Runtime, dataset/model byte identities, seed selection and exclusions are recorded only to the extent present in the snapshot; missing fields are unknown.",
                ],
            }
        )
    artifacts = [
        evidence / "mode_diversity_training.json",
        evidence / "verified_samples_completed_cohort.jsonl.gz",
        directory / "historical_bindings_v1.json",
        *(evidence / "snapshots").glob("*.json"),
        *(destination / "configs").glob("*.json"),
        *(
            destination / "ops" / name
            for name in (
                "build_mode_diversity_training.py",
                "followup_metrics.py",
                "export_experiment_packages.py",
            )
        ),
    ]
    catalog = {
        "schema": "remax-result-packages-v1",
        "artifacts": {
            str(p.relative_to(destination)): sha(p) for p in sorted(artifacts)
        },
        "evaluation_protocol": expected["definition"],
        "scope": "Saved-key analysis is runnable on CPU. No historical training score is claimed reproduced by a maintained recipe.",
        "packages": packages + snapshots,
        "training_export_audit": {
            "promoted_training_recipes": [],
            "level3_summary_source": {
                "path": str(source_snapshot.relative_to(source)),
                "sha256": sha(source_snapshot),
                "campaign": "e122",
                "model": "Qwen2.5-0.5B-Instruct",
            },
            "level3_qwen3b_plan": {
                "path": str(plan_path.relative_to(source)),
                "sha256": sha(plan_path),
                "launched_runs": len(plan["runs"]),
                "status": plan.get("status"),
                "admission": plan.get("admission"),
            },
            "decision": "Do not promote incomplete or changed historical training chains; retain separately runnable analysis packages and explicit gaps.",
        },
    }
    write(directory / "catalog_v1.json", catalog)
    return {
        "packages": len(packages),
        "snapshots": len(snapshots),
        "bound_cells": len(launches),
        "runtime_audits": {p: r["status"] for p, r in runtimes.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source_root.resolve()), indent=2))
