"""Read-only pre/post-refactor GPU comparison; never restores across source identities."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("resume_gpu", ROOT / "ops/resume_gpu.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
import torch
from remax.core.checkpoints import validate_checkpoint
from remax.input_identity import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument(
        "--methods", nargs="+", choices=audit.METHODS, default=list(audit.METHODS)
    )
    args = p.parse_args()
    report = {
        "schema": "remax-refactor-parity-v1",
        "tolerances": audit.TOLERANCES,
        "compared_step": 6,
        "methods": {},
        "source_identity_policy": "Offline comparison only; source hashes must differ, all other identity fields must match. No cross-source restore.",
    }
    for base in (args.before, args.after):
        complete = audit.completed_methods(
            json.loads((base / "experiment.json").read_text())
        )
        if not set(args.methods) <= set(complete):
            raise ValueError(f"{base}: requested method has no complete experiment")
    for method in args.methods:
        folders = [base / f"{method}-whole" for base in (args.before, args.after)]
        configs = [
            json.loads((f / "effective_config.json").read_text()) for f in folders
        ]
        identities = [dict(c["resume_identity"]) for c in configs]
        sources = [i.pop("source_sha256") for i in identities]
        if sources[0] == sources[1]:
            raise ValueError("expected distinct source revisions")
        if identities[0] != identities[1]:
            raise ValueError(f"{method}: non-source run identity differs")
        roots = []
        for folder in folders:
            candidates = list(folder.glob("*/checkpoints"))
            if len(candidates) != 1:
                raise ValueError(f"{folder}: expected exactly one training run")
            roots.append(candidates[0].parent)
        for name in ("resume_decisions.jsonl", "eval_mode_coverage_draws.jsonl"):
            records = [audit.load_records(r / name) for r in roots]
            if name == "resume_decisions.jsonl" and any(
                [r["step"] for r in rows] != list(range(1, 7)) for rows in records
            ):
                raise ValueError(f"{method}: incomplete six-update decision trace")
            if records[0] != records[1]:
                raise ValueError(f"{method}: {name} differs")
        paths = [r / "checkpoints/step_00006" for r in roots]
        for path, config in zip(paths, configs):
            validate_checkpoint(path, config["resume_identity"])
        print("Comparing", method, "model/client state", flush=True)
        states = [
            torch.load(
                next(p.glob("*model_states.pt")), map_location="cpu", weights_only=False
            )
            for p in paths
        ]
        if any(state["global_steps"] != 6 for state in states):
            raise ValueError(
                f"{method}: checkpoint does not contain six optimizer updates"
            )
        audit.compare_tree(
            states[0]["module"],
            states[1]["module"],
            **audit.TOLERANCES["model"],
            path="model",
        )
        for key in (
            "lr_scheduler",
            "global_steps",
            "global_samples",
            "engine_micro_steps",
            "online_canonical_bank_state",
            "rng_state",
            "steps",
            "global_step",
            "policy_sgd_step",
            "query_step",
            "prompt_consumed",
            "prompt_epoch",
            "prompt_batches_consumed_total",
            "last_evaluated_global_step",
        ):
            audit.compare_tree(states[0][key], states[1][key], path=key)
        del states
        print("Comparing", method, "optimizer", flush=True)
        optim = [
            torch.load(
                next(p.glob("*optim_states.pt")), map_location="cpu", weights_only=False
            )
            for p in paths
        ]
        audit.compare_tree(
            optim[0], optim[1], **audit.TOLERANCES["optimizer"], path="optimizer"
        )
        del optim
        report["methods"][method] = {
            "status": "pass",
            "identical_non_source_identity": True,
            "identical_decisions_and_evaluation": True,
            "state_fields_exact": True,
            "model_and_optimizer_within_tolerance": True,
            "configuration_sha256": [
                digest(f / "effective_config.json") for f in folders
            ],
            "checkpoint_manifest_sha256": [
                digest(p / "resume_manifest.json") for p in paths
            ],
        }
    target = args.after / "pre_refactor_comparison.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(target, flush=True)


if __name__ == "__main__":
    main()
