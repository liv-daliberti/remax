"""Run and compare six-update uninterrupted/resumed jobs for each maintained method."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("drgrpo", "redr", "maxrl", "remax")
# Declared before comparisons; decision/bank/RNG/scheduler comparisons are exact.
TOLERANCES = {
    "model": {"atol": 1e-6, "rtol": 1e-6},
    "optimizer": {"atol": 1e-8, "rtol": 1e-5},
}


def load_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(args):
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "remax-resume-gpu-v1", "tolerances": TOLERANCES, "runs": {}}
    (work / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for method in args.methods:
        recipe = json.loads(
            (ROOT / f"configs/{method}_{args.domain}_05b.json").read_text()
        )
        overrides = {
            "MAX_TRAIN": "3",
            "NUM_PROMPT_EPOCH": "2",
            "MAX_PROMPT_EPOCHS": "2",
            "EVAL_STEPS": "2",
            "EVAL_PROMPT_INTERVAL": "2",
            "EVAL_MODE_COVERAGE_K": "2",
            "EVAL_MODE_COVERAGE_DRAWS": "1",
            "RESUME_STEPS": "2",
            "RESUME_FROM": "2",
            "SAVE_CKPT": "1",
            "MAX_RESUME_NUM": "3",
            "PRUNE_RESUME_ON_SUCCESS": "0",
            "AUTO_RESUME": "0",
        }
        recipe["environment"].update({"OAT_ZERO_" + k: v for k, v in overrides.items()})
        config = work / f"{method}.json"
        config.write_text(json.dumps(recipe, indent=2) + "\n")
        checkpoint = None
        for label in ("whole", "resumed"):
            output = work / f"{method}-{label}"
            command = [
                sys.executable,
                str(ROOT / "ops/run_recipe.py"),
                str(config),
                "--data-root",
                str(args.data_root.resolve()),
                "--model",
                str(args.model.resolve()),
                "--output",
                str(output),
                "--execute",
            ]
            if checkpoint:
                command += ["--resume", str(checkpoint)]
            print(
                f"Running {method}/{label}; log: {work}/{method}-{label}.log",
                flush=True,
            )
            start = time.time()
            with (work / f"{method}-{label}.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=args.timeout)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            if code:
                raise RuntimeError(
                    f"{method}/{label} failed: {work}/{method}-{label}.log"
                )
            manifest["runs"][f"{method}-{label}"] = {
                "elapsed_seconds": time.time() - start,
                "returncode": code,
            }
            (work / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if label == "whole":
                matches = list(output.glob("*/checkpoints/step_00002"))
                if len(matches) != 1:
                    raise ValueError("missing unique committed interruption boundary")
                checkpoint = matches[0]
    audit(work)


def compare_tree(a, b, *, atol=0.0, rtol=0.0, path="state"):
    import numpy as np
    import torch

    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"{path}: tensor structure differs")
        # DeepSpeed flattens hundreds of millions of parameters into one
        # tensor. Bound comparison temporaries instead of materializing several
        # full-size difference/mask arrays alongside both optimizer states.
        left, right = a.reshape(-1), b.reshape(-1)
        for start in range(0, left.numel(), 1_000_000):
            x, y = left[start : start + 1_000_000], right[start : start + 1_000_000]
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"{path}: nonfinite tensor state")
            if not torch.allclose(x, y, atol=atol, rtol=rtol, equal_nan=False):
                raise ValueError(f"{path}: tensor differs beyond tolerance")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise ValueError(f"{path}: RNG array differs")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError(f"{path}: keys differ")
        for key in a:
            compare_tree(a[key], b[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f"{path}: length differs")
        for i, (x, y) in enumerate(zip(a, b)):
            compare_tree(x, y, atol=atol, rtol=rtol, path=f"{path}[{i}]")
    elif (
        type(a) is type(b)
        and type(a).__module__ == "deepspeed.runtime.fp16.loss_scaler"
    ):
        compare_tree(vars(a), vars(b), atol=atol, rtol=rtol, path=f"{path}.state")
    elif a != b:
        raise ValueError(f"{path}: {a!r} != {b!r}")


def completed_methods(experiment):
    if experiment.get("schema") != "remax-resume-gpu-v1":
        raise ValueError("unsupported resume experiment manifest")
    runs = experiment.get("runs", {})
    methods = [m for m in METHODS if f"{m}-whole" in runs]
    expected = {f"{m}-{label}" for m in methods for label in ("whole", "resumed")}
    if (
        not methods
        or set(runs) != expected
        or any(r["returncode"] != 0 for r in runs.values())
    ):
        raise ValueError("audit requires complete successful whole/resumed pairs")
    if experiment.get("tolerances") != TOLERANCES:
        raise ValueError(
            "experiment tolerances differ from the declared audit contract"
        )
    return methods


def audit(work):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import torch
    from remax.checkpointing import validate_checkpoint
    from remax.input_identity import digest

    experiment = json.loads((work / "experiment.json").read_text())
    report = {
        "schema": "remax-resume-equivalence-v1",
        "tolerances": TOLERANCES,
        "methods": {},
        "scope": "six updates across two epochs; explicit step-2 restart; single GPU",
    }
    for method in completed_methods(experiment):
        roots = []
        for label in ("whole", "resumed"):
            markers = list(
                (work / f"{method}-{label}").glob("*/TRAINING_COMPLETE.json")
            )
            if not markers:  # completion marker is named by the storage finalizer
                candidates = list((work / f"{method}-{label}").glob("*/checkpoints"))
                if len(candidates) != 1:
                    raise ValueError("missing run checkpoint directory")
                roots.append(candidates[0].parent)
            else:
                roots.append(markers[0].parent)
        whole, resumed = roots
        original = load_records(whole / "resume_decisions.jsonl")
        continued = load_records(resumed / "resume_decisions.jsonl")
        if [r["step"] for r in original] != list(range(1, 7)) or [
            r["step"] for r in continued
        ] != list(range(3, 7)):
            raise ValueError(f"{method}: incomplete six-update trajectory")
        if [r for r in original if r["step"] > 2] != continued:
            raise ValueError(f"{method}: sampling/bank/replay/data decisions diverged")
        baseline_eval = [
            r
            for r in load_records(whole / "eval_mode_coverage_draws.jsonl")
            if r["step"] > 2
        ]
        if {r["step"] for r in baseline_eval} != {4, 6}:
            raise ValueError(f"{method}: missing scheduled evaluations")
        if baseline_eval != load_records(resumed / "eval_mode_coverage_draws.jsonl"):
            raise ValueError(f"{method}: evaluation cadence or outcomes diverged")
        for step in (4, 6):
            print(f"Auditing {method}, checkpoint {step}", flush=True)
            folders = [root / "checkpoints" / f"step_{step:05d}" for root in roots]
            manifests = [validate_checkpoint(f) for f in folders]
            if manifests[0]["identity"] != manifests[1]["identity"]:
                raise ValueError("resume identities differ")
            states = [
                torch.load(
                    next(f.glob("*model_states.pt")),
                    map_location="cpu",
                    weights_only=False,
                )
                for f in folders
            ]
            if any(state["global_steps"] != step for state in states):
                raise ValueError(
                    "checkpoint does not contain the expected optimizer updates"
                )
            compare_tree(
                states[0]["module"],
                states[1]["module"],
                **TOLERANCES["model"],
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
                compare_tree(states[0][key], states[1][key], path=key)
            del states
            optimizers = [
                torch.load(
                    next(f.glob("*optim_states.pt")),
                    map_location="cpu",
                    weights_only=False,
                )
                for f in folders
            ]
            compare_tree(
                optimizers[0],
                optimizers[1],
                **TOLERANCES["optimizer"],
                path="optimizer",
            )
            del optimizers
        report["methods"][method] = {
            "updates": 6,
            "fresh_responses": sum(len(r["responses"]) for r in original),
            "accepted_responses": sum(
                any(v != 0 for v in response["rewards"])
                for r in original
                for response in r["responses"]
            ),
            "replay_groups": sum(len(r["replay"]) for r in original),
            "run_identity_sha256": manifests[0]["identity_sha256"],
            "resume_after": 2,
            "compared_checkpoint_steps": [4, 6],
            "identical_decisions": True,
            "identical_evaluation": True,
            "identical_bank_scheduler_rng": True,
            "model_optimizer_within_tolerance": True,
            "artifacts": {
                str(p.relative_to(work)): digest(p)
                for root in roots
                for p in (
                    root / "resume_decisions.jsonl",
                    root / "eval_mode_coverage_draws.jsonl",
                )
            },
        }
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--workdir", type=Path, required=True)
    runp.add_argument("--model", type=Path, required=True)
    runp.add_argument("--data-root", type=Path, required=True)
    runp.add_argument(
        "--domain",
        choices=(
            "pantry_plan",
            "countdown",
            "graph_coloring",
            "mathir",
            "python_factors",
        ),
        default="pantry_plan",
    )
    runp.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    runp.add_argument("--timeout", type=int, default=1800)
    auditp = sub.add_parser("audit")
    auditp.add_argument("--workdir", type=Path, required=True)
    args = p.parse_args()
    if args.command == "run":
        run(args)
    else:
        audit(args.workdir.resolve())


if __name__ == "__main__":
    main()
