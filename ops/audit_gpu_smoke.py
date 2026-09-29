"""Fail closed unless GPU artifacts demonstrate the complete four-method workflow."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from gpu_smoke import METHODS, digest, validate_inputs, write_json

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_metrics(rows, method):
    trained = [r for r in rows if r.get("train/canonical_replay_score_passes", 0) > 0]
    require(trained, f"{method}: no teacher-forced replay scoring")
    control = method in ("drgrpo", "maxrl")
    for row in trained:
        for key in (
            "train/canonical_replay_weighted_loss",
            "train/canonical_replay_raw_weighted_loss",
            "train/canonical_replay_applied_score_gradient_l2",
        ):
            require(
                key in row and math.isfinite(row[key]),
                f"{method}: missing/nonfinite {key}",
            )
        require(
            row["train/canonical_replay_compute_only"] == float(control),
            f"{method}: wrong replay policy",
        )
        if control:
            require(
                row["train/canonical_replay_weighted_loss"] == 0
                and row["train/canonical_replay_applied_score_gradient_l2"] == 0,
                f"{method}: control applies replay gradient",
            )
    require(
        any(r["train/canonical_replay_raw_weighted_loss"] > 0 for r in trained),
        f"{method}: empty replay computation",
    )
    if not control:
        require(
            any(
                r["train/canonical_replay_applied_score_gradient_l2"] > 0
                for r in trained
            ),
            f"{method}: replay gradient never applied",
        )
    require(
        max(r.get("trainer/policy_sgd_step", 0) for r in rows) >= 4,
        f"{method}: fewer than four optimizer updates",
    )
    return {
        "optimizer_updates": max(r.get("trainer/policy_sgd_step", 0) for r in rows),
        "replay_score_passes_max": max(
            r["train/canonical_replay_score_passes"] for r in trained
        ),
        "replay_raw_loss_max": max(
            r["train/canonical_replay_raw_weighted_loss"] for r in trained
        ),
        "replay_applied_loss_max": max(
            r["train/canonical_replay_weighted_loss"] for r in trained
        ),
        "replay_applied_score_gradient_l2_max": max(
            r["train/canonical_replay_applied_score_gradient_l2"] for r in trained
        ),
    }


def audit_draws(draws):
    count = 0
    for draw in draws:
        require(len(draw["prompts"]) == 2, "Incomplete evaluation prompt set")
        for prompt in draw["prompts"]:
            infos = prompt["verifier_infos"]
            require(
                len(infos) == draw["sample_count"], "Incomplete verifier diagnostics"
            )
            for info in infos:
                diagnostic = info["verifier"]
                require(
                    diagnostic["status"] in ("correct", "incorrect", "malformed"),
                    "Fatal evaluator status",
                )
                count += 1
    require(count > 0, "No verified evaluation responses")
    return count


def changed_parameters(base, export):
    from safetensors import safe_open

    total = 0
    largest = 0.0
    with safe_open(
        str(base / "model.safetensors"), framework="pt", device="cpu"
    ) as original:
        for shard in export.glob("*.safetensors"):
            with safe_open(str(shard), framework="pt", device="cpu") as final:
                for key in final.keys():
                    require(key in original.keys(), f"Unexpected parameter {key}")
                    a, b = original.get_tensor(key), final.get_tensor(key)
                    require(a.shape == b.shape, f"Parameter shape changed: {key}")
                    total += int((a != b).sum())
                    difference = float((a.float() - b.float()).abs().max())
                    require(
                        math.isfinite(difference), f"Nonfinite parameter update: {key}"
                    )
                    largest = max(largest, difference)
    require(
        total > 0 and math.isfinite(largest),
        "No finite numerical parameter update in exported model",
    )
    return {
        "changed_parameter_elements": total,
        "max_absolute_parameter_change": largest,
    }


def audit(work):
    import torch
    from remax.online_canonical_bank import OnlineCanonicalBank

    inputs = validate_inputs(work)
    result = {"schema": "remax-gpu-smoke-report-v1", "methods": {}}
    for method in METHODS:
        exit_record = json.loads((work / f"{method}-exit.json").read_text())
        require(exit_record["returncode"] == 0, f"{method}: process failed")
        completions = list((work / method).rglob("TRAINING_COMPLETE.json"))
        require(len(completions) == 1, f"{method}: missing completion marker")
        completion = json.loads(completions[0].read_text())
        attempt = Path(completion["terminal_attempt"])
        require(
            not list(attempt.rglob("evaluation_failures.jsonl")),
            f"{method}: failed evaluation",
        )
        metrics = audit_metrics(records(attempt / "train_metrics.jsonl"), method)
        with (work / f"{method}-gpu.csv").open() as stream:
            samples = list(csv.reader(stream))
        require(
            len(samples) > 1 and samples[0][0] == "timestamp",
            f"{method}: missing GPU resource samples",
        )
        metrics["peak_gpu_memory_mib"] = max(float(row[1]) for row in samples[1:])
        metrics["gpu_memory_samples"] = len(samples) - 1
        metrics["verified_evaluation_responses"] = audit_draws(
            records(attempt / "eval_mode_coverage_draws.jsonl")
        )
        checkpoint = attempt / "checkpoints"
        tag = (checkpoint / "latest").read_text().strip()
        model_states = list((checkpoint / tag).glob("*model_states.pt"))
        optimizer_states = list((checkpoint / tag).glob("*optim_states.pt"))
        require(
            len(model_states) == 1 and len(optimizer_states) == 1,
            f"{method}: incomplete ZeRO checkpoint",
        )
        # Only load checkpoints produced by this run, never arbitrary downloads.
        state = torch.load(model_states[0], map_location="cpu", weights_only=False)
        bank_state = state["online_canonical_bank_state"]
        bank = OnlineCanonicalBank(
            entropy_alpha=0, retain_exemplars=True, global_replay_groups_per_step=1
        )
        bank.load_state_dict(bank_state)
        require(
            bank.tracked_prompt_count > 0 and bank.tracked_outcome_count > 0,
            f"{method}: empty saved bank",
        )
        metrics.update(
            {
                "checkpoint_step": state["steps"],
                "bank_prompts": bank.tracked_prompt_count,
                "bank_outcomes": bank.tracked_outcome_count,
                "elapsed_seconds": exit_record["elapsed_seconds"],
            }
        )
        del state
        optimizer = torch.load(
            optimizer_states[0], map_location="cpu", weights_only=False
        )
        adam_states = optimizer["optimizer_state_dict"]["base_optimizer_state"]["state"]
        optimizer_steps = [float(value["step"]) for value in adam_states.values()]
        require(
            optimizer_steps
            and all(value == metrics["optimizer_updates"] for value in optimizer_steps),
            f"{method}: saved optimizer progress differs",
        )
        metrics["saved_optimizer_updates"] = optimizer_steps
        del optimizer, adam_states
        metrics.update(
            changed_parameters(
                Path(inputs["model"]), Path(completion["terminal_export"])
            )
        )
        draws = records(attempt / "eval_mode_coverage_draws.jsonl")
        final_evaluation_step = max(d["step"] for d in draws)
        metrics["terminal_evaluation_step"] = final_evaluation_step
        metrics["terminal_export_step"] = completion["terminal_step"]
        metrics["terminal_evaluation"] = [
            {
                "kind": d["evaluation_kind"],
                "draw": d["draw_index"],
                "metrics": d["metrics"],
            }
            for d in draws
            if d["step"] == final_evaluation_step
        ]
        metrics["artifact_sha256"] = {
            str(p.relative_to(work)): digest(p)
            for p in (
                attempt / "train_metrics.jsonl",
                attempt / "eval_mode_coverage_draws.jsonl",
                completions[0],
                model_states[0],
                optimizer_states[0],
            )
        }
        result["methods"][method] = metrics
    restored = json.loads((work / "remax-restored-exit.json").read_text())
    require(restored["returncode"] == 0, "Checkpoint reload failed")
    markers = list((work / "remax-restored").rglob("EVAL_ONLY_COMPLETE.json"))
    require(len(markers) == 1, "Missing restored evaluation completion marker")
    resumed_draws = records(markers[0].parent / "eval_mode_coverage_draws.jsonl")
    audit_draws(resumed_draws)
    original_path = next((work / "remax").rglob("eval_mode_coverage_draws.jsonl"))
    step = result["methods"]["remax"]["checkpoint_step"]
    original_draws = [r for r in records(original_path) if r["step"] == step]
    # Compare saved responses, rewards, keys, diagnostics and per-draw metrics.
    require(
        original_draws == resumed_draws,
        "Restored checkpoint evaluation differs from its saved-step evaluation",
    )
    result["restored_evaluation"] = {
        "checkpoint_step": step,
        "exact_draw_records_match": True,
        "elapsed_seconds": restored["elapsed_seconds"],
    }
    hardware = json.loads((work / "environment.json").read_text())
    result["hardware"] = {
        k: hardware[k]
        for k in ("python", "platform", "gpu", "gpu_memory_bytes", "cuda_runtime")
    }
    result["model"] = {
        "id": inputs["model_id"],
        "revision": inputs["model_revision"],
        "sha256": inputs["model_sha256"],
    }
    result["datasets"] = inputs["splits"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.workdir.resolve())
    write_json(args.workdir / "report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
