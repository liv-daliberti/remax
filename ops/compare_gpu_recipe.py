"""Compare the maintained pantry recipe with retained historical launch evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

from remax.launcher import environment

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests/fixtures/gpu_historical_pantry_v1.json"


def compare(recipe, reference):
    checked = {}
    differences = {}
    for key, historical in reference["registered_environment"].items():
        current = recipe["environment"].get(key)
        if current != historical:
            differences[key] = {"historical": historical, "current": current}
        checked[key] = current
    if differences:
        raise ValueError(
            f"Registered method settings differ from historical ledger: {differences}"
        )
    # Inspect the actual portable shell command for the opt-in recovery flag.
    with tempfile.TemporaryDirectory(prefix="remax-recipe-") as temporary:
        folder = Path(temporary)
        (folder / "train").mkdir()
        (folder / "eval").mkdir()
        env = environment(
            recipe,
            data_root=folder,
            model=folder / "model",
            output=folder / "output",
            seed=recipe["seed"],
        )
        rendered = subprocess.check_output(
            ["bash", str(ROOT / "ops/train.sh")], env=env, text=True
        )
    command = shlex.split(
        next(
            line.split(":", 1)[1]
            for line in rendered.splitlines()
            if line.startswith("[train] command:")
        )
    )
    runtime_differences = {}
    for field, previous in reference["runtime_fields"].items():
        key = "OAT_ZERO_" + field.upper()
        if field == "save_ckpt":
            current = "--save-ckpt" in command
        elif key in recipe["environment"]:
            current = recipe["environment"][key]
        else:
            continue
        try:
            equal = float(previous) == float(current)
        except (ValueError, TypeError):
            equal = str(previous).lower() == str(current).lower()
        if not equal:
            runtime_differences[field] = {"historical": previous, "current": current}
    expected = {
        "eval_steps": {"historical": 96, "current": "192"},
        "save_ckpt": {"historical": True, "current": False},
    }
    if runtime_differences != expected:
        raise ValueError(
            f"Unexpected runtime configuration differences: {runtime_differences}"
        )
    return {
        "schema": "remax-historical-config-comparison-v1",
        "recipe": recipe["name"],
        "registered_environment_fields_matched": len(checked),
        "runtime_differences": runtime_differences,
        "interpretation": "Historical wrapper capped evaluation at a quarter pass (96 steps); portable recipe explicitly evaluates each half pass (192). Recovery checkpoint writes were enabled historically but require an explicit OAT_ZERO_SAVE_CKPT=1 in the portable launcher (the smoke enables this). Automatic checkpoint discovery is not implemented by the portable launcher. Training method settings match. This is not a historical score or bitwise trajectory reproduction.",
        "historical_sources": reference["sources"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = compare(
        json.loads((ROOT / "configs/remax_pantry_plan_05b.json").read_text()),
        json.loads(REFERENCE.read_text()),
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
