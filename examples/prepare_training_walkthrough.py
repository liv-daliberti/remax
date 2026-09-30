"""Write a six-update Re:Max recipe; authenticate full inputs at launch as usual."""

import argparse
import json
from pathlib import Path

from remax.launcher import resolve_recipe
from remax.recipes import Recipe


def prepare(path: Path) -> None:
    recipe = json.loads(resolve_recipe("remax_pantry_plan_05b").read_text())
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
    Recipe.from_dict(recipe)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(recipe, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
