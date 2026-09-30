"""The newcomer examples must use actual grading and the maintained replay API."""

import json
from pathlib import Path
import runpy

from remax.walkthrough import walkthrough


def test_walkthrough_uses_verified_modes_and_response_only_scoring():
    result = walkthrough()
    assert result == {
        "statuses": ["correct", "correct", "incorrect"],
        "retained_modes": 2,
        "prompt_tokens_scored": 0,
        "parameters_updated": True,
        "bank_restored": True,
        "scope": "CPU replay API example; not a complete RL training run",
    }


def test_short_recipe_runs_six_updates_with_retained_recovery(tmp_path):
    import pytest
    from remax.launcher import environment, render
    from remax.recipes import Recipe

    script = Path(__file__).parents[1] / "examples/prepare_training_walkthrough.py"
    prepare = runpy.run_path(str(script))["prepare"]
    output = tmp_path / "recipe.json"
    prepare(output)
    recipe = Recipe.load(output)
    assert recipe.settings.num_prompt_epoch * recipe.settings.max_train == 6
    assert recipe.settings.save_ckpt
    assert recipe.settings.resume_steps == recipe.settings.eval_steps == 2
    assert recipe.settings.max_resume_num == 3
    assert not recipe.settings.prune_resume_on_success
    assert recipe.settings.eval_mode_coverage_k == 2
    assert recipe.settings.eval_mode_coverage_draws == 1
    command, _ = render(
        environment(
            json.loads(output.read_text()),
            data_root=tmp_path / "data",
            model=tmp_path / "model",
            output=tmp_path / "run",
            seed=43,
        )
    )
    assert "--save-ckpt" in command
    assert command[command.index("--max-train") + 1] == "3"
    with pytest.raises(FileExistsError):
        prepare(output)
