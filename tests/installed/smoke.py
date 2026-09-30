"""Run with python -I in a core-only venv, outside the checkout (no pytest needed)."""

import importlib.abc
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in {
            "oat",
            "vllm",
            "deepspeed",
            "transformers",
            "datasets",
        }:
            raise AssertionError(f"core imported a training dependency: {fullname}")
        if fullname.startswith(("remax.experiments", "remax.integrations")):
            raise AssertionError(f"core imported a training implementation: {fullname}")


sys.meta_path.insert(0, RejectTraining())
import torch
import remax
from remax import core
from remax.core.checkpoints import commit_checkpoint, validate_checkpoint
from remax.input_identity import registry
from remax.launcher import environment, recipe_names, resolve_recipe, render
from remax.recipes import Recipe


def command(*args):
    return subprocess.check_output([sys.executable, "-I", "-m", *args], text=True)


class InstalledWorkflow(unittest.TestCase):
    def test_installation_is_real_and_core_only(self):
        self.assertTrue(sys.flags.isolated)
        self.assertNotIn("PYTHONPATH", os.environ)
        self.assertTrue(
            Path(remax.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        )
        direct = json.loads(
            metadata.distribution("remax-rl").read_text("direct_url.json") or "{}"
        )
        self.assertFalse(direct.get("dir_info", {}).get("editable", False))
        for name in ("oat-llm", "vllm", "deepspeed", "transformers", "datasets"):
            with self.assertRaises(metadata.PackageNotFoundError):
                metadata.version(name)
        self.assertEqual(len(registry()["splits"]), 10)

    def test_documented_commands(self):
        expected = {
            "maxrl_advantages": [[1.0, -1.0]],
            "replay_loss": 2.0,
            "score_gradients": [-0.25, -0.25, -0.5],
        }
        self.assertEqual(json.loads(command("remax", "demo")), expected)
        console = Path(sys.executable).parent / "remax"
        self.assertEqual(
            json.loads(subprocess.check_output([str(console), "demo"], text=True)),
            expected,
        )
        self.assertEqual(
            command("remax", "--version").strip(), metadata.version("remax-rl")
        )
        self.assertEqual(len(command("remax", "recipes").splitlines()), 20)
        recipe = json.loads(command("remax", "recipes", "remax_countdown_05b"))
        self.assertEqual(recipe["method"], "remax")
        json.loads(command("remax", "environment"))
        command("remax.launcher", "--help")
        result = subprocess.run(
            [sys.executable, "-I", "-m", "remax", "environment", "--training"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertTrue(
            "training requires" in result.stderr
            or "missing training dependency" in result.stderr
        )

    def test_result_packages_without_checkout(self):
        rows = command("remax", "results", "list").splitlines()
        self.assertEqual(len(rows), 101)
        self.assertEqual(
            sum(row.startswith("saved_key_reproducible\t") for row in rows), 95
        )
        report = json.loads(command("remax", "results", "verify"))
        self.assertFalse(report["numerical_reproduction_performed"])
        package = json.loads(
            command("remax", "results", "show", "level2/qwen05b/countdown/replay_maxrl")
        )
        self.assertEqual(len(package["cells"]), 5)
        self.assertIsNone(package["recipe"]["path"])

    def test_all_bundled_recipe_commands(self):
        names = recipe_names()
        self.assertEqual(len(names), 20)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in names:
                with self.subTest(recipe=name):
                    raw = json.loads(resolve_recipe(name).read_text())
                    recipe = Recipe.from_dict(raw)
                    env = environment(
                        raw,
                        data_root=root / "data",
                        model=root / "model",
                        output=root / "output",
                        seed=47,
                    )
                    self.assertNotIn("PYTHONPATH", env)
                    cmd, _ = render(env)
                    self.assertEqual(
                        cmd[:3], [sys.executable, "-m", "remax.train_zero_math"]
                    )
                    self.assertEqual(
                        "--maxrl-task-objective" in cmd,
                        recipe.method in ("remax", "maxrl"),
                    )
                    self.assertEqual(
                        "--online-canonical-replay-compute-only" in cmd,
                        recipe.method in ("drgrpo", "maxrl"),
                    )
            preview = command(
                "remax.launcher",
                "remax_countdown_05b",
                "--data-root",
                str(root / "data"),
                "--model",
                str(root / "model"),
                "--output",
                str(root / "output"),
                "--render-only",
            )
            self.assertIn("UNVERIFIED", preview)
            self.assertFalse((root / "output").exists())

    def test_verification_bank_scoring_and_checkpoint(self):
        from remax.benchmark import (
            grade_reference_response,
            require_scorable,
            EvaluationFailure,
        )

        reference = {"verifier": "countdown", "numbers": [3, 6, 9], "target": 18}
        result = grade_reference_response("(3+(6+9))", reference)
        self.assertTrue(result["verified"])
        with self.assertRaises(EvaluationFailure):
            require_scorable({"status": "worker_failure", "detail": "unavailable"})
        bank = core.OnlineCanonicalBank(
            entropy_alpha=0, retain_exemplars=True, global_replay_groups_per_step=1
        )
        bank.score_and_update(
            prompt_token_ids=[[1]] * 2,
            outcome_keys=[result["canonical_key"]] * 2,
            task_rewards=[1, 1],
            active_mask=[1, 1],
            num_samples=2,
            response_token_ids=[[2, 3]] * 2,
        )
        groups = bank.scheduled_global_replay_groups(min_modes=1)
        batch = core.materialize_canonical_replay_batch(
            groups, pad_token_id=0, device=torch.device("cpu")
        )
        self.assertEqual(batch.group_sizes, (1,))
        scores = torch.tensor([-2.0], requires_grad=True)
        core.canonical_replay_uniform_verified_likelihood_loss(
            scores, batch.group_sizes
        ).loss.backward()
        self.assertEqual(scores.grad.tolist(), [-1.0])
        with tempfile.TemporaryDirectory() as folder:

            def writer(path):
                torch.save(bank.state_dict(), path / "bank.pt")

            checkpoint = commit_checkpoint(
                Path(folder), step=1, identity={"example": 1}, writer=writer, keep=1
            )
            self.assertEqual(validate_checkpoint(checkpoint, {"example": 1})["step"], 1)
            restored = core.OnlineCanonicalBank(
                entropy_alpha=0, retain_exemplars=True, global_replay_groups_per_step=1
            )
            restored.load_state_dict(
                torch.load(checkpoint / "bank.pt", weights_only=False)
            )
            self.assertEqual(restored.state_dict(), bank.state_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
