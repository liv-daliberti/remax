"""Dependency boundaries and numerical parity of the maintained extraction."""

import ast
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
import remax

from remax.integrations.oat.selection import EXPERIMENT_SWITCHES, historical_reasons
from tests.training_harness import learner_types, make_learner, run_step, snapshot

ROOT = Path(__file__).parents[1]
SPEC = json.loads((ROOT / "tests/fixtures/training_v1/inputs.json").read_text())


class RejectExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("remax.experiments.") and ".oat" not in fullname:
            raise AssertionError(f"maintained path imported {fullname}")
        if fullname.startswith("remax.experiments.oat."):
            raise AssertionError(f"maintained path imported {fullname}")
        return None


def test_core_imports_without_training_or_experiments():
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'oat', 'vllm', 'deepspeed', 'transformers'} or fullname.startswith(('remax.integrations', 'remax.experiments')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from remax.core.bank import OnlineCanonicalBank
from remax.core.objectives import binary_maxrl_advantages, canonical_replay_uniform_verified_likelihood_loss
from remax.core.scoring import materialize_canonical_replay_batch
from remax.core.checkpoints import commit_checkpoint
assert OnlineCanonicalBank(entropy_alpha=0).state_dict()
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_maintained_training_never_imports_comparators():
    # Remove already loaded experiments to make this independent of test order.
    saved = {k: v for k, v in sys.modules.items() if k.startswith("remax.experiments.")}
    for name in saved:
        sys.modules.pop(name)
    guard = RejectExperiments()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with learner_types() as types:
            sys.meta_path.insert(0, guard)
            for method in ("drgrpo", "redr", "maxrl", "remax"):
                learner = make_learner(types, SPEC, method)
                for step in SPEC["steps"]:
                    run_step(learner, SPEC, step)
                    assert learner._remax_adapter == "maintained"
                    assert learner._remax_adapter_reasons == ()
    finally:
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        sys.modules.update(saved)
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("method", ["drgrpo", "redr", "maxrl", "remax"])
@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_extracted_and_historical_updates_are_identical(method, microbatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with learner_types() as types:
            from remax.experiments.oat.grpo import HistoricalGrpoMixin

            maintained = make_learner(types, SPEC, method, microbatch)
            historical = make_learner(types, SPEC, method, microbatch)
            # Force both historical entry points; inner dispatch must not mask a difference.
            for name in (
                "_grpo_learning_step_with_progress",
                "_baseline_update_with_precomputed_advantages",
            ):
                setattr(
                    historical,
                    name,
                    MethodType(getattr(HistoricalGrpoMixin, name), historical),
                )
            for step in SPEC["steps"]:
                rng = np.random.get_state(), torch.get_rng_state()
                actual = run_step(maintained, SPEC, step)
                np.random.set_state(rng[0])
                torch.set_rng_state(rng[1])
                expected = run_step(historical, SPEC, step)
                assert snapshot(maintained, actual) == snapshot(historical, expected)
                assert maintained.model.calls == historical.model.calls
                assert actual.keys() == expected.keys()
                for key in actual:
                    if key != "get_grad_norm_time":
                        torch.testing.assert_close(
                            actual[key], expected[key], rtol=0, atol=0, equal_nan=True
                        )
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("field", EXPERIMENT_SWITCHES)
def test_comparator_switches_cannot_enter_maintained_path(field):
    learner = SimpleNamespace(args=SimpleNamespace(**{field: 1}))
    assert field in historical_reasons(learner)


@pytest.mark.parametrize(
    "field,value",
    [
        ("critic_type", "ppo"),
        ("xdr_tau", 0.1),
        ("online_canonical_key_mode", "verified_route"),
        ("online_canonical_replay_objective", "bank_balance"),
        ("online_canonical_replay_key_weighting", "fresh_frequency"),
    ],
)
def test_comparator_modes_cannot_enter_maintained_path(field, value):
    args = SimpleNamespace(online_canonical_replay=True, **{field: value})
    assert field in historical_reasons(SimpleNamespace(args=args))


def test_identity_bound_run_cannot_fall_back_to_historical_adapter():
    with learner_types() as types:
        learner = make_learner(types, SPEC, "remax")
        learner.args.dapo_enabled = True
        learner.args._remax_resume_identity = {"protocol": "test"}
        with pytest.raises(RuntimeError, match="requires historical adapter"):
            run_step(learner, SPEC, SPEC["steps"][0])
        assert not learner.strategy.updates
        assert learner._online_canonical_bank.tracked_prompt_count == 0


def test_legacy_imports_preserve_class_and_function_identity():
    from remax.online_canonical_bank import OnlineCanonicalBank as legacy
    from remax.core.bank import OnlineCanonicalBank
    from remax.maxrl import binary_maxrl_advantages as old_objective
    from remax.core.objectives import binary_maxrl_advantages
    from remax.rlep import RLEPExperiencePool as old_pool
    from remax.experiments.rlep import RLEPExperiencePool

    assert legacy is OnlineCanonicalBank
    assert old_objective is binary_maxrl_advantages
    assert old_pool is RLEPExperiencePool


def test_core_has_no_training_or_comparator_imports():
    for path in (Path(remax.__file__).parent / "core").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = node.module or ""
                assert not any(
                    part in target.split(".")
                    for part in (
                        "learner",
                        "integrations",
                        "experiments",
                        "oat",
                        "vllm",
                        "deepspeed",
                    )
                ), (path, target)
