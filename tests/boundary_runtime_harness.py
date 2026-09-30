"""Import real actor/evaluator modules with only GPU/OAT interfaces substituted."""

from contextlib import contextmanager
import importlib
import sys
from types import ModuleType, SimpleNamespace

from tests.training_harness import learner_types


@contextmanager
def runtime_modules():
    with learner_types():
        names = (
            "pandas",
            "tree",
            "tqdm",
            "oat.interface",
            "oat.types",
            "oat.utils.data",
            "oat.utils.distributed",
            "oat.oracles",
            "oat.oracles.base",
            "vllm",
            "vllm.envs",
            "transformers",
        )
        shims = {name: ModuleType(name) for name in names}
        shims["tqdm"].tqdm = lambda iterable, **kw: iterable
        shims["oat.interface"].lp = SimpleNamespace()
        shims["oat.types"].TrajectoryData = type("TrajectoryData", (), {})
        shims["oat.types"].Metric = dict
        for name in ("PromptDataset", "load_data_from_disk_or_hf"):
            setattr(shims["oat.utils.data"], name, lambda *a, **k: None)
        for name in (
            "init_process_group",
            "node_ip_address_from_perspective",
            "torch_type_codec",
        ):
            setattr(shims["oat.utils.distributed"], name, lambda *a, **k: None)
        shims["oat.utils.distributed"].WorkerWrap = type("WorkerWrap", (), {})
        for name in ("PreferenceOracleBase", "RewardOracleBase"):
            setattr(shims["oat.oracles.base"], name, type(name, (), {}))
        shims["transformers"].AutoTokenizer = type("AutoTokenizer", (), {})
        shims["vllm"].envs = shims["vllm.envs"]
        sys.modules["oat.algorithms.ppo"].PPOActor = type("PPOActor", (), {})
        imported = (
            "remax.actor",
            "remax.learner.run",
            "remax.vllm_worker",
            "remax.trajectory_dataset",
        )
        missing = object()
        saved = {name: sys.modules.get(name, missing) for name in (*names, *imported)}
        parents = [
            (importlib.import_module(n.rpartition(".")[0]), n.rpartition(".")[2])
            for n in imported
        ]
        attrs = [
            (parent, attr, getattr(parent, attr, missing)) for parent, attr in parents
        ]
        for name in imported:
            sys.modules.pop(name, None)
        sys.modules.update(shims)
        try:
            yield importlib.import_module("remax.actor"), importlib.import_module(
                "remax.learner.run"
            )
        finally:
            for name, value in saved.items():
                if value is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
            for parent, attr, value in attrs:
                if value is missing:
                    if hasattr(parent, attr):
                        delattr(parent, attr)
                else:
                    setattr(parent, attr, value)
