"""Adversarial recipe/input checks must fail before a training process is started."""

from dataclasses import make_dataclass
import json
from pathlib import Path
import sys
import types

import pytest

import remax.input_identity as identity
from remax.launch_record import (
    configuration_digest,
    finalize_launch,
    json_value,
    write_json,
)
from remax.recipes import Recipe, strict_json
from remax import launcher as run_recipe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw():
    return json.loads((ROOT / "configs/remax_pantry_plan_05b.json").read_text())


@pytest.mark.parametrize(
    "change",
    [
        {"typo": 1},
        {"level": True},
        {"level": 2},
        {"domain": "missing"},
        {"method": "unknown"},
        {"model_revision": "main"},
        {"model_id": "another/model"},
        {"seed": True},
        {"seed": 48},
        {"registered_seeds": [43, 44]},
        {"resolved_wrapper_defaults": {"typo": "ignored"}},
    ],
)
def test_rejects_unknown_or_incompatible_recipe(raw, change):
    raw.update(change)
    with pytest.raises(ValueError):
        Recipe.from_dict(raw)


@pytest.mark.parametrize(
    "key,value",
    [
        ("TYPO", "1"),
        ("LEARNING_RATE", "nan"),
        ("LEARNING_RATE", "-1"),
        ("LEARNING_RATE", 0.1),
        ("NUM_SAMPLES", "1.0"),
        ("NUM_SAMPLES", "0"),
        ("NUM_SAMPLES", "1"),
        ("SEED", "44"),
        ("MAX_TRAIN", "385"),
        ("MAXRL_TASK_OBJECTIVE", "0"),
        ("ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY", "1"),
        ("ONLINE_CANONICAL_REPLAY", "false"),
        ("ONLINE_CANONICAL_REPLAY", "0"),
        ("ONLINE_CANONICAL_REPLAY_OBJECTIVE", "bank_balance"),
        ("ONLINE_CANONICAL_REPLAY_ALPHA", "0"),
        ("XDR_TAU", "nan"),
        ("XDR_TAU", "0.5"),
        ("PROMPT_TEMPLATE", "qwen_boxed"),
        ("CANONICAL_ACTION_TASK", "none"),
        ("CANONICAL_GRAPH_ACTION_COUNT", "3"),
        ("GENERATE_MAX_LENGTH", "192"),
        ("MAX_MODEL_LEN", "640"),
        ("TEST_SPLIT", "train"),
        ("INPUT_KEY", "other"),
        ("POLICY_ENTROPY_COEF", "0.1"),
        ("VERIFIER_VERSION", "math_verify"),
        ("TOP_P", "1.1"),
        ("VLLM_GPU_RATIO", "1"),
        ("ZERO_STAGE", "4"),
        ("TRAIN_BATCH_SIZE_PER_DEVICE", "3"),
        ("NUM_PROMPT_EPOCH", "9"),
        ("EVAL_PROMPT_INTERVAL", "96"),
        ("RESUME_DIR", "/some/checkpoint"),
    ],
)
def test_rejects_invalid_training_settings(raw, key, value):
    raw["environment"]["OAT_ZERO_" + key] = value
    with pytest.raises(ValueError):
        Recipe.from_dict(raw)


def test_required_fields_cannot_fall_back_silently(raw):
    del raw["environment"]["OAT_ZERO_PROMPT_TEMPLATE"]
    with pytest.raises(ValueError, match="incomplete"):
        Recipe.from_dict(raw)


@pytest.mark.parametrize(
    "text",
    ['{"seed":43,"seed":44}', '{"a":NaN}', '{"a":Infinity}', '{"a":{"x":1,"x":2}}'],
)
def test_json_rejects_ambiguity(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_legacy_omissions_have_explicit_typed_defaults():
    raw = json.loads((ROOT / "configs/redr_countdown_05b.json").read_text())
    recipe = Recipe.from_dict(raw)
    assert recipe.settings.critic_type == "drgrpo"
    assert recipe.settings.maxrl_task_objective is False
    assert recipe.settings.save_ckpt is False
    raw["environment"]["OAT_ZERO_SAVE_CKPT"] = "1"
    assert Recipe.from_dict(raw).settings.save_ckpt is True
    with pytest.raises(ValueError, match="registered_seeds"):
        recipe.choose_seed(48)


@pytest.mark.parametrize(
    "key,value",
    [
        ("OAT_ZERO_SEED", "43"),
        ("OAT_ZERO_PROMPT_TEMPLATE", "qwen_boxed"),
        ("OAT_ZERO_OPS_SNAPSHOT_ROOT", "/different/code"),
        ("SAVE_PATH", "/another/run"),
        ("VLLM_USE_V1", "1"),
        ("VLLM_ATTENTION_BACKEND", "FLASHINFER"),
        ("PYTHONPATH", "/another/package"),
        ("PYTHONHOME", "/another/python"),
        ("BASH_ENV", "/injected/script"),
        ("LD_PRELOAD", "/injected/library"),
        ("REMAX_LAUNCH_REQUEST", "/old/record"),
        ("HF_HUB_OFFLINE", "0"),
        ("NCCL_ALGO", "Tree"),
        ("MODEBENCH_PROMPT_CONDITION", "another"),
    ],
)
def test_inherited_configuration_fails(raw, key, value, tmp_path, monkeypatch):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="inherited setting"):
        run_recipe.environment(
            raw, data_root=tmp_path, model=tmp_path, output=tmp_path, seed=43
        )


def test_environment_excludes_unrelated_secrets_and_records_explicit_flags(
    raw, tmp_path, monkeypatch
):
    monkeypatch.setenv("PRIVATE_API_KEY", "should-not-be-saved")
    monkeypatch.setenv("PYTHON", sys.executable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-device")
    env = run_recipe.environment(
        raw, data_root=tmp_path, model=tmp_path, output=tmp_path, seed=44
    )
    assert "PRIVATE_API_KEY" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-device"
    assert env["OAT_ZERO_SEED"] == "44"
    assert env["VLLM_USE_V1"] == "0"
    assert env["OAT_ZERO_SAVE_CKPT"] == "0"


@pytest.fixture
def trusted_model(tmp_path, monkeypatch, raw):
    model = tmp_path / "model"
    model.mkdir()
    hashes = {}
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        path = model / name
        path.write_text("trusted " + name)
        hashes[name] = identity.digest(path)
    entry = {"id": raw["model_id"], "revision": raw["model_revision"], "sha256": hashes}
    monkeypatch.setattr(identity, "registry", lambda: {"model": entry})
    return model, Recipe.from_dict(raw)


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json", "model.safetensors"])
def test_model_bytes_must_match_release_registry(trusted_model, name):
    model, recipe = trusted_model
    assert (
        identity.authenticate_model(model, recipe)["revision"] == recipe.model_revision
    )
    (model / name).write_text("different contents")
    with pytest.raises(ValueError, match="model identity mismatch"):
        identity.authenticate_model(model, recipe)


def test_renaming_snapshot_or_adding_adapter_cannot_authenticate(trusted_model):
    model, recipe = trusted_model
    renamed = model.with_name(recipe.model_revision)
    model.rename(renamed)
    (renamed / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="unregistered model file"):
        identity.authenticate_model(renamed, recipe)


def test_missing_model_file_is_fatal(trusted_model):
    model, recipe = trusted_model
    (model / "config.json").unlink()
    with pytest.raises(ValueError, match="model identity mismatch"):
        identity.authenticate_model(model, recipe)


@pytest.fixture
def rows():
    return [
        {"problem": "prompt A", "answer": "reference A"},
        {"problem": "prompt B", "answer": "reference B"},
    ]


def row_record(rows, split="train"):
    return {
        "config_name": "level1_pantry_plan",
        "split": split,
        "rows": len(rows),
        "source_column_order": ["problem", "answer"],
        "materialized_rows_sha256": identity.rows_digest(rows),
    }


@pytest.mark.parametrize(
    "mutation", ["order", "prompt", "reference", "missing", "duplicate", "column"]
)
def test_dataset_semantics_order_and_columns_are_authenticated(rows, mutation):
    expected = row_record(rows)
    assert identity.authenticate_rows(rows, ["problem", "answer"], expected) == expected
    columns = ["problem", "answer"]
    if mutation == "order":
        rows.reverse()
    elif mutation == "prompt":
        rows[0]["problem"] = "prompt with answer hint"
    elif mutation == "reference":
        rows[0]["answer"] = "different reference"
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = rows[0]
    elif mutation == "column":
        columns.reverse()
    with pytest.raises(ValueError, match="dataset"):
        identity.authenticate_rows(rows, columns, expected)


def test_disk_manifest_cannot_reauthenticate_changed_dataset(
    raw, rows, tmp_path, monkeypatch
):
    class Dataset(list):
        column_names = ["problem", "answer"]

    class DatasetDict(dict):
        pass

    expected = [row_record(rows, s) for s in ("train", "eval")]
    rows[0]["problem"] = "tampered prompt"
    (tmp_path / "manifest.json").write_text(
        json.dumps({"rows_sha256": identity.rows_digest(rows)})
    )
    monkeypatch.setattr(identity, "registry", lambda: {"splits": expected})
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            DatasetDict=DatasetDict,
            load_from_disk=lambda _: DatasetDict(train=Dataset(rows)),
        ),
    )
    with pytest.raises(ValueError, match="dataset identity mismatch"):
        identity.authenticate_data(tmp_path, Recipe.from_dict(raw))


@pytest.mark.parametrize("subset", ["eval", "multi_answer", "another"])
def test_datasetdict_subset_must_match_split(raw, rows, tmp_path, monkeypatch, subset):
    class DatasetDict(dict):
        pass

    monkeypatch.setattr(identity, "registry", lambda: {"splits": [row_record(rows)]})
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            DatasetDict=DatasetDict,
            load_from_disk=lambda _: DatasetDict({subset: rows}),
        ),
    )
    with pytest.raises(ValueError, match="exactly DatasetDict subset train"):
        identity.authenticate_data(tmp_path, Recipe.from_dict(raw))


def test_authentication_failure_never_renders_or_starts_training(
    raw, tmp_path, monkeypatch
):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(raw))

    def fail(*args):
        raise ValueError("wrong dataset")

    monkeypatch.setattr(run_recipe, "authenticate_inputs", fail)
    monkeypatch.setattr(run_recipe, "validate_runtime", lambda: None)

    def forbidden(*args):
        pytest.fail("training command started before input validation")

    monkeypatch.setattr(run_recipe, "render", forbidden)
    output = tmp_path / "output"
    with pytest.raises(SystemExit):
        run_recipe.main(
            [
                str(recipe),
                "--data-root",
                str(tmp_path),
                "--model",
                str(tmp_path),
                "--output",
                str(output),
                "--execute",
            ]
        )
    assert not output.exists()


@pytest.fixture
def launch(raw, tmp_path, monkeypatch):
    from dataclasses import asdict

    settings = json_value(asdict(Recipe.from_dict(raw).settings))
    data = tmp_path / "data"
    data.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    source = tmp_path / "source.py"
    source.write_text("source")
    weights = model / "model.safetensors"
    weights.write_bytes(b"weights")
    request = {
        "command": [sys.executable, "-m", "remax.train_zero_math", "--seed", "44"],
        "typed_settings": settings,
        "seed": 44,
        "source_root": str(tmp_path),
        "source_sha256": {"source.py": identity.digest(source)},
        "inputs": {
            "model": {
                "path": str(model),
                "sha256": {"model.safetensors": identity.digest(weights)},
            },
            "data": {"path": str(data), "sha256": {}, "splits": {}},
            "prompt": {},
            "registry_sha256": "trusted",
        },
        "launch_environment": {"OMP_NUM_THREADS": "1"},
        "validate_only": True,
    }
    request["request_sha256"] = configuration_digest(request)
    path = tmp_path / "launch_request.json"
    write_json(path, request)
    monkeypatch.setenv("REMAX_LAUNCH_REQUEST", str(path))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(sys, "argv", ["train_zero_math.py", "--seed", "44"])
    values = {
        **settings,
        "seed": 44,
        "pretrain": str(model),
        "prompt_data": str(data / "train"),
        "eval_data": str(data / "eval"),
        "save_path": str(tmp_path),
        "unmentioned_default": 17,
        "gpus": 1,
        "asynchronous": False,
        "buffer_clear_every": 1,
        "dump_all_buffer": False,
        "resume_dir": None,
        "resume_tag": None,
    }
    args = make_dataclass("Args", [(k, object) for k in values])(**values)
    return args, tmp_path


def test_complete_record_contains_runtime_defaults_and_seed_override(launch):
    args, path = launch
    assert finalize_launch(args) is True
    record = strict_json((path / "effective_config.json").read_text())
    assert record["effective_arguments"]["seed"] == 44
    assert record["effective_arguments"]["unmentioned_default"] == 17
    assert record["effective_arguments_sha256"] == configuration_digest(
        record["effective_arguments"]
    )
    assert record["dependencies"]
    assert record["status"] == "validated"
    with pytest.raises(ValueError, match="overwrite"):
        finalize_launch(args)


@pytest.mark.parametrize(
    "mutation",
    [
        "arguments",
        "defaults",
        "source",
        "model",
        "request",
        "environment",
        "adapter",
        "extra_data",
    ],
)
def test_runtime_boundary_rejects_drift_before_training(launch, mutation, monkeypatch):
    args, path = launch
    if mutation == "arguments":
        monkeypatch.setattr(sys, "argv", ["train_zero_math.py", "--seed", "43"])
    elif mutation == "defaults":
        args.online_canonical_replay_compute_only = True
    elif mutation == "source":
        (path / "source.py").write_text("changed source")
    elif mutation == "model":
        (path / "model/model.safetensors").write_bytes(b"changed model")
    elif mutation == "adapter":
        (path / "model/adapter_config.json").write_text("{}")
    elif mutation == "extra_data":
        (path / "data/extra.arrow").write_bytes(b"new data")
    elif mutation == "environment":
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
    elif mutation == "request":
        p = path / "launch_request.json"
        r = json.loads(p.read_text())
        r["seed"] = 43
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        finalize_launch(args)
    assert not (path / "effective_config.json").exists()


def test_runtime_check_is_opt_in_for_legacy_entrypoint(monkeypatch):
    monkeypatch.delenv("REMAX_LAUNCH_REQUEST", raising=False)
    assert finalize_launch(object()) is False


def test_execution_records_configuration_before_returning_to_training(launch):
    args, path = launch
    p = path / "launch_request.json"
    request = strict_json(p.read_text())
    request["validate_only"] = False
    request["request_sha256"] = configuration_digest(
        {k: v for k, v in request.items() if k != "request_sha256"}
    )
    write_json(p, request)
    assert finalize_launch(args) is False
    assert (path / "effective_config.json").is_file()


def test_runtime_pin_rejects_dependency_drift(monkeypatch):
    import remax.launch_record as records
    import platform

    monkeypatch.setattr(sys, "version_info", (3, 10, 19))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(records.metadata, "version", lambda _: "different-version")
    with pytest.raises(ValueError, match="incompatible torch"):
        records.validate_runtime()


@pytest.mark.parametrize(
    "change", [None, "identity", "corruption", "selector", "implicit"]
)
def test_explicit_resume_is_checked_at_preworker_boundary(launch, change):
    from remax.checkpointing import commit_checkpoint

    args, path = launch
    assert finalize_launch(args)
    effective = path / "effective_config.json"
    identity_record = strict_json(effective.read_text())["resume_identity"]
    effective.unlink()
    if change == "identity":
        identity_record["arguments"]["num_prompt_epoch"] += 1
    checkpoint = commit_checkpoint(
        path / "checkpoints",
        step=1,
        identity=identity_record,
        keep=1,
        writer=lambda folder: (folder / "state.bin").write_bytes(b"state"),
    )
    args.resume_dir = str(checkpoint.parent)
    args.resume_tag = checkpoint.name
    p = path / "launch_request.json"
    request = strict_json(p.read_text())
    if change != "implicit":
        request["resume_checkpoint"] = str(checkpoint)
    request["request_sha256"] = configuration_digest(
        {k: v for k, v in request.items() if k != "request_sha256"}
    )
    write_json(p, request)
    if change == "corruption":
        (checkpoint / "state.bin").write_bytes(b"changed")
    if change == "selector":
        args.resume_tag = "step_00002"
    if change is None:
        assert finalize_launch(args)
        assert strict_json(effective.read_text())["resume_identity"] == identity_record
    else:
        with pytest.raises(ValueError):
            finalize_launch(args)
        assert not effective.exists()


@pytest.mark.parametrize(
    "field,value",
    [("gpus", 2), ("asynchronous", True), ("dump_all_buffer", True)],
)
def test_unsupported_resume_layout_fails_before_workers(launch, field, value):
    args, path = launch
    setattr(args, field, value)
    with pytest.raises(ValueError, match="requires"):
        finalize_launch(args)
    assert not (path / "effective_config.json").exists()
