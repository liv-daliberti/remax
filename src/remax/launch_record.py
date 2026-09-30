"""Write resolved training arguments before any actor, model or learner is built."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from .input_identity import digest, validate_model_layout
from .recipes import strict_json


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("NaN in effective configuration")
        return "inf" if value > 0 else "-inf"
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    encoded = (
        json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def configuration_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def finalize_launch(args) -> bool:
    """Return True for a requested validation-only launch; otherwise continue training.

    Legacy direct invocations without a launch request retain their existing behavior.
    """
    request_path = os.environ.get("REMAX_LAUNCH_REQUEST")
    if not request_path:
        return False
    path = Path(request_path)
    request = strict_json(path.read_text())
    if request["request_sha256"] != configuration_digest(
        {k: v for k, v in request.items() if k != "request_sha256"}
    ):
        raise ValueError("launch request changed before training")
    expected_argv = request["command"][3:]  # python -m remax.train_zero_math
    if sys.argv[1:] != expected_argv:
        raise ValueError("training command differs from authenticated launch request")
    for name, expected in request["source_sha256"].items():
        if digest(Path(request["source_root"]) / name) != expected:
            raise ValueError(f"launch source changed: {name}")
    # Recheck local bytes at the runtime boundary: do not trust a receipt alone.
    model = request["inputs"]["model"]
    validate_model_layout(Path(model["path"]), model["sha256"])
    data = request["inputs"]["data"]
    data_root = Path(data["path"])
    actual_files = {
        str(p.relative_to(data_root)) for p in data_root.rglob("*") if p.is_file()
    }
    if actual_files != set(data["sha256"]):
        raise ValueError("dataset file inventory changed before training")
    for identity in (request["inputs"]["model"], request["inputs"]["data"]):
        for name, expected in identity["sha256"].items():
            if digest(Path(identity["path"]) / name) != expected:
                raise ValueError(f"authenticated input changed before training: {name}")
    for key, value in request["launch_environment"].items():
        # The shell prepends the Python libdir; package imports use the installation.
        if key == "PATH":
            value = str(Path(sys.executable).parent) + os.pathsep + value
        if (
            key not in ("LD_LIBRARY_PATH",)
            and os.environ.get(key) != value
        ):
            raise ValueError(f"inherited setting changed before training: {key}")
    effective = json_value(args)
    metadata_only = {"auto_resume", "eval_prompt_interval", "max_prompt_epochs"}
    expected_settings = {**request["typed_settings"], "seed": request["seed"]}
    for name, expected in expected_settings.items():
        if name not in metadata_only and effective.get(name) != expected:
            raise ValueError(
                f"resolved setting differs from recipe: {name}: {effective.get(name)!r} != {expected!r}"
            )
    for name, expected in {
        "pretrain": request["inputs"]["model"]["path"],
        "prompt_data": str(Path(request["inputs"]["data"]["path"]) / "train"),
        "eval_data": str(Path(request["inputs"]["data"]["path"]) / "eval"),
        "save_path": str(path.parent),
    }.items():
        if effective.get(name) != expected:
            raise ValueError(
                f"resolved input path differs from authenticated request: {name}"
            )
    record = {
        **request,
        "schema": "remax-effective-configuration-v1",
        "status": "validated",
        "effective_arguments": effective,
        "effective_arguments_sha256": configuration_digest(effective),
        "python": sys.version,
        "dependencies": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in metadata.distributions()
                if d.metadata["Name"]
            )
        ),
        "runtime_environment": {
            k: v
            for k, v in os.environ.items()
            if k in request["launch_environment"]
            or k
            in (
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "REMAX_LAUNCH_REQUEST",
                "HF_HOME",
                "TRITON_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR",
            )
            or k.startswith(("VLLM_", "OAT_ZERO_"))
        },
    }
    from .checkpointing import (
        run_identity,
        validate_checkpoint,
        validate_resume_contract,
    )

    validate_resume_contract(args)
    import torch

    record["hardware"] = {
        "cuda": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
    }
    identity = run_identity(record)
    selected = request.get("resume_checkpoint")
    if selected:
        validate_checkpoint(Path(selected), identity)
        if (
            args.resume_dir != str(Path(selected).parent)
            or args.resume_tag != Path(selected).name
        ):
            raise ValueError(
                "resolved resume selector differs from authenticated checkpoint"
            )
    elif args.resume_dir or args.resume_tag:
        raise ValueError("resume requires an explicit authenticated checkpoint")
    record["resume_identity"] = identity
    # OAT passes this dataclass object to its worker processes. These private
    # operational attributes are not CLI-overridable scientific settings.
    args._remax_resume_identity = identity
    destination = path.parent / "effective_config.json"
    if destination.exists():
        raise ValueError("refusing to overwrite effective_config.json")
    write_json(destination, record)
    print(
        f"[recipe] validated complete effective configuration: {destination}",
        flush=True,
    )
    return request["validate_only"]


def validate_runtime() -> dict[str, str]:
    """The maintained training runtime is intentionally narrower than the core API."""
    import platform

    if (
        sys.version_info[:2] != (3, 10)
        or sys.platform != "linux"
        or platform.machine() != "x86_64"
        or platform.python_implementation() != "CPython"
    ):
        raise ValueError(
            "maintained training requires Linux x86_64 / Python 3.10; use ops/setup_gpu_environment.sh"
        )
    required = {
        "torch": "2.6.0",
        "transformers": "4.51.3",
        "vllm": "0.8.4",
        "oat-llm": "0.1.3.post1",
        "deepspeed": "0.16.8",
        "datasets": "2.16.1",
        "numpy": "1.26.4",
        "pyarrow": "11.0.0",
        "modebench": "0.4.0",
    }
    actual = {}
    for name, version in required.items():
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise ValueError(
                f"missing training dependency {name}; use ops/setup_gpu_environment.sh"
            ) from error
        if actual[name].split("+")[0] != version:
            raise ValueError(
                f"incompatible {name}: expected {version}, installed {actual[name]}"
            )
    from .benchmark_identity import validate_modebench_release

    validate_modebench_release()
    import torch

    if torch.version.cuda != "12.4":
        raise ValueError("maintained training requires the PyTorch CUDA 12.4 runtime")
    return actual
