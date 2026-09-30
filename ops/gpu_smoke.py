"""Prepare, run, and audit a four-method GPU smoke test using production training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from remax.launcher import environment

ROOT = Path(__file__).resolve().parents[1]

METHODS = ("drgrpo", "redr", "maxrl", "remax")
# Method, optimization, sampling, prompt, and replay settings stay registered.
OVERRIDES = {
    "OAT_ZERO_MAX_TRAIN": "4",
    "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
    "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
    "OAT_ZERO_EVAL_STEPS": "2",
    "OAT_ZERO_EVAL_PROMPT_INTERVAL": "2",
    "OAT_ZERO_SAVE_STEPS": "2",
    "OAT_ZERO_SAVE_FROM": "2",
    "OAT_ZERO_RESUME_STEPS": "2",
    "OAT_ZERO_RESUME_FROM": "2",
    "OAT_ZERO_SAVE_CKPT": "1",
    "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS": "0",
    "OAT_ZERO_AUTO_RESUME": "0",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def recipe(method):
    return json.loads((ROOT / f"configs/{method}_pantry_plan_05b.json").read_text())


def prepare(args):
    from datasets import DatasetDict
    from huggingface_hub import HfApi, snapshot_download
    from modebench.data import load_split, split_record

    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (work / "inputs.json").exists() or (work / "data").exists():
        raise ValueError("Use a fresh work directory for preparation")
    registered = recipe("remax")
    model = args.model
    if model is None:
        model = Path(
            snapshot_download(
                registered["model_id"], revision=registered["model_revision"]
            )
        )
    model = model.resolve()
    # Verify cached and downloaded files against immutable Hub Git/LFS identities.
    upstream = HfApi().model_info(
        registered["model_id"],
        revision=registered["model_revision"],
        files_metadata=True,
    )
    identities = {}
    for entry in upstream.siblings:
        if entry.rfilename.endswith((".json", ".safetensors", ".txt")):
            path = model / entry.rfilename
            actual = digest(path)
            if entry.lfs:
                if actual != entry.lfs.sha256:
                    raise ValueError(f"Model LFS hash mismatch: {entry.rfilename}")
            else:
                content = path.read_bytes()
                if (
                    hashlib.sha1(
                        b"blob " + str(len(content)).encode() + b"\0" + content
                    ).hexdigest()
                    != entry.blob_id
                ):
                    raise ValueError(f"Model Git hash mismatch: {entry.rfilename}")
            identities[entry.rfilename] = actual
    splits = {}
    for split, subset, count in [("train", "train", 4), ("eval", "multi_answer", 2)]:
        source = load_split(
            args.modebench_data, "level1_pantry_plan", split, frozen=True
        )
        DatasetDict({subset: source.select(range(count))}).save_to_disk(
            str(work / "data" / split)
        )
        splits[split] = {
            "source": split_record(
                args.modebench_data, "level1_pantry_plan", split, frozen=True
            ),
            "row_indices": list(range(count)),
        }
    write_json(
        work / "inputs.json",
        {
            "schema": "remax-gpu-smoke-inputs-v1",
            "model": str(model),
            "model_id": registered["model_id"],
            "model_revision": upstream.sha,
            "model_sha256": identities,
            "splits": splits,
            "data_sha256": {
                str(p.relative_to(work)): digest(p)
                for p in sorted((work / "data").rglob("*"))
                if p.is_file()
            },
        },
    )
    print(
        f"Prepared {work}: four train rows, two evaluation rows; verified model and frozen data."
    )


def validate_inputs(work):
    inputs = json.loads((work / "inputs.json").read_text())
    for name, expected in inputs["model_sha256"].items():
        if digest(Path(inputs["model"]) / name) != expected:
            raise ValueError(f"Model changed since preparation: {name}")
    for name, expected in inputs["data_sha256"].items():
        if digest(work / name) != expected:
            raise ValueError(f"Dataset changed since preparation: {name}")
    return inputs


def smoke_environment(method, work, inputs, *, restore=None):
    registered = recipe(method)
    env = environment(
        registered,
        data_root=work / "data",
        model=Path(inputs["model"]),
        output=work / (method if restore is None else method + "-restored"),
        seed=43,
    )
    env.update(OVERRIDES)
    env.update(
        {
            "OAT_ZERO_DRY_RUN": "0",
            "OAT_ZERO_FIXED_EXP_SUFFIX": "gpu-smoke",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if restore is not None:
        env.update(
            {
                "OAT_ZERO_RESUME_DIR": str(restore),
                "OAT_ZERO_RESUME_TAG": (restore / "latest").read_text().strip(),
                "OAT_ZERO_EVAL_ONLY": "1",
            }
        )
    return env


def launch(method, work, inputs, *, restore=None):
    name = method if restore is None else method + "-restored"
    if (work / name).exists() or (work / f"{name}-launch.json").exists():
        raise ValueError(
            f"Refusing to overwrite {work / name}; choose a fresh work directory"
        )
    env = smoke_environment(method, work, inputs, restore=restore)
    write_json(
        work / f"{name}-launch.json",
        {
            "recipe": recipe(method),
            "overrides": OVERRIDES,
            "effective_environment": {
                k: v
                for k, v in env.items()
                if k.startswith("OAT_ZERO_") or k in ("SAVE_PATH", "VLLM_USE_V1")
            },
            "source_sha256": {
                str(p.relative_to(Path(env["OAT_ZERO_REPO_ROOT"]))): digest(p)
                for p in sorted(Path(env["OAT_ZERO_REPO_ROOT"]).rglob("*"))
                if p.suffix in (".py", ".sh")
            },
        },
    )
    start = time.time()
    print(f'Running {name}; live log: {work / (name + ".log")}', flush=True)
    import torch

    gpu_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix(
        "GPU-"
    )
    with (
        (work / f"{name}.log").open("w") as log,
        (work / f"{name}-gpu.csv").open("w") as samples,
    ):
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--id=" + gpu_uuid,
                "--query-gpu=timestamp,memory.used,utilization.gpu",
                "--format=csv,nounits",
                "-l",
                "1",
            ],
            stdout=samples,
            stderr=subprocess.STDOUT,
        )
        try:
            command = ["bash", str(Path(env["OAT_ZERO_OPS_SNAPSHOT_ROOT"]) / "train.sh")]
            with subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as process:
                try:
                    returncode = process.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    returncode = 124
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            completed = subprocess.CompletedProcess(command, returncode)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    write_json(
        work / f"{name}-exit.json",
        {"returncode": completed.returncode, "elapsed_seconds": time.time() - start},
    )
    completed.check_returncode()


def run(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Run inside an allocation with an NVIDIA GPU")
    work = args.workdir.resolve()
    inputs = validate_inputs(work)
    for method in args.methods:
        if (work / method).exists() or (work / f"{method}-launch.json").exists():
            raise ValueError(
                f"Existing {method} attempt; prepare a fresh work directory"
            )
    write_json(
        work / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
            "venv_config": (Path(sys.prefix) / "pyvenv.cfg").read_text(),
        },
    )
    for method in args.methods:
        launch(method, work, inputs)
    if "remax" in args.methods:
        checkpoints = list((work / "remax").rglob("checkpoints"))
        if len(checkpoints) != 1:
            raise RuntimeError("Expected one Re:Max recovery checkpoint root")
        launch("remax", work, inputs, restore=checkpoints[0])
    print(
        f"Training finished. Run python ops/audit_gpu_smoke.py --workdir {work} to audit artifacts.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--modebench-data", type=Path, required=True)
    p.add_argument(
        "--model",
        type=Path,
        help="Existing pinned snapshot, verified against the Hub; otherwise download it",
    )
    p = sub.add_parser("run")
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
