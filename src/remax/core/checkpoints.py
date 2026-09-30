"""Committed, identity-bound optimizer-boundary checkpoints (single learner rank)."""

from __future__ import annotations

import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Callable

from ..input_identity import digest
from ..launch_record import configuration_digest, write_json
from ..recipes import strict_json

PROTOCOL = "remax-resume-v1"
MANIFEST = "resume_manifest.json"


def run_identity(record: dict) -> dict:
    """Bind scientific arguments, content identities, code, runtime and hardware.

    Only artifact locations and explicit restore selectors may change. A resumed
    run retains the original total training/scheduler horizon.
    """
    args = dict(record["effective_arguments"])
    for key in ("save_path", "resume_dir", "resume_tag"):
        args.pop(key, None)
    for key in (
        "pretrain",
        "ref_pretrain",
        "critic_pretrain",
        "prompt_data",
        "eval_data",
    ):
        if key in args:
            args[key] = "<authenticated-input>"
    inputs = record["inputs"]
    return {
        "protocol": PROTOCOL,
        "arguments": args,
        "model": {k: v for k, v in inputs["model"].items() if k != "path"},
        "data": inputs["data"]["splits"],
        "prompt": inputs["prompt"],
        "registry_sha256": inputs["registry_sha256"],
        "source_sha256": record["source_sha256"],
        "dependencies": record["dependencies"],
        "hardware": record.get("hardware", {}),
    }


def validate_resume_contract(args) -> None:
    expected = {
        "gpus": 1,
        "collocate": True,
        "asynchronous": False,
        "rollout_batch_size": 1,
        "rollout_batch_size_per_device": 1,
        "num_ppo_epochs": 1,
        "sync_params_every": 1,
        "dump_all_buffer": False,
    }
    for name, value in expected.items():
        if getattr(args, name, None) != value:
            raise ValueError(f"{PROTOCOL} requires {name}={value!r}")
    if args.train_batch_size != args.num_samples:
        raise ValueError(
            "resume requires one complete sample group per optimizer boundary"
        )
    if args.save_ckpt and args.max_resume_num < 1:
        raise ValueError("resume checkpoint retention must be positive")


def capture_rng() -> dict:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    import numpy as np
    import torch

    if not isinstance(state, dict) or set(state) != {
        "python",
        "numpy",
        "torch",
        "cuda",
    }:
        raise ValueError("checkpoint missing complete random-number state")
    if len(state["cuda"]) != (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    ):
        raise ValueError("checkpoint CUDA random-state topology mismatch")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def prepare_prompt_iterator(loader, *, seed: int, epoch: int):
    """DataLoader iterator creation must not consume the policy's global RNG."""
    import torch

    loader.generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    return iter(loader)


def _files(folder: Path) -> dict[str, str]:
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint cannot contain symlinks")
        if path.is_file() and path.relative_to(folder).as_posix() != MANIFEST:
            result[str(path.relative_to(folder))] = digest(path)
    return result


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_checkpoint(folder: Path, expected_identity: dict | None = None) -> dict:
    """Validate JSON and every file before any pickle or model state is loaded."""
    folder = Path(folder)
    if folder.is_symlink() or not re.fullmatch(r"step_[0-9]+", folder.name):
        raise ValueError("resume requires an explicit committed step directory")
    try:
        manifest = strict_json((folder / MANIFEST).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(
            "checkpoint is incomplete or has no commit manifest"
        ) from error
    if (
        set(manifest) != {"schema", "step", "identity", "identity_sha256", "files"}
        or manifest["schema"] != PROTOCOL
    ):
        raise ValueError("unsupported checkpoint commit manifest")
    if (
        type(manifest["step"]) is not int
        or manifest["step"] < 1
        or manifest["step"] != int(folder.name[5:])
    ):
        raise ValueError("checkpoint step identity mismatch")
    if manifest["identity_sha256"] != configuration_digest(manifest["identity"]):
        raise ValueError("checkpoint identity digest mismatch")
    if expected_identity is not None and manifest["identity"] != expected_identity:
        raise ValueError(
            "incompatible resume: configuration, inputs, code, runtime or hardware differ"
        )
    if not manifest["files"] or _files(folder) != manifest["files"]:
        raise ValueError("checkpoint file inventory or hash mismatch")
    return manifest


def commit_checkpoint(
    root: Path, *, step: int, identity: dict, writer: Callable[[Path], None], keep: int
) -> Path:
    """Stage, flush, commit by same-filesystem rename, then publish latest/prune.

    Writer failure cannot modify the previous checkpoint or latest pointer.
    SIGKILL may leave an ignored .pending directory, never a loadable checkpoint.
    """
    root = Path(root)
    if type(step) is not int or step < 1 or keep < 1:
        raise ValueError("checkpoint step and retention must be positive")
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step_{step:05d}"
    if target.exists():
        raise ValueError("refusing to overwrite a committed checkpoint")
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    try:
        writer(staging)
        files = _files(staging)
        if not files:
            raise ValueError("checkpoint writer produced no state")
        for name in files:
            with (staging / name).open("rb") as stream:
                os.fsync(stream.fileno())
        manifest = {
            "schema": PROTOCOL,
            "step": step,
            "identity": identity,
            "identity_sha256": configuration_digest(identity),
            "files": files,
        }
        write_json(staging / MANIFEST, manifest)
        # Flush nested directory entries before committing the root entry.
        for directory in sorted(
            (p for p in staging.rglob("*") if p.is_dir()), reverse=True
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        os.rename(staging, target)
        _fsync_directory(root)
        fd, temporary = tempfile.mkstemp(prefix=".latest-", dir=root)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(target.name + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, root / "latest")
            _fsync_directory(root)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        committed = sorted(
            (
                p
                for p in root.glob("step_*")
                if p.is_dir() and not p.is_symlink() and (p / MANIFEST).is_file()
            ),
            key=lambda p: int(p.name[5:]),
            reverse=True,
        )
        for obsolete in committed[keep:]:
            shutil.rmtree(obsolete)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_client_state(
    state: dict, *, identity: dict, step: int, batches_per_epoch: int, epochs: int
) -> None:
    required = {
        "steps",
        "global_step",
        "policy_sgd_step",
        "query_step",
        "prompt_consumed",
        "prompt_epoch",
        "prompt_batches_consumed_total",
        "update_interval",
        "online_canonical_bank_state",
        "resume_protocol",
        "resume_identity_sha256",
        "rng_state",
        "last_evaluated_global_step",
        "engine_micro_steps",
        "rollout_buffer",
    }
    if not isinstance(state, dict) or not required <= set(state):
        raise ValueError("checkpoint missing required full-run state")
    if state["resume_protocol"] != PROTOCOL or state[
        "resume_identity_sha256"
    ] != configuration_digest(identity):
        raise ValueError("checkpoint client-state identity mismatch")
    for key in (
        "steps",
        "global_step",
        "query_step",
        "prompt_consumed",
        "prompt_epoch",
        "prompt_batches_consumed_total",
        "update_interval",
        "engine_micro_steps",
    ):
        if type(state[key]) is not int or state[key] < 0:
            raise ValueError(f"invalid checkpoint counter: {key}")
    if (
        state["steps"] != step
        or state["prompt_batches_consumed_total"] != step
        or state["global_step"] != step
        or state["policy_sgd_step"] != step
        or state["update_interval"] != 1
    ):
        raise ValueError("checkpoint is not a complete optimizer boundary")
    if (
        not 0 < step <= batches_per_epoch * epochs
        or state["prompt_epoch"] != step // batches_per_epoch
    ):
        raise ValueError("checkpoint data cursor or training horizon mismatch")
    if not isinstance(state["rollout_buffer"], list):
        raise ValueError("checkpoint rollout buffer is malformed")
    settings = identity.get("arguments", {})
    if "num_samples" in settings:
        if (
            state["query_step"] != step * settings["num_samples"]
            or state["prompt_consumed"] != step * settings["num_samples"]
        ):
            raise ValueError(
                "checkpoint query/sample counters do not match data position"
            )
        micro_steps = (
            step * settings["num_samples"] // settings["train_batch_size_per_device"]
        )
        if state["engine_micro_steps"] != micro_steps:
            raise ValueError("checkpoint accumulation cursor mismatch")
    last_eval = state["last_evaluated_global_step"]
    if last_eval is not None and (
        type(last_eval) is not int or not 0 <= last_eval <= step
    ):
        raise ValueError("invalid checkpoint evaluation cursor")
    if not isinstance(state["rng_state"], dict) or set(state["rng_state"]) != {
        "python",
        "numpy",
        "torch",
        "cuda",
    }:
        raise ValueError("checkpoint missing random-number state")
