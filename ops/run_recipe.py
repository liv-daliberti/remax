"""Validate a recipe and authenticate its inputs; --execute starts local training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from remax.input_identity import authenticate_inputs, digest
from remax.launch_record import (
    json_value,
    configuration_digest,
    write_json,
    validate_runtime,
)
from remax.recipes import Recipe, strict_json

ROOT = Path(__file__).resolve().parents[1]

# Infrastructure only. Arbitrary process variables never reach training.
INFRASTRUCTURE = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CXX",
    "CC",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
}
FIXED_ENVIRONMENT = {
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_NO_USAGE_STATS": "1",
    "TRANSFORMERS_NO_TF": "1",
    "USE_TF": "0",
    "USE_FLAX": "0",
    "OMP_NUM_THREADS": "1",
    "WANDB_MODE": "offline",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
}
SENSITIVE_PREFIXES = (
    "OAT_",
    "REMAX_",
    "VLLM_",
    "PYTHON",
    "TORCH_",
    "PYTORCH_",
    "NCCL_",
    "CUBLAS_",
    "CUDNN_",
    "HF_",
    "TRANSFORMERS_",
    "WANDB_",
    "TOKENIZERS_",
    "DS_",
    "DEEPSPEED_",
    "MODEBENCH_",
)


def environment(recipe, *, data_root, model, output, seed):
    """Strict environment builder shared by maintained and bounded smoke launchers."""
    typed = recipe if isinstance(recipe, Recipe) else Recipe.from_dict(recipe)
    typed.validate()
    seed = typed.choose_seed(seed)
    fixed = {**FIXED_ENVIRONMENT}
    # Leave V1 unset for noncanonical runs: vLLM performs its documented capability
    # selection. Inherited VLLM_USE_V1 is rejected there rather than forced.
    if typed.settings.canonical_action_task != "none":
        fixed["VLLM_USE_V1"] = "0"
    for key, value in os.environ.items():
        if (
            key == "PYTHON"
            and shutil.which(value)
            and Path(shutil.which(value)).absolute() == Path(sys.executable).absolute()
        ):
            continue  # Make's explicit interpreter selection agrees with this process.
        if key in INFRASTRUCTURE:
            continue
        if key in fixed and value == fixed[key]:
            continue
        if (
            key in fixed
            or key.startswith(SENSITIVE_PREFIXES)
            or key
            in (
                "SAVE_PATH",
                "BASH_ENV",
                "ENV",
                "LD_PRELOAD",
                "USE_TF",
                "USE_FLAX",
                "OMP_NUM_THREADS",
            )
        ):
            raise ValueError(
                f"inherited setting {key} conflicts with the strict launcher; unset it and put supported settings in the recipe"
            )
    env = {k: v for k, v in os.environ.items() if k in INFRASTRUCTURE}
    env.update(fixed)
    # Preserve historical string representations for the comparison/smoke helpers;
    # fill the four historical omissions and opt-in checkpoint flag explicitly.
    env.update(typed.settings.environment())
    if isinstance(recipe, dict):
        env.update(recipe["environment"])
    env.update(
        {
            "OAT_ZERO_REPO_ROOT": str(ROOT),
            "OAT_ZERO_PYTHON": sys.executable,
            "OAT_ZERO_SOURCE_ROOT": str(ROOT / "src"),
            "OAT_ZERO_PRETRAIN": str(model),
            "OAT_ZERO_PROMPT_DATA": str(data_root / "train"),
            "OAT_ZERO_EVAL_DATA": str(data_root / "eval"),
            "OAT_ZERO_SEED": str(seed),
            "SAVE_PATH": str(output),
            "OAT_ZERO_DRY_RUN": "1",
        }
    )
    return env


def render(env):
    result = subprocess.run(
        ["bash", str(ROOT / "ops/train.sh")],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    commands = [
        shlex.split(line.split(":", 1)[1])
        for line in result.stdout.splitlines()
        if line.startswith("[train] command:")
    ]
    if len(commands) != 1 or commands[0][:3] != [
        sys.executable,
        "-m",
        "remax.train_zero_math",
    ]:
        raise ValueError("launcher did not render the expected training entry point")
    return commands[0], result.stdout


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recipe", type=Path)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--model",
        type=Path,
        required=True,
        help="local snapshot authenticated against the pinned model registry",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--resume",
        type=Path,
        help="explicit committed checkpoint step directory; total recipe horizon must stay unchanged",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="resolve all runtime defaults and validate hardware; exit before building models/workers",
    )
    mode.add_argument(
        "--render-only",
        action="store_true",
        help="unverified command preview; no input authentication, runtime imports or launch record",
    )
    args = p.parse_args(argv)
    try:
        raw = strict_json(args.recipe.read_text())
        recipe = Recipe.from_dict(raw)
        seed = recipe.choose_seed(args.seed)
        data_root, model, output = (
            path.resolve() for path in (args.data_root, args.model, args.output)
        )
        env = environment(
            raw, data_root=data_root, model=model, output=output, seed=seed
        )
        if not args.render_only and output.exists():
            raise ValueError(
                "output already exists; choose a fresh directory (automatic resume is not supported)"
            )
        if args.execute or args.validate_only:
            validate_runtime()
        inputs = (
            None if args.render_only else authenticate_inputs(data_root, model, recipe)
        )
        selected_checkpoint = None
        if args.resume:
            if args.render_only:
                raise ValueError("resume cannot be combined with an unverified preview")
            from remax.checkpointing import validate_checkpoint

            if args.resume.is_symlink():
                raise ValueError(
                    "resume requires an explicit checkpoint directory, not a symlink"
                )
            selected_checkpoint = args.resume.resolve()
            validate_checkpoint(selected_checkpoint)
            env["OAT_ZERO_RESUME_DIR"] = str(selected_checkpoint.parent)
            env["OAT_ZERO_RESUME_TAG"] = selected_checkpoint.name
        command, rendered = render(env)
        if args.render_only:
            print(
                "[recipe] UNVERIFIED preview: model/dataset identities and runtime defaults have not been checked."
            )
            print(rendered, end="")
            return
        output.mkdir(parents=True, exist_ok=False)
        source_hashes = {
            str(path.relative_to(ROOT)): digest(path)
            for base in ("src/remax", "ops")
            for path in sorted((ROOT / base).rglob("*"))
            if path.suffix in (".py", ".sh", ".json")
        }
        # Dry-run is a preview switch only. The recorded launch environment is the
        # exact environment handed to the shell for validation or execution.
        env["OAT_ZERO_DRY_RUN"] = "0"
        request_path = output / "launch_request.json"
        env["REMAX_LAUNCH_REQUEST"] = str(request_path)
        request = {
            "schema": "remax-launch-request-v1",
            "status": "inputs_authenticated",
            "recipe": raw,
            "recipe_sha256": digest(args.recipe),
            "typed_settings": json_value(asdict(recipe.settings)),
            "seed": seed,
            "inputs": inputs,
            "command": command,
            "launch_environment": env,
            "source_root": str(ROOT),
            "source_sha256": source_hashes,
            "validate_only": args.validate_only,
            "resume_checkpoint": (
                str(selected_checkpoint) if selected_checkpoint else None
            ),
            "historical_metadata": {
                "auto_resume_requested": recipe.settings.auto_resume,
                "auto_resume_implemented": False,
            },
        }
        request["request_sha256"] = configuration_digest(request)
        write_json(request_path, request)
        print(
            f"[recipe] authenticated inputs; launch request: {request_path}", flush=True
        )
        if args.execute or args.validate_only:
            # The child writes effective_config.json after BOTH argument validators
            # and before run_zero_math_rl can initialize training.
            subprocess.run(["bash", str(ROOT / "ops/train.sh")], env=env, check=True)
        else:
            print(rendered, end="")
            print(
                "[recipe] Runtime defaults/hardware remain unchecked. Use --validate-only or --execute with a fresh output directory."
            )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
