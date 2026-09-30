"""Authenticate local inputs against release-owned identities, without network access."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
from typing import Iterable, Mapping, Any

from .recipes import Recipe


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def registry() -> dict:
    return json.loads(files("remax").joinpath("input_registry_v1.json").read_text())


def rows_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    """Canonical ordered materialized rows (compact sorted JSON; no trailing newline)."""
    h = hashlib.sha256()
    for index, row in enumerate(rows):
        if index:
            h.update(b"\n")
        h.update(
            json.dumps(
                dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
    return h.hexdigest()


def validate_model_layout(model: Path, expected_names) -> None:
    allowed = set(expected_names) | {"README.md", "LICENSE", ".gitattributes"}
    # Extra tokenizer, adapter, configuration or weight files can change loading.
    for path in model.rglob("*"):
        relative = path.relative_to(model)
        if relative.parts[:2] == (".cache", "huggingface"):
            continue  # Hugging Face local-dir download receipts, not model inputs.
        if path.is_file() and str(relative) not in allowed:
            raise ValueError(f"unregistered model file: {relative}")


def authenticate_model(model: Path, recipe: Recipe) -> dict:
    expected = registry()["model"]
    if (recipe.model_id, recipe.model_revision) != (
        expected["id"],
        expected["revision"],
    ):
        raise ValueError("model is absent from the trusted input registry")
    validate_model_layout(model, expected["sha256"])
    for name, checksum in expected["sha256"].items():
        path = model / name
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"model identity mismatch: {name}")
    return {
        "path": str(model),
        "id": expected["id"],
        "revision": expected["revision"],
        "sha256": expected["sha256"],
    }


def authenticate_rows(rows, columns, expected: dict) -> dict:
    if list(columns) != expected["source_column_order"]:
        raise ValueError(
            f'dataset columns mismatch: {expected["config_name"]}/{expected["split"]}'
        )
    if (
        len(rows) != expected["rows"]
        or rows_digest(rows) != expected["materialized_rows_sha256"]
    ):
        raise ValueError(
            f'dataset identity mismatch: {expected["config_name"]}/{expected["split"]}'
        )
    return dict(expected)


def authenticate_data(data_root: Path, recipe: Recipe) -> dict:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as error:
        raise ValueError(
            "input authentication requires the training environment (datasets)"
        ) from error
    config = f"level{recipe.level}_{recipe.domain}"
    identities = {}
    for split, subset in [("train", "train"), ("eval", "multi_answer")]:
        matches = [
            s
            for s in registry()["splits"]
            if s["config_name"] == config and s["split"] == split
        ]
        if len(matches) != 1:
            raise ValueError(f"unsupported frozen split: {config}/{split}")
        loaded = load_from_disk(str(data_root / split))
        if not isinstance(loaded, DatasetDict) or set(loaded) != {subset}:
            raise ValueError(
                f"{split} must contain exactly DatasetDict subset {subset}"
            )
        identities[split] = authenticate_rows(
            loaded[subset], loaded[subset].column_names, matches[0]
        )
    return {
        "path": str(data_root),
        "splits": identities,
        "sha256": {
            str(p.relative_to(data_root)): digest(p)
            for p in sorted(data_root.rglob("*"))
            if p.is_file()
        },
    }


def authenticate_inputs(data_root: Path, model: Path, recipe: Recipe) -> dict:
    return {
        "registry_sha256": hashlib.sha256(
            files("remax").joinpath("input_registry_v1.json").read_bytes()
        ).hexdigest(),
        "data": authenticate_data(data_root, recipe),
        "model": authenticate_model(model, recipe),
        "prompt": {
            "level": recipe.level,
            "domain": recipe.domain,
            "template": recipe.settings.prompt_template,
            "canonical_action_task": recipe.settings.canonical_action_task,
            "input_key": recipe.settings.input_key,
            "output_key": recipe.settings.output_key,
        },
    }
