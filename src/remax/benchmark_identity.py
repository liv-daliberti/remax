"""Authenticate the qualified ModeBench runtime independently of install transport."""

from importlib import metadata
import hashlib
import json
from pathlib import Path


def validate_modebench_release(*, package_root=None):
    expected = json.loads(
        Path(__file__).with_name("modebench_release_v1.json").read_text()
    )
    if metadata.version("modebench") != expected["version"]:
        raise ValueError("ModeBench version differs from the qualified package release")
    root = (
        Path(package_root)
        if package_root is not None
        else Path(metadata.distribution("modebench").locate_file("modebench"))
    )
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".typed"}
    }
    if files != expected["files"]:
        changed = sorted(
            k
            for k in files.keys() | expected["files"].keys()
            if files.get(k) != expected["files"].get(k)
        )
        raise ValueError(
            f"ModeBench package content differs from the qualified release: {changed}"
        )
    return {"version": expected["version"], "verified_files": len(files)}
