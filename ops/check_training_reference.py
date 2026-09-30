"""Check frozen training identities, optionally against the pull request base.

Existing fixture bytes may not be changed or removed in a refactor. New
versioned cases may be added. This is separate from refreshable release hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "tests/fixtures/"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def check(root=ROOT, base_ref=None):
    versions = sorted((root / PREFIX).glob("training_v*/manifest.json"))
    if not versions:
        raise ValueError("no frozen training references found")
    checked = 0
    for manifest_path in versions:
        manifest = json.loads(manifest_path.read_text())
        if set(manifest["files"]) != {"inputs.json", "expected.json"}:
            raise ValueError(f"incomplete training reference: {manifest_path}")
        for name, expected in manifest["files"].items():
            path = manifest_path.parent / name
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(f"changed frozen training reference: {path}")
            checked += 1
    for path in sorted(
        [
            *(root / PREFIX).glob("benchmark_boundary_v*.json"),
            *(root / PREFIX).glob("gpu_historical_*_v*.json"),
        ]
    ):
        lock = path.with_suffix(".sha256")
        if (
            not lock.is_file()
            or lock.read_text().strip() != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError(f"changed frozen benchmark boundary: {path}")
        checked += 1
    preserved = 0
    if base_ref:
        # Resolve the ref first: an invalid/missing base must never look like an
        # empty initial reference set. CI fetches the full base history.
        commit = (
            git(root, "rev-parse", "--verify", f"{base_ref}^{{commit}}")
            .decode()
            .strip()
        )
        paths = (
            git(root, "ls-tree", "-r", "--name-only", commit, "--", PREFIX, "src/remax")
            .decode()
            .splitlines()
        )
        for relative in paths:
            parts = Path(relative).parts
            input_registry = relative.startswith(
                "src/remax/input_registry_v"
            ) and relative.endswith(".json")
            if not input_registry and (
                len(parts) < 3
                or not parts[2].startswith(
                    ("training_v", "benchmark_boundary_v", "gpu_historical_")
                )
            ):
                continue
            original = git(root, "show", f"{commit}:{relative}")
            path = root / relative
            if not path.is_file() or path.read_bytes() != original:
                raise ValueError(
                    f"PR changes frozen training reference {relative}; preserve it and add a versioned case"
                )
            preserved += 1
    return {"verified_reference_files": checked, "preserved_base_files": preserved}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        help="PR base commit; existing references must remain byte-identical",
    )
    args = parser.parse_args()
    print(json.dumps(check(base_ref=args.base_ref)))
