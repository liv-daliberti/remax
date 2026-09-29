#!/usr/bin/env bash
# Tested Linux x86_64 / CPython 3.10 / PyTorch CUDA 12.4 environment.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${1:-$root/.venv}"
bootstrap_python="${PYTHON:-python3.10}"
"$bootstrap_python" -c 'import platform,sys; assert sys.version_info[:2] == (3,10) and platform.system() == "Linux" and platform.machine() == "x86_64" and platform.python_implementation() == "CPython", "GPU lock requires CPython 3.10 on Linux x86_64"'
if [[ -e "$env_dir" ]]; then
  echo "Use a new environment directory: $env_dir" >&2
  exit 1
fi
mkdir -p "$root/outputs/install-tmp"
export TMPDIR="${TMPDIR:-$root/outputs/install-tmp}"
"$bootstrap_python" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --no-cache-dir pip==25.2 setuptools==75.8.0 wheel==0.45.1
"$env_dir/bin/python" -m pip install --no-cache-dir -r "$root/requirements-gpu-py310-cu124.txt"
"$env_dir/bin/python" -m pip install --no-cache-dir --no-deps "$root"
"$env_dir/bin/python" -m pip check
printf 'Activate with: source %q/bin/activate\n' "$env_dir"
