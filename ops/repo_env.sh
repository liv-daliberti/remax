#!/usr/bin/env bash
# Portable cache and source configuration. CUDA/toolchain selection belongs to the environment.
_remax_root="${OAT_ZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$_remax_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$_remax_root/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$_remax_root/.cache/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$_remax_root/.cache/torch_extensions}"
unset _remax_root
