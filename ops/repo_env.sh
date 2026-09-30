#!/usr/bin/env bash
# Portable cache and source configuration. CUDA/toolchain selection belongs to the environment.
_remax_root="${XDG_CACHE_HOME:-$HOME/.cache}/remax"
export HF_HOME="${HF_HOME:-$_remax_root/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$_remax_root/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$_remax_root/torch_extensions}"
unset _remax_root
