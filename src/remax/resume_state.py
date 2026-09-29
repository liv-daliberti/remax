from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def _coerce_int(
    state: Mapping[str, Any] | None,
    key: str,
    default: int,
) -> int:
    if state is None or key not in state:
        return int(default)
    try:
        return int(state[key])
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(
    state: Mapping[str, Any] | None,
    key: str,
    default: float,
) -> float:
    if state is None or key not in state:
        return float(default)
    try:
        return float(state[key])
    except (TypeError, ValueError):
        return float(default)


def resolve_resume_progress_state(
    *,
    resume_states: Mapping[str, Any] | None,
    resume_step: int,
    update_interval: int,
    num_prompt_epochs: int,
    num_prompt_batches_per_epoch: int,
    rollout_batch_size: int,
) -> dict[str, int | float]:
    checkpoint_step = max(
        max(int(resume_step), 0),
        _coerce_int(resume_states, "steps", max(int(resume_step), 0)),
    )
    prompt_batches_consumed_total = max(
        checkpoint_step,
        _coerce_int(
            resume_states,
            "prompt_batches_consumed_total",
            checkpoint_step,
        ),
    )

    safe_num_prompt_epochs = max(int(num_prompt_epochs), 0)
    safe_batches_per_epoch = max(int(num_prompt_batches_per_epoch), 0)
    safe_rollout_batch_size = max(int(rollout_batch_size), 0)
    safe_update_interval = max(int(update_interval), 1)

    total_prompt_batches = safe_num_prompt_epochs * safe_batches_per_epoch
    if total_prompt_batches > 0:
        prompt_batches_consumed_total = min(
            prompt_batches_consumed_total,
            total_prompt_batches,
        )

    if safe_batches_per_epoch > 0:
        start_prompt_epoch, start_batch_offset = divmod(
            prompt_batches_consumed_total,
            safe_batches_per_epoch,
        )
    else:
        start_prompt_epoch, start_batch_offset = 0, 0

    default_completed_prompt_epochs = min(start_prompt_epoch, safe_num_prompt_epochs)
    prompt_epoch = min(
        _coerce_int(
            resume_states,
            "prompt_epoch",
            default_completed_prompt_epochs,
        ),
        default_completed_prompt_epochs,
    )

    default_query_step = prompt_batches_consumed_total * safe_rollout_batch_size
    default_global_step = checkpoint_step // safe_update_interval
    default_policy_sgd_step = float(default_global_step)

    return {
        "checkpoint_step": checkpoint_step,
        "next_step": max(1, checkpoint_step + 1),
        "prompt_batches_consumed_total": prompt_batches_consumed_total,
        "start_prompt_epoch": min(start_prompt_epoch, safe_num_prompt_epochs),
        "start_batch_offset": start_batch_offset,
        "prompt_epoch": max(prompt_epoch, 0),
        "query_step": max(
            _coerce_int(resume_states, "query_step", default_query_step),
            0,
        ),
        "prompt_consumed": max(
            _coerce_int(resume_states, "prompt_consumed", default_query_step),
            0,
        ),
        "global_step": max(
            _coerce_int(resume_states, "global_step", default_global_step),
            0,
        ),
        "policy_sgd_step": max(
            _coerce_float(
                resume_states,
                "policy_sgd_step",
                default_policy_sgd_step,
            ),
            0.0,
        ),
        "last_eval_query_step": max(
            _coerce_int(resume_states, "last_eval_query_step", 0),
            0,
        ),
    }


def discover_local_wandb_resume_run(
    *,
    wandb_run_roots: Sequence[str | os.PathLike[str]],
    resume_dir: str | None,
    resume_tag: str | None = None,
    saved_run_name: str | None = None,
    current_run_name: str | None = None,
) -> tuple[str | None, str | None]:
    checkpoint_dir_candidates: list[tuple[str, str]] = []

    def _register_checkpoint_dir(path: str | os.PathLike[str] | None) -> None:
        if not path:
            return
        normalized = os.path.normpath(str(path))
        checkpoint_path = Path(normalized)
        run_name = (
            checkpoint_path.parent.name
            if checkpoint_path.name == "checkpoints"
            else checkpoint_path.name
        )
        candidate = (normalized, run_name)
        if candidate not in checkpoint_dir_candidates:
            checkpoint_dir_candidates.append(candidate)

    if resume_dir:
        resume_root = Path(os.path.normpath(str(resume_dir)))
        _register_checkpoint_dir(resume_root)
        if resume_root.exists():
            _register_checkpoint_dir(resume_root.resolve())
            step_refs = []
            if resume_tag:
                step_refs.append(str(resume_tag))
            step_refs.append("latest")
            for step_ref in step_refs:
                step_path = resume_root / step_ref
                if not step_path.exists():
                    continue
                resolved_step_path = step_path.resolve()
                _register_checkpoint_dir(resolved_step_path.parent)

    candidate_run_names: list[str] = []
    for _, checkpoint_run_name in checkpoint_dir_candidates:
        if checkpoint_run_name not in candidate_run_names:
            candidate_run_names.append(checkpoint_run_name)
    for raw_name in (saved_run_name, current_run_name):
        if not raw_name:
            continue
        normalized_name = str(raw_name).strip()
        if normalized_name and normalized_name not in candidate_run_names:
            candidate_run_names.append(normalized_name)

    normalized_roots: list[Path] = []
    for root in wandb_run_roots:
        root_path = Path(root)
        if root_path.exists() and root_path not in normalized_roots:
            normalized_roots.append(root_path)

    discovered_logs: list[tuple[Path, str]] = []
    for root in normalized_roots:
        for output_log in sorted(root.glob("run-*/files/output.log"), reverse=True):
            try:
                log_text = output_log.read_text(errors="ignore")
            except OSError:
                continue
            discovered_logs.append((output_log, log_text))

    for checkpoint_dir, checkpoint_run_name in checkpoint_dir_candidates:
        checkpoint_save_needles = (
            f"Saving model checkpoint: {checkpoint_dir}/step_",
            f"zero checkpoint saved {checkpoint_dir}/step_",
            f"Saved {checkpoint_dir}/step_",
        )
        for output_log, log_text in discovered_logs:
            if any(needle in log_text for needle in checkpoint_save_needles):
                run_dir_name = output_log.parents[1].name
                _, _, run_id = run_dir_name.rpartition("-")
                if run_id:
                    return run_id, checkpoint_run_name

    for output_log, log_text in discovered_logs:
        for candidate_run_name in candidate_run_names:
            if candidate_run_name in log_text:
                run_dir_name = output_log.parents[1].name
                _, _, run_id = run_dir_name.rpartition("-")
                if run_id:
                    return run_id, candidate_run_name

    return None, None
