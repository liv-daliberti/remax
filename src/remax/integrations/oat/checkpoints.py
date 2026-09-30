"""OAT checkpoints responsibilities, preserving historical state and ordering."""

from __future__ import annotations
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any
import torch.distributed as dist
from ...core.checkpoints import (
    PROTOCOL as RESUME_PROTOCOL,
    capture_rng,
    restore_rng,
    validate_client_state,
    commit_checkpoint,
)
from ...launch_record import configuration_digest
from ...resume_state import resolve_resume_progress_state


class OatCheckpointsMixin:
    def _checkpoint_client_state(self) -> dict[str, Any]:
        self._ensure_training_progress_state()
        client_state: dict[str, Any] = {
            "global_step": int(self.global_step),
            "policy_sgd_step": float(self.policy_sgd_step),
            "query_step": int(self.query_step),
            "prompt_consumed": int(self.prompt_consumed),
            "prompt_epoch": int(self.prompt_epoch),
            "steps": int(self.steps),
            "prompt_batches_consumed_total": int(self._prompt_batches_consumed_total),
            "update_interval": int(self.update_interval),
            "progress_metric_baselines": dict(self._progress_metric_baselines),
            "progress_metric_previous": dict(self._progress_metric_previous),
            "progress_metric_bests": dict(self._progress_metric_bests),
            "progress_metric_best_steps": dict(self._progress_metric_best_steps),
        }
        if self._actor_reward_ema is not None:
            client_state["actor_reward_ema"] = float(self._actor_reward_ema)
        if self._actor_reward_ema_start is not None:
            client_state["actor_reward_ema_start"] = float(self._actor_reward_ema_start)
        tau_controller = getattr(self, "_xdr_tau_controller", None)
        if tau_controller is not None:
            client_state["xdr_tau_controller_state"] = tau_controller.state_dict()
        maxent_controller = getattr(self, "_maxent_alpha_controller", None)
        if maxent_controller is not None:
            client_state["maxent_alpha_controller_state"] = (
                maxent_controller.state_dict()
            )
        length_controller = getattr(self, "_maxent_length_controller", None)
        if length_controller is not None:
            client_state["maxent_length_controller_state"] = (
                length_controller.state_dict()
            )
        diayn_tracker = getattr(self, "_diayn_mi_tracker", None)
        if diayn_tracker is not None:
            client_state["diayn_mi_tracker_state"] = diayn_tracker.state_dict()
        semantic_shannon_tracker = getattr(self, "_semantic_shannon_tracker", None)
        if semantic_shannon_tracker is not None:
            client_state["semantic_shannon_tracker_state"] = (
                semantic_shannon_tracker.state_dict()
            )
        semantic_rms_controller = getattr(self, "_semantic_rms_controller", None)
        if semantic_rms_controller is not None:
            # The adapted coefficient is run state, not a hyperparameter: a
            # resume that restarted it at the base dose would silently rerun
            # the controller's warmup partway through training.
            client_state["semantic_rms_controller_state"] = (
                semantic_rms_controller.state_dict()
            )
        online_canonical_bank = getattr(self, "_online_canonical_bank", None)
        if online_canonical_bank is not None:
            client_state["online_canonical_bank_state"] = (
                online_canonical_bank.state_dict()
            )
        rlep_online_pool = getattr(self, "_rlep_experience_pool", None)
        if rlep_online_pool is not None:
            from ...experiments.rlep import OnlineRLEPExperiencePool

            if isinstance(rlep_online_pool, OnlineRLEPExperiencePool):
                # Historical frequency replay retains its own pool schema.
                client_state["rlep_online_pool_state"] = rlep_online_pool.state_dict()
        proposal_starvation_controller = getattr(
            self,
            "_proposal_starvation_controller",
            None,
        )
        if proposal_starvation_controller is not None:
            client_state["proposal_starvation_controller_state"] = (
                proposal_starvation_controller.state_dict()
            )
        verified_route_library = getattr(
            self,
            "_verified_route_library",
            None,
        )
        if verified_route_library is not None:
            from ...verified_route_library import VerifiedRouteLibrary

            if not isinstance(verified_route_library, VerifiedRouteLibrary):
                raise RuntimeError("invalid verified route library")
            client_state["verified_route_library_state"] = (
                verified_route_library.state_dict()
            )
        math_strategy_canonicalizer = getattr(
            self, "_math_strategy_canonicalizer", None
        )
        if math_strategy_canonicalizer is not None:
            client_state["math_strategy_canonicalizer_state"] = (
                math_strategy_canonicalizer.state_dict()
            )
        online_canonical_controller = getattr(
            self, "_online_canonical_alpha_controller", None
        )
        if online_canonical_controller is not None:
            client_state["online_canonical_alpha_controller_state"] = (
                online_canonical_controller.state_dict()
            )
        if hasattr(self, "last_eval_query_step"):
            client_state["last_eval_query_step"] = int(self.last_eval_query_step)
        if hasattr(self, "_pending_eval"):
            client_state["_pending_eval"] = bool(self._pending_eval)
        if self._wandb_run_id:
            client_state["wandb_run_id"] = self._wandb_run_id
        if self._wandb_run_name:
            client_state["wandb_run_name"] = self._wandb_run_name
        identity = getattr(self.args, "_remax_resume_identity", None)
        if identity is not None:
            client_state.update(
                {
                    "resume_protocol": RESUME_PROTOCOL,
                    "resume_identity_sha256": configuration_digest(identity),
                    "rng_state": capture_rng(),
                    "engine_micro_steps": int(self.model.model.micro_steps),
                    "rollout_buffer": list(self.pi_buffer),
                    "last_evaluated_global_step": getattr(
                        self, "_last_evaluated_global_step", None
                    ),
                    # Saving follows a consumed batch, before the outer loop can
                    # advance the epoch counter. Store the logical next position.
                    "prompt_epoch": int(self._prompt_batches_consumed_total)
                    // len(self.prompts_dataloader),
                }
            )
        return client_state

    def _infer_resume_step(self, resume_states: dict[str, Any] | None) -> int:
        """Recover the loaded checkpoint's step without restarting its counters."""
        saved_step = None
        if isinstance(resume_states, dict) and "steps" in resume_states:
            raw_step = resume_states["steps"]
            try:
                saved_step = int(raw_step)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("Checkpoint has invalid client-state steps") from exc
            if (
                isinstance(raw_step, bool)
                or saved_step < 0
                or str(raw_step) not in (str(saved_step), str(float(saved_step)))
            ):
                raise RuntimeError("Checkpoint has invalid client-state steps")

        tag = getattr(self.args, "resume_tag", None)
        if not tag and getattr(self.args, "resume_dir", None):
            latest = Path(self.args.resume_dir) / "latest"
            if latest.is_symlink():
                tag = latest.resolve().name
            elif latest.is_file():
                tag = latest.read_text(encoding="utf-8").strip()
        tag_step = None
        if tag:
            tag_name = Path(str(tag)).name
            if tag_name.startswith("step_") and tag_name[5:].isdigit():
                tag_step = int(tag_name[5:])
        if saved_step is not None:
            if tag_step is not None and tag_step != saved_step:
                raise RuntimeError(
                    f"Checkpoint step mismatch: client state={saved_step}, tag={tag_step}"
                )
            return saved_step
        if tag_step is not None:
            return tag_step
        raise RuntimeError(
            "Cannot infer resumed checkpoint step from client state or tag"
        )

    def _restore_prompt_progress(
        self,
        resume_states: dict[str, Any] | None,
        resume_step: int,
    ) -> tuple[int, int, int]:
        saved_update_interval = None
        if isinstance(resume_states, dict) and "update_interval" in resume_states:
            try:
                saved_update_interval = int(resume_states["update_interval"])
            except (TypeError, ValueError):
                saved_update_interval = None
        if saved_update_interval is not None and saved_update_interval != int(
            self.update_interval
        ):
            logging.warning(
                "Checkpoint update_interval=%s does not match current update_interval=%s; "
                "continuing with the current setting.",
                saved_update_interval,
                int(self.update_interval),
            )

        progress_state = resolve_resume_progress_state(
            resume_states=resume_states,
            resume_step=resume_step,
            update_interval=int(self.update_interval),
            num_prompt_epochs=int(self.args.num_prompt_epoch),
            num_prompt_batches_per_epoch=int(len(self.prompts_dataloader)),
            rollout_batch_size=int(self.args.rollout_batch_size),
        )
        self.steps = int(progress_state["checkpoint_step"])
        self.global_step = int(progress_state["global_step"])
        self.policy_sgd_step = float(progress_state["policy_sgd_step"])
        self.query_step = int(progress_state["query_step"])
        self.prompt_consumed = int(progress_state["prompt_consumed"])
        self.prompt_epoch = int(progress_state["prompt_epoch"])
        self._prompt_batches_consumed_total = int(
            progress_state["prompt_batches_consumed_total"]
        )

        last_eval_query_step = int(progress_state["last_eval_query_step"])
        if last_eval_query_step > 0:
            self.last_eval_query_step = last_eval_query_step
        if isinstance(resume_states, dict) and "_pending_eval" in resume_states:
            self._pending_eval = bool(resume_states["_pending_eval"])
        diayn_tracker = getattr(self, "_diayn_mi_tracker", None)
        saved_diayn_state = (
            resume_states.get("diayn_mi_tracker_state")
            if isinstance(resume_states, dict)
            else None
        )
        if diayn_tracker is not None:
            if saved_diayn_state is None:
                raise RuntimeError(
                    "DIAYN run cannot resume without discriminator state"
                )
            diayn_tracker.load_state_dict(saved_diayn_state)
        elif saved_diayn_state is not None:
            raise RuntimeError(
                "non-DIAYN run cannot resume a DIAYN discriminator checkpoint"
            )
        self._restore_training_progress_state(resume_states)

        if self.steps % max(1, int(self.update_interval)) != 0:
            logging.warning(
                "Checkpoint step %s lands mid-update interval %s; optimizer step state "
                "resumes, but any in-memory rollout buffer from the unfinished interval "
                "cannot be reconstructed from checkpoints alone.",
                self.steps,
                int(self.update_interval),
            )

        used_saved_prompt_cursor = (
            isinstance(resume_states, dict)
            and "prompt_batches_consumed_total" in resume_states
        )
        log_fn = logging.info if used_saved_prompt_cursor else logging.warning
        log_fn(
            "%s prompt traversal state: checkpoint_step=%s completed_batches=%s "
            "start_epoch=%s start_batch_offset=%s query_step=%s prompt_consumed=%s",
            ("Restored" if used_saved_prompt_cursor else "Inferred fallback"),
            self.steps,
            self._prompt_batches_consumed_total,
            int(progress_state["start_prompt_epoch"]),
            int(progress_state["start_batch_offset"]),
            self.query_step,
            self.prompt_consumed,
        )

        return (
            int(progress_state["next_step"]),
            int(progress_state["start_prompt_epoch"]),
            int(progress_state["start_batch_offset"]),
        )

    @staticmethod
    def _checkpoint_step(path: Path) -> int:
        try:
            return int(path.name.removeprefix("step_"))
        except ValueError:
            return -1

    @staticmethod
    def _storage_barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _save_resume_checkpoint(self) -> None:
        """Atomically replace rolling recovery state without a zero-copy gap."""
        checkpoint_root = Path(self.save_path) / "checkpoints"
        tag = "step_{:05d}".format(self.steps)
        keep = int(self.args.max_resume_num)
        identity = getattr(self.args, "_remax_resume_identity", None)
        if identity is not None:
            if dist.get_world_size() != 1 or int(self.update_interval) != 1:
                raise RuntimeError(
                    "resume checkpoint requires a single-rank completed update"
                )
            state = self._checkpoint_client_state()
            validate_client_state(
                state,
                identity=identity,
                step=int(self.steps),
                batches_per_epoch=len(self.prompts_dataloader),
                epochs=int(self.args.num_prompt_epoch),
            )

            def write(staging):
                # Bypass OAT's pre-write rotation: the previous commit must
                # survive even a process kill halfway through this write.
                result = self.model.model.save_checkpoint(
                    str(staging.parent),
                    tag=staging.name,
                    client_state=state,
                    save_latest=False,
                )
                if result is False:
                    raise RuntimeError("DeepSpeed checkpoint writer failed")

            try:
                committed = commit_checkpoint(
                    checkpoint_root,
                    step=int(self.steps),
                    identity=identity,
                    writer=write,
                    keep=keep,
                )
            finally:
                restore_rng(state["rng_state"])
            logging.info("Committed full-run checkpoint %s", committed)
            return

        # OAT rotates before writing.  Giving it one temporary extra slot keeps
        # the previous valid checkpoint alive until the new distributed write
        # has returned on every rank; only then do we enforce the real limit.
        self.strategy.save_ckpt(
            self.model.model,
            str(checkpoint_root),
            tag=tag,
            max_num=keep + 1,
            max_mem=int(self.args.max_resume_mem),
            client_state=self._checkpoint_client_state(),
        )
        self._storage_barrier()
        if self.strategy.is_rank_0() and checkpoint_root.is_dir():
            checkpoints = sorted(
                (
                    path
                    for path in checkpoint_root.iterdir()
                    if path.is_dir()
                    and not path.is_symlink()
                    and self._checkpoint_step(path) >= 0
                ),
                key=lambda path: (self._checkpoint_step(path), path.stat().st_mtime_ns),
                reverse=True,
            )
            for obsolete in checkpoints[keep:]:
                shutil.rmtree(obsolete)
                logging.info("Deleted superseded resume checkpoint %s", obsolete)
        self._storage_barrier()

    def _write_eval_only_marker(self) -> None:
        """Record that an eval-only measurement completed, and under what settings.

        The decoding-frontier aggregator refuses any cell without this marker,
        so a job that died between loading the checkpoint and finishing its
        draws cannot be mistaken for a measured point.
        """
        self._storage_barrier()
        if self.strategy.is_rank_0():
            attempt_root = Path(self.save_path).resolve()
            payload = {
                "schema": "oat_zero_eval_only_complete_v1",
                "completed_at_unix": time.time(),
                "attempt_root": str(attempt_root),
                "pretrain": str(self.args.pretrain),
                "eval_mode_coverage_k": int(self.args.eval_mode_coverage_k),
                "eval_mode_coverage_temperature": float(
                    self.args.eval_mode_coverage_temperature
                ),
                "eval_mode_coverage_top_p": float(
                    getattr(self.args, "eval_mode_coverage_top_p", 1.0)
                ),
                "eval_mode_coverage_draws": int(self.args.eval_mode_coverage_draws),
                "eval_mode_coverage_seed": int(self.args.eval_mode_coverage_seed),
                "eval_temperature": float(self.args.eval_temperature),
                "test_split": str(self.args.test_split),
                "eval_data": str(self.args.eval_data),
                "prompt_template": str(self.args.prompt_template),
                "optimizer_steps": 0,
            }
            marker = attempt_root / "EVAL_ONLY_COMPLETE.json"
            temporary = attempt_root / f".{marker.name}.{os.getpid()}.tmp"
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, marker)
        self._storage_barrier()

    def _finalize_successful_storage(self) -> None:
        """Mark a completed run and retire optimizer state from all attempts."""
        self._storage_barrier()
        if self.strategy.is_rank_0():
            run_root = Path(self.args.save_path).resolve()
            attempt_root = Path(self.save_path).resolve()
            if attempt_root.parent != run_root:
                raise RuntimeError(
                    f"attempt path {attempt_root} is not directly beneath {run_root}"
                )

            removed: list[str] = []
            cleanup_errors: list[str] = []
            terminal_export = attempt_root / "saved_models" / f"step_{self.steps:05d}"
            export_required = int(self.args.export_steps) >= 0
            export_ready = not export_required or terminal_export.is_dir()
            if not export_ready:
                cleanup_errors.append(
                    f"terminal export missing; preserved resume state: {terminal_export}"
                )
            if bool(self.args.prune_resume_on_success) and export_ready:
                for checkpoint_root in sorted(run_root.glob("*/checkpoints")):
                    resolved = checkpoint_root.resolve()
                    if (
                        checkpoint_root.is_symlink()
                        or resolved.parent.parent != run_root
                        or resolved.name != "checkpoints"
                    ):
                        cleanup_errors.append(f"refused unsafe path: {checkpoint_root}")
                        continue
                    try:
                        shutil.rmtree(resolved)
                        removed.append(str(resolved))
                    except OSError as exc:
                        cleanup_errors.append(f"{resolved}: {exc}")
                        logging.exception(
                            "Could not retire completed-run checkpoint %s", resolved
                        )

            completion = {
                "schema": "oat_zero_training_complete_v1",
                "completed_at_unix": time.time(),
                "terminal_step": int(self.steps),
                "terminal_attempt": str(attempt_root),
                "terminal_export": (str(terminal_export) if export_required else None),
                "resume_checkpoints_pruned": bool(
                    self.args.prune_resume_on_success and export_ready
                ),
                "removed_checkpoint_roots": removed,
                "cleanup_errors": cleanup_errors,
            }
            marker = run_root / "TRAINING_COMPLETE.json"
            temporary = run_root / f".{marker.name}.{os.getpid()}.tmp"
            temporary.write_text(
                json.dumps(completion, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, marker)
        self._storage_barrier()

    def _storage_actions(
        self, *, terminal: bool, allow_scheduled_save: bool
    ) -> tuple[bool, bool]:
        """Return ``(export_model, save_resume_state)`` for this boundary."""
        export_steps = int(self.args.export_steps)
        resume_steps = int(self.args.resume_steps)
        should_export = (terminal and export_steps >= 0) or (
            allow_scheduled_save
            and export_steps > 0
            and self.steps > 0
            and self._should_do(export_steps)
            and self.steps >= int(self.args.export_from)
        )
        should_resume = (
            bool(self.args.save_ckpt)
            and not terminal
            and allow_scheduled_save
            and resume_steps > 0
            and self.steps > 0
            and self._should_do(resume_steps)
            and self.steps >= int(self.args.resume_from)
        )
        return should_export, should_resume
