"""OAT progress responsibilities, preserving historical state and ordering."""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any
import numpy as np
from ...launch_record import configuration_digest
from ...resume_state import discover_local_wandb_resume_run


class OatProgressMixin:
    def _ensure_training_progress_state(self) -> None:
        if hasattr(self, "_progress_metric_baselines"):
            return
        self._progress_metric_baselines: dict[str, float] = {}
        self._progress_metric_previous: dict[str, float] = {}
        self._progress_metric_bests: dict[str, float] = {}
        self._progress_metric_best_steps: dict[str, int] = {}
        self._actor_reward_ema: float | None = None
        self._actor_reward_ema_start: float | None = None

    @staticmethod
    def _coerce_log_float(value: Any) -> float | None:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "item"):
                value = value.item()
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        if scalar != scalar or scalar in {float("inf"), float("-inf")}:
            return None
        return scalar

    def _restore_training_progress_state(
        self,
        resume_states: dict[str, Any] | None,
    ) -> None:
        self._ensure_training_progress_state()
        if not isinstance(resume_states, dict):
            return
        for attr, key in (
            ("_progress_metric_baselines", "progress_metric_baselines"),
            ("_progress_metric_previous", "progress_metric_previous"),
            ("_progress_metric_bests", "progress_metric_bests"),
            ("_progress_metric_best_steps", "progress_metric_best_steps"),
        ):
            saved = resume_states.get(key)
            if isinstance(saved, dict):
                setattr(
                    self,
                    attr,
                    {
                        str(saved_key): (
                            int(saved_value)
                            if attr == "_progress_metric_best_steps"
                            else float(saved_value)
                        )
                        for saved_key, saved_value in saved.items()
                    },
                )
        self._actor_reward_ema = self._coerce_log_float(
            resume_states.get("actor_reward_ema")
        )
        self._actor_reward_ema_start = self._coerce_log_float(
            resume_states.get("actor_reward_ema_start")
        )
        tau_controller = getattr(self, "_xdr_tau_controller", None)
        saved_controller = resume_states.get("xdr_tau_controller_state")
        if tau_controller is not None and isinstance(saved_controller, dict):
            tau_controller.load_state_dict(saved_controller)
        maxent_controller = getattr(self, "_maxent_alpha_controller", None)
        saved_maxent_controller = resume_states.get("maxent_alpha_controller_state")
        if maxent_controller is not None:
            if not isinstance(saved_maxent_controller, dict):
                raise ValueError(
                    "adaptive MaxEnt checkpoint is missing alpha-controller state"
                )
            maxent_controller.load_state_dict(saved_maxent_controller)
        elif saved_maxent_controller is not None:
            raise ValueError(
                "fixed-alpha run cannot resume an entropy-adaptive checkpoint"
            )
        length_controller = getattr(self, "_maxent_length_controller", None)
        length_state_key = "maxent_length_controller_state"
        saved_length_controller = resume_states.get(length_state_key)
        if length_controller is not None:
            if not isinstance(saved_length_controller, dict):
                raise ValueError(
                    "constrained MaxEnt checkpoint is missing length-controller state"
                )
            length_controller.load_state_dict(saved_length_controller)
        elif saved_length_controller is not None:
            raise ValueError(
                "unconstrained run cannot resume a length-constrained checkpoint"
            )
        semantic_shannon_tracker = getattr(self, "_semantic_shannon_tracker", None)
        semantic_shannon_state_key = "semantic_shannon_tracker_state"
        saved_semantic_shannon = resume_states.get(semantic_shannon_state_key)
        if semantic_shannon_tracker is not None:
            if not isinstance(saved_semantic_shannon, dict):
                raise ValueError(
                    "semantic Shannon run cannot resume without estimator state"
                )
            rms_controller = getattr(self, "_semantic_rms_controller", None)
            saved_rms = resume_states.get("semantic_rms_controller_state")
            if rms_controller is not None:
                if not isinstance(saved_rms, dict):
                    raise ValueError(
                        "adaptive semantic run cannot resume without controller state"
                    )
                # The tracker stores the coefficient that was active when the
                # checkpoint was written, while a fresh adaptive run starts at
                # the registered base coefficient. Restore the controller
                # first so the tracker's strict contract check compares the
                # two checkpoint states rather than comparing evolved state to
                # the base configuration.
                rms_controller.load_state_dict(saved_rms)
                semantic_shannon_tracker.coefficient = float(
                    rms_controller.current_coefficient
                )
            elif saved_rms is not None:
                raise ValueError(
                    "fixed-coefficient run cannot resume an adaptive checkpoint"
                )
            semantic_shannon_tracker.load_state_dict(saved_semantic_shannon)
        elif saved_semantic_shannon is not None:
            raise ValueError(
                "non-semantic-Shannon run cannot resume a semantic Shannon checkpoint"
            )
        online_canonical_bank = getattr(self, "_online_canonical_bank", None)
        online_canonical_state_key = "online_canonical_bank_state"
        saved_online_canonical = resume_states.get(online_canonical_state_key)
        if online_canonical_bank is not None:
            if isinstance(saved_online_canonical, dict):
                online_canonical_bank.load_state_dict(saved_online_canonical)
            elif online_canonical_bank.objective_active or bool(
                getattr(
                    online_canonical_bank,
                    "retain_exemplars",
                    False,
                )
            ):
                raise ValueError(
                    "online canonical run cannot resume without bank state"
                )
            else:
                logging.warning(
                    "legacy Dr.GRPO checkpoint has no passive verified-discovery "
                    "state; cumulative tracking restarts from zero"
                )
        elif saved_online_canonical is not None:
            raise ValueError(
                "regular run cannot resume an online canonical bank checkpoint"
            )
        saved_rlep_online_pool = resume_states.get("rlep_online_pool_state")
        if bool(getattr(self.args, "rlep_online_pool", False)):
            from ...experiments.rlep import OnlineRLEPExperiencePool

            if not isinstance(saved_rlep_online_pool, dict):
                raise ValueError(
                    "RLEP online-pool run cannot resume without its pool state"
                )
            pool = OnlineRLEPExperiencePool(
                minimum=int(getattr(self.args, "rlep_replay_count", 0) or 0)
            )
            pool.load_state_dict(saved_rlep_online_pool)
            self._rlep_experience_pool = pool
        elif saved_rlep_online_pool is not None:
            raise ValueError("regular run cannot resume an RLEP online pool checkpoint")
        proposal_starvation_controller = getattr(
            self,
            "_proposal_starvation_controller",
            None,
        )
        saved_proposal_starvation = resume_states.get(
            "proposal_starvation_controller_state"
        )
        if proposal_starvation_controller is not None:
            if not isinstance(saved_proposal_starvation, dict):
                raise ValueError(
                    "proposal-starvation run cannot resume without controller state"
                )
            proposal_starvation_controller.load_state_dict(saved_proposal_starvation)
        elif saved_proposal_starvation is not None:
            raise ValueError(
                "fixed proposal run cannot resume a starvation-controller checkpoint"
            )
        verified_route_library = getattr(
            self,
            "_verified_route_library",
            None,
        )
        verified_route_state_key = "verified_route_library_state"
        saved_verified_routes = resume_states.get(verified_route_state_key)
        if verified_route_library is not None:
            from ...verified_route_library import VerifiedRouteLibrary

            if not isinstance(verified_route_library, VerifiedRouteLibrary):
                raise RuntimeError("invalid verified route library")
            if not isinstance(saved_verified_routes, dict):
                raise ValueError(
                    "verified-route run cannot resume without route-library state"
                )
            verified_route_library.load_state_dict(saved_verified_routes)
        elif saved_verified_routes is not None:
            raise ValueError(
                "non-route run cannot resume a verified route library checkpoint"
            )
        math_strategy_canonicalizer = getattr(
            self, "_math_strategy_canonicalizer", None
        )
        math_strategy_state_key = "math_strategy_canonicalizer_state"
        saved_math_strategy = resume_states.get(math_strategy_state_key)
        if math_strategy_canonicalizer is not None:
            if not isinstance(saved_math_strategy, dict):
                raise ValueError(
                    "MATH strategy run cannot resume without canonicalizer state"
                )
            math_strategy_canonicalizer.load_state_dict(saved_math_strategy)
        elif saved_math_strategy is not None:
            raise ValueError(
                "non-MATH-strategy run cannot resume a strategy checkpoint"
            )
        online_canonical_controller = getattr(
            self, "_online_canonical_alpha_controller", None
        )
        online_canonical_controller_state_key = (
            "online_canonical_alpha_controller_state"
        )
        saved_online_canonical_controller = resume_states.get(
            online_canonical_controller_state_key
        )
        if online_canonical_controller is not None:
            if not isinstance(saved_online_canonical_controller, dict):
                raise ValueError(
                    "adaptive online canonical checkpoint is missing "
                    "alpha-controller state"
                )
            online_canonical_controller.load_state_dict(
                saved_online_canonical_controller
            )
        elif saved_online_canonical_controller is not None:
            raise ValueError(
                "fixed-alpha online canonical run cannot resume an adaptive "
                "online canonical checkpoint"
            )

    def _record_progress_metric(
        self,
        logs_dict: dict[str, Any],
        *,
        source_key: str,
        output_prefix: str,
        higher_is_better: bool = True,
    ) -> None:
        self._ensure_training_progress_state()
        value = self._coerce_log_float(logs_dict.get(source_key))
        if value is None:
            return

        baselines = self._progress_metric_baselines
        previous = self._progress_metric_previous
        bests = self._progress_metric_bests
        best_steps = self._progress_metric_best_steps

        baseline = baselines.setdefault(source_key, value)
        prev = previous.get(source_key, value)
        best = bests.get(source_key, value)
        is_new_best = value > best if higher_is_better else value < best
        if source_key not in bests or is_new_best:
            best = value
            bests[source_key] = value
            best_steps[source_key] = int(self.steps)
        previous[source_key] = value

        logs_dict[f"{output_prefix}/value"] = value
        logs_dict[f"{output_prefix}/gain_from_start"] = value - baseline
        logs_dict[f"{output_prefix}/gain_from_prev"] = value - prev
        logs_dict[f"{output_prefix}/best"] = best
        logs_dict[f"{output_prefix}/best_gain_from_start"] = best - baseline
        logs_dict[f"{output_prefix}/best_step"] = int(best_steps[source_key])
        logs_dict[f"{output_prefix}/steps_since_best"] = int(
            self.steps - best_steps[source_key]
        )

    def _add_learning_progress_metrics(self, logs_dict: dict[str, Any]) -> None:
        self._record_progress_metric(
            logs_dict,
            source_key="eval/average/accuracy",
            output_prefix="xdr/progress/eval_accuracy",
        )
        self._record_progress_metric(
            logs_dict,
            source_key="eval/average/score",
            output_prefix="xdr/progress/eval_score",
        )
        self._record_progress_metric(
            logs_dict,
            source_key="actor/rewards",
            output_prefix="xdr/progress/rollout_reward",
        )

        actor_reward = self._coerce_log_float(logs_dict.get("actor/rewards"))
        if actor_reward is None:
            return
        self._ensure_training_progress_state()
        if self._actor_reward_ema is None:
            self._actor_reward_ema = actor_reward
            self._actor_reward_ema_start = actor_reward
        else:
            self._actor_reward_ema = 0.9 * self._actor_reward_ema + 0.1 * actor_reward
        logs_dict["xdr/progress/rollout_reward_ema"] = self._actor_reward_ema
        if self._actor_reward_ema_start is not None:
            logs_dict["xdr/progress/rollout_reward_ema_gain_from_start"] = (
                self._actor_reward_ema - self._actor_reward_ema_start
            )

    def _format_compact_training_sample(self) -> str | None:
        """Return a compact rollout sample summary for console logs."""

        if not self.pi_buffer:
            return None
        sample = np.random.choice(self.pi_buffer)
        prompt = str(getattr(sample, "prompt", "") or "")
        response = str(getattr(sample, "response", "") or "")
        response_ids = getattr(sample, "response_ids", None)
        rewards = getattr(sample, "rewards", None)
        info = getattr(sample, "info", None)
        response_tokens = len(response_ids) if response_ids is not None else None
        reward_mean = None
        if rewards is not None:
            try:
                reward_values = list(rewards)
                if reward_values:
                    reward_mean = sum(float(value) for value in reward_values) / len(
                        reward_values
                    )
            except (TypeError, ValueError):
                reward_mean = None
        pieces = [
            f"prompt_chars={len(prompt)}",
            f"response_chars={len(response)}",
        ]
        if response_tokens is not None:
            pieces.append(f"response_tokens={response_tokens}")
        if reward_mean is not None:
            pieces.append(f"reward_mean={reward_mean:.4f}")
        if isinstance(info, dict):
            actor_reward = info.get("actor/rewards")
            if actor_reward is not None:
                try:
                    pieces.append(f"actor_reward={float(actor_reward):.4f}")
                except (TypeError, ValueError):
                    pass
        return "Training sample summary: " + " ".join(pieces)

    def _init_wandb(self, resume_states: dict[str, Any] | None = None) -> None:
        if not self._requested_use_wb or not self.strategy.is_rank_0():
            return
        if self._wandb is not None:
            return

        import wandb

        if not wandb.api.api_key and isinstance(self._requested_use_wb, str):
            wandb.login(key=self._requested_use_wb)

        env_run_id = os.environ.get("OAT_ZERO_WANDB_RUN_ID") or os.environ.get(
            "WANDB_RUN_ID"
        )
        saved_run_id = None
        saved_run_name = None
        if isinstance(resume_states, dict):
            raw_run_id = resume_states.get("wandb_run_id")
            if isinstance(raw_run_id, str) and raw_run_id:
                saved_run_id = raw_run_id
            raw_run_name = resume_states.get("wandb_run_name")
            if isinstance(raw_run_name, str) and raw_run_name:
                saved_run_name = raw_run_name

        discovered_run_id = None
        discovered_run_name = None
        if not env_run_id and not saved_run_id and self.args.resume_dir:
            wandb_run_roots = []
            env_wandb_dir = os.environ.get("WANDB_DIR")
            if env_wandb_dir:
                wandb_run_roots.append(Path(env_wandb_dir) / "runs" / "wandb")
            wandb_run_roots.extend(
                [
                    Path.cwd() / "var" / "wandb" / "runs" / "wandb",
                    Path.cwd() / "wandb" / "runs" / "wandb",
                ]
            )
            discovered_run_id, discovered_run_name = discover_local_wandb_resume_run(
                wandb_run_roots=wandb_run_roots,
                resume_dir=self.args.resume_dir,
                resume_tag=self.args.resume_tag,
                saved_run_name=saved_run_name,
                current_run_name=self._wandb_run_name,
            )
            if discovered_run_id:
                logging.info(
                    "Recovered W&B resume run id %s from local run logs for %s",
                    discovered_run_id,
                    self.args.resume_dir,
                )

        run_id = env_run_id or saved_run_id or discovered_run_id
        if env_run_id:
            if saved_run_id or discovered_run_id:
                logging.info(
                    "Using explicit W&B run id %s; skipping resumed/discovered lineage.",
                    env_run_id,
                )
            run_name = self._wandb_run_name
        else:
            run_name = saved_run_name or discovered_run_name or self._wandb_run_name
        init_kwargs: dict[str, Any] = {
            "entity": self.args.wb_org,
            "project": self.args.wb_project,
            "group": self.args.wb_group,
            "name": run_name,
            "config": self.args.__dict__,
            "reinit": True,
        }
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"

        self._wandb = wandb
        wandb.init(**init_kwargs)
        if wandb.run is not None:
            self._wandb_run_id = wandb.run.id
            self._wandb_run_name = wandb.run.name
        else:
            self._wandb_run_id = run_id
            self._wandb_run_name = run_name

        # Use the actual training step for chart alignment across resumes.
        wandb.define_metric("trainer/step")
        wandb.define_metric("*", step_metric="trainer/step")

    def _append_resume_decision(self, feedback_data, raw_prompts):
        """Exact discrete audit trail; no wall-clock or approximate loss fields."""

        def tokens(value):
            return [int(x) for x in value]

        record = {
            "step": int(self.steps),
            "data_position": int(self._prompt_batches_consumed_total),
            "prompt_sha256": configuration_digest(list(raw_prompts)),
            "responses": [
                {
                    "tokens": tokens(row.response_ids),
                    "rewards": [float(x) for x in row.rewards],
                }
                for row in feedback_data
            ],
            "bank_sha256": configuration_digest(
                self._online_canonical_bank.state_dict()
            ),
            "replay": getattr(self, "_resume_replay_decisions", []),
        }
        with (Path(self.save_path) / "resume_decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _append_train_metrics_jsonl(self, logs_dict: dict[str, Any]) -> None:
        """Persist per-step scalar training metrics to train_metrics.jsonl.

        The comparative recipes run with wandb disabled, but the exploration
        side-car figure needs per-step aggregation diagnostics
        (train/agg_eff_rollouts, train/agg_incorrect_mass) from every arm, so
        rank 0 appends one JSON record per logging step under save_path. The
        sink is best-effort: logging must never kill a training run, so any
        failure is warned once and then ignored.
        """
        try:
            record: dict[str, float] = {}
            for key, value in logs_dict.items():
                scalar = self._coerce_log_float(value)
                if scalar is not None:
                    record[str(key)] = scalar
            if not record:
                return
            path = os.path.join(self.save_path, "train_metrics.jsonl")
            with open(path, "a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            if not getattr(self, "_train_metrics_jsonl_failed_warned", False):
                logging.exception(
                    "Failed to append train_metrics.jsonl; continuing without "
                    "the per-step metrics sink."
                )
                self._train_metrics_jsonl_failed_warned = True

    def _append_mode_coverage_draw_jsonl(self, record: dict[str, Any]) -> None:
        """Durably retain every sampled evaluation outcome on rank zero."""

        path = os.path.join(self.save_path, "eval_mode_coverage_draws.jsonl")
        with open(path, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, sort_keys=True) + "\n")
