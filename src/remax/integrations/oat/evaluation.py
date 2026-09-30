"""OAT evaluation responsibilities, preserving historical state and ordering."""

from __future__ import annotations
import json
import logging
import os
from typing import Any
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
from ...answer_options import conditional_answer_repr, crossfit_option_answer_mi
from ...benchmark import EvaluationFailure, failure, validate_reward_batch
from ...logging_utils import filter_wandb_logs
from .support import (
    MODE_COVERAGE_METRICS,
    OPTION_BINDING_METRICS,
    _compute_mode_coverage_metrics,
    _mode_coverage_log_key,
    _parse_answer_mode_count,
    _parse_public_seed_key,
    _summarize_mode_coverage_draws,
)


class OatEvaluationMixin:
    def eval_and_log(
        self,
        train_info,
        eval=False,
        save=False,
        allow_scheduled_save=True,
    ):
        eval_info = {}
        forced_eval = self.args.eval_steps > 0 and eval
        should_eval = forced_eval or self._should_do(self.args.eval_steps)
        duplicate_terminal_eval = (
            bool(save)
            and forced_eval
            and getattr(self, "_last_evaluated_global_step", None)
            == int(self.global_step)
        )
        if duplicate_terminal_eval:
            # The run loop increments ``steps`` after every consumed prompt,
            # including the final one.  If that final update already landed
            # on a scheduled evaluation boundary, the terminal call below
            # therefore has a new bookkeeping step but the exact same policy.
            # Keep terminal export/storage semantics while avoiding a second
            # full deterministic evaluation of unchanged weights.
            should_eval = False
            logging.info(
                "Skipping duplicate terminal evaluation at step %s; "
                "policy global_step %s was already evaluated.",
                self.steps,
                self.global_step,
            )
        should_export, should_resume = self._storage_actions(
            terminal=bool(save), allow_scheduled_save=allow_scheduled_save
        )

        if should_resume and should_eval and self.strategy.is_rank_0():
            logging.info(
                "Recovery and evaluation boundary at step %s.",
                self.steps,
            )

        strict_checkpoint = (
            getattr(self.args, "_remax_resume_identity", None) is not None
        )
        if should_resume and not strict_checkpoint:
            self._save_resume_checkpoint()

        if should_export:
            self.strategy.save_model(
                self.model,
                self.tokenizer,
                os.path.join(self.save_path, "saved_models"),
                tag="step_{:05d}".format(self.steps),
                max_num=int(self.args.max_export_num),
                max_mem=int(self.args.max_export_mem),
            )

        if should_eval:
            eval_info = self.evaluate(self.eval_prompts_dataloader, self.steps)
            self._last_evaluated_global_step = int(self.global_step)

        if eval_info or self.steps % self.args.logging_steps == 0:
            misc_info = self.get_misc_info()
            misc_info["lr"] = self.scheduler.get_last_lr()[0]

            misc_info = {
                "misc/%s" % k: v
                for k, v in {
                    **misc_info,
                }.items()
            }
            logs_dict = {**train_info, **eval_info, **self.actor_info, **misc_info}
            logs_dict = self.strategy.all_reduce(logs_dict)
            logs_dict.update(
                self.strategy.all_reduce(
                    {
                        "misc/query_step": self.query_step,
                        "misc/prompt_consumed": self.prompt_consumed,
                    },
                    op="sum",
                )
            )
            logs_dict["trainer/step"] = int(self.steps)
            logs_dict["trainer/global_step"] = int(self.global_step)
            logs_dict["trainer/policy_sgd_step"] = float(self.policy_sgd_step)
            self._add_learning_progress_metrics(logs_dict)

            if self.strategy.is_rank_0():
                sample_summary = self._format_compact_training_sample()
                if sample_summary:
                    self.strategy.print(sample_summary)
                self.strategy.pprint(logs_dict)
                self._append_train_metrics_jsonl(logs_dict)
                if self._wandb is not None:
                    self._wandb.log(
                        filter_wandb_logs(logs_dict),
                        step=int(self.steps),
                    )

        if should_resume and strict_checkpoint:
            # Commit after evaluation and progress logging so a restored boundary
            # neither repeats evaluation nor changes logging's RNG consumption.
            self._save_resume_checkpoint()

    def evaluate(self, dataloader, steps):
        try:
            return self._evaluate_verified(dataloader, steps)
        except Exception as error:
            diagnostic = (
                error.diagnostic
                if isinstance(error, EvaluationFailure)
                else failure(
                    "worker_failure", f"{type(error).__name__}: {error}"
                ).diagnostic
            )
            if self.strategy.is_rank_0():
                path = os.path.join(self.save_path, "evaluation_failures.jsonl")
                with open(path, "a", encoding="utf-8") as sink:
                    sink.write(
                        json.dumps(
                            {
                                "status": "failed",
                                "step": steps,
                                "metrics": None,
                                "diagnostic": diagnostic,
                            }
                        )
                        + "\n"
                    )
                    sink.flush()
                    os.fsync(sink.fileno())
            raise

    def _evaluate_verified(self, dataloader, steps):
        # Discard the default eval dataloader, and run eval on multiple benchmarks.
        del dataloader
        all_metrics = {}
        accuracies = []
        scores = []
        lens = []
        total_benchmarks = len(self.eval_dataset_dict)
        for benchmark_idx, (benchmark_name, dataset) in enumerate(
            self.eval_dataset_dict.items(), start=1
        ):
            eval_prompts_dataloader = DataLoader(
                dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.eval_dataloader_collate_fn,
            )
            if self.strategy.is_rank_0():
                logging.info(
                    "Starting eval benchmark %s/%s: %s (%s prompts, %s batches) at step %s",
                    benchmark_idx,
                    total_benchmarks,
                    benchmark_name,
                    len(dataset),
                    len(eval_prompts_dataloader),
                    steps,
                )
            metrics = super().evaluate(
                eval_prompts_dataloader, f"{steps}_{benchmark_name}"
            )
            if self.strategy.is_rank_0():
                logging.info(
                    "Finished eval benchmark %s/%s: %s accuracy=%.4f score=%.4f avg_len=%.2f at step %s",
                    benchmark_idx,
                    total_benchmarks,
                    benchmark_name,
                    float(metrics["eval/accuracy"]),
                    float(metrics["eval/score"]),
                    float(metrics["eval/response_tok_len"]),
                    steps,
                )
            all_metrics.update(
                {
                    k.replace("eval/", f"eval/{benchmark_name}/"): v
                    for k, v in metrics.items()
                }
            )
            accuracies.append(metrics["eval/accuracy"])
            scores.append(metrics["eval/score"])
            lens.append(metrics["eval/response_tok_len"])
        all_metrics.update(
            {
                "eval/average/accuracy": np.mean(accuracies),
                "eval/average/score": np.mean(scores),
                "eval/average/response_tok_len": np.mean(lens),
            }
        )

        if self.args.eval_mode_coverage_k > 0:
            mc_coverages = []
            for benchmark_name, dataset in self.eval_dataset_dict.items():
                mc = self.evaluate_mode_coverage(
                    dataset,
                    benchmark_name,
                    steps,
                    k=self.args.eval_mode_coverage_k,
                    temperature=self.args.eval_mode_coverage_temperature,
                )
                all_metrics.update(mc)
                cov_key = f"eval/{benchmark_name}/sampled_mode_coverage_at_{self.args.eval_mode_coverage_k}"
                if cov_key in mc:
                    mc_coverages.append(mc[cov_key])
            if mc_coverages:
                all_metrics[
                    f"eval/average/sampled_mode_coverage_at_{self.args.eval_mode_coverage_k}"
                ] = float(np.mean(mc_coverages))

        return all_metrics

    def evaluate_mode_coverage(
        self,
        dataset,
        benchmark_name: str,
        steps: int,
        *,
        k: int,
        temperature: float,
    ) -> dict[str, float]:
        """Sampled mode-coverage eval: K completions per prompt at T=temperature.

        Runs on the live vLLM engine without saving a checkpoint. All distributed
        ranks participate in the barriers; only rank 0 drives actor inference.
        Results are broadcast from rank 0 so that the subsequent all_reduce in
        eval_and_log produces correct per-run values.
        """
        self._pre_evaluate()
        draw_count = int(getattr(self.args, "eval_mode_coverage_draws", 4))
        seed_base = int(getattr(self.args, "eval_mode_coverage_seed", 1001))
        try:
            diagnostic = None
            try:
                metrics = self._run_sampled_mode_coverage(
                    dataset,
                    benchmark_name,
                    steps,
                    k=k,
                    temperature=temperature,
                    draw_count=draw_count,
                    seed_base=seed_base,
                )
            except Exception as error:
                metrics = {}
                diagnostic = (
                    error.diagnostic
                    if isinstance(error, EvaluationFailure)
                    else failure(
                        "worker_failure", f"{type(error).__name__}: {error}"
                    ).diagnostic
                )
            if (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            ):
                failures = [None] * dist.get_world_size()
                dist.all_gather_object(failures, diagnostic)
                diagnostic = next((item for item in failures if item is not None), None)
            if diagnostic is not None:
                raise EvaluationFailure(diagnostic)
        finally:
            self._post_evaluate()
        # All ranks must call broadcast the same number of times (once per key).
        # Non-rank-0 processes return {} from _run_sampled_mode_coverage, so
        # pre-populate the canonical keys with 0.0 on every rank before broadcast.
        canonical_keys = []
        metrics_to_broadcast = [(metric, "neutral") for metric in MODE_COVERAGE_METRICS]
        run_args = getattr(self, "args", None)
        if int(getattr(run_args, "diayn_num_options", 0) or 0) > 1:
            metrics_to_broadcast.extend(
                (metric, "latent") for metric in MODE_COVERAGE_METRICS
            )
            metrics_to_broadcast.extend(
                (metric, "neutral") for metric in OPTION_BINDING_METRICS
            )
        for metric, metric_namespace in metrics_to_broadcast:
            key = _mode_coverage_log_key(
                benchmark_name,
                metric,
                k,
                metric_namespace=metric_namespace,
            )
            canonical_keys.extend(
                [
                    key,
                    f"{key}_draw_count",
                    f"{key}_draw_std",
                    f"{key}_draw_se",
                    f"{key}_draw_min",
                    f"{key}_draw_max",
                    *(f"{key}_draw_{index}" for index in range(draw_count)),
                ]
            )
        for key in canonical_keys:
            metrics.setdefault(key, 0.0)
        metrics = self.strategy.broadcast(metrics)
        return metrics

    def _run_sampled_mode_coverage(
        self,
        dataset,
        benchmark_name: str,
        steps: int,
        *,
        k: int,
        temperature: float,
        draw_count: int,
        seed_base: int,
    ) -> dict[str, float]:
        if not self.strategy.is_rank_0():
            return {}

        # Nucleus truncation is a property of the sampled draws only; the
        # greedy trace below is unaffected by it and keeps the untruncated
        # surface so its value stays comparable across a decoding sweep.
        coverage_top_p = float(
            getattr(getattr(self, "args", None), "eval_mode_coverage_top_p", 1.0) or 1.0
        )
        # Each draw reserves its own block of K child streams; see
        # ``eval_mode_coverage_disjoint_draws``.
        stride = (
            k
            if bool(getattr(self.args, "eval_mode_coverage_disjoint_draws", True))
            else 1
        )
        logging.info(
            "Starting sampled mode-coverage eval %s/%s: %s "
            "(%s prompts, draws=%s, K=%s, T=%s, top_p=%s, seeds=%s..%s stride=%s) at step %s",
            benchmark_name,
            len(self.eval_dataset_dict),
            benchmark_name,
            len(dataset),
            draw_count,
            k,
            temperature,
            coverage_top_p,
            seed_base,
            seed_base + (draw_count - 1) * stride + k - 1,
            stride,
            steps,
        )
        greedy_mean, greedy_prompt_outcomes = self._run_sampled_mode_coverage_draw(
            dataset,
            k=1,
            temperature=0.0,
            seed=0,
            condition_on_answer_options=False,
        )
        if greedy_mean:
            self._append_mode_coverage_draw_jsonl(
                {
                    "schema_version": 1,
                    "evaluation_kind": "deterministic_greedy_trace_neutral",
                    "benchmark": benchmark_name,
                    "step": int(steps),
                    "draw_index": None,
                    "seed": 0,
                    "sample_count": 1,
                    "temperature": 0.0,
                    "metrics": greedy_mean,
                    "prompts": greedy_prompt_outcomes,
                }
            )
        draw_means: list[dict[str, float]] = []
        draw_prompt_outcomes: list[list[dict[str, Any]]] = []
        for draw_index in range(draw_count):
            draw_seed = seed_base + draw_index * stride
            mean, prompt_outcomes = self._run_sampled_mode_coverage_draw(
                dataset,
                k=k,
                temperature=temperature,
                seed=draw_seed,
                condition_on_answer_options=False,
                top_p=coverage_top_p,
            )
            if not mean:
                continue
            draw_means.append(mean)
            draw_prompt_outcomes.append(prompt_outcomes)
            self._append_mode_coverage_draw_jsonl(
                {
                    "schema_version": 1,
                    "evaluation_kind": "fixed_seed_sampled_k_neutral",
                    "benchmark": benchmark_name,
                    "step": int(steps),
                    "draw_index": draw_index,
                    "seed": draw_seed,
                    "sample_count": int(k),
                    "temperature": float(temperature),
                    "top_p": coverage_top_p,
                    "metrics": mean,
                    "prompts": prompt_outcomes,
                }
            )

        if not draw_means:
            return {}
        metrics = _summarize_mode_coverage_draws(draw_means, benchmark_name, k)
        run_args = getattr(self, "args", None)
        if int(getattr(run_args, "diayn_num_options", 0) or 0) > 1:
            latent_draw_means: list[dict[str, float]] = []
            latent_draw_prompt_outcomes: list[list[dict[str, Any]]] = []
            for draw_index in range(draw_count):
                draw_seed = seed_base + draw_index * stride
                latent_mean, latent_prompt_outcomes = (
                    self._run_sampled_mode_coverage_draw(
                        dataset,
                        k=k,
                        temperature=temperature,
                        seed=draw_seed,
                        condition_on_answer_options=True,
                        top_p=coverage_top_p,
                    )
                )
                if not latent_mean:
                    continue
                latent_draw_means.append(latent_mean)
                latent_draw_prompt_outcomes.append(latent_prompt_outcomes)
                self._append_mode_coverage_draw_jsonl(
                    {
                        "schema_version": 2,
                        "evaluation_kind": ("fixed_seed_sampled_k_latent_binding"),
                        "benchmark": benchmark_name,
                        "step": int(steps),
                        "draw_index": draw_index,
                        "seed": draw_seed,
                        "seed_derivation": (
                            "draw_seed*1000000 + prompt_index*num_options + z"
                        ),
                        "sample_count": int(k),
                        "temperature": float(temperature),
                        "num_options": int(run_args.diayn_num_options),
                        "metrics": latent_mean,
                        "prompts": latent_prompt_outcomes,
                    }
                )
            if not latent_draw_means:
                raise RuntimeError(
                    "DIAYN latent-conditioned evaluation produced no draws"
                )
            metrics.update(
                _summarize_mode_coverage_draws(
                    latent_draw_means,
                    benchmark_name,
                    k,
                    metric_namespace="latent",
                )
            )
            answer_reprs_by_draw: list[list[str | None]] = []
            option_ids_by_draw: list[list[int | None]] = []
            correct_by_draw: list[list[bool]] = []
            for outcomes in latent_draw_prompt_outcomes:
                answer_rows: list[str | None] = []
                option_rows: list[int | None] = []
                correct_rows: list[bool] = []
                for outcome in outcomes:
                    prompt_index = int(outcome["prompt_index"])
                    answer_rows.extend(
                        conditional_answer_repr(prompt_index, key)
                        for key in outcome["answer_keys"]
                    )
                    option_rows.extend(outcome["option_ids"])
                    correct_rows.extend(
                        float(value) > 0.0 for value in outcome["rewards"]
                    )
                answer_reprs_by_draw.append(answer_rows)
                option_ids_by_draw.append(option_rows)
                correct_by_draw.append(correct_rows)
            crossfit_metrics = crossfit_option_answer_mi(
                answer_reprs_by_draw=answer_reprs_by_draw,
                option_ids_by_draw=option_ids_by_draw,
                correct_by_draw=correct_by_draw,
                num_options=int(run_args.diayn_num_options),
                smoothing=float(run_args.diayn_mi_smoothing),
            )
            metrics.update(
                _summarize_mode_coverage_draws(
                    crossfit_metrics,
                    benchmark_name,
                    k,
                )
            )
            self._append_mode_coverage_draw_jsonl(
                {
                    "schema_version": 2,
                    "evaluation_kind": "crossfit_option_binding",
                    "benchmark": benchmark_name,
                    "step": int(steps),
                    "draw_count": len(latent_draw_means),
                    "num_options": int(run_args.diayn_num_options),
                    "metrics_by_held_out_draw": crossfit_metrics,
                }
            )
        coverage_key = _mode_coverage_log_key(benchmark_name, "mode_coverage_at_k", k)
        pass_key = _mode_coverage_log_key(benchmark_name, "any_correct_at_k", k)
        mean_key = _mode_coverage_log_key(benchmark_name, "mean_at_k", k)
        distinct_key = _mode_coverage_log_key(
            benchmark_name, "distinct_correct_modes_at_k", k
        )

        logging.info(
            "Finished sampled mode-coverage eval %s: "
            "mode_cov@%s=%.4f±%.4f any_correct@%s=%.4f "
            "mean@%s=%.4f distinct@%s=%.4f across %s fixed draws at step %s",
            benchmark_name,
            k,
            metrics[coverage_key],
            metrics[f"{coverage_key}_draw_se"],
            k,
            metrics[pass_key],
            k,
            metrics[mean_key],
            k,
            metrics[distinct_key],
            len(draw_means),
            steps,
        )
        if int(getattr(run_args, "diayn_num_options", 0) or 0) > 1:
            latent_coverage_key = _mode_coverage_log_key(
                benchmark_name,
                "mode_coverage_at_k",
                k,
                metric_namespace="latent",
            )
            binding_key = _mode_coverage_log_key(
                benchmark_name,
                "option_answer_mi_lower_bound_nats",
                k,
            )
            logging.info(
                "Finished latent-conditioned binding eval %s: "
                "latent_mode_cov@%s=%.4f MI_lb@%s=%.4f with independent "
                "(draw,prompt,z) seeds at step %s",
                benchmark_name,
                k,
                metrics[latent_coverage_key],
                k,
                metrics[binding_key],
                steps,
            )
        return metrics

    def _run_sampled_mode_coverage_draw(
        self,
        dataset,
        *,
        k: int,
        temperature: float,
        seed: int,
        condition_on_answer_options: bool = False,
        top_p: float = 1.0,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Evaluate one fixed-seed K draw and return every prompt outcome."""

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.eval_dataloader_collate_fn,
        )
        per_prompt_metrics: list[dict[str, float]] = []
        prompt_outcomes: list[dict[str, Any]] = []
        futs: list = []
        pending: list[tuple[list[str], list[int], list[str]]] = []
        prompt_offset = 0

        for batch_index, (batch_formatted, batch_raw, batch_refs) in enumerate(
            dataloader
        ):
            actor = self.actors[batch_index % len(self.actors)]
            refs_batch = list(batch_refs)
            indices = list(range(prompt_offset, prompt_offset + len(refs_batch)))
            prompt_offset += len(refs_batch)
            futs.append(
                actor.futures.generate_for_mode_coverage(
                    list(batch_formatted),
                    refs_batch,
                    k,
                    temperature,
                    seed,
                    condition_on_answer_options,
                    indices,
                    float(top_p),
                )
            )
            pending.append((refs_batch, indices, list(batch_raw)))
            if len(futs) == len(self.actors) or batch_index == len(dataloader) - 1:
                for fut, (queued_refs, queued_indices, queued_prompts) in zip(
                    futs, pending
                ):
                    result = fut.result()
                    verifier_rows = result.get("verifier_infos")
                    if verifier_rows is None or len(verifier_rows) != len(queued_refs):
                        raise failure(
                            "worker_failure",
                            "missing sampled-evaluation verifier diagnostics",
                        )
                    for rewards_row, infos_row, ref in zip(
                        result["rewards"], verifier_rows, queued_refs
                    ):
                        validate_reward_batch(
                            rewards_row,
                            infos_row,
                            count=k,
                            context="sampled_evaluation",
                            references=[ref] * k,
                        )
                    if any(
                        len(result.get(key, [])) != len(queued_refs)
                        for key in ("rewards", "answer_keys", "responses")
                    ):
                        raise failure(
                            "worker_failure", "incomplete sampled-evaluation payload"
                        )
                    option_rows = result.get("option_ids")
                    if option_rows is None:
                        option_rows = [
                            [None] * len(reward_rows)
                            for reward_rows in result["rewards"]
                        ]
                    request_seed_rows = result.get("request_seeds_by_prompt")
                    if request_seed_rows is None:
                        request_seed_rows = [[] for _ in result["rewards"]]
                    if len(option_rows) != len(queued_refs) or len(
                        request_seed_rows
                    ) != len(queued_refs):
                        raise failure(
                            "worker_failure", "incomplete sampled-evaluation metadata"
                        )
                    for (
                        rewards,
                        answer_keys,
                        responses,
                        option_ids,
                        request_seeds,
                        verifier_infos,
                        ref,
                        prompt_index,
                        prompt,
                    ) in zip(
                        result["rewards"],
                        result["answer_keys"],
                        result["responses"],
                        option_rows,
                        request_seed_rows,
                        verifier_rows,
                        queued_refs,
                        queued_indices,
                        queued_prompts,
                    ):
                        if (
                            len(answer_keys) != k
                            or len(responses) != k
                            or len(option_ids) != k
                        ):
                            raise failure(
                                "worker_failure",
                                "incomplete sampled-evaluation response/key rows",
                            )
                        for key, info in zip(answer_keys, verifier_infos):
                            diagnostic = info.get("verifier")
                            if (
                                diagnostic is not None
                                and key != diagnostic["canonical_key"]
                            ):
                                raise failure(
                                    "worker_failure",
                                    "evaluation identity disagrees with verifier",
                                )
                        mode_count = _parse_answer_mode_count(ref)
                        prompt_metrics = _compute_mode_coverage_metrics(
                            rewards,
                            answer_keys,
                            mode_count,
                            _parse_public_seed_key(ref),
                        )
                        per_prompt_metrics.append(prompt_metrics)
                        prompt_outcomes.append(
                            {
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "reference": ref,
                                "answer_mode_count": mode_count,
                                "verifier_infos": verifier_infos,
                                "responses": [str(response) for response in responses],
                                "rewards": [float(value) for value in rewards],
                                "answer_keys": [
                                    None if key is None else str(key)
                                    for key in answer_keys
                                ],
                                "option_ids": [
                                    None if value is None else int(value)
                                    for value in option_ids
                                ],
                                "request_seeds_by_option": [
                                    int(value) for value in request_seeds
                                ],
                                "metrics": prompt_metrics,
                            }
                        )
                futs.clear()
                pending.clear()

        if not per_prompt_metrics:
            return {}, []
        mean = {
            key: float(
                sum(metrics[key] for metrics in per_prompt_metrics)
                / len(per_prompt_metrics)
            )
            for key in per_prompt_metrics[0]
        }
        return mean, prompt_outcomes
