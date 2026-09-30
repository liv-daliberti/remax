"""OAT lifecycle responsibilities, preserving historical state and ordering."""

from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from oat.interface import lp
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from ...core.checkpoints import (
    restore_rng,
    prepare_prompt_iterator,
    validate_checkpoint,
    validate_client_state,
)
from ...args import resolve_canonical_action_task
from .support import _derive_freeform_request_seed


class OatLifecycleMixin:
    def learn(self, learning_round: int):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier()
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.strategy,
        )
        if learning_round == 1:
            # The upstream PPO learner prints the full first training example.
            # For long math trajectories this can dominate startup time and bury
            # the actual learner-step logs without changing the optimization.
            self.strategy.print("Training example omitted for compact startup logs")

        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=(True if self.args.critic_type == "ppo" else False),
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        step_bar = tqdm(
            range(len(dataloader)),
            desc="Train steps",
            disable=not self.strategy.is_rank_0(),
        )
        learn_batch_time = []

        self.model.train()
        if self.critic is not None:
            self.critic.train()
        st = time.time()

        logging.info(
            f"start learn() buffer_len={len(self.pi_buffer)} dl_len={len(dataloader)}"
        )
        for data in dataloader:
            if local_sgd_steps > self.args.max_sgd_steps:
                break
            infos = self.learning_step(data)
            replication_factor = (
                self.strategy.world_size
                if bool(
                    getattr(self.args, "canonical_graph_learner_sampling", False)
                    or getattr(self.args, "replicated_freeform_sampling", False)
                )
                else 1
            )
            self.policy_sgd_step += (
                len(dataset)
                * self.args.num_ppo_epochs
                / self.args.train_batch_size_per_device
                / self.strategy.grad_acc_step
                / replication_factor
            )
            learn_batch_time.append(time.time() - st)
            step_bar.update()

            self.global_step += 1
            if self.global_step % self.strategy.grad_acc_step == 0:
                self.gradient_update_elapse = time.time() - self.gradient_update_st
                st = time.time()
                self.gradient_update_st = time.time()

                local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            "total_time": time.time() - st,
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        pending_canonical_prompt = getattr(
            self, "_canonical_entropy_prompt_pending", None
        )
        if pending_canonical_prompt is not None:
            train_info.update(
                self._compute_exact_canonical_sequence_entropy(pending_canonical_prompt)
            )
            self._canonical_entropy_prompt_pending = None
        self._update_xdr_tau_controller(train_info)
        self._update_maxent_alpha_controller(train_info)
        self._update_maxent_length_controller(train_info)
        self._update_online_canonical_alpha_controller(train_info)
        # Keep distributed logging reductions aligned even when optional metrics
        # are populated by different local minibatch conditions.
        train_info = {key: train_info[key] for key in sorted(train_info)}
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        logging.info("finish learn()")
        return train_info

    def learning_step(self, trajectory):
        return self._grpo_learning_step_with_progress(trajectory)

    def run(self):
        self._init(self.args, self.actors)
        self._init_local_actor_weight_sync()

        resume_step = 0
        next_step = 1
        start_prompt_epoch = 0
        start_batch_offset = 0
        resume_states: dict[str, Any] | None = None
        identity = getattr(self.args, "_remax_resume_identity", None)
        if self.args.resume_dir:
            if identity is not None:
                checkpoint = Path(self.args.resume_dir) / self.args.resume_tag
                committed = validate_checkpoint(checkpoint, identity)
            _, resume_states = self.strategy.load_ckpt(
                self.model.model,
                self.args.resume_dir,
                self.args.resume_tag,
            )
            resume_step = self._infer_resume_step(resume_states)
            if identity is not None:
                validate_client_state(
                    resume_states,
                    identity=identity,
                    step=committed["step"],
                    batches_per_epoch=len(self.prompts_dataloader),
                    epochs=int(self.args.num_prompt_epoch),
                )
                self._last_evaluated_global_step = resume_states[
                    "last_evaluated_global_step"
                ]
                self.model.model.micro_steps = resume_states["engine_micro_steps"]
                if len(resume_states["rollout_buffer"]) > self.pi_buffer.maxlen:
                    raise ValueError(
                        "checkpoint rollout buffer exceeds configured capacity"
                    )
                self.pi_buffer.extend(resume_states["rollout_buffer"])
            next_step, start_prompt_epoch, start_batch_offset = (
                self._restore_prompt_progress(resume_states, resume_step)
            )
        else:
            self.steps = 0
            self._prompt_batches_consumed_total = 0

        early_stop = False
        self.start_time = time.time()

        self.actor_info = {}
        train_info: dict[str, Any] = {}
        self._init_wandb(resume_states)

        if self.args.resume_dir:
            # The learner now holds the restored checkpoint, but actors were
            # initialized from ``args.pretrain``.  Synchronize before either
            # the initial evaluation or the first resumed rollout; otherwise
            # both would use stale base-model weights until the next update.
            logging.info(
                "resume actor weight sync start checkpoint_step=%s", self.steps
            )
            self.sync_params_to_actors()
            logging.info("resume actor weight sync done checkpoint_step=%s", self.steps)

        if bool(getattr(self.args, "eval_only", False)):
            # Measure the loaded policy through the ordinary training
            # evaluation path and stop. No rollout, optimizer step, export, or
            # resume checkpoint occurs, so the reported cell is a pure function
            # of the checkpoint and the decoding settings. Unlike the initial
            # evaluation below, this runs even under ``debug``: an eval-only
            # job that silently produced no evaluation would be indistinguishable
            # from a completed one.
            self.eval_and_log({}, eval=True, save=False, allow_scheduled_save=False)
            self._write_eval_only_marker()
            # Tear the program down exactly as the training path does. Without
            # this the measurement finishes but the job holds its GPU until the
            # scheduler's time limit.
            if self.strategy.is_rank_0():
                self._wandb.finish() if self._wandb else None
                lp.stop()
            return

        if not self.strategy.args.debug and not (
            identity is not None and self.args.resume_dir
        ):
            # The checkpoint already exists at a resumed boundary. Rewriting
            # the same multi-gigabyte model/optimizer state before the initial
            # evaluation creates avoidable I/O contention across a recovery
            # cohort and cannot improve recoverability.
            self.eval_and_log(
                {},
                eval=True,
                save=False,
                allow_scheduled_save=not bool(self.args.resume_dir),
            )

        # Construction, model loading, actor synchronization and logging may
        # consume RNG. Restore only after all startup work, before next sampling.
        if identity is not None and resume_states is not None:
            restore_rng(resume_states["rng_state"])
        self.steps = next_step
        self.gradient_update_st = time.time()
        for p_ep in range(start_prompt_epoch, self.args.num_prompt_epoch):
            batch_offset = start_batch_offset if p_ep == start_prompt_epoch else 0
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(p_ep)
                self.strategy.print(f"Set DistributedSampler at epoch {p_ep}")
            if batch_offset > 0:
                self.strategy.print(
                    "Skipping "
                    f"{batch_offset} already-consumed prompt batches in epoch {p_ep}"
                )
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
                initial=batch_offset,
            )

            prompt_iterator = (
                prepare_prompt_iterator(
                    self.prompts_dataloader, seed=self.args.seed, epoch=p_ep
                )
                if identity is not None
                else iter(self.prompts_dataloader)
            )
            for batch_idx, (processed_prompts, raw_prompts, refs) in enumerate(
                prompt_iterator
            ):
                if batch_idx < batch_offset:
                    continue
                if early_stop:
                    break
                self._prompt_batches_consumed_total += 1
                if resolve_canonical_action_task(self.args) != "none":
                    if len(processed_prompts) != 1:
                        raise RuntimeError(
                            "exact canonical entropy requires one current prompt"
                        )
                    self._canonical_entropy_prompt_pending = processed_prompts[0]

                learner_sampling = bool(
                    getattr(self.args, "canonical_graph_learner_sampling", False)
                )
                replicated_freeform_sampling = bool(
                    getattr(self.args, "replicated_freeform_sampling", False)
                )
                pre_learning_done = False
                if learner_sampling:
                    if self.steps % self.update_interval != 0:
                        raise RuntimeError(
                            "canonical learner sampling reached a rollout without an "
                            "immediate optimizer update"
                        )
                    logging.info(
                        "pre-learning start before canonical learner sampling step=%s",
                        self.steps,
                    )
                    self._pre_learning()
                    pre_learning_done = True
                    feedback_data, self.actor_info = (
                        self._sample_canonical_feedback_with_learner(
                            raw_prompts,
                            processed_prompts,
                            refs,
                        )
                    )
                    if bool(
                        getattr(
                            self.args,
                            "online_canonical_counterfactual_proposals",
                            False,
                        )
                    ):
                        self.actor_info = (
                            self._sample_and_admit_canonical_counterfactual_proposals(
                                raw_prompts=raw_prompts,
                                processed_prompts=processed_prompts,
                                refs=refs,
                                neutral_feedback=feedback_data,
                                actor_info=self.actor_info,
                            )
                        )
                elif replicated_freeform_sampling:
                    feedback_data, self.actor_info = (
                        self._sample_replicated_freeform_feedback(
                            raw_prompts,
                            processed_prompts,
                            refs,
                        )
                    )
                elif bool(getattr(self.args, "dapo_enabled", False)):
                    feedback_data, self.actor_info = (
                        self._collect_dapo_dynamic_feedback(
                            raw_prompts,
                            processed_prompts,
                            refs,
                        )
                    )
                elif identity is not None:
                    # Bind each vLLM request to its data position; engine-global
                    # sampling RNG/request counters cannot be restored reliably.
                    request_seed = _derive_freeform_request_seed(
                        base_seed=int(self.args.seed),
                        prompt_batch_index=int(self._prompt_batches_consumed_total),
                        stream="neutral",
                    )
                    # Cache hits change vLLM prefill batch shapes. A new actor
                    # has no cache, so every position-bound request starts cold.
                    if self.args.enable_prefix_caching:
                        self.actors[0].reset_prefix_cache()
                    started = time.time()
                    handle = self.actors[0].step(
                        raw_prompts, processed_prompts, refs, sampling_seed=request_seed
                    )
                    feedback_data = self.collector.ipc_client.deserialize_ipc(handle)
                    if len(feedback_data) != int(self.args.num_samples):
                        raise RuntimeError(
                            "resume-bound actor returned an incomplete sample group"
                        )
                    self.actor_info = self.collector.get_metrics(
                        time.time() - started, feedback_data
                    )
                else:
                    feedback_data, self.actor_info = self.collector.collect_feedback(
                        raw_prompts,
                        processed_prompts,
                        refs,
                        self._same_actor_group,
                    )
                dist.barrier()

                if feedback_data is None:
                    if pre_learning_done:
                        self._post_learning()
                    continue
                replication_factor = (
                    dist.get_world_size()
                    if (learner_sampling or replicated_freeform_sampling)
                    and dist.get_world_size() > 1
                    else 1
                )
                if len(feedback_data) % replication_factor != 0:
                    raise RuntimeError(
                        "replicated feedback does not divide across learner ranks"
                    )
                unique_local_count = len(feedback_data) // replication_factor
                self.prompt_consumed += unique_local_count

                self.process_feedback_data(feedback_data)
                if replication_factor > 1:
                    self.query_step -= len(feedback_data) - unique_local_count

                if (
                    self.args.dump_replay_every > 0
                    and self.steps % self.args.dump_replay_every == 0
                ):
                    if not self.strategy.is_rank_0():
                        dist.gather_object(self.pi_buffer)
                    else:
                        gather_all_buffer = [None] * self.strategy.world_size
                        dist.gather_object(self.pi_buffer, gather_all_buffer)
                        pd.to_pickle(
                            (processed_prompts, refs, gather_all_buffer),
                            os.path.join(
                                self.save_path,
                                f"buffer_step{self.steps:05}.pkl",
                            ),
                        )

                if self.steps % self.update_interval == 0:
                    if not pre_learning_done:
                        logging.info("pre-learning start step=%s", self.steps)
                        self._pre_learning()
                    logging.info("learn start step=%s", self.steps)
                    train_info = self.learn(self.steps // self.update_interval)
                    logging.info("post-learning start step=%s", self.steps)
                    self._post_learning()
                    logging.info("post-learning done step=%s", self.steps)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.sync_params_every == 0:
                        logging.info("sync params to actors start step=%s", self.steps)
                        self.sync_params_to_actors()
                        logging.info("sync params to actors done step=%s", self.steps)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                    if identity is not None:
                        self._append_resume_decision(feedback_data, raw_prompts)
                    logging.info("eval/log start step=%s", self.steps)
                    self.eval_and_log(train_info)
                    logging.info("eval/log done step=%s", self.steps)

                progress_bar.update()
                self.steps += 1

                if self.get_current_query() > self.args.max_queries:
                    early_stop = True

            self.prompt_epoch = p_ep + 1
            if early_stop:
                break

        self.eval_and_log(train_info, eval=True, save=True)

        if self.args.dump_all_buffer:  # For debug purpose.
            if not self.strategy.is_rank_0():
                dist.gather_object(self.all_buffer)
            else:
                gather_all_buffer = [None] * self.strategy.world_size
                dist.gather_object(self.all_buffer, gather_all_buffer)
                pd.to_pickle(
                    gather_all_buffer,
                    os.path.join(self.save_path, "all_buffer.pkl"),
                )

        self._finalize_successful_storage()

        if self.strategy.is_rank_0():
            self._wandb.finish() if self._wandb else None
            lp.stop()
