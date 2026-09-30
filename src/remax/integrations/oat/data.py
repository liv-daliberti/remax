"""OAT data responsibilities, preserving historical state and ordering."""

from __future__ import annotations
import logging
from oat.utils.data import PromptDataset, load_data_from_disk_or_hf
from torch.utils.data import DistributedSampler
from ...args import resolve_canonical_action_task
from modebench.templates import (
    CANONICAL_TASK_PROMPT_TEMPLATES,
    apply_prompt_template_to_example,
    collate_eval_prompt_items,
    validate_canonical_prompt_materialization,
)


class OatDataMixin:
    def prepare_data(self, strategy, tokenizer):
        prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data)
        prompts_data = prompt_dataset[self.args.train_split].select(
            range(min(self.args.max_train, len(prompt_dataset[self.args.train_split])))
        )
        raw_questions = list(prompts_data[self.args.input_key])

        # Prepare the data: templated questions & gt final answers.
        # Do not persist or consume Dataset.map caches here. This repository
        # intentionally runs several prompt contracts over the same frozen raw
        # rows; a stale cached transform would silently change the policy's
        # conditioning context while leaving the data path unchanged.
        prompts_data = prompts_data.map(
            apply_prompt_template_to_example,
            fn_kwargs={
                "input_key": self.args.input_key,
                "prompt_template": self.args.prompt_template,
            },
            load_from_cache_file=False,
            keep_in_memory=True,
        )
        canonical_task = resolve_canonical_action_task(self.args)
        if canonical_task != "none":
            allowed_templates = CANONICAL_TASK_PROMPT_TEMPLATES[canonical_task]
            if self.args.prompt_template not in allowed_templates:
                raise RuntimeError(
                    f"canonical task {canonical_task} requires one of "
                    f"{sorted(allowed_templates)}; got "
                    f"{self.args.prompt_template}"
                )
            validate_canonical_prompt_materialization(
                self.args.prompt_template,
                raw_questions,
                list(prompts_data[self.args.input_key]),
            )
            logging.info(
                "canonical %s prompt materialization verified: rows=%d "
                "template=%s dataset_map_cache=disabled",
                canonical_task,
                len(prompts_data),
                self.args.prompt_template,
            )

        self.prompts_dataset = PromptDataset(
            prompts_data,
            tokenizer,
            strategy,
            input_key=self.args.input_key,
            output_key=self.args.output_key,
            apply_chat_template=False,  # Because we have applied already.
            get_reference=True,
        )
        if canonical_task != "none" and len(self.prompts_dataset) != len(prompts_data):
            raise RuntimeError(
                "canonical prompt tokenization dropped rows under "
                f"prompt_max_length={self.args.prompt_max_length}: "
                f"rendered={len(prompts_data)} retained={len(self.prompts_dataset)}"
            )
        prompt_sampler = None
        if bool(
            getattr(self.args, "canonical_graph_learner_sampling", False)
            or getattr(self.args, "replicated_freeform_sampling", False)
        ):
            # Every learner rank follows the same audited prompt order. The
            # complete candidate group is partitioned only for backpropagation.
            prompt_sampler = DistributedSampler(
                self.prompts_dataset,
                num_replicas=1,
                rank=0,
                shuffle=True,
                seed=int(self.args.seed),
                drop_last=True,
            )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            self.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
            sampler=prompt_sampler,
        )
        self.eval_prompts_dataset = self.eval_prompts_dataloader = (
            None  # We use our own `self.eval_dataset_dict`.
        )

    def eval_dataloader_collate_fn(self, item_list):
        return collate_eval_prompt_items(
            item_list,
            prompt_template=self.args.prompt_template,
        )
