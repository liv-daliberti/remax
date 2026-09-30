"""Initialize OAT infrastructure and the maintained verified bank."""

from __future__ import annotations
import functools
import logging
import os
from typing import List
from datasets import load_from_disk
from oat.actors.base import ActorBase
from oat.utils.ops import masked_sum
from ...args import ZeroMathArgs, resolve_canonical_action_task
from ...canonical_actions import resolve_canonical_action_space
from ...runtime import patch_oat_learner_datetime, resolve_fixed_oat_exp_suffix
from ...trajectory_dataset import ZeroMathTrajectoryDataset

from .selection import historical_reasons
from ...core.bank import OnlineCanonicalBank


class ZeroMathInitMixin:
    def _init(self, args: ZeroMathArgs, actors: List[ActorBase]) -> None:
        requested_use_wb = args.use_wb
        args.use_wb = False
        fixed_exp_suffix = resolve_fixed_oat_exp_suffix()
        if fixed_exp_suffix:
            logging.info("Using fixed OAT experiment suffix %s", fixed_exp_suffix)
        with patch_oat_learner_datetime(fixed_exp_suffix):
            super()._init(args, actors)

        self.dataset_builder = ZeroMathTrajectoryDataset
        args.use_wb = requested_use_wb
        if hasattr(self, "strategy") and hasattr(self.strategy, "args"):
            self.strategy.args.use_wb = requested_use_wb
        self.eval_dataset_dict = load_from_disk(args.eval_data)
        if args.test_split != "all":
            self.eval_dataset_dict = {
                key: value
                for key, value in self.eval_dataset_dict.items()
                if key in args.test_split
            }
        if bool(args.online_evaluation) and not self.eval_dataset_dict:
            raise ValueError(
                "online evaluation selected zero datasets for "
                f"test_split={args.test_split!r}"
            )
        self.args = args
        self._requested_use_wb = requested_use_wb
        self._wandb = None
        self._wandb_run_id: str | None = None
        self._wandb_run_name = os.path.basename(self.save_path)
        self.masked_aggregator = functools.partial(
            masked_sum, constant_normalizer=args.generate_max_length
        )
        self._baseline_grad_norm_logging_disabled_warned = False
        self._invalid_scoring_token_ids_warned_contexts = set()
        self._invalid_logit_columns_warned_contexts = set()
        self._prompt_batches_consumed_total = 0
        self._canonical_action_space = None
        self._canonical_action_token_ids: tuple[int, ...] | None = None
        self._canonical_action_token_ids_by_position: (
            tuple[tuple[int, ...], ...] | None
        ) = None
        canonical_task = resolve_canonical_action_task(args)
        if canonical_task != "none":
            self._canonical_action_space = resolve_canonical_action_space(
                self.tokenizer, canonical_task
            )
            self._canonical_action_token_ids = (
                self._canonical_action_space.union_token_ids
            )
            self._canonical_action_token_ids_by_position = (
                self._canonical_action_space.token_ids_by_position
            )
            logging.info(
                "canonical %s policy: position_token_ids=%s sequence_count=%d "
                "max_entropy=%.9f horizon=%d "
                "tokenizer=%s tokenizer_class=%s vocab_size=%s",
                canonical_task,
                self._canonical_action_token_ids_by_position,
                self._canonical_action_space.sequence_count,
                self._canonical_action_space.max_sequence_entropy,
                self._canonical_action_space.horizon,
                getattr(self.tokenizer, "name_or_path", args.pretrain),
                type(self.tokenizer).__name__,
                len(self.tokenizer),
            )

        reasons = historical_reasons(self)
        self._remax_adapter = "historical" if reasons else "maintained"
        self._remax_adapter_reasons = reasons
        logging.info(
            "ReMax adapter=%s historical_reasons=%s", self._remax_adapter, reasons
        )
        if reasons:
            if getattr(args, "_remax_resume_identity", None) is not None:
                raise RuntimeError(
                    f"maintained recipe requires historical adapter: {reasons}"
                )
            from ...experiments.oat.initialization import initialize_extensions

            initialize_extensions(self, args)
        else:
            initialize_replay(self, args)


def initialize_replay(self, args):
    """Construct maintained state; empty historical fields preserve checkpoint schema."""
    for name in (
        "_xdr_tau_controller",
        "_maxent_alpha_controller",
        "_maxent_length_controller",
        "_diayn_mi_tracker",
        "_semantic_shannon_tracker",
        "_semantic_rms_controller",
        "_online_canonical_bank",
        "_proposal_starvation_controller",
        "_verified_route_library",
        "_math_strategy_canonicalizer",
        "_online_canonical_alpha_controller",
    ):
        setattr(self, name, None)
    if args.online_canonical_replay or getattr(
        args, "verified_discovery_tracking", True
    ):
        self._online_canonical_bank = OnlineCanonicalBank(
            entropy_alpha=0.0,
            pseudocount=float(args.online_canonical_bank_pseudocount),
            surprisal_clip=float(args.online_canonical_bank_surprisal_clip),
            retain_exemplars=bool(args.online_canonical_replay),
            replay_capacity=int(args.online_canonical_replay_capacity),
            global_replay_groups_per_step=int(
                args.online_canonical_replay_global_groups_per_step
            ),
            global_replay_bootstrap_steps=int(
                args.online_canonical_replay_global_bootstrap_steps
            ),
            separate_proposal_objective_support=bool(
                getattr(
                    args,
                    "online_canonical_counterfactual_separate_objective_support",
                    False,
                )
            ),
            proposal_replay_priority_visits=int(
                getattr(
                    args,
                    "online_canonical_proposal_replay_priority_visits",
                    0,
                )
            ),
            proposal_replay_priority_multiplier=float(
                getattr(
                    args,
                    "online_canonical_proposal_replay_priority_multiplier",
                    1.0,
                )
            ),
            proposal_retention_tracking=bool(
                getattr(
                    args,
                    "online_canonical_proposal_retention_tracking",
                    False,
                )
            ),
            proposal_adaptive_retention_priority=bool(
                getattr(
                    args,
                    "online_canonical_proposal_adaptive_retention_priority",
                    False,
                )
            ),
            proposal_retention_max_missed_rollout_opportunities=int(
                getattr(
                    args,
                    "online_canonical_proposal_retention_max_missed_rollout_opportunities",
                    2,
                )
            ),
            proposal_retention_max_mean_logprob_drop=float(
                getattr(
                    args,
                    "online_canonical_proposal_retention_max_mean_logprob_drop",
                    0.5,
                )
            ),
            proposal_retention_refresh_visits=int(
                getattr(
                    args,
                    "online_canonical_proposal_retention_refresh_visits",
                    4,
                )
            ),
            proposal_retention_score_cooldown_observations=int(
                getattr(
                    args,
                    "online_canonical_proposal_retention_score_cooldown_observations",
                    2,
                )
            ),
        )
