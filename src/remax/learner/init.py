"""Learner initialization for the single Dr.GRPO/xDr.GRPO path."""

from __future__ import annotations

import functools
import logging
import math
import os
from typing import List

from datasets import load_from_disk
from oat.actors.base import ActorBase
from oat.utils.ops import masked_sum

from ..args import ZeroMathArgs, resolve_canonical_action_task
from ..answer_options import AnswerOptionMITracker
from ..canonical_actions import (
    canonical_action_strings_by_position,
    resolve_canonical_action_space,
)
from ..maxent_controllers import (
    MaxEntDualController,
    MaxEntInverseController,
    MaxEntProportionalController,
)
from ..maxent_length_controller import MaxEntLengthController
from ..math_strategy_canonicalizer import MathStrategyCanonicalizer
from ..online_canonical_bank import OnlineCanonicalBank
from ..online_canonical_controller import (
    OnlineCanonicalDualController,
    OnlineCanonicalPolicyEntropyController,
)
from ..proposal_starvation import ProposalStarvationController
from ..runtime import patch_oat_learner_datetime, resolve_fixed_oat_exp_suffix
from ..semantic_shannon import SemanticShannonTracker
from ..trajectory_dataset import ZeroMathTrajectoryDataset
from ..verified_route_library import VerifiedRouteLibrary
from ..xdr_tau_controller import XdrTauController
from ..xdr_sac_dual_controller import XdrSacDualController


def build_maxent_controllers(
    args: ZeroMathArgs,
) -> tuple[
    MaxEntProportionalController
    | MaxEntDualController
    | MaxEntInverseController
    | None,
    MaxEntLengthController | None,
]:
    """Construct the independent entropy-alpha and expected-length controls."""

    canonical_task = resolve_canonical_action_task(args)
    canonical_max_entropy = None
    if canonical_task != "none":
        canonical_max_entropy = math.log(
            math.prod(
                len(support)
                for support in canonical_action_strings_by_position(canonical_task)
            )
        )
    if canonical_task != "none":
        configured_target = (
            float(args.maxent_control_target_entropy)
            if float(args.maxent_control_target_ratio) > 0
            else float(args.maxent_dual_target_entropy)
            if float(args.maxent_dual_target_ratio) > 0
            else None
        )
        if configured_target is not None and configured_target <= 0:
            raise ValueError(
                "adaptive canonical MaxEnt requires an explicit positive "
                "configured entropy target"
            )
        if (
            configured_target is not None
            and canonical_max_entropy is not None
            and configured_target > canonical_max_entropy + 1e-9
        ):
            raise ValueError(
                "canonical configured entropy target exceeds exact log-support "
                f"maximum {canonical_max_entropy:.12g}"
            )
    if bool(getattr(args, "maxent_observe_masked_mean_entropy", False)):
        # Explicitly requested: the target was measured from `train/entropy`,
        # so the controller must read that same estimator. This overrides the
        # objective-derived choice below, including the canonical one.
        controller_units = "masked_mean_token_nats_v1"
        controller_metric = "entropy"
    elif canonical_task != "none":
        controller_units = "canonical_action_nats_exact_v1"
        controller_metric = "canonical_exact_sequence_entropy"
    elif str(getattr(args, "maxent_objective", "sequence")) == (
        "conditional_token_mean"
    ):
        controller_units = "conditional_content_token_nats_mean_v1"
        controller_metric = "maxent_conditional_token_entropy"
    else:
        controller_units = "sequence_nats_v1"
        controller_metric = "maxent_sequence_entropy"
    alpha_controller: (
        MaxEntProportionalController
        | MaxEntDualController
        | MaxEntInverseController
        | None
    ) = None
    if float(args.maxent_control_target_ratio) > 0:
        alpha_controller = MaxEntProportionalController(
            base_alpha=float(args.maxent_alpha),
            max_alpha=float(args.maxent_control_max_alpha),
            target_ratio=float(args.maxent_control_target_ratio),
            warmup_steps=int(args.maxent_control_warmup_steps),
            ema_decay=float(args.maxent_control_ema_decay),
            gain=float(args.maxent_control_gain),
            configured_target_entropy=float(args.maxent_control_target_entropy),
            entropy_units=controller_units,
            observation_metric_key=controller_metric,
        )
    elif float(args.maxent_dual_target_ratio) > 0:
        alpha_controller = MaxEntDualController(
            base_alpha=float(args.maxent_alpha),
            min_alpha=float(args.maxent_dual_min_alpha),
            max_alpha=float(args.maxent_dual_max_alpha),
            target_ratio=float(args.maxent_dual_target_ratio),
            warmup_steps=int(args.maxent_dual_warmup_steps),
            alpha_lr=float(args.maxent_dual_alpha_lr),
            ema_decay=float(args.maxent_dual_ema_decay),
            configured_target_entropy=float(args.maxent_dual_target_entropy),
            entropy_units=controller_units,
            observation_metric_key=controller_metric,
        )
    elif bool(getattr(args, "maxent_inverse_adaptation", False)):
        alpha_controller = MaxEntInverseController(
            base_alpha=float(args.maxent_alpha),
            warmup_steps=int(args.maxent_inverse_warmup_steps),
            ema_decay=float(args.maxent_inverse_ema_decay),
            entropy_units=controller_units,
            observation_metric_key=controller_metric,
        )

    length_controller = None
    if float(args.maxent_length_target) > 0:
        length_controller = MaxEntLengthController(
            target_length=float(args.maxent_length_target),
            init_lambda=float(args.maxent_length_lambda_init),
            max_lambda=float(args.maxent_length_lambda_max),
            dual_lr=float(args.maxent_length_dual_lr),
            ema_decay=float(args.maxent_length_ema_decay),
        )
    return alpha_controller, length_controller


def build_semantic_shannon_tracker(
    args: ZeroMathArgs,
) -> SemanticShannonTracker | None:
    """Construct the semantic estimator, including explicit zero-dose controls."""

    coefficient = float(getattr(args, "semantic_shannon_coef", 0.0) or 0.0)
    zero_coefficient_control = bool(
        getattr(args, "semantic_shannon_allow_zero_coefficient_control", False)
    )
    if coefficient <= 0.0 and not zero_coefficient_control:
        return None
    return SemanticShannonTracker(
        coefficient=coefficient,
        surprisal_clip=float(args.semantic_shannon_surprisal_clip),
        pseudocount=float(args.semantic_shannon_pseudocount),
        quality_gated_advantage=bool(
            getattr(args, "semantic_shannon_quality_gated_advantage", False)
        ),
        quality_gated_cap=float(
            getattr(args, "semantic_shannon_quality_gated_cap", 0.05)
        ),
        success_conditioned_signed_advantage=bool(
            getattr(
                args,
                "semantic_shannon_success_conditioned_signed_advantage",
                False,
            )
        ),
        success_conditioned_signed_cap=float(
            getattr(args, "semantic_shannon_success_conditioned_signed_cap", 0.05)
        ),
        success_conditioned_group_centered_advantage=bool(
            getattr(
                args,
                "semantic_shannon_success_conditioned_group_centered_advantage",
                False,
            )
        ),
        success_conditioned_verified_support_advantage=bool(
            getattr(
                args,
                "semantic_shannon_success_conditioned_verified_support_advantage",
                False,
            )
        ),
    )


class ZeroMathInitMixin:
    """Initialize OAT state, exact-answer data, and Dr.GRPO normalization."""

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
        self._xdr_tau_controller: XdrTauController | None = None
        controller_base_tau = float(args.xdr_tau)
        if float(args.xdr_tau_control_target_ratio) > 0:
            self._xdr_tau_controller = XdrTauController(
                base_tau=controller_base_tau,
                min_tau=float(args.xdr_tau_control_min),
                target_ratio=float(args.xdr_tau_control_target_ratio),
                warmup_steps=int(args.xdr_tau_control_warmup_steps),
                ema_decay=float(args.xdr_tau_control_ema_decay),
                gain=float(args.xdr_tau_control_gain),
            )
        elif float(args.xdr_sac_dual_target_ratio) > 0:
            self._xdr_tau_controller = XdrSacDualController(
                base_tau=controller_base_tau,
                min_tau=float(args.xdr_sac_dual_min_tau),
                max_tau=float(args.xdr_sac_dual_max_tau),
                target_ratio=float(args.xdr_sac_dual_target_ratio),
                warmup_steps=int(args.xdr_sac_dual_warmup_steps),
                alpha_lr=float(args.xdr_sac_dual_alpha_lr),
            )
        (
            self._maxent_alpha_controller,
            self._maxent_length_controller,
        ) = build_maxent_controllers(args)
        if isinstance(self._maxent_alpha_controller, MaxEntInverseController):
            logging.info(
                "direct inverse MaxEnt control enabled: "
                "reference_alpha=%.9g warmup_steps=%d ema_decay=%.6g "
                "sensor=%s units=%s "
                "rule=unprojected_warmup_inverse_direct_entropy_v1",
                float(args.maxent_alpha),
                int(args.maxent_inverse_warmup_steps),
                float(args.maxent_inverse_ema_decay),
                self._maxent_alpha_controller.observation_metric_key,
                self._maxent_alpha_controller.entropy_units,
            )
        self._diayn_mi_tracker: AnswerOptionMITracker | None = None
        if int(getattr(args, "diayn_num_options", 0) or 0) > 1:
            self._diayn_mi_tracker = AnswerOptionMITracker(
                num_options=int(args.diayn_num_options),
                ema_decay=float(args.diayn_mi_ema_decay),
                smoothing=float(args.diayn_mi_smoothing),
                bonus_clip=float(args.diayn_mi_bonus_clip),
                leave_one_out=bool(args.diayn_mi_leave_one_out),
            )
            logging.info(
                "DIAYN answer-option MI enabled: options=%s beta=%.6g "
                "ema_decay=%.3f correct_only=%s leave_one_out=%s",
                int(args.diayn_num_options),
                float(args.diayn_mi_beta),
                float(args.diayn_mi_ema_decay),
                bool(args.diayn_mi_correct_only),
                bool(args.diayn_mi_leave_one_out),
            )
        semantic_shannon_coef = float(
            getattr(args, "semantic_shannon_coef", 0.0) or 0.0
        )
        self._semantic_shannon_tracker = build_semantic_shannon_tracker(args)
        if self._semantic_shannon_tracker is not None:
            logging.info(
                "semantic Shannon shaping enabled: coefficient=%.6g "
                "surprisal_clip=%.6g pseudocount=%.6g "
                "separate_advantage=%s quality_gated_advantage=%s "
                "success_conditioned_signed_advantage=%s "
                "success_conditioned_group_centered_advantage=%s "
                "success_conditioned_verified_support_advantage=%s "
                "verified_support_include_replay_bank=%s "
                "semantic_estimator=mode_selected coefficient_control=fixed",
                semantic_shannon_coef,
                float(args.semantic_shannon_surprisal_clip),
                float(args.semantic_shannon_pseudocount),
                bool(getattr(args, "semantic_shannon_separate_advantage", False)),
                bool(getattr(args, "semantic_shannon_quality_gated_advantage", False)),
                bool(
                    getattr(
                        args,
                        "semantic_shannon_success_conditioned_signed_advantage",
                        False,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "semantic_shannon_success_conditioned_group_centered_advantage",
                        False,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "semantic_shannon_success_conditioned_verified_support_advantage",
                        False,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "semantic_shannon_verified_support_include_replay_bank",
                        False,
                    )
                ),
            )
        self._semantic_rms_controller = None
        if semantic_shannon_coef > 0 and bool(
            getattr(args, "semantic_rms_control", False)
        ):
            from ..semantic_rms_controller import SemanticRmsController

            self._semantic_rms_controller = SemanticRmsController(
                base_coefficient=semantic_shannon_coef,
                target_ratio=float(args.semantic_rms_target_ratio),
                min_coefficient=float(args.semantic_rms_min_coefficient),
                max_coefficient=float(args.semantic_rms_max_coefficient),
                ema_decay=float(args.semantic_rms_ema_decay),
                gain=float(args.semantic_rms_gain),
                max_step_ratio=float(args.semantic_rms_max_step_ratio),
                warmup_steps=int(args.semantic_rms_warmup_steps),
                min_eligible_fraction=float(args.semantic_rms_min_eligible_fraction),
            )
            logging.info(
                "adaptive semantic MaxEnt enabled: target_ratio=%.6g "
                "bounds=[%.6g, %.6g] gain=%.6g ema=%.6g warmup=%d "
                "start_coefficient=%.6g",
                float(args.semantic_rms_target_ratio),
                float(args.semantic_rms_min_coefficient),
                float(args.semantic_rms_max_coefficient),
                float(args.semantic_rms_gain),
                float(args.semantic_rms_ema_decay),
                int(args.semantic_rms_warmup_steps),
                semantic_shannon_coef,
            )

        self._online_canonical_bank: OnlineCanonicalBank | None = None
        self._proposal_starvation_controller: (
            ProposalStarvationController | None
        ) = None
        self._verified_route_library: VerifiedRouteLibrary | None = None
        self._math_strategy_canonicalizer: MathStrategyCanonicalizer | None = None
        self._online_canonical_alpha_controller: (
            OnlineCanonicalDualController
            | OnlineCanonicalPolicyEntropyController
            | None
        ) = None
        online_canonical_bank_alpha = float(
            getattr(args, "online_canonical_bank_alpha", 0.0) or 0.0
        )
        online_canonical_replay = bool(getattr(args, "online_canonical_replay", False))
        online_canonical_objective_active = (
            online_canonical_bank_alpha > 0.0 or online_canonical_replay
        )
        verified_discovery_tracking = bool(
            getattr(args, "verified_discovery_tracking", True)
        )
        if online_canonical_objective_active or verified_discovery_tracking:
            self._online_canonical_bank = OnlineCanonicalBank(
                entropy_alpha=online_canonical_bank_alpha,
                pseudocount=float(args.online_canonical_bank_pseudocount),
                surprisal_clip=float(args.online_canonical_bank_surprisal_clip),
                retain_exemplars=online_canonical_replay,
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
            if online_canonical_objective_active:
                logging.info(
                    "online canonical bank enabled: alpha=%.6g "
                    "pseudocount=%.6g surprisal_clip=%.6g key_mode=%s "
                    "estimator=verified_prompt_bank_loo_v1",
                    online_canonical_bank_alpha,
                    float(args.online_canonical_bank_pseudocount),
                    float(args.online_canonical_bank_surprisal_clip),
                    str(args.online_canonical_key_mode),
                )
            else:
                logging.info(
                    "passive verified-discovery tracking enabled for ordinary "
                    "Dr.GRPO: key_mode=%s objective_influence=zero",
                    str(args.online_canonical_key_mode),
                )
            if online_canonical_replay:
                logging.info(
                    "verified canonical replay enabled: "
                    "balance_alpha=%.6g mass_alpha=%.6g capacity=%d "
                    "global_groups_per_step=%d global_bootstrap_steps=%d "
                    "objective=%s key_weighting=%s coefficient_control=fixed "
                    "gold_support_feedback=none",
                    float(args.online_canonical_replay_alpha),
                    float(args.online_canonical_replay_mass_alpha),
                    int(args.online_canonical_replay_capacity),
                    int(args.online_canonical_replay_global_groups_per_step),
                    int(args.online_canonical_replay_global_bootstrap_steps),
                    str(args.online_canonical_replay_objective),
                    str(args.online_canonical_replay_key_weighting),
                )
                if bool(
                    getattr(
                        args,
                        "online_canonical_counterfactual_proposals",
                        False,
                    )
                ):
                    if bool(
                        getattr(
                            args,
                            "online_canonical_counterfactual_starvation_fallback",
                            False,
                        )
                    ):
                        self._proposal_starvation_controller = ProposalStarvationController(
                            base_max_attempts=int(
                                args.online_canonical_counterfactual_max_attempts
                            ),
                            patience_updates=int(
                                args.online_canonical_counterfactual_starvation_patience_updates
                            ),
                            fallback_max_attempts=int(
                                args.online_canonical_counterfactual_starvation_fallback_max_attempts
                            ),
                            burst_updates=int(
                                args.online_canonical_counterfactual_starvation_burst_updates
                            ),
                            cooldown_updates=int(
                                args.online_canonical_counterfactual_starvation_cooldown_updates
                            ),
                        )
                        logging.info(
                            "target-free proposal starvation fallback enabled: "
                            "patience_eligible_updates=%d base_attempts=%d "
                            "fallback_attempts=%d burst_updates=%d "
                            "cooldown_updates=%d reset=new_verified_admission_only "
                            "gold_support_feedback=none "
                            "desired_mode_count_feedback=none "
                            "evaluation_feedback=none",
                            int(
                                args.online_canonical_counterfactual_starvation_patience_updates
                            ),
                            int(args.online_canonical_counterfactual_max_attempts),
                            int(
                                args.online_canonical_counterfactual_starvation_fallback_max_attempts
                            ),
                            int(
                                args.online_canonical_counterfactual_starvation_burst_updates
                            ),
                            int(
                                args.online_canonical_counterfactual_starvation_cooldown_updates
                            ),
                        )
                    logging.info(
                        "verified open-set proposals enabled: "
                        "proposal_groups_per_eligible_prompt=up_to_%d "
                        "proposal_width=num_samples "
                        "original_prompt_temperature=%.6g "
                        "proposal_temperature_step=%s "
                        "transform_proposals=%s "
                        "exact_grammar_transforms=%s "
                        "admission=novel_validator_and_task_positive_only "
                        "proposal_rows_to_ppo=0 "
                        "objective_support_separated=%s "
                        "gold_support_feedback=none "
                        "desired_mode_count_feedback=none "
                        "evaluation_feedback=none",
                        int(args.online_canonical_counterfactual_max_attempts),
                        float(
                            args.online_canonical_counterfactual_sampling_temperature
                        ),
                        (
                            "0"
                            if str(args.online_canonical_key_mode) == "verified_route"
                            else "0.2"
                        ),
                        bool(
                            getattr(
                                args,
                                "online_canonical_counterfactual_transform_proposals",
                                True,
                            )
                        ),
                        bool(
                            getattr(
                                args,
                                "online_canonical_counterfactual_exact_grammar_transforms",
                                False,
                            )
                        ),
                        bool(
                            getattr(
                                args,
                                "online_canonical_counterfactual_separate_objective_support",
                                False,
                            )
                        ),
                    )
            if str(args.online_canonical_key_mode) == "verified_route":
                self._verified_route_library = VerifiedRouteLibrary(
                    replay_groups_per_step=(
                        int(args.online_canonical_replay_global_groups_per_step)
                        if online_canonical_replay
                        else 0
                    ),
                    replay_capacity_per_route=int(
                        args.verified_route_replay_capacity_per_route
                    ),
                    recurring_min_neutral_prompts=int(
                        args.verified_route_recurring_min_neutral_prompts
                    ),
                    proposal_max_mean_logprob_drop=float(
                        args.verified_route_proposal_max_mean_logprob_drop
                    ),
                )
                logging.info(
                    "verified route library enabled: replay_groups_per_step=%d "
                    "capacity_per_route=%d recurring_min_neutral_prompts=%d "
                    "proposal_max_mean_logprob_drop=%.6g",
                    self._verified_route_library.replay_groups_per_step,
                    self._verified_route_library.replay_capacity_per_route,
                    (self._verified_route_library.recurring_min_neutral_prompts),
                    (self._verified_route_library.proposal_max_mean_logprob_drop),
                )
            if str(args.online_canonical_key_mode) == "math_strategy_qwen72":
                self._math_strategy_canonicalizer = MathStrategyCanonicalizer(
                    endpoint=str(args.math_strategy_endpoint),
                    model=str(args.math_strategy_model),
                    timeout_seconds=int(args.math_strategy_timeout_seconds),
                    max_workers=int(args.math_strategy_workers),
                    max_item_chars=int(args.math_strategy_max_item_chars),
                    allow_unstructured_menu_inference=bool(
                        getattr(
                            args,
                            "math_strategy_allow_unstructured_inference",
                            False,
                        )
                    ),
                )
                logging.info(
                    "validator-bound MATH strategy canonicalizer enabled: "
                    "model=%s endpoint=%s integrity_passes=2 "
                    "runtime_partition_passes=0 "
                    "runtime_pairwise_veto_passes=0 "
                    "seeds=470721,470722 "
                    "support_rule=exact_precalibrated_menu_combo_v18 "
                    "fail_closed task_reward_gate=%s "
                    "unstructured_menu_inference=%s",
                    str(args.math_strategy_model),
                    str(args.math_strategy_endpoint),
                    bool(
                        getattr(
                            args,
                            "math_strategy_gate_task_reward",
                            False,
                        )
                    ),
                    bool(
                        getattr(
                            args,
                            "math_strategy_allow_unstructured_inference",
                            False,
                        )
                    ),
                )
            online_canonical_dual_target = float(
                getattr(args, "online_canonical_dual_target_ratio", 0.0) or 0.0
            )
            if online_canonical_dual_target > 0:
                self._online_canonical_alpha_controller = OnlineCanonicalDualController(
                    base_alpha=online_canonical_bank_alpha,
                    min_alpha=float(args.online_canonical_dual_min_alpha),
                    max_alpha=float(args.online_canonical_dual_max_alpha),
                    target_ratio=online_canonical_dual_target,
                    alpha_lr=float(args.online_canonical_dual_alpha_lr),
                    ema_decay=float(args.online_canonical_dual_ema_decay),
                )
                logging.info(
                    "online canonical Haarnoja control enabled: "
                    "normalized_target=%.6g base_alpha=%.6g "
                    "min_alpha=%.6g max_alpha=%.6g alpha_lr=%.6g "
                    "ema_decay=%.6g "
                    "sensor=exact_postupdate_H_over_log_support_v1",
                    online_canonical_dual_target,
                    online_canonical_bank_alpha,
                    float(args.online_canonical_dual_min_alpha),
                    float(args.online_canonical_dual_max_alpha),
                    float(args.online_canonical_dual_alpha_lr),
                    float(args.online_canonical_dual_ema_decay),
                )
            elif bool(
                getattr(
                    args,
                    "online_canonical_policy_entropy_adaptation",
                    False,
                )
            ):
                self._online_canonical_alpha_controller = (
                    OnlineCanonicalPolicyEntropyController(
                        base_alpha=online_canonical_bank_alpha,
                        warmup_steps=int(
                            args.online_canonical_policy_entropy_warmup_steps
                        ),
                        ema_decay=float(args.online_canonical_policy_entropy_ema_decay),
                    )
                )
                logging.info(
                    "online canonical policy-entropy adaptation enabled: "
                    "reference_alpha=%.6g warmup_steps=%d "
                    "ema_decay=%.6g "
                    "sensor=masked_mean_policy_token_entropy_v1 "
                    "rule=unprojected_relative_to_own_warmup_v1",
                    online_canonical_bank_alpha,
                    int(args.online_canonical_policy_entropy_warmup_steps),
                    float(args.online_canonical_policy_entropy_ema_decay),
                )
