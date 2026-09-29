"""Arguments for the Dr.GRPO/xDr.GRPO training surface."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from oat.algorithms.ppo import PPOArgs

from modebench.templates import (
    CANONICAL_DIGIT_TEMPLATE_ROLES,
    CANONICAL_TASK_PROMPT_TEMPLATES,
    prompt_template_role,
)


@dataclass
class ZeroMathArgs(PPOArgs):
    """OAT PPO arguments plus the small set of knobs used in this project."""

    prompt_template: Literal[
        "qwen_boxed",
        "qwen_countdown_digits",
        "qwen_graph_digits",
        "qwen_pantry_support_mask",
        "qwen_math",
        "qwen_math_route",
        "qwen_level2_countdown",
        "qwen_level2_python_factors",
        "qwen_level2_mathir",
        "qwen_level2_pantry",
        # Falcon-surface twins render the same contracts for the
        # Falcon3-*-Instruct external-validity cohort.
        "falcon_boxed",
        "falcon_countdown_digits",
        "falcon_graph_digits",
        "falcon_pantry_support_mask",
        "falcon_math",
        "falcon_math_route",
        "no",
        "r1",
    ] = field(default="qwen_math")
    test_split: str = "all"
    verifier_version: Literal["fast", "math_verify"] = field(default="fast")
    modebench_domain: Literal[
        "none", "countdown", "graph_coloring", "python_factors", "mathir", "pantry_plan"
    ] = field(default="none")
    modebench_syntax_profile: Literal[
        "none", "domain_legal_v1", "countdown_legal_v3"
    ] = field(default="none")

    # Dr.GRPO is recovered exactly when xdr_tau is infinite. Finite values
    # apply detached candidate weights softmax(U/tau); zero is the exact
    # argmax-set limit.
    xdr_tau: float = math.inf
    # Replace the ordinary group-centered binary task advantage with the
    # finite-rollout MaxRL estimator. Verified replay remains a separate
    # additive current-policy likelihood objective.
    maxrl_task_objective: bool = False
    # E44 composes reward-directed xDr aggregation with a separately added
    # semantic policy-gradient advantage. When enabled, xDr's detached row
    # weights are computed from the ordinary task advantage captured before
    # semantic augmentation; the actor still optimizes the combined advantage.
    # This prevents semantic MaxEnt from being counted once in the advantage
    # and again through the aggregation weights.
    xdr_task_advantage_weights: bool = False
    xdr_mode_adaptive: bool = False
    # Optional label-free feedback controller over xDr's aggregation
    # temperature. A positive target ratio enables the controller; zero keeps
    # the fixed-tau treatment exactly unchanged.
    xdr_tau_control_target_ratio: float = 0.0
    xdr_tau_control_warmup_steps: int = 64
    xdr_tau_control_min: float = 0.005
    xdr_tau_control_ema_decay: float = 0.9
    xdr_tau_control_gain: float = 20.0
    # Distinct Haarnoja-style dual controller.  It learns a positive inverse-
    # tau strength with Adam from the signed entropy-target error, rather than
    # applying the one-sided proportional rule above.
    xdr_sac_dual_target_ratio: float = 0.0
    xdr_sac_dual_warmup_steps: int = 64
    xdr_sac_dual_min_tau: float = 0.005
    xdr_sac_dual_max_tau: float = 0.5
    xdr_sac_dual_alpha_lr: float = 0.003

    # Direct on-policy maximum-entropy Dr.GRPO.  The coefficient multiplies
    # raw completion-policy sequence entropy H(pi(.|x)).  Entropy is
    # differentiated at sampled prefixes rather than injected into the
    # group-relative reward advantage.  The two optional controllers observe
    # the same raw sequence-entropy quantity used by the actor objective.
    maxent_alpha: float = 0.0
    # ``sequence`` is the literal trajectory-entropy formulation retained for
    # E11--E20. ``conditional_token_mean`` is E21's free-form, length-neutral
    # formulation: at each sampled response state it maximizes entropy over
    # non-EOS content tokens, does not importance-differentiate state
    # visitation, and averages within each response before averaging rows.
    maxent_objective: Literal["sequence", "conditional_token_mean"] = "sequence"
    maxent_control_target_ratio: float = 0.0
    maxent_control_target_entropy: float = 0.0
    maxent_control_warmup_steps: int = 64
    maxent_control_max_alpha: float = 0.5
    maxent_control_ema_decay: float = 0.9
    maxent_control_gain: float = 1.0
    # Regulate the dual controller on the masked-mean token entropy the run
    # already reports as `train/entropy`, rather than on the objective's own
    # estimator. Required whenever the target was measured from that telemetry.
    maxent_observe_masked_mean_entropy: bool = False
    maxent_dual_target_ratio: float = 0.0
    maxent_dual_target_entropy: float = 0.0
    maxent_dual_warmup_steps: int = 64
    maxent_dual_min_alpha: float = 0.005
    maxent_dual_max_alpha: float = 0.5
    maxent_dual_alpha_lr: float = 0.003
    # Smooth the per-prompt-group entropy sensor before dual Adam. A zero
    # decay explicitly recovers the historical instantaneous-feedback rule.
    maxent_dual_ema_decay: float = 0.7
    # Projection-free, memoryless inverse control of the coefficient on the
    # direct MaxEnt objective. It calibrates exclusively from the same entropy
    # quantity differentiated by that objective, never from evaluation labels
    # or the canonical bank. The reference dose is maxent_alpha.
    maxent_inverse_adaptation: bool = False
    maxent_inverse_warmup_steps: int = 64
    maxent_inverse_ema_decay: float = 0.9
    # Optional constrained-MaxEnt length dual. The actor optimizes
    # E[R] + alpha H - lambda(E[L] - target), with raw generated response
    # length L and a separate projected dual controller over lambda. It may
    # accompany fixed, proportional, or Haarnoja-controlled entropy alpha; a
    # zero target leaves those established MaxEnt paths unchanged.
    maxent_length_target: float = 0.0
    maxent_length_lambda_init: float = 0.0
    maxent_length_lambda_max: float = 0.02
    maxent_length_ema_decay: float = 0.9
    maxent_length_dual_lr: float = 0.0002

    # DIAYN-style answer-option binding.  When enabled, each rollout group is
    # split across K latent answer options by augmenting the prompt with z.
    # The learner adds beta * (log q(z | answer_repr) - log 1/K) to terminal
    # reward, where q is an EMA discriminator over extracted final answers.
    diayn_num_options: int = 0
    diayn_mi_beta: float = 0.0
    diayn_mi_ema_decay: float = 0.9
    diayn_mi_smoothing: float = 1.0
    diayn_mi_bonus_clip: float = 5.0
    diayn_mi_correct_only: bool = True
    diayn_mi_leave_one_out: bool = False

    # Group-local outcome-collision shaping for free-form Dr.GRPO. Every
    # observed canonical answer, including one shared INVALID key for parse
    # failures, receives -(coefficient / G) for each same-key peer in its
    # candidate group. Zero recovers the unmodified reward exactly.
    outcome_collision_coef: float = 0.0
    # Opt-in free-form semantic policy-gradient variant. When true, the
    # outcome-collision penalty is not added to terminal reward before
    # Dr.GRPO's group baseline. Instead, the learner adds that detached
    # leave-one-out penalty once to the already group-centered sequence
    # advantage. False preserves E37's reward-shaping behavior exactly.
    outcome_collision_outside_centering: bool = False

    # Prompt-specific predictive Shannon-surprisal shaping over observed
    # canonical answers. Counts from prior groups and current leave-one-out
    # peers define a smoothed answer probability with one unseen bucket. The
    # bounded non-positive bonus preserves task-reward ordering. Zero disables
    # the tracker and recovers the unmodified reward exactly.
    semantic_shannon_coef: float = 0.0
    # Explicit component-ablation surface. When enabled, the semantic estimator
    # may remain fully configured at coefficient zero so a control can share
    # the exact runtime variant and flag surface with a semantic-on arm.
    semantic_shannon_allow_zero_coefficient_control: bool = False
    semantic_shannon_surprisal_clip: float = 5.0
    semantic_shannon_pseudocount: float = 1.0
    # E41 keeps ordinary task reward as the only input to Dr.GRPO's group
    # baseline. It adds a detached semantic advantage afterward, centered
    # under the prompt-local predictive distribution rather than the current
    # candidate group's empirical mean. False preserves E38 exactly.
    semantic_shannon_separate_advantage: bool = False
    # Optional safety gate for a future separate-advantage treatment. Only
    # reward-positive rows with parseable answer keys can receive semantic
    # pressure or enter the predictive history. Negative raw semantic
    # advantages are zeroed, and positive ones are capped explicitly.
    # False preserves both E38 and E41 exactly.
    semantic_shannon_quality_gated_advantage: bool = False
    semantic_shannon_quality_gated_cap: float = 0.05
    # Success-conditioned semantic MaxEnt uses the catalogue-free open-set
    # predictor with one structural unseen bucket and this fixed coefficient.
    # Only active, parseable, reward-positive rows receive its signed pressure.
    semantic_shannon_success_conditioned_signed_advantage: bool = False
    semantic_shannon_success_conditioned_signed_cap: float = 0.05
    # Conservative verified-support conditional-entropy gradient surrogate.
    # It retains open-set surprisal for ranking sampled successful modes, then
    # centers only over sampled eligible rows. The artificial unseen bucket
    # cannot create downward pressure unless a distinct verified mode is also
    # present in the current group.
    semantic_shannon_success_conditioned_group_centered_advantage: bool = False
    # Predictor-centered conditional entropy over observed verifier-positive
    # support only. Unlike v6, persistent history supplies the control variate,
    # so a rare singleton remains actuated once another verified mode is known.
    semantic_shannon_success_conditioned_verified_support_advantage: bool = False
    # Optionally expand the predictor support with validator-positive replay
    # exemplars, including proposal-only discoveries. Membership is imported;
    # proposal rows never become on-policy frequency counts.
    semantic_shannon_verified_support_include_replay_bank: bool = False

    # Uniform-Correct Policy Optimization (Lochab et al., 2026). A positive
    # value reallocates each prompt group's existing positive advantage mass
    # toward low-probability verifier-positive rollouts. The paper default is
    # tau=.2; zero is exactly inactive.
    ucpo_tau: float = 0.0

    # Group-Aware Policy Optimization (Anschel et al., EMNLP 2025). When
    # enabled, the group's frequency-aware reward replaces the binary task
    # reward before centering. `gapo_support_index` is the frozen map from a
    # prompt's reference onto its enumerated ModeBench support L; GAPO assumes
    # a known valid set and this campaign can supply one, so the size is read
    # rather than estimated. `gapo_reward_scale` selects the published [-1, 1]
    # span or its affine image on the control's unit span; see gapo.py.
    gapo_enabled: bool = False
    gapo_support_index: str = ""
    gapo_reward_scale: str = "unit"

    # SetPO set-level diversity shaping (Li et al., 2026). A positive
    # coefficient adds lambda times each row's leave-one-out marginal
    # contribution to the kernelized set diversity of its rollout group. The
    # kernel is cosine similarity between frozen sentence embeddings loaded
    # from `setpo_embedder_path`; no gradient reaches the embedder.
    setpo_coefficient: float = 0.0
    setpo_embedder_path: str = ""
    setpo_embed_batch_size: int = 64

    # Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO; Yu et
    # al., 2025). The direct baseline uses standard GRPO advantages, the
    # paper's asymmetric .20/.28 PPO clip, token-level loss aggregation,
    # accuracy-group dynamic sampling, and soft overlong reward shaping.
    # Dynamic sampling draws at most ten generation batches to fill this
    # campaign's one-prompt training batch, matching the official recipe's
    # fail-closed generation ceiling.
    dapo_enabled: bool = False
    dapo_clip_low: float = 0.20
    dapo_clip_high: float = 0.28
    dapo_max_num_gen_batches: int = 10
    dapo_overlong_buffer_ratio: float = 0.20
    dapo_overlong_penalty_factor: float = 1.0

    # RLEP-Dr comparative baseline. The experience root contains four fixed
    # 16-sample T=.7/top-p=.95 verifier traces collected from the paired seed
    # model. A positive count mixes that many verified trajectories into each
    # fresh prompt group under one common reward baseline.
    rlep_experience_root: str = ""
    rlep_replay_count: int = 0
    # Preserve every fresh prompt update and use replay only where the frozen
    # seed-specific pool contains the complete prompt-matched replay dose.
    # False preserves E98's original all-prompts hard gate exactly.
    rlep_sparse_fallback: bool = False
    # E135: RLEP's update and eligibility rule, with the pool filled online
    # from the learner's own verified fresh rollouts instead of harvested once
    # from an RL-trained seed policy. Requires a positive replay count and no
    # experience root; prompts with fewer than `rlep_replay_count` stored
    # successes take the unchanged 16-row Dr.GRPO update, as the sparse
    # offline arm does.
    rlep_online_pool: bool = False

    # E88 adaptive semantic MaxEnt. The coefficient is moved so that the
    # realized ratio of semantic-advantage RMS to task-advantage RMS tracks
    # `semantic_rms_target_ratio`. The controller observes only those two
    # quantities and the eligible fraction; it never sees evaluation behavior.
    # It refuses an observation rather than extrapolating when the semantic
    # signal is degenerate, which is what keeps the singleton case safe.
    semantic_rms_control: bool = False
    semantic_rms_target_ratio: float = 0.05
    semantic_rms_min_coefficient: float = 0.02
    semantic_rms_max_coefficient: float = 0.40
    semantic_rms_ema_decay: float = 0.98
    semantic_rms_gain: float = 0.5
    semantic_rms_max_step_ratio: float = 1.1
    semantic_rms_warmup_steps: int = 64
    semantic_rms_min_eligible_fraction: float = 0.05

    # Online growing-support canonical MaxEnt. A prompt-local bank contains
    # only validator-positive canonical strategies produced on-policy. The
    # learner adds a bounded bank-entropy score after ordinary Dr.GRPO task
    # centering.
    # The bank also runs passively by default for ordinary Dr.GRPO so normal
    # runs report cumulative verified discoveries and mean verified support
    # per prompt without altering rewards, advantages, or gradients.
    verified_discovery_tracking: bool = True
    online_canonical_bank_alpha: float = 0.0
    online_canonical_bank_pseudocount: float = 1.0
    online_canonical_bank_surprisal_clip: float = 5.0
    # Haarnoja-style log-alpha control against the exact post-update ratio
    # H(q_x) / log |B_x^+|. A zero target preserves fixed-alpha E44 exactly.
    online_canonical_dual_target_ratio: float = 0.0
    online_canonical_dual_min_alpha: float = 0.005
    # Positive infinity disables only the controller's upper projection.
    online_canonical_dual_max_alpha: float = 0.5
    online_canonical_dual_alpha_lr: float = 0.003
    online_canonical_dual_ema_decay: float = 0.9
    # E51's projection-free alternative to the canonical-bank Haarnoja dual. The
    # model's masked-mean token entropy is calibrated separately in every run.
    # After warmup, alpha is the base dose times the warmup-mean/entropy-EMA
    # ratio with no upper or lower projection, so falling entropy raises alpha.
    online_canonical_policy_entropy_adaptation: bool = False
    online_canonical_policy_entropy_warmup_steps: int = 64
    online_canonical_policy_entropy_ema_decay: float = 0.9
    # Default-off verified exemplar replay. Once a prompt has at least two
    # observed validator-positive modes, teacher-forced model scores over one
    # stored exemplar per mode are balanced with KL(U_bank || q_model).
    # Replay and optional balance use the fixed coefficients below.
    online_canonical_replay: bool = False
    online_canonical_replay_alpha: float = 0.1
    # Default-off bank normalization of the replay dose. Uniform replay splits
    # `alpha` across the banked modes, so each mode receives `alpha / n` and a
    # prompt that has discovered *more* modes protects each one *less* -- the
    # opposite of the per-mode recurrent dose the retention argument relies on.
    # When enabled the dose becomes `per_mode_coefficient * n`, holding per-mode
    # pressure at a constant. No new ceiling is introduced: `n` is already
    # bounded by `online_canonical_replay_capacity`, so the dose is bounded by
    # `per_mode_coefficient * capacity`.
    online_canonical_replay_bank_normalized: bool = False
    # Registered as .10 / E[n], where E[n] = 3.08 is the mean bank occupancy
    # measured over the 25 fixed-dose replay cells of E78. That choice holds
    # total replay mass equal to the fixed arm in expectation, so the arm
    # redistributes a matched dose across bank sizes rather than adding one.
    online_canonical_replay_per_mode_coefficient: float = 0.0325
    # "bank_balance" is E53's conditioned-bank reverse KL. The successor
    # "verified_likelihood" keeps the same target-free sensor but gives the
    # actuator a non-zero common verified-mode score gradient.
    online_canonical_replay_objective: Literal[
        "bank_balance",
        "verified_likelihood",
        "verified_likelihood_per_rollout",
        "split_mass_balance_per_rollout",
    ] = "bank_balance"
    # Weight the same materialized replay exemplars either uniformly over
    # retained keys or by cumulative fresh on-policy observation frequency.
    # This changes only the within-bank target vector.
    online_canonical_replay_key_weighting: Literal[
        "uniform",
        "fresh_frequency",
    ] = "uniform"
    # A compute budget, not a semantic-support target. The default equals the
    # standard rollout width and never changes in response to evaluation data.
    online_canonical_replay_capacity: int = 16
    # Freeze membership and fresh-observation counts at this learner step while
    # continuing to replay and score the retained exemplars. Zero disables the
    # freeze. This is an audit instrument for identity-level survival, not an
    # adaptive training rule; it never reads outcomes or evaluation metrics.
    online_canonical_replay_bank_freeze_step: int = 0
    # Default-off cross-prompt scheduling. A positive value replays this many
    # model-discovered verified prompt banks per optimizer update in persistent
    # round-robin order. It is a fixed compute budget, not a support target.
    online_canonical_replay_global_groups_per_step: int = 0
    # Optional finite global cold-start phase. When positive, cross-prompt
    # replay is used for exactly this many non-empty optimizer updates and then
    # replay returns to the current prompt. Zero preserves either prompt-local
    # replay (global_groups_per_step=0) or unlimited global replay. The phase is
    # checkpointed and never observes evaluation or exhaustive support.
    online_canonical_replay_global_bootstrap_steps: int = 0
    online_canonical_replay_mass_alpha: float = 0.1
    # Default-off score-space safety cap for the split mass/balance objective.
    # For each prompt bank, use the largest requested balance scale for which
    # mass + balance assigns no positive derivative to any verified sequence
    # score. This keeps every direct replay update retention-preserving while
    # still applying as much known-bank equalization as the mass term permits.
    online_canonical_replay_retention_safe_balance: bool = False
    # Compute-matched negative control: retain and teacher-force the same
    # verified replay banks, including the backward traversal, but replace the
    # replay score derivative by exact zeros before it reaches the optimizer.
    # Ordinary task-reward gradients are unchanged.
    online_canonical_replay_compute_only: bool = False
    # Optional model-self-proposal actuator. Once the current prompt has one
    # model-generated validator-positive outcome, sample a fixed-budget
    # temperature sweep from the untouched original task prompt and admit only
    # genuinely new executable outcomes. Proposal rows are discarded before
    # PPO. The retry budget and proposal temperatures are search-compute
    # parameters, not support or entropy targets.
    online_canonical_counterfactual_proposals: bool = False
    # Compute-matched proposal control. Generate, validate, and canonicalize
    # proposal candidates through the ordinary explorer, but discard the
    # resulting admission payload before it can mutate support, replay,
    # retention, or policy-gradient state. Proposal rows still never enter PPO.
    online_canonical_counterfactual_admission_compute_only: bool = False
    # Keep proposal-derived replay exemplars out of the on-policy count table
    # used by the canonical entropy advantage. This permits a literal
    # E58 objective plus a replay-support actuator without off-policy proposal
    # outcomes changing any neutral-rollout advantage.
    online_canonical_counterfactual_separate_objective_support: bool = False
    # Restrict support-only proposals to a singleton verified bank. At most
    # one new verified outcome is admitted; no adaptive sensor is consulted.
    online_canonical_counterfactual_singleton_only: bool = False
    # Try deterministic validator-preserving transformations before the
    # isolated original-prompt sampler. Default-on preserves every historical
    # proposal variant; a task may disable it when its declared exact support
    # excludes validator-accepted transformation keys.
    online_canonical_counterfactual_transform_proposals: bool = True
    # Separately named, default-off Countdown actuator. It searches a fixed
    # radius-two neighborhood in the public easy3 action grammar, then applies
    # the ordinary validator. It never reads the enumerated solution set.
    online_canonical_counterfactual_exact_grammar_transforms: bool = False
    online_canonical_counterfactual_anchor_max_tokens: int = 256
    online_canonical_counterfactual_max_attempts: int = 3
    online_canonical_counterfactual_sampling_temperature: float = 1.0
    # Optional target-free reliability fallback. The controller counts only
    # eligible original-prompt proposal updates with no genuinely new verified
    # bank admission. After ``patience_updates`` it applies a bounded burst at
    # ``fallback_max_attempts``, followed by a fixed cooldown if discovery still
    # fails. Any new verified admission resets the schedule. It never observes
    # evaluation, exhaustive support, desired mode counts, or entropy targets.
    online_canonical_counterfactual_starvation_fallback: bool = False
    online_canonical_counterfactual_starvation_patience_updates: int = 64
    online_canonical_counterfactual_starvation_fallback_max_attempts: int = 4
    online_canonical_counterfactual_starvation_burst_updates: int = 16
    online_canonical_counterfactual_starvation_cooldown_updates: int = 48
    # Optional compute-matching surface. Every prompt update issues exactly
    # this many additional sampling requests with isolated proposal seeds.
    # Proposal-enabled arms may inspect up to ``max_attempts`` groups; all
    # remaining rows are discarded before banks, replay, and PPO.
    online_canonical_counterfactual_fixed_control_groups: int = 0
    # Newly admitted proposal-only outcomes may receive a fixed number of
    # replay visits with elevated *mass* weight. The weights are normalized
    # within the full prompt bank, so the balance loss continues to compare the
    # complete bank and the total replay-mass budget is unchanged.
    online_canonical_proposal_replay_priority_visits: int = 0
    online_canonical_proposal_replay_priority_multiplier: float = 1.0
    # Default-off successor to E102/E103. Track every proposal-admitted
    # exemplar's later verifier-positive neutral-rollout frequency and its
    # teacher-forced replay likelihood. The optional controller refreshes only
    # bounded mass-replay priority when either training-only signal weakens.
    online_canonical_proposal_retention_tracking: bool = False
    online_canonical_proposal_adaptive_retention_priority: bool = False
    online_canonical_proposal_retention_max_missed_rollout_opportunities: int = 2
    online_canonical_proposal_retention_max_mean_logprob_drop: float = 0.5
    online_canonical_proposal_retention_refresh_visits: int = 4
    online_canonical_proposal_retention_score_cooldown_observations: int = 2
    online_canonical_key_mode: Literal[
        "modebench_outcome",
        "math_verified_answer",
        "math_strategy_qwen72",
        "verified_route",
    ] = "modebench_outcome"
    # E69 maintains a prompt-local endpoint bank plus a separate global route
    # library. Routes replay only after independent neutral reproduction on at
    # least this many prompts. Proposal admission is support-only and must pass
    # an anchor-relative mean-token-log-probability trust check.
    verified_route_replay_capacity_per_route: int = 16
    verified_route_recurring_min_neutral_prompts: int = 2
    verified_route_proposal_max_mean_logprob_drop: float = 2.0
    # ``math_verified_answer`` is an external-validity track, not a
    # multi-mode strategy claim. The ordinary MATH verifier maps every
    # reward-positive solution for a prompt to one shared ``correct`` outcome.
    # Consequently verified-mass replay may anchor a discovered solution, but
    # known-mode balance remains structurally ineligible unless a future
    # executable proof/strategy verifier supplies distinct outcome keys.
    # E47-calibrated semantic strategy IDs for validator-positive MATH only.
    # The endpoint is an OpenAI-compatible /v1 service for the frozen 72B
    # judge. Two independent temperature-zero permutations are mandatory.
    math_strategy_endpoint: str = ""
    math_strategy_model: str = "qwen2.5-72b"
    math_strategy_timeout_seconds: int = 600
    math_strategy_workers: int = 4
    math_strategy_max_item_chars: int = 4000
    # E49T bootstrap successor: answer-positive natural derivations may be
    # mapped to one exact frozen menu route by two unanimous semantic audits.
    # This never creates an open-set strategy and is off by default.
    math_strategy_allow_unstructured_inference: bool = False
    # For finite-menu MATH prompts, only an answer-positive response that
    # passes the exact declaration parser and both semantic execution audits
    # retains its task reward. This matched contract can be enabled for both
    # Dr.GRPO and canonical-MaxEnt arms.
    math_strategy_gate_task_reward: bool = False

    # E14's finite canonical graph policy. Rollouts contain exactly
    # canonical_graph_action_count stochastic actions, each chosen from the
    # one-token support {"1", "2", "3"}; termination is deterministic.
    canonical_graph_actions: bool = False
    # Generic task selector for new finite policies. ``canonical_graph_actions``
    # remains the exact backward-compatible E14 switch.
    canonical_action_task: Literal[
        "none",
        "graph_coloring",
        "countdown",
        "pantry_support_mask",
    ] = "none"
    canonical_graph_action_count: int = 3
    canonical_graph_learner_sampling: bool = False
    canonical_graph_fixed_shape_sampling: bool = False
    # Four-rank free-form execution that preserves one prompt and one complete
    # num_samples candidate group per optimizer update. The rollout is
    # generated once, replicated for group-relative statistics, and sharded
    # only for the backward pass.
    replicated_freeform_sampling: bool = False
    # Pair each learner rank with one collocated one-GPU actor for concurrent
    # model-weight broadcasts. This avoids serializing the full 7B model from
    # rank zero into four tensor-parallel actor workers.
    local_actor_weight_sync: bool = False
    # vLLM sleep level 2 discards actor weights instead of copying four full
    # models into host RAM. The learner remaps empty weight storage, broadcasts
    # the updated policy, and restores only the KV cache after each update.
    vllm_sleep_level: int = 1

    # Controls retained because they are reported in the paper.
    policy_entropy_coef: float = 0.0
    seed_entropy_alpha: float = 0.0

    eval_mode_coverage_k: int = 0
    eval_mode_coverage_temperature: float = 1.0
    # Nucleus truncation applied to the sampled mode-coverage draws only. The
    # default 1.0 is the untruncated decoding surface every reported cell used;
    # the E72 decoding frontier sweeps it to test whether wider or narrower
    # decoding repairs a collapsed policy. Greedy draws ignore it.
    eval_mode_coverage_top_p: float = 1.0
    # Repeated, fixed-seed K-draws expose Monte Carlo evaluation variance.
    # The established headline keys remain the mean across draws; every raw
    # draw and its spread are logged separately by the learner.
    eval_mode_coverage_draws: int = 4
    eval_mode_coverage_seed: int = 1001
    # vLLM 0.8.4 V0 expands an ``n=K`` request with seed ``s`` into children
    # ``s..s+K-1``. Seeding consecutive draws at ``base + draw_index`` therefore
    # makes them share children: four draws of eight span eleven distinct
    # streams, not thirty-two, and every success-conditional statistic computed
    # by pooling a prompt's draws counts those repeats as independent. Striding
    # by K keeps each draw's block disjoint. Set False only to reproduce a run
    # recorded before this was fixed.
    eval_mode_coverage_disjoint_draws: bool = True
    # E72: evaluate the loaded policy exactly once through the ordinary
    # training evaluation path, then exit before any rollout, optimizer step,
    # export, or resume checkpoint. This is how a frozen checkpoint is measured
    # on a new decoding setting without re-implementing the metric.
    eval_only: bool = False
    baseline_zero_adv_response_tokens: int = 8

    # Storage lifecycle.  Evaluation is intentionally independent from both
    # model export and resumable DeepSpeed state.  ``export_steps=0`` means
    # terminal-only; a negative value disables exports entirely.  Resume
    # checkpoints are opt-in at the Python surface and resolved to one prompt
    # epoch by the shared experiment launcher.
    export_steps: int = 0
    export_from: int = 0
    resume_steps: int = -1
    resume_from: int = 0
    max_export_num: int = 1
    max_resume_num: int = 1
    max_export_mem: int = 64
    max_resume_mem: int = 256
    prune_resume_on_success: bool = True


def resolve_canonical_action_task(args: ZeroMathArgs) -> str:
    """Resolve the generic task selector and E14's legacy graph boolean."""

    requested = str(getattr(args, "canonical_action_task", "none"))
    if requested not in {
        "none",
        "graph_coloring",
        "countdown",
        "pantry_support_mask",
    }:
        raise ValueError(
            "canonical_action_task must be none, graph_coloring, countdown, "
            "or pantry_support_mask"
        )
    legacy_graph = bool(getattr(args, "canonical_graph_actions", False))
    if legacy_graph and requested not in {"none", "graph_coloring"}:
        raise ValueError("canonical_graph_actions conflicts with canonical_action_task")
    return "graph_coloring" if legacy_graph else requested


def validate_zero_math_args(args: ZeroMathArgs) -> ZeroMathArgs:
    """Reject configurations outside the single supported training surface."""

    canonical_task = resolve_canonical_action_task(args)
    modebench_domain = str(getattr(args, "modebench_domain", "none"))
    syntax_profile = str(getattr(args, "modebench_syntax_profile", "none"))
    level2_contracts = {
        "graph_coloring": ("qwen_boxed", "none"),
        "countdown": ("qwen_level2_countdown", "countdown_legal_v3"),
        "python_factors": ("qwen_level2_python_factors", "domain_legal_v1"),
        "mathir": ("qwen_level2_mathir", "domain_legal_v1"),
        "pantry_plan": ("qwen_level2_pantry", "domain_legal_v1"),
    }
    if modebench_domain == "none":
        if syntax_profile != "none":
            raise ValueError("ModeBench syntax profile requires an explicit domain")
    else:
        required_template, required_syntax = level2_contracts[modebench_domain]
        if (args.prompt_template, syntax_profile) != (required_template, required_syntax):
            raise ValueError(
                "Level-2 prompt/syntax contract mismatch: "
                f"domain={modebench_domain} requires "
                f"({required_template}, {required_syntax})"
            )
        if canonical_task != "none":
            raise ValueError("Level-2 guided syntax and canonical actions are separate policies")
        if int(getattr(args, "diayn_num_options", 0) or 0) != 0:
            raise ValueError("Level-2 guided syntax does not admit DIAYN options")
    # Dr.GRPO is the training surface for every treatment arm. Plain GRPO is
    # admitted for control arms only, so the paper can show whether correct-mode
    # collapse depends on Dr.GRPO's debiasing; the canonical bank and the
    # semantic arms still assert drgrpo below.
    if args.critic_type not in ("drgrpo", "grpo"):
        raise ValueError("This project supports critic_type in {drgrpo, grpo}")
    if args.num_samples <= 1:
        raise ValueError("Dr.GRPO requires num_samples > 1")
    if bool(getattr(args, "maxrl_task_objective", False)):
        if args.critic_type != "drgrpo":
            raise ValueError("binary MaxRL requires critic_type=drgrpo")
        if bool(getattr(args, "dapo_enabled", False)):
            raise ValueError("binary MaxRL and DAPO are separate objectives")
        if int(getattr(args, "rlep_replay_count", 0) or 0) != 0:
            raise ValueError("binary MaxRL does not admit RLEP reward mixing")
    ucpo_tau = float(getattr(args, "ucpo_tau", 0.0) or 0.0)
    if not math.isfinite(ucpo_tau) or not 0.0 <= ucpo_tau <= 1.0:
        raise ValueError("ucpo_tau must be finite and in [0, 1]")
    dapo_enabled = bool(getattr(args, "dapo_enabled", False))
    dapo_clip_low = float(getattr(args, "dapo_clip_low", 0.20))
    dapo_clip_high = float(getattr(args, "dapo_clip_high", 0.28))
    dapo_max_num_gen_batches = int(
        getattr(args, "dapo_max_num_gen_batches", 10)
    )
    dapo_overlong_buffer_ratio = float(
        getattr(args, "dapo_overlong_buffer_ratio", 0.20)
    )
    dapo_overlong_penalty_factor = float(
        getattr(args, "dapo_overlong_penalty_factor", 1.0)
    )
    if (
        not math.isfinite(dapo_clip_low)
        or not math.isfinite(dapo_clip_high)
        or not 0.0 < dapo_clip_low < 1.0
        or not dapo_clip_low <= dapo_clip_high < 1.0
    ):
        raise ValueError(
            "DAPO clipping requires 0 < dapo_clip_low <= dapo_clip_high < 1"
        )
    if dapo_max_num_gen_batches <= 0:
        raise ValueError("dapo_max_num_gen_batches must be positive")
    if (
        not math.isfinite(dapo_overlong_buffer_ratio)
        or not 0.0 < dapo_overlong_buffer_ratio < 1.0
    ):
        raise ValueError(
            "dapo_overlong_buffer_ratio must be finite and in (0, 1)"
        )
    if (
        not math.isfinite(dapo_overlong_penalty_factor)
        or dapo_overlong_penalty_factor < 0.0
    ):
        raise ValueError(
            "dapo_overlong_penalty_factor must be finite and non-negative"
        )
    if dapo_enabled:
        if args.critic_type != "grpo":
            raise ValueError("DAPO requires critic_type=grpo")
        if int(args.rollout_batch_size) != 1 or int(
            args.rollout_batch_size_per_device
        ) != 1:
            raise ValueError(
                "this registered DAPO adapter requires one training prompt per batch"
            )
        if int(args.train_batch_size) != int(args.num_samples) or int(
            args.train_batch_size_per_device
        ) != int(args.num_samples):
            raise ValueError(
                "DAPO requires one complete rollout group per optimizer batch"
            )
        if ucpo_tau > 0.0:
            raise ValueError("DAPO and UCPO are separate comparative baselines")
    rlep_root = str(getattr(args, "rlep_experience_root", "") or "")
    rlep_count = int(getattr(args, "rlep_replay_count", 0) or 0)
    rlep_online = bool(getattr(args, "rlep_online_pool", False))
    if rlep_count < 0:
        raise ValueError("rlep_replay_count must be non-negative")
    if rlep_online:
        if rlep_root:
            raise ValueError(
                "rlep_online_pool fills its pool from fresh rollouts and takes "
                "no rlep_experience_root"
            )
        if not rlep_count:
            raise ValueError("rlep_online_pool requires a positive rlep_replay_count")
    elif bool(rlep_root) != bool(rlep_count):
        raise ValueError(
            "rlep_experience_root and rlep_replay_count must be enabled together"
        )
    if bool(getattr(args, "rlep_sparse_fallback", False)) and not rlep_count:
        raise ValueError(
            "rlep_sparse_fallback requires an enabled RLEP experience pool"
        )
    if dapo_enabled and rlep_count:
        raise ValueError("DAPO and RLEP-Dr are separate comparative baselines")
    if rlep_count:
        if args.critic_type != "drgrpo":
            raise ValueError("RLEP-Dr requires critic_type=drgrpo")
        if int(args.num_samples) != 16 or int(args.rollout_batch_size) != 1:
            raise ValueError("RLEP-Dr requires one fresh 16-row prompt group")
        if ucpo_tau > 0.0:
            raise ValueError("RLEP-Dr and UCPO are separate comparative baselines")
        if bool(getattr(args, "online_canonical_replay", False)):
            raise ValueError("RLEP-Dr cannot be combined with canonical replay")
    gapo_enabled = bool(getattr(args, "gapo_enabled", False))
    gapo_support_index = str(getattr(args, "gapo_support_index", "") or "")
    gapo_reward_scale = str(getattr(args, "gapo_reward_scale", "unit") or "unit")
    setpo_coefficient = float(getattr(args, "setpo_coefficient", 0.0) or 0.0)
    setpo_embedder_path = str(getattr(args, "setpo_embedder_path", "") or "")
    setpo_embed_batch_size = int(getattr(args, "setpo_embed_batch_size", 64) or 64)
    if gapo_enabled:
        if args.critic_type != "drgrpo":
            raise ValueError("GAPO rides this campaign's Dr.GRPO backbone")
        if not gapo_support_index:
            raise ValueError(
                "GAPO requires gapo_support_index: the enumerated support is "
                "read from a frozen index, never estimated from the group"
            )
        if gapo_reward_scale not in ("unit", "paper"):
            raise ValueError("gapo_reward_scale must be 'unit' or 'paper'")
        for other, label in (
            (ucpo_tau > 0.0, "UCPO"),
            (dapo_enabled, "DAPO"),
            (bool(rlep_count), "RLEP-Dr"),
            (setpo_coefficient > 0.0, "SetPO"),
        ):
            if other:
                raise ValueError(f"GAPO and {label} are separate comparative baselines")
        if bool(getattr(args, "maxrl_task_objective", False)):
            raise ValueError("GAPO replaces the task reward and excludes MaxRL")
        if float(getattr(args, "outcome_collision_coef", 0.0) or 0.0) != 0.0:
            raise ValueError(
                "GAPO already penalizes within-group frequency; the outcome "
                "collision bonus would double it"
            )
    elif gapo_support_index:
        raise ValueError("gapo_support_index requires gapo_enabled")
    if not math.isfinite(setpo_coefficient) or setpo_coefficient < 0.0:
        raise ValueError("setpo_coefficient must be finite and non-negative")
    if setpo_coefficient > 0.0:
        if args.critic_type != "drgrpo":
            raise ValueError("SetPO rides this campaign's Dr.GRPO backbone")
        if not setpo_embedder_path:
            raise ValueError(
                "SetPO requires setpo_embedder_path: the similarity kernel is "
                "a frozen local snapshot, not a network download"
            )
        if setpo_embed_batch_size <= 0:
            raise ValueError("setpo_embed_batch_size must be positive")
        for other, label in (
            (ucpo_tau > 0.0, "UCPO"),
            (dapo_enabled, "DAPO"),
            (bool(rlep_count), "RLEP-Dr"),
        ):
            if other:
                raise ValueError(
                    f"SetPO and {label} are separate comparative baselines"
                )
    elif setpo_embedder_path:
        raise ValueError("setpo_embedder_path requires a positive setpo_coefficient")
    if dapo_enabled:
        incompatible = {
            "finite xDr tau": math.isfinite(float(args.xdr_tau)),
            "direct MaxEnt": float(getattr(args, "maxent_alpha", 0.0) or 0.0)
            != 0.0,
            "token entropy": float(
                getattr(args, "policy_entropy_coef", 0.0) or 0.0
            )
            != 0.0,
            "SEED": float(getattr(args, "seed_entropy_alpha", 0.0) or 0.0)
            != 0.0,
            "semantic Shannon": float(
                getattr(args, "semantic_shannon_coef", 0.0) or 0.0
            )
            != 0.0,
            "outcome collision": float(
                getattr(args, "outcome_collision_coef", 0.0) or 0.0
            )
            != 0.0,
            "DIAYN": float(getattr(args, "diayn_mi_beta", 0.0) or 0.0)
            != 0.0,
            "canonical bank objective": float(
                getattr(args, "online_canonical_bank_alpha", 0.0) or 0.0
            )
            != 0.0,
            "canonical replay": bool(
                getattr(args, "online_canonical_replay", False)
            ),
            "verified discovery tracking": bool(
                getattr(args, "verified_discovery_tracking", True)
            ),
        }
        active = sorted(name for name, enabled in incompatible.items() if enabled)
        if active:
            raise ValueError(
                "DAPO must be an isolated direct baseline; incompatible "
                f"components are active: {active}"
            )
    for name in ("export_from", "resume_from"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    if int(args.export_steps) < -1:
        raise ValueError("export_steps must be -1, 0, or a positive interval")
    if int(args.resume_steps) == 0 or int(args.resume_steps) < -1:
        raise ValueError("resume_steps must be -1 or a positive interval")
    for name in (
        "max_export_num",
        "max_resume_num",
        "max_export_mem",
        "max_resume_mem",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if math.isnan(float(args.xdr_tau)) or float(args.xdr_tau) < 0:
        raise ValueError(
            "xdr_tau must be non-negative (0 = argmax-set limit; inf = Dr.GRPO)"
        )
    maxent_alpha = float(args.maxent_alpha)
    if not math.isfinite(maxent_alpha) or maxent_alpha < 0:
        raise ValueError("maxent_alpha must be finite and non-negative")
    if maxent_alpha > 0:
        if math.isfinite(float(args.xdr_tau)):
            raise ValueError(
                "direct MaxEnt and signed-surrogate xDr are separate treatments"
            )
        if args.xdr_mode_adaptive:
            raise ValueError(
                "direct MaxEnt and mode-adaptive xDr are separate treatments"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError("direct MaxEnt and SEED are separate treatments")
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "direct MaxEnt and the legacy token-entropy control are separate treatments"
            )
    maxent_objective = str(getattr(args, "maxent_objective", "sequence"))
    if maxent_objective not in {"sequence", "conditional_token_mean"}:
        raise ValueError("maxent_objective must be sequence or conditional_token_mean")
    diayn_num_options = int(getattr(args, "diayn_num_options", 0) or 0)
    diayn_beta = float(getattr(args, "diayn_mi_beta", 0.0) or 0.0)
    if diayn_num_options < 0:
        raise ValueError("diayn_num_options must be non-negative")
    if not math.isfinite(diayn_beta) or diayn_beta < 0:
        raise ValueError("diayn_mi_beta must be finite and non-negative")
    if diayn_num_options <= 1 and diayn_beta > 0:
        raise ValueError("diayn_mi_beta requires diayn_num_options > 1")
    outcome_collision_coef = float(getattr(args, "outcome_collision_coef", 0.0) or 0.0)
    outcome_collision_outside_centering = bool(
        getattr(args, "outcome_collision_outside_centering", False)
    )
    if not math.isfinite(outcome_collision_coef) or outcome_collision_coef < 0:
        raise ValueError("outcome_collision_coef must be finite and non-negative")
    if outcome_collision_outside_centering and outcome_collision_coef <= 0:
        raise ValueError(
            "outcome_collision_outside_centering requires a positive "
            "outcome_collision_coef"
        )
    semantic_shannon_coef = float(getattr(args, "semantic_shannon_coef", 0.0) or 0.0)
    semantic_shannon_allow_zero_coefficient_control = bool(
        getattr(args, "semantic_shannon_allow_zero_coefficient_control", False)
    )
    semantic_shannon_surprisal_clip = float(
        getattr(args, "semantic_shannon_surprisal_clip", 5.0)
    )
    semantic_shannon_pseudocount = float(
        getattr(args, "semantic_shannon_pseudocount", 1.0)
    )
    semantic_shannon_separate_advantage = bool(
        getattr(args, "semantic_shannon_separate_advantage", False)
    )
    semantic_shannon_quality_gated_advantage = bool(
        getattr(args, "semantic_shannon_quality_gated_advantage", False)
    )
    semantic_shannon_quality_gated_cap = float(
        getattr(args, "semantic_shannon_quality_gated_cap", 0.05)
    )
    semantic_shannon_success_conditioned_signed_advantage = bool(
        getattr(
            args,
            "semantic_shannon_success_conditioned_signed_advantage",
            False,
        )
    )
    semantic_shannon_success_conditioned_signed_cap = float(
        getattr(
            args,
            "semantic_shannon_success_conditioned_signed_cap",
            0.05,
        )
    )
    semantic_shannon_success_conditioned_group_centered_advantage = bool(
        getattr(
            args,
            "semantic_shannon_success_conditioned_group_centered_advantage",
            False,
        )
    )
    semantic_shannon_success_conditioned_verified_support_advantage = bool(
        getattr(
            args,
            "semantic_shannon_success_conditioned_verified_support_advantage",
            False,
        )
    )
    semantic_shannon_verified_support_include_replay_bank = bool(
        getattr(
            args,
            "semantic_shannon_verified_support_include_replay_bank",
            False,
        )
    )
    semantic_shannon_success_conditioned_advantage = bool(
        semantic_shannon_success_conditioned_signed_advantage
        or semantic_shannon_success_conditioned_group_centered_advantage
        or semantic_shannon_success_conditioned_verified_support_advantage
    )
    online_canonical_bank_alpha = float(
        getattr(args, "online_canonical_bank_alpha", 0.0) or 0.0
    )
    online_canonical_bank_pseudocount = float(
        getattr(args, "online_canonical_bank_pseudocount", 1.0)
    )
    online_canonical_bank_surprisal_clip = float(
        getattr(args, "online_canonical_bank_surprisal_clip", 5.0)
    )
    online_canonical_dual_target_ratio = float(
        getattr(args, "online_canonical_dual_target_ratio", 0.0) or 0.0
    )
    online_canonical_dual_min_alpha = float(
        getattr(args, "online_canonical_dual_min_alpha", 0.005)
    )
    online_canonical_dual_max_alpha = float(
        getattr(args, "online_canonical_dual_max_alpha", 0.5)
    )
    online_canonical_dual_alpha_lr = float(
        getattr(args, "online_canonical_dual_alpha_lr", 0.003)
    )
    online_canonical_dual_ema_decay = float(
        getattr(args, "online_canonical_dual_ema_decay", 0.9)
    )
    online_canonical_policy_entropy_adaptation = bool(
        getattr(
            args,
            "online_canonical_policy_entropy_adaptation",
            False,
        )
    )
    online_canonical_policy_entropy_warmup_steps = int(
        getattr(
            args,
            "online_canonical_policy_entropy_warmup_steps",
            64,
        )
    )
    online_canonical_policy_entropy_ema_decay = float(
        getattr(
            args,
            "online_canonical_policy_entropy_ema_decay",
            0.9,
        )
    )
    online_canonical_replay = bool(getattr(args, "online_canonical_replay", False))
    online_canonical_replay_alpha = float(
        getattr(args, "online_canonical_replay_alpha", 0.1)
    )
    online_canonical_replay_key_weighting = str(
        getattr(args, "online_canonical_replay_key_weighting", "uniform")
    )
    online_canonical_replay_objective = str(
        getattr(
            args,
            "online_canonical_replay_objective",
            "bank_balance",
        )
    )
    online_canonical_replay_capacity = int(
        getattr(args, "online_canonical_replay_capacity", 16)
    )
    online_canonical_replay_bank_freeze_step = int(
        getattr(args, "online_canonical_replay_bank_freeze_step", 0)
    )
    online_canonical_replay_global_groups_per_step = int(
        getattr(
            args,
            "online_canonical_replay_global_groups_per_step",
            0,
        )
    )
    online_canonical_replay_global_bootstrap_steps = int(
        getattr(
            args,
            "online_canonical_replay_global_bootstrap_steps",
            0,
        )
    )
    online_canonical_replay_mass_alpha = float(
        getattr(args, "online_canonical_replay_mass_alpha", 0.1)
    )
    online_canonical_replay_retention_safe_balance = bool(
        getattr(
            args,
            "online_canonical_replay_retention_safe_balance",
            False,
        )
    )
    online_canonical_counterfactual_proposals = bool(
        getattr(
            args,
            "online_canonical_counterfactual_proposals",
            False,
        )
    )
    online_canonical_counterfactual_admission_compute_only = bool(
        getattr(
            args,
            "online_canonical_counterfactual_admission_compute_only",
            False,
        )
    )
    online_canonical_counterfactual_separate_objective_support = bool(
        getattr(
            args,
            "online_canonical_counterfactual_separate_objective_support",
            False,
        )
    )
    online_canonical_counterfactual_singleton_only = bool(
        getattr(
            args,
            "online_canonical_counterfactual_singleton_only",
            False,
        )
    )
    online_canonical_counterfactual_anchor_max_tokens = int(
        getattr(
            args,
            "online_canonical_counterfactual_anchor_max_tokens",
            256,
        )
    )
    online_canonical_key_mode = str(
        getattr(args, "online_canonical_key_mode", "modebench_outcome")
    )
    verified_route_replay_capacity_per_route = int(
        getattr(args, "verified_route_replay_capacity_per_route", 16)
    )
    verified_route_recurring_min_neutral_prompts = int(
        getattr(args, "verified_route_recurring_min_neutral_prompts", 2)
    )
    verified_route_proposal_max_mean_logprob_drop = float(
        getattr(args, "verified_route_proposal_max_mean_logprob_drop", 2.0)
    )
    math_strategy_gate_task_reward = bool(
        getattr(args, "math_strategy_gate_task_reward", False)
    )
    math_strategy_allow_unstructured_inference = bool(
        getattr(args, "math_strategy_allow_unstructured_inference", False)
    )
    online_canonical_bank_active = online_canonical_bank_alpha > 0.0
    online_canonical_objective_active = (
        online_canonical_bank_active or online_canonical_replay
    )
    if not math.isfinite(online_canonical_bank_alpha) or (
        online_canonical_bank_alpha < 0
    ):
        raise ValueError(
            "online_canonical_bank_alpha must be finite and non-negative"
        )
    for name, value in (
        ("online_canonical_bank_pseudocount", online_canonical_bank_pseudocount),
        (
            "online_canonical_bank_surprisal_clip",
            online_canonical_bank_surprisal_clip,
        ),
        ("online_canonical_dual_min_alpha", online_canonical_dual_min_alpha),
        ("online_canonical_dual_alpha_lr", online_canonical_dual_alpha_lr),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        math.isnan(online_canonical_dual_max_alpha)
        or online_canonical_dual_max_alpha <= 0
    ):
        raise ValueError("online_canonical_dual_max_alpha must be positive or +inf")
    if (
        not math.isfinite(online_canonical_dual_target_ratio)
        or not 0 <= online_canonical_dual_target_ratio <= 1
    ):
        raise ValueError(
            "online_canonical_dual_target_ratio must be finite and in [0, 1]"
        )
    if (
        not math.isfinite(online_canonical_dual_ema_decay)
        or not 0 <= online_canonical_dual_ema_decay < 1
    ):
        raise ValueError("online_canonical_dual_ema_decay must be finite and in [0, 1)")
    if online_canonical_policy_entropy_warmup_steps <= 0:
        raise ValueError(
            "online_canonical_policy_entropy_warmup_steps must be positive"
        )
    if (
        not math.isfinite(online_canonical_policy_entropy_ema_decay)
        or not 0 <= online_canonical_policy_entropy_ema_decay < 1
    ):
        raise ValueError(
            "online_canonical_policy_entropy_ema_decay must be finite and in [0, 1)"
        )
    if not math.isfinite(online_canonical_replay_alpha) or (
        online_canonical_replay_alpha < 0
        or (
            online_canonical_replay_alpha == 0
            and online_canonical_replay_objective
            != "split_mass_balance_per_rollout"
        )
    ):
        raise ValueError(
            "online_canonical_replay_alpha must be finite and positive, or "
            "exactly zero for a split mass/balance ablation"
        )
    if online_canonical_replay_objective not in {
        "bank_balance",
        "verified_likelihood",
        "verified_likelihood_per_rollout",
        "split_mass_balance_per_rollout",
    }:
        raise ValueError(
            "online_canonical_replay_objective must be bank_balance or "
            "verified_likelihood or verified_likelihood_per_rollout or "
            "split_mass_balance_per_rollout"
        )
    if online_canonical_replay_key_weighting not in {
        "uniform",
        "fresh_frequency",
    }:
        raise ValueError(
            "online_canonical_replay_key_weighting must be uniform or "
            "fresh_frequency"
        )
    if (
        online_canonical_replay_key_weighting != "uniform"
        and not online_canonical_replay
    ):
        raise ValueError(
            "non-uniform online_canonical_replay_key_weighting requires "
            "online_canonical_replay"
        )
    # ``bank_balance`` is the only objective that cannot score a one-mode bank:
    # its KL(U_g || softmax) term is undefined on a singleton group and
    # ``canonical_replay_uniform_loss`` refuses one. The mass objectives this
    # campaign trains under already score singleton groups -- every prompt that
    # has discovered exactly one mode is replayed as one -- so a capacity of
    # one is a legal configuration for them and is what the E130 mode-agnostic
    # ablation requests. The floor therefore follows the objective rather than
    # standing over all of them.
    minimum_capacity = 2 if online_canonical_replay_objective == "bank_balance" else 1
    if online_canonical_replay_capacity < minimum_capacity:
        raise ValueError(
            "online_canonical_replay_capacity must be at least "
            f"{minimum_capacity} for objective "
            f"{online_canonical_replay_objective}"
        )
    if online_canonical_replay_bank_freeze_step < 0:
        raise ValueError(
            "online_canonical_replay_bank_freeze_step must be non-negative"
        )
    if online_canonical_replay_bank_freeze_step > 0 and not online_canonical_replay:
        raise ValueError(
            "online_canonical_replay_bank_freeze_step requires canonical replay"
        )
    online_canonical_replay_bank_normalized = bool(
        getattr(args, "online_canonical_replay_bank_normalized", False)
    )
    online_canonical_replay_per_mode_coefficient = float(
        getattr(args, "online_canonical_replay_per_mode_coefficient", 0.0325)
    )
    if online_canonical_replay_bank_normalized and not online_canonical_replay:
        raise ValueError(
            "online_canonical_replay_bank_normalized requires "
            "online_canonical_replay"
        )
    if not math.isfinite(online_canonical_replay_per_mode_coefficient) or (
        online_canonical_replay_per_mode_coefficient <= 0.0
    ):
        raise ValueError(
            "online_canonical_replay_per_mode_coefficient must be finite "
            "and positive"
        )
    if online_canonical_replay_bank_normalized and bool(
        getattr(args, "online_canonical_replay_compute_only", False)
    ):
        # The compute-matched control zeroes the replay derivative, so a dose
        # rule would be measured on a term that never reaches the optimizer.
        raise ValueError(
            "online_canonical_replay_bank_normalized conflicts with "
            "online_canonical_replay_compute_only"
        )
    if online_canonical_replay_global_groups_per_step < 0:
        raise ValueError(
            "online_canonical_replay_global_groups_per_step must be non-negative"
        )
    if online_canonical_replay_global_bootstrap_steps < 0:
        raise ValueError(
            "online_canonical_replay_global_bootstrap_steps must be non-negative"
        )
    if (
        online_canonical_replay_global_groups_per_step > 0
        and not online_canonical_replay
    ):
        raise ValueError(
            "online_canonical_replay_global_groups_per_step requires replay"
        )
    if (
        online_canonical_replay_global_bootstrap_steps > 0
        and online_canonical_replay_global_groups_per_step <= 0
    ):
        raise ValueError(
            "online_canonical_replay_global_bootstrap_steps requires "
            "positive global groups per step"
        )
    if not math.isfinite(online_canonical_replay_mass_alpha) or (
        online_canonical_replay_mass_alpha < 0
        or (
            online_canonical_replay_mass_alpha == 0
            and online_canonical_replay_objective
            != "split_mass_balance_per_rollout"
        )
    ):
        raise ValueError(
            "online_canonical_replay_mass_alpha must be finite and positive, "
            "or exactly zero for a split mass/balance ablation"
        )
    if online_canonical_replay_retention_safe_balance:
        if not online_canonical_replay:
            raise ValueError("retention-safe balance requires canonical replay")
        if online_canonical_replay_objective != "split_mass_balance_per_rollout":
            raise ValueError(
                "retention-safe balance requires split_mass_balance_per_rollout"
            )
        if online_canonical_replay_mass_alpha <= 0.0:
            raise ValueError(
                "retention-safe balance requires positive verified-mass pressure"
            )
    if (
        bool(getattr(args, "online_canonical_replay_compute_only", False))
        and not online_canonical_replay
    ):
        raise ValueError(
            "online_canonical_replay_compute_only requires canonical replay"
        )
    if online_canonical_counterfactual_anchor_max_tokens <= 0:
        raise ValueError(
            "online_canonical_counterfactual_anchor_max_tokens must be positive"
        )
    online_canonical_counterfactual_max_attempts = int(
        getattr(
            args,
            "online_canonical_counterfactual_max_attempts",
            3,
        )
    )
    if online_canonical_counterfactual_max_attempts <= 0:
        raise ValueError(
            "online_canonical_counterfactual_max_attempts must be positive"
        )
    online_canonical_counterfactual_sampling_temperature = float(
        getattr(
            args,
            "online_canonical_counterfactual_sampling_temperature",
            1.0,
        )
    )
    online_canonical_counterfactual_starvation_fallback = bool(
        getattr(
            args,
            "online_canonical_counterfactual_starvation_fallback",
            False,
        )
    )
    online_canonical_counterfactual_starvation_patience_updates = int(
        getattr(
            args,
            "online_canonical_counterfactual_starvation_patience_updates",
            64,
        )
    )
    online_canonical_counterfactual_starvation_fallback_max_attempts = int(
        getattr(
            args,
            "online_canonical_counterfactual_starvation_fallback_max_attempts",
            4,
        )
    )
    online_canonical_counterfactual_starvation_burst_updates = int(
        getattr(
            args,
            "online_canonical_counterfactual_starvation_burst_updates",
            16,
        )
    )
    online_canonical_counterfactual_starvation_cooldown_updates = int(
        getattr(
            args,
            "online_canonical_counterfactual_starvation_cooldown_updates",
            48,
        )
    )
    online_canonical_counterfactual_fixed_control_groups = int(
        getattr(
            args,
            "online_canonical_counterfactual_fixed_control_groups",
            0,
        )
    )
    online_canonical_proposal_replay_priority_visits = int(
        getattr(
            args,
            "online_canonical_proposal_replay_priority_visits",
            0,
        )
    )
    online_canonical_proposal_replay_priority_multiplier = float(
        getattr(
            args,
            "online_canonical_proposal_replay_priority_multiplier",
            1.0,
        )
    )
    online_canonical_proposal_retention_tracking = bool(
        getattr(args, "online_canonical_proposal_retention_tracking", False)
    )
    online_canonical_proposal_adaptive_retention_priority = bool(
        getattr(
            args,
            "online_canonical_proposal_adaptive_retention_priority",
            False,
        )
    )
    online_canonical_proposal_retention_max_missed_rollout_opportunities = int(
        getattr(
            args,
            "online_canonical_proposal_retention_max_missed_rollout_opportunities",
            2,
        )
    )
    online_canonical_proposal_retention_max_mean_logprob_drop = float(
        getattr(
            args,
            "online_canonical_proposal_retention_max_mean_logprob_drop",
            0.5,
        )
    )
    online_canonical_proposal_retention_refresh_visits = int(
        getattr(
            args,
            "online_canonical_proposal_retention_refresh_visits",
            4,
        )
    )
    online_canonical_proposal_retention_score_cooldown_observations = int(
        getattr(
            args,
            "online_canonical_proposal_retention_score_cooldown_observations",
            2,
        )
    )
    if online_canonical_counterfactual_fixed_control_groups < 0:
        raise ValueError(
            "online_canonical_counterfactual_fixed_control_groups must be "
            "non-negative"
        )
    if online_canonical_proposal_replay_priority_visits < 0:
        raise ValueError("proposal replay priority visits must be non-negative")
    if (
        not math.isfinite(online_canonical_proposal_replay_priority_multiplier)
        or online_canonical_proposal_replay_priority_multiplier < 1.0
    ):
        raise ValueError(
            "proposal replay priority multiplier must be finite and at least one"
        )
    if online_canonical_proposal_replay_priority_visits > 0:
        if not online_canonical_counterfactual_proposals:
            raise ValueError(
                "proposal replay priority requires counterfactual proposals"
            )
        if not online_canonical_counterfactual_separate_objective_support:
            raise ValueError(
                "proposal replay priority requires separate proposal support"
            )
        if online_canonical_proposal_replay_priority_multiplier <= 1.0:
            raise ValueError(
                "positive proposal priority visits require multiplier greater than one"
            )
    elif not math.isclose(
        online_canonical_proposal_replay_priority_multiplier,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "proposal priority multiplier must be one when priority is disabled"
        )
    for name, value in (
        (
            "online_canonical_proposal_retention_max_missed_rollout_opportunities",
            online_canonical_proposal_retention_max_missed_rollout_opportunities,
        ),
        (
            "online_canonical_proposal_retention_refresh_visits",
            online_canonical_proposal_retention_refresh_visits,
        ),
        (
            "online_canonical_proposal_retention_score_cooldown_observations",
            online_canonical_proposal_retention_score_cooldown_observations,
        ),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        not math.isfinite(
            online_canonical_proposal_retention_max_mean_logprob_drop
        )
        or online_canonical_proposal_retention_max_mean_logprob_drop <= 0.0
    ):
        raise ValueError(
            "online_canonical_proposal_retention_max_mean_logprob_drop must be "
            "finite and positive"
        )
    if online_canonical_proposal_adaptive_retention_priority and not (
        online_canonical_proposal_retention_tracking
    ):
        raise ValueError(
            "adaptive proposal retention priority requires retention tracking"
        )
    if online_canonical_proposal_retention_tracking:
        if not online_canonical_counterfactual_proposals:
            raise ValueError(
                "proposal retention tracking requires counterfactual proposals"
            )
        if not online_canonical_counterfactual_separate_objective_support:
            raise ValueError(
                "proposal retention tracking requires separate proposal support"
            )
    if online_canonical_proposal_adaptive_retention_priority:
        if online_canonical_proposal_replay_priority_visits <= 0:
            raise ValueError(
                "adaptive proposal retention priority requires proposal priority"
            )
        if online_canonical_replay_global_groups_per_step <= 0:
            raise ValueError(
                "adaptive proposal retention priority requires global replay"
            )
        if (
            online_canonical_proposal_retention_refresh_visits
            > online_canonical_proposal_replay_priority_visits
        ):
            raise ValueError(
                "retention refresh visits may not exceed initial priority visits"
            )
        if online_canonical_replay_objective not in {
            "verified_likelihood",
            "verified_likelihood_per_rollout",
            "split_mass_balance_per_rollout",
        }:
            raise ValueError(
                "adaptive proposal retention priority requires a replay mass objective"
            )
        if bool(getattr(args, "online_canonical_replay_compute_only", False)):
            raise ValueError(
                "adaptive proposal retention priority conflicts with compute-only replay"
            )
    if (
        online_canonical_counterfactual_fixed_control_groups > 0
        and not bool(getattr(args, "replicated_freeform_sampling", False))
    ):
        raise ValueError(
            "fixed counterfactual control groups require replicated "
            "free-form sampling"
        )
    if (
        online_canonical_counterfactual_proposals
        and 0 < online_canonical_counterfactual_fixed_control_groups
        < online_canonical_counterfactual_max_attempts
    ):
        raise ValueError(
            "fixed counterfactual control groups must cover every proposal "
            "attempt"
        )
    if (
        not math.isfinite(online_canonical_counterfactual_sampling_temperature)
        or online_canonical_counterfactual_sampling_temperature <= 0
    ):
        raise ValueError(
            "online_canonical_counterfactual_sampling_temperature must be "
            "finite and positive"
        )
    for name, value, minimum in (
        (
            "online_canonical_counterfactual_starvation_patience_updates",
            online_canonical_counterfactual_starvation_patience_updates,
            1,
        ),
        (
            "online_canonical_counterfactual_starvation_fallback_max_attempts",
            online_canonical_counterfactual_starvation_fallback_max_attempts,
            1,
        ),
        (
            "online_canonical_counterfactual_starvation_burst_updates",
            online_canonical_counterfactual_starvation_burst_updates,
            1,
        ),
        (
            "online_canonical_counterfactual_starvation_cooldown_updates",
            online_canonical_counterfactual_starvation_cooldown_updates,
            0,
        ),
    ):
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if online_canonical_counterfactual_starvation_fallback:
        if not online_canonical_counterfactual_proposals:
            raise ValueError(
                "proposal starvation fallback requires counterfactual proposals"
            )
        if (
            online_canonical_counterfactual_starvation_fallback_max_attempts
            <= online_canonical_counterfactual_max_attempts
        ):
            raise ValueError(
                "proposal starvation fallback max attempts must exceed the base "
                "proposal max attempts"
            )
        if (
            online_canonical_counterfactual_fixed_control_groups > 0
            and online_canonical_counterfactual_fixed_control_groups
            < online_canonical_counterfactual_starvation_fallback_max_attempts
        ):
            raise ValueError(
                "fixed counterfactual control groups must cover every fallback "
                "proposal attempt"
            )
    if online_canonical_replay:
        if not bool(getattr(args, "verified_discovery_tracking", True)):
            raise ValueError(
                "online canonical replay requires verified discovery tracking"
            )
        if online_canonical_key_mode not in {
            "modebench_outcome",
            "math_verified_answer",
            "verified_route",
        }:
            raise ValueError(
                "online canonical replay currently requires "
                "online_canonical_key_mode=modebench_outcome or "
                "math_verified_answer or verified_route"
            )
    if online_canonical_counterfactual_proposals:
        if not online_canonical_replay:
            raise ValueError(
                "counterfactual canonical proposals require canonical replay"
            )
        if online_canonical_key_mode not in {
            "modebench_outcome",
            "verified_route",
        }:
            raise ValueError(
                "counterfactual canonical proposals require executable "
                "ModeBench outcome keys or verified route identities"
            )
        if online_canonical_bank_alpha != 0.0 and not (
            online_canonical_counterfactual_separate_objective_support
        ):
            raise ValueError(
                "proposal support cannot feed an on-policy canonical-bank advantage"
            )
        if not bool(getattr(args, "online_evaluation", False)):
            raise ValueError(
                "counterfactual canonical proposals require online validation"
            )
    elif online_canonical_counterfactual_separate_objective_support:
        raise ValueError(
            "separate counterfactual objective support requires "
            "counterfactual proposals"
        )
    if (
        online_canonical_counterfactual_admission_compute_only
        and not online_canonical_counterfactual_proposals
    ):
        raise ValueError(
            "counterfactual admission compute-only requires counterfactual "
            "proposals"
        )
    if (
        online_canonical_counterfactual_singleton_only
        and not online_canonical_counterfactual_proposals
    ):
        raise ValueError(
            "singleton-only support expansion requires counterfactual proposals"
        )
    if (
        online_canonical_policy_entropy_adaptation
        and online_canonical_dual_target_ratio > 0
    ):
        raise ValueError(
            "canonical policy-entropy adaptation and Haarnoja dual control "
            "are separate treatments"
        )
    if online_canonical_policy_entropy_adaptation and bool(
        getattr(args, "maxent_inverse_adaptation", False)
    ):
        raise ValueError(
            "direct inverse MaxEnt and canonical policy-entropy adaptation "
            "are separate treatments"
        )
    if online_canonical_policy_entropy_adaptation:
        if not online_canonical_bank_active or online_canonical_bank_alpha <= 0:
            raise ValueError(
                "canonical policy-entropy adaptation requires a positive bank alpha"
            )
    if online_canonical_dual_target_ratio > 0:
        if not online_canonical_bank_active or online_canonical_bank_alpha <= 0:
            raise ValueError(
                "online canonical dual control requires a positive bank alpha"
            )
        if online_canonical_dual_min_alpha > online_canonical_bank_alpha:
            raise ValueError(
                "online_canonical_dual_min_alpha must not exceed bank alpha"
            )
        if online_canonical_dual_max_alpha < online_canonical_bank_alpha:
            raise ValueError(
                "online_canonical_dual_max_alpha must be at least bank alpha"
            )
    if online_canonical_key_mode not in {
        "modebench_outcome",
        "math_verified_answer",
        "math_strategy_qwen72",
        "verified_route",
    }:
        raise ValueError(
            "online_canonical_key_mode must be modebench_outcome, "
            "math_verified_answer, math_strategy_qwen72, or verified_route"
        )
    if verified_route_replay_capacity_per_route <= 0:
        raise ValueError("verified_route_replay_capacity_per_route must be positive")
    if verified_route_recurring_min_neutral_prompts < 2:
        raise ValueError(
            "verified_route_recurring_min_neutral_prompts must be at least two"
        )
    if (
        not math.isfinite(verified_route_proposal_max_mean_logprob_drop)
        or verified_route_proposal_max_mean_logprob_drop < 0
    ):
        raise ValueError(
            "verified_route_proposal_max_mean_logprob_drop must be finite and "
            "non-negative"
        )
    if online_canonical_key_mode == "verified_route":
        template_role = prompt_template_role(args.prompt_template)
        modebench_contract = (
            template_role == "boxed"
            and args.verifier_version == "fast"
            and args.test_split == "multi_answer"
        )
        math_route_contract = (
            template_role == "math_route"
            and args.verifier_version == "math_verify"
            and args.test_split == "math_dev"
        )
        if not (modebench_contract or math_route_contract):
            raise ValueError(
                "verified-route mode requires either executable ModeBench "
                "(*_boxed, fast, multi_answer) or sealed route-development "
                "MATH (*_math_route, math_verify, math_dev)"
            )
        if online_canonical_bank_alpha != 0.0:
            raise ValueError(
                "verified-route neutral learning keeps canonical advantages "
                "at zero and uses conservative replay only"
            )
        if online_canonical_counterfactual_proposals and not (
            online_canonical_counterfactual_separate_objective_support
        ):
            raise ValueError(
                "verified-route proposals require a separate support store"
            )
        if online_canonical_counterfactual_proposals and not math.isclose(
            online_canonical_counterfactual_sampling_temperature,
            float(args.temperature),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "verified-route proposal and neutral temperatures must match "
                "for the anchor-relative likelihood trust check"
            )
        if online_canonical_replay:
            if online_canonical_replay_objective != ("verified_likelihood_per_rollout"):
                raise ValueError(
                    "verified-route replay requires verified_likelihood_per_rollout"
                )
            if online_canonical_replay_global_groups_per_step != 1:
                raise ValueError(
                    "verified-route replay requires exactly one fixed-budget "
                    "global replay group per optimizer update"
                )
            if online_canonical_replay_global_bootstrap_steps != 0:
                raise ValueError(
                    "verified-route replay does not use a finite endpoint "
                    "bootstrap phase"
                )
    if math_strategy_gate_task_reward:
        if online_canonical_key_mode != "math_strategy_qwen72":
            raise ValueError(
                "math strategy task-reward gating requires "
                "online_canonical_key_mode=math_strategy_qwen72"
            )
        if not bool(getattr(args, "verified_discovery_tracking", True)):
            raise ValueError(
                "math strategy task-reward gating requires canonical "
                "tracking so every positive response is audited"
            )
    if math_strategy_allow_unstructured_inference and (
        online_canonical_key_mode != "math_strategy_qwen72"
    ):
        raise ValueError(
            "unstructured MATH strategy inference requires "
            "online_canonical_key_mode=math_strategy_qwen72"
        )
    if online_canonical_objective_active:
        if args.critic_type != "drgrpo":
            raise ValueError("online canonical bank requires critic_type=drgrpo")
        if online_canonical_key_mode == "modebench_outcome":
            template_role = prompt_template_role(args.prompt_template)
            executable_modebench_contract = (
                args.verifier_version == "fast"
                and args.test_split == "multi_answer"
                and (
                    template_role == "boxed"
                    or (
                        canonical_task == "pantry_support_mask"
                        and template_role == "pantry_support_mask"
                    )
                )
            )
            if not executable_modebench_contract:
                raise ValueError(
                    "online canonical banks require executable ModeBench "
                    "validation with the task-bound prompt template, "
                    "verifier_version=fast, and test_split=multi_answer"
                )
        elif online_canonical_key_mode == "math_verified_answer":
            if (
                prompt_template_role(args.prompt_template) != "math"
                or args.verifier_version != "math_verify"
                or args.test_split not in {"math", "math_dev"}
            ):
                raise ValueError(
                    "verified-answer MATH replay requires "
                    "prompt_template=*_math, "
                    "verifier_version=math_verify, and "
                    "test_split=math or math_dev"
                )
        elif online_canonical_key_mode == "math_strategy_qwen72" and (
            args.prompt_template != "qwen_math"
            or args.verifier_version != "math_verify"
            or args.test_split != "math"
        ):
            raise ValueError(
                "MATH strategy banks require validator-bound "
                "prompt_template=qwen_math, verifier_version=math_verify, "
                "and test_split=math"
            )
    if online_canonical_key_mode == "math_strategy_qwen72" and (
        online_canonical_bank_active
        or bool(getattr(args, "verified_discovery_tracking", True))
        or math_strategy_gate_task_reward
    ):
        if not str(getattr(args, "math_strategy_endpoint", "")).strip():
            raise ValueError("math_strategy_qwen72 requires math_strategy_endpoint")
        if int(getattr(args, "math_strategy_timeout_seconds", 600)) <= 0:
            raise ValueError("math_strategy_timeout_seconds must be positive")
        if int(getattr(args, "math_strategy_workers", 4)) <= 0:
            raise ValueError("math_strategy_workers must be positive")
        if int(getattr(args, "math_strategy_max_item_chars", 4000)) < 600:
            raise ValueError("math_strategy_max_item_chars must be at least 600")
    if not math.isfinite(semantic_shannon_coef) or semantic_shannon_coef < 0:
        raise ValueError("semantic_shannon_coef must be finite and non-negative")
    if (
        not math.isfinite(semantic_shannon_surprisal_clip)
        or semantic_shannon_surprisal_clip <= 0
    ):
        raise ValueError("semantic_shannon_surprisal_clip must be finite and positive")
    if (
        not math.isfinite(semantic_shannon_pseudocount)
        or semantic_shannon_pseudocount <= 0
    ):
        raise ValueError("semantic_shannon_pseudocount must be finite and positive")
    if (
        semantic_shannon_separate_advantage
        and semantic_shannon_coef <= 0
        and not semantic_shannon_allow_zero_coefficient_control
    ):
        raise ValueError(
            "semantic_shannon_separate_advantage requires a positive "
            "semantic_shannon_coef unless the explicit zero-coefficient "
            "control surface is enabled"
        )
    if (
        semantic_shannon_allow_zero_coefficient_control
        and not semantic_shannon_separate_advantage
    ):
        raise ValueError(
            "semantic zero-coefficient control requires "
            "semantic_shannon_separate_advantage"
        )
    if semantic_shannon_separate_advantage and args.critic_type != "drgrpo":
        raise ValueError(
            "semantic_shannon_separate_advantage requires critic_type=drgrpo"
        )
    if (
        not math.isfinite(semantic_shannon_quality_gated_cap)
        or semantic_shannon_quality_gated_cap <= 0
    ):
        raise ValueError(
            "semantic_shannon_quality_gated_cap must be finite and positive"
        )
    if (
        semantic_shannon_quality_gated_advantage
        and not semantic_shannon_separate_advantage
    ):
        raise ValueError(
            "semantic_shannon_quality_gated_advantage requires "
            "semantic_shannon_separate_advantage"
        )
    if (
        not math.isfinite(semantic_shannon_success_conditioned_signed_cap)
        or semantic_shannon_success_conditioned_signed_cap <= 0
    ):
        raise ValueError(
            "semantic_shannon_success_conditioned_signed_cap must be finite "
            "and positive"
        )
    if (
        semantic_shannon_success_conditioned_signed_advantage
        and not semantic_shannon_separate_advantage
    ):
        raise ValueError(
            "semantic_shannon_success_conditioned_signed_advantage requires "
            "semantic_shannon_separate_advantage"
        )
    if (
        semantic_shannon_success_conditioned_group_centered_advantage
        and not semantic_shannon_separate_advantage
    ):
        raise ValueError(
            "semantic_shannon_success_conditioned_group_centered_advantage "
            "requires semantic_shannon_separate_advantage"
        )
    if (
        semantic_shannon_success_conditioned_verified_support_advantage
        and not semantic_shannon_separate_advantage
    ):
        raise ValueError(
            "semantic_shannon_success_conditioned_verified_support_advantage "
            "requires semantic_shannon_separate_advantage"
        )
    if semantic_shannon_verified_support_include_replay_bank:
        if not semantic_shannon_success_conditioned_verified_support_advantage:
            raise ValueError(
                "replay-bank semantic support requires verified-support mode"
            )
        if not online_canonical_replay:
            raise ValueError(
                "replay-bank semantic support requires online canonical replay"
            )

    if bool(getattr(args, "semantic_rms_control", False)):

        if not semantic_shannon_success_conditioned_signed_advantage:
            raise ValueError(
                "semantic_rms_control requires the success-conditioned signed "
                "semantic advantage"
            )
        if semantic_shannon_coef <= 0:
            raise ValueError(
                "semantic_rms_control requires a positive starting "
                "semantic_shannon_coef"
            )
        ratio = float(getattr(args, "semantic_rms_target_ratio", 0.0))
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("semantic_rms_target_ratio must be finite and positive")
        low = float(getattr(args, "semantic_rms_min_coefficient", 0.0))
        high = float(getattr(args, "semantic_rms_max_coefficient", 0.0))
        if not 0 < low <= high:
            raise ValueError(
                "semantic_rms coefficient bounds must satisfy 0 < min <= max"
            )
        if not low <= semantic_shannon_coef <= high:
            raise ValueError(
                "semantic_shannon_coef must start inside the registered "
                "semantic_rms coefficient bounds"
            )
        decay = float(getattr(args, "semantic_rms_ema_decay", 0.0))
        if not 0 <= decay < 1:
            raise ValueError("semantic_rms_ema_decay must be in [0, 1)")
        if float(getattr(args, "semantic_rms_gain", 0.0)) <= 0:
            raise ValueError("semantic_rms_gain must be positive")
        if float(getattr(args, "semantic_rms_max_step_ratio", 0.0)) <= 1:
            raise ValueError("semantic_rms_max_step_ratio must exceed 1")
        if int(getattr(args, "semantic_rms_warmup_steps", -1)) < 0:
            raise ValueError("semantic_rms_warmup_steps must be non-negative")
    if sum(
        (
            semantic_shannon_success_conditioned_signed_advantage,
            semantic_shannon_success_conditioned_group_centered_advantage,
            semantic_shannon_success_conditioned_verified_support_advantage,
            semantic_shannon_quality_gated_advantage,
        )
    ) > 1:
        raise ValueError(
            "semantic Shannon quality-gated, success-conditioned signed, "
            "group-centered, and verified-support advantages are separate "
            "treatments"
        )
    xdr_task_advantage_weights = bool(
        getattr(args, "xdr_task_advantage_weights", False)
    )
    if xdr_task_advantage_weights:
        if not math.isfinite(float(args.xdr_tau)) or float(args.xdr_tau) <= 0:
            raise ValueError(
                "xdr_task_advantage_weights requires a finite positive xdr_tau"
            )
        if not semantic_shannon_success_conditioned_signed_advantage:
            raise ValueError(
                "xdr_task_advantage_weights requires the success-conditioned "
                "signed semantic advantage"
            )
        if bool(getattr(args, "xdr_mode_adaptive", False)):
            raise ValueError(
                "xdr_task_advantage_weights and mode-adaptive xDr are "
                "separate treatments"
            )
    if outcome_collision_coef > 0:
        if semantic_shannon_coef > 0:
            raise ValueError(
                "outcome-collision and semantic Shannon shaping are separate treatments"
            )
        if diayn_num_options > 1:
            raise ValueError(
                "outcome-collision shaping and DIAYN answer options are "
                "separate treatments"
            )
        if maxent_alpha > 0:
            raise ValueError(
                "outcome-collision shaping and direct MaxEnt are separate treatments"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError(
                "outcome-collision shaping and SEED are separate treatments"
            )
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "outcome-collision shaping and token entropy are separate treatments"
            )
        if math.isfinite(float(args.xdr_tau)):
            raise ValueError(
                "outcome-collision shaping and signed-surrogate xDr are "
                "separate treatments"
            )
    if semantic_shannon_coef > 0:
        if diayn_num_options > 1:
            raise ValueError(
                "semantic Shannon shaping and DIAYN answer options are "
                "separate treatments"
            )
        open_set_maxent_composition = (
            semantic_shannon_success_conditioned_advantage
            and bool(getattr(args, "maxent_inverse_adaptation", False))
            and str(getattr(args, "maxent_objective", "sequence"))
            == "conditional_token_mean"
        )
        if maxent_alpha > 0 and not open_set_maxent_composition:
            raise ValueError(
                "semantic Shannon shaping and direct MaxEnt are separate "
                "treatments except for open-set semantic MaxEnt "
                "with fixed conditional-token MaxEnt"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError(
                "semantic Shannon shaping and SEED are separate treatments"
            )
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "semantic Shannon shaping and token entropy are separate treatments"
            )
        if math.isfinite(float(args.xdr_tau)) and not xdr_task_advantage_weights:
            raise ValueError(
                "semantic Shannon shaping and signed-surrogate xDr are "
                "separate treatments"
            )
    if online_canonical_objective_active:
        # Open-set semantic MaxEnt composes with either replay objective. The
        # split objective is the historical E43/E56 pairing; the uniform
        # verified-likelihood objective is the E78 replay arm, and admitting it
        # is what lets semantic MaxEnt be measured against verified replay
        # without also inheriting the balance loss.
        open_set_replay_composition = (
            semantic_shannon_success_conditioned_advantage
            and online_canonical_replay
            and online_canonical_replay_objective
            in {
                "split_mass_balance_per_rollout",
                "verified_likelihood_per_rollout",
            }
        )
        if semantic_shannon_coef > 0 and not open_set_replay_composition:
            raise ValueError(
                "online canonical bank and semantic Shannon are separate "
                "treatments except for open-set semantic MaxEnt "
                "with split mass/balance or uniform verified-likelihood replay"
            )
        if outcome_collision_coef > 0:
            raise ValueError(
                "online canonical bank and outcome collision are separate treatments"
            )
        if diayn_num_options > 1:
            raise ValueError("online canonical bank and DIAYN are separate treatments")
        inverse_fixed_canonical_hybrid = (
            bool(getattr(args, "maxent_inverse_adaptation", False))
            and str(getattr(args, "maxent_objective", "sequence"))
            == "conditional_token_mean"
            and online_canonical_dual_target_ratio == 0
            and not online_canonical_policy_entropy_adaptation
        )
        if maxent_alpha > 0 and not inverse_fixed_canonical_hybrid:
            raise ValueError(
                "online canonical bank and token-policy MaxEnt are separate "
                "treatments except for fixed conditional-token MaxEnt with "
                "a fixed canonical coefficient"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError("online canonical bank and SEED are separate treatments")
        # A token-entropy term acts on the fresh-rollout loss only, so it
        # composes additively with (a) the compute-matched control, whose
        # traversal is inert, and (b) Re:Dr's verified-likelihood replay term,
        # which is a separate loss on stored rows. E134 needs both: the tuned
        # control and Re:Dr at that control's setting (user decision,
        # 2026-09-27). A live bank coefficient or the split mass/balance
        # replay objective stays a separate treatment.
        inert_canonical_traversal = bool(
            getattr(args, "online_canonical_replay_compute_only", False)
        ) and not online_canonical_bank_active
        verified_likelihood_replay = (
            not online_canonical_bank_active
            and online_canonical_replay
            and str(getattr(args, "online_canonical_replay_objective", ""))
            == "verified_likelihood_per_rollout"
        )
        if float(args.policy_entropy_coef) > 0 and not (
            inert_canonical_traversal or verified_likelihood_replay
        ):
            raise ValueError(
                "online canonical bank and token entropy are separate treatments"
            )
        if math.isfinite(float(args.xdr_tau)):
            raise ValueError(
                "online canonical bank and signed-surrogate xDr are separate treatments"
            )
    if diayn_num_options > 1:
        if int(args.num_samples) % diayn_num_options != 0:
            raise ValueError("num_samples must divide evenly across DIAYN options")
        if maxent_alpha > 0:
            raise ValueError(
                "DIAYN answer-option MI and direct MaxEnt are separate treatments"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError("DIAYN answer-option MI and SEED are separate treatments")
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "DIAYN answer-option MI and token-entropy control are separate treatments"
            )
        if math.isfinite(float(args.xdr_tau)):
            raise ValueError(
                "DIAYN answer-option MI and signed-surrogate xDr are separate treatments"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError("DIAYN answer-option MI and SEED are separate treatments")
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "DIAYN answer-option MI and token entropy are separate treatments"
            )
        coverage_k = int(getattr(args, "eval_mode_coverage_k", 0) or 0)
        if coverage_k > 0 and coverage_k % diayn_num_options != 0:
            raise ValueError(
                "eval_mode_coverage_k must divide evenly across DIAYN options"
            )
        for name in (
            "diayn_mi_ema_decay",
            "diayn_mi_smoothing",
            "diayn_mi_bonus_clip",
        ):
            value = float(getattr(args, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= float(args.diayn_mi_ema_decay) < 1:
            raise ValueError("diayn_mi_ema_decay must be in [0, 1)")
        if float(args.diayn_mi_smoothing) <= 0:
            raise ValueError("diayn_mi_smoothing must be positive")
        if float(args.diayn_mi_bonus_clip) <= 0:
            raise ValueError("diayn_mi_bonus_clip must be positive")
    canonical_actions = canonical_task != "none"
    canonical_action_count = int(args.canonical_graph_action_count)
    canonical_learner_sampling = bool(args.canonical_graph_learner_sampling)
    canonical_fixed_shape_sampling = bool(args.canonical_graph_fixed_shape_sampling)
    replicated_freeform_sampling = bool(args.replicated_freeform_sampling)
    if replicated_freeform_sampling:
        if canonical_actions:
            raise ValueError(
                "replicated_freeform_sampling cannot use canonical actions"
            )
        if int(args.rollout_batch_size_per_device) != 1:
            raise ValueError(
                "replicated free-form sampling requires rollout_batch_size_per_device=1"
            )
        if int(args.num_gpus_per_actor) <= 1 and not bool(args.local_actor_weight_sync):
            raise ValueError(
                "replicated free-form sampling requires a multi-GPU actor or "
                "local_actor_weight_sync"
            )
    if online_canonical_counterfactual_proposals:
        canonical_pantry_proposals = (
            canonical_task == "pantry_support_mask"
            and canonical_learner_sampling
            and canonical_fixed_shape_sampling
        )
        if not replicated_freeform_sampling and not canonical_pantry_proposals:
            raise ValueError(
                "counterfactual canonical proposals require replicated "
                "free-form sampling or the fixed-shape Pantry learner sampler"
            )
    if bool(args.local_actor_weight_sync):
        if not replicated_freeform_sampling:
            raise ValueError(
                "local_actor_weight_sync requires replicated_freeform_sampling"
            )
        if int(args.num_gpus_per_actor) != 1:
            raise ValueError("local_actor_weight_sync requires num_gpus_per_actor=1")
    if int(args.vllm_sleep_level) not in {1, 2}:
        raise ValueError("vllm_sleep_level must be 1 or 2")
    if int(args.vllm_sleep_level) == 2:
        if not bool(args.vllm_sleep):
            raise ValueError("vllm_sleep_level=2 requires vllm_sleep")
        if not bool(args.local_actor_weight_sync):
            raise ValueError("vllm_sleep_level=2 requires local_actor_weight_sync")
        if int(args.sync_params_every) != 1:
            raise ValueError("vllm_sleep_level=2 requires sync_params_every=1")
    if canonical_actions:
        pantry_verified_maxent = canonical_task == "pantry_support_mask"
        if online_canonical_bank_active and not pantry_verified_maxent:
            raise ValueError("online growing-support banks require free-form rollouts")
        if outcome_collision_coef > 0:
            raise ValueError(
                "outcome-collision shaping currently requires free-form rollouts"
            )
        if semantic_shannon_coef > 0 and not pantry_verified_maxent:
            raise ValueError(
                "semantic Shannon shaping currently requires free-form rollouts"
            )
        if diayn_num_options > 1:
            raise ValueError(
                "DIAYN answer options currently require free-form rollouts"
            )
        if maxent_objective != "sequence":
            raise ValueError(
                "canonical finite policies require maxent_objective=sequence"
            )
        # The task pins the contract role; the chat surface is free so the same
        # canonical policy can run on either base-model family.
        allowed_templates = CANONICAL_TASK_PROMPT_TEMPLATES[canonical_task]
        if args.prompt_template not in allowed_templates:
            raise ValueError(
                f"canonical {canonical_task} actions require "
                f"prompt_template in {sorted(allowed_templates)}; "
                f"got {args.prompt_template}"
            )
        required_action_count = 6 if canonical_task == "pantry_support_mask" else 3
        if canonical_action_count != required_action_count:
            raise ValueError(
                "canonical finite policy action count mismatch: "
                f"task={canonical_task} canonical_graph_action_count="
                f"{required_action_count} required; observed={canonical_action_count}"
            )
        if args.test_split != "multi_answer":
            raise ValueError(
                "canonical finite policies require test_split=multi_answer"
            )
        if float(args.maxent_length_target) > 0:
            raise ValueError(
                "canonical fixed-horizon actions cannot use the response-length controller"
            )
        if float(args.top_p) != 1.0:
            raise ValueError("canonical actions require top_p=1")
        if int(args.top_k) != -1:
            raise ValueError("canonical actions require top_k=-1")
        if not math.isfinite(float(args.temperature)) or float(args.temperature) <= 0:
            raise ValueError("canonical actions require finite positive temperature")
        if canonical_learner_sampling and int(args.rollout_batch_size) != 1:
            raise ValueError(
                "canonical learner-side sampling requires rollout_batch_size=1"
            )
        if canonical_learner_sampling and not canonical_fixed_shape_sampling:
            raise ValueError(
                "canonical learner-side sampling requires the frozen fixed-shape "
                "causal-placeholder path"
            )
    elif prompt_template_role(args.prompt_template) in (
        CANONICAL_DIGIT_TEMPLATE_ROLES
    ):
        raise ValueError(
            "canonical digit prompt templates require canonical_action_task"
        )
    elif canonical_learner_sampling:
        raise ValueError(
            "canonical_graph_learner_sampling requires canonical_graph_actions "
            "or canonical_action_task"
        )
    elif canonical_fixed_shape_sampling:
        raise ValueError(
            "canonical_graph_fixed_shape_sampling requires canonical graph actions "
            "or canonical_action_task"
        )
    if canonical_fixed_shape_sampling and not canonical_learner_sampling:
        raise ValueError(
            "canonical_graph_fixed_shape_sampling requires learner-side sampling"
        )
    if math.isfinite(float(args.xdr_tau)) and getattr(args, "reinforce_update", False):
        raise ValueError(
            "xDr treatments require the maintained non-REINFORCE learner path"
        )
    if args.xdr_mode_adaptive and (
        not math.isfinite(float(args.xdr_tau)) or float(args.xdr_tau) <= 0
    ):
        raise ValueError("xdr_mode_adaptive requires a finite positive xdr_tau")
    target_ratio = float(args.xdr_tau_control_target_ratio)
    if not math.isfinite(target_ratio) or not 0 <= target_ratio <= 1:
        raise ValueError("xdr_tau_control_target_ratio must be in [0, 1]")
    controller_base_tau = float(args.xdr_tau)
    if target_ratio > 0:
        if not math.isfinite(controller_base_tau) or controller_base_tau <= 0:
            raise ValueError(
                "xDr tau control requires a finite positive base temperature"
            )
        if args.xdr_mode_adaptive:
            raise ValueError(
                "xdr_mode_adaptive and xDr tau control are separate treatments"
            )
        tau_min = float(args.xdr_tau_control_min)
        if not math.isfinite(tau_min) or tau_min <= 0 or tau_min > controller_base_tau:
            raise ValueError("xdr_tau_control_min must be in (0, xdr_tau]")
        if int(args.xdr_tau_control_warmup_steps) <= 0:
            raise ValueError("xdr_tau_control_warmup_steps must be positive")
        ema_decay = float(args.xdr_tau_control_ema_decay)
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("xdr_tau_control_ema_decay must be in [0, 1)")
        gain = float(args.xdr_tau_control_gain)
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("xdr_tau_control_gain must be finite and positive")
    sac_target_ratio = float(args.xdr_sac_dual_target_ratio)
    if not math.isfinite(sac_target_ratio) or not 0 <= sac_target_ratio <= 1:
        raise ValueError("xdr_sac_dual_target_ratio must be in [0, 1]")
    if target_ratio > 0 and sac_target_ratio > 0:
        raise ValueError(
            "proportional and SAC-dual xDr controllers are separate treatments"
        )
    if sac_target_ratio > 0:
        if not math.isfinite(controller_base_tau) or controller_base_tau <= 0:
            raise ValueError(
                "xDr SAC-dual control requires a finite positive base temperature"
            )
        if args.xdr_mode_adaptive:
            raise ValueError(
                "xdr_mode_adaptive and xDr SAC-dual control are separate treatments"
            )
        min_tau = float(args.xdr_sac_dual_min_tau)
        max_tau = float(args.xdr_sac_dual_max_tau)
        if not math.isfinite(min_tau) or min_tau <= 0 or min_tau > controller_base_tau:
            raise ValueError("xdr_sac_dual_min_tau must be in (0, xdr_tau]")
        if not math.isfinite(max_tau) or max_tau < controller_base_tau:
            raise ValueError(
                "xdr_sac_dual_max_tau must be at least the base temperature"
            )
        if int(args.xdr_sac_dual_warmup_steps) <= 0:
            raise ValueError("xdr_sac_dual_warmup_steps must be positive")
        alpha_lr = float(args.xdr_sac_dual_alpha_lr)
        if not math.isfinite(alpha_lr) or alpha_lr <= 0:
            raise ValueError("xdr_sac_dual_alpha_lr must be finite and positive")

    maxent_control_ratio = float(args.maxent_control_target_ratio)
    maxent_dual_ratio = float(args.maxent_dual_target_ratio)
    maxent_inverse_adaptation = bool(args.maxent_inverse_adaptation)
    maxent_control_target_entropy = float(args.maxent_control_target_entropy)
    maxent_dual_target_entropy = float(args.maxent_dual_target_entropy)
    for name, value in (
        ("maxent_control_target_ratio", maxent_control_ratio),
        ("maxent_dual_target_ratio", maxent_dual_ratio),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    for name, value in (
        ("maxent_control_target_entropy", maxent_control_target_entropy),
        ("maxent_dual_target_entropy", maxent_dual_target_entropy),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    active_maxent_controllers = sum(
        (
            maxent_control_ratio > 0,
            maxent_dual_ratio > 0,
            maxent_inverse_adaptation,
        )
    )
    if active_maxent_controllers > 1:
        raise ValueError(
            "proportional, Haarnoja-dual, and inverse MaxEnt controllers are "
            "separate treatments"
        )
    if maxent_control_target_entropy > 0 and maxent_control_ratio == 0:
        raise ValueError(
            "maxent_control_target_entropy requires proportional MaxEnt control"
        )
    if maxent_dual_target_entropy > 0 and maxent_dual_ratio == 0:
        raise ValueError(
            "maxent_dual_target_entropy requires Haarnoja-dual MaxEnt control"
        )
    if canonical_task != "none" and maxent_control_ratio > 0:
        if maxent_control_target_entropy <= 0:
            raise ValueError(
                "canonical proportional MaxEnt control requires an explicit "
                "positive maxent_control_target_entropy"
            )
    if canonical_task != "none" and maxent_dual_ratio > 0:
        if maxent_dual_target_entropy <= 0:
            raise ValueError(
                "canonical Haarnoja-dual MaxEnt control requires an explicit "
                "positive maxent_dual_target_entropy"
            )
    if canonical_task != "none":
        canonical_sequence_count = {
            "graph_coloring": 27,
            "countdown": 108,
            "pantry_support_mask": 64,
        }[canonical_task]
        canonical_max_entropy = math.log(canonical_sequence_count)
        for name, value in (
            ("maxent_control_target_entropy", maxent_control_target_entropy),
            ("maxent_dual_target_entropy", maxent_dual_target_entropy),
        ):
            if value > canonical_max_entropy + 1e-9:
                raise ValueError(
                    f"{name} cannot exceed the canonical {canonical_task} "
                    f"maximum log-support entropy {canonical_max_entropy:.12g}"
                )
    if active_maxent_controllers and maxent_alpha <= 0:
        raise ValueError("MaxEnt control requires a finite positive maxent_alpha")
    if maxent_inverse_adaptation:
        if int(args.maxent_inverse_warmup_steps) <= 0:
            raise ValueError("maxent_inverse_warmup_steps must be positive")
        inverse_ema_decay = float(args.maxent_inverse_ema_decay)
        if not math.isfinite(inverse_ema_decay) or not 0 <= inverse_ema_decay < 1:
            raise ValueError("maxent_inverse_ema_decay must be finite and in [0, 1)")
        if bool(args.online_canonical_policy_entropy_adaptation):
            raise ValueError(
                "direct inverse MaxEnt and canonical policy-entropy "
                "adaptation are separate treatments"
            )
    if maxent_control_ratio > 0:
        if int(args.maxent_control_warmup_steps) <= 0:
            raise ValueError("maxent_control_warmup_steps must be positive")
        max_alpha = float(args.maxent_control_max_alpha)
        if not math.isfinite(max_alpha) or max_alpha < maxent_alpha:
            raise ValueError("maxent_control_max_alpha must be at least maxent_alpha")
        ema_decay = float(args.maxent_control_ema_decay)
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("maxent_control_ema_decay must be in [0, 1)")
        gain = float(args.maxent_control_gain)
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("maxent_control_gain must be finite and positive")
    if maxent_dual_ratio > 0:
        if int(args.maxent_dual_warmup_steps) <= 0:
            raise ValueError("maxent_dual_warmup_steps must be positive")
        min_alpha = float(args.maxent_dual_min_alpha)
        max_alpha = float(args.maxent_dual_max_alpha)
        if not math.isfinite(min_alpha) or min_alpha <= 0 or min_alpha > maxent_alpha:
            raise ValueError("maxent_dual_min_alpha must be in (0, maxent_alpha]")
        if not math.isfinite(max_alpha) or max_alpha < maxent_alpha:
            raise ValueError("maxent_dual_max_alpha must be at least maxent_alpha")
        alpha_lr = float(args.maxent_dual_alpha_lr)
        if not math.isfinite(alpha_lr) or alpha_lr <= 0:
            raise ValueError("maxent_dual_alpha_lr must be finite and positive")
        ema_decay = float(args.maxent_dual_ema_decay)
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("maxent_dual_ema_decay must be in [0, 1)")
    maxent_length_target = float(args.maxent_length_target)
    if not math.isfinite(maxent_length_target) or maxent_length_target < 0:
        raise ValueError("maxent_length_target must be finite and non-negative")
    if maxent_length_target > 0:
        if maxent_alpha <= 0:
            raise ValueError("MaxEnt length control requires positive maxent_alpha")
        horizon = float(args.generate_max_length)
        if maxent_length_target < 1 or maxent_length_target > horizon:
            raise ValueError("maxent_length_target must be in [1, generate_max_length]")
        initial_lambda = float(args.maxent_length_lambda_init)
        max_lambda = float(args.maxent_length_lambda_max)
        if (
            not math.isfinite(initial_lambda)
            or initial_lambda < 0
            or not math.isfinite(max_lambda)
            or max_lambda <= 0
            or initial_lambda > max_lambda
        ):
            raise ValueError(
                "maxent_length_lambda_init must be in [0, maxent_length_lambda_max]"
            )
        length_ema_decay = float(args.maxent_length_ema_decay)
        if not math.isfinite(length_ema_decay) or not 0 <= length_ema_decay < 1:
            raise ValueError("maxent_length_ema_decay must be in [0, 1)")
        length_dual_lr = float(args.maxent_length_dual_lr)
        if not math.isfinite(length_dual_lr) or length_dual_lr <= 0:
            raise ValueError("maxent_length_dual_lr must be finite and positive")
        if bool(args.ignore_no_eos):
            raise ValueError(
                "MaxEnt length control must retain horizon-truncated no-EOS rows"
            )
    if float(args.policy_entropy_coef) < 0:
        raise ValueError("policy_entropy_coef must be non-negative")
    if float(args.seed_entropy_alpha) < 0:
        raise ValueError("seed_entropy_alpha must be non-negative")
    if math.isfinite(float(args.xdr_tau)) and float(args.seed_entropy_alpha) > 0:
        raise ValueError("xDr and SEED are separate arms; enable at most one")
    if int(args.eval_mode_coverage_k) < 0:
        raise ValueError("eval_mode_coverage_k must be non-negative")
    if float(args.eval_mode_coverage_temperature) < 0:
        raise ValueError("eval_mode_coverage_temperature must be non-negative")
    if int(args.eval_mode_coverage_draws) < 1:
        raise ValueError("eval_mode_coverage_draws must be positive")
    if int(args.eval_mode_coverage_seed) < 0:
        raise ValueError("eval_mode_coverage_seed must be non-negative")
    coverage_top_p = float(getattr(args, "eval_mode_coverage_top_p", 1.0))
    if not math.isfinite(coverage_top_p) or not 0 < coverage_top_p <= 1:
        raise ValueError("eval_mode_coverage_top_p must be finite and in (0, 1]")
    if bool(getattr(args, "eval_only", False)) and int(args.eval_mode_coverage_k) <= 0:
        # An eval-only run exists to measure sampled mode coverage. Without a
        # positive K it would load a checkpoint, evaluate greedy accuracy only,
        # and silently return no frontier point.
        raise ValueError("eval_only requires a positive eval_mode_coverage_k")
    if int(args.baseline_zero_adv_response_tokens) < 0:
        raise ValueError("baseline_zero_adv_response_tokens must be non-negative")
    return args
