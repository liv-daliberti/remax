#!/usr/bin/env bash
# Canonical Dr.GRPO/xDr.GRPO training entry point. Configuration is supplied
# through OAT_ZERO_* variables so Slurm arms can share this exact command.
set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPS_SNAPSHOT_ROOT="${OAT_ZERO_OPS_SNAPSHOT_ROOT:-$ROOT_DIR/ops}"
source "$OPS_SNAPSHOT_ROOT/repo_env.sh"

PYTHON_BIN="${OAT_ZERO_PYTHON:-$(command -v python)}"
PYTHON_LIB_DIR="${OAT_ZERO_PYTHON_LIB_DIR:-}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing training Python: $PYTHON_BIN" >&2
  exit 1
fi
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
# Launchpad's native Courier extension links libpython explicitly. Conda-backed
# venvs do not necessarily put that library on the dynamic loader's search path.
if [[ -z "$PYTHON_LIB_DIR" ]]; then
  PYTHON_LIB_DIR="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
fi
if [[ -n "$PYTHON_LIB_DIR" ]]; then export LD_LIBRARY_PATH="$PYTHON_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_NO_USAGE_STATS=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

PRETRAIN="${OAT_ZERO_PRETRAIN:-Qwen/Qwen2.5-0.5B-Instruct}"
PROMPT_DATA="${OAT_ZERO_PROMPT_DATA:?set OAT_ZERO_PROMPT_DATA}"
EVAL_DATA="${OAT_ZERO_EVAL_DATA:?set OAT_ZERO_EVAL_DATA}"
SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/var/data/xdr_$(date +%Y%m%d_%H%M%S)}"
XDR_TAU="${OAT_ZERO_XDR_TAU:-inf}"
XDR_TASK_ADVANTAGE_WEIGHTS="${OAT_ZERO_XDR_TASK_ADVANTAGE_WEIGHTS:-0}"
XDR_TAU_CONTROL_TARGET_RATIO="${OAT_ZERO_XDR_TAU_CONTROL_TARGET_RATIO:-0.0}"
XDR_TAU_CONTROL_WARMUP_STEPS="${OAT_ZERO_XDR_TAU_CONTROL_WARMUP_STEPS:-64}"
XDR_TAU_CONTROL_MIN="${OAT_ZERO_XDR_TAU_CONTROL_MIN:-0.005}"
XDR_TAU_CONTROL_EMA_DECAY="${OAT_ZERO_XDR_TAU_CONTROL_EMA_DECAY:-0.9}"
XDR_TAU_CONTROL_GAIN="${OAT_ZERO_XDR_TAU_CONTROL_GAIN:-20.0}"
XDR_SAC_DUAL_TARGET_RATIO="${OAT_ZERO_XDR_SAC_DUAL_TARGET_RATIO:-0.0}"
XDR_SAC_DUAL_WARMUP_STEPS="${OAT_ZERO_XDR_SAC_DUAL_WARMUP_STEPS:-64}"
XDR_SAC_DUAL_MIN_TAU="${OAT_ZERO_XDR_SAC_DUAL_MIN_TAU:-0.005}"
XDR_SAC_DUAL_MAX_TAU="${OAT_ZERO_XDR_SAC_DUAL_MAX_TAU:-0.5}"
XDR_SAC_DUAL_ALPHA_LR="${OAT_ZERO_XDR_SAC_DUAL_ALPHA_LR:-0.003}"
MAXENT_ALPHA="${OAT_ZERO_MAXENT_ALPHA:-0.0}"
MAXENT_OBJECTIVE="${OAT_ZERO_MAXENT_OBJECTIVE:-sequence}"
MAXENT_OBSERVE_MASKED_MEAN="${OAT_ZERO_MAXENT_OBSERVE_MASKED_MEAN_ENTROPY:-0}"
MAXENT_CONTROL_TARGET_RATIO="${OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO:-0.0}"
MAXENT_CONTROL_TARGET_ENTROPY="${OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY:-0.0}"
MAXENT_CONTROL_WARMUP_STEPS="${OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS:-64}"
MAXENT_CONTROL_MAX_ALPHA="${OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA:-0.5}"
MAXENT_CONTROL_EMA_DECAY="${OAT_ZERO_MAXENT_CONTROL_EMA_DECAY:-0.9}"
MAXENT_CONTROL_GAIN="${OAT_ZERO_MAXENT_CONTROL_GAIN:-1.0}"
MAXENT_DUAL_TARGET_RATIO="${OAT_ZERO_MAXENT_DUAL_TARGET_RATIO:-0.0}"
MAXENT_DUAL_TARGET_ENTROPY="${OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY:-0.0}"
MAXENT_DUAL_WARMUP_STEPS="${OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS:-64}"
MAXENT_DUAL_MIN_ALPHA="${OAT_ZERO_MAXENT_DUAL_MIN_ALPHA:-0.005}"
MAXENT_DUAL_MAX_ALPHA="${OAT_ZERO_MAXENT_DUAL_MAX_ALPHA:-0.5}"
MAXENT_DUAL_ALPHA_LR="${OAT_ZERO_MAXENT_DUAL_ALPHA_LR:-0.003}"
MAXENT_DUAL_EMA_DECAY="${OAT_ZERO_MAXENT_DUAL_EMA_DECAY:-0.7}"
MAXENT_INVERSE_ADAPTATION="${OAT_ZERO_MAXENT_INVERSE_ADAPTATION:-0}"
MAXENT_INVERSE_WARMUP_STEPS="${OAT_ZERO_MAXENT_INVERSE_WARMUP_STEPS:-64}"
MAXENT_INVERSE_EMA_DECAY="${OAT_ZERO_MAXENT_INVERSE_EMA_DECAY:-0.9}"
MAXENT_LENGTH_TARGET="${OAT_ZERO_MAXENT_LENGTH_TARGET:-0.0}"
MAXENT_LENGTH_LAMBDA_INIT="${OAT_ZERO_MAXENT_LENGTH_LAMBDA_INIT:-0.0}"
MAXENT_LENGTH_LAMBDA_MAX="${OAT_ZERO_MAXENT_LENGTH_LAMBDA_MAX:-0.02}"
MAXENT_LENGTH_EMA_DECAY="${OAT_ZERO_MAXENT_LENGTH_EMA_DECAY:-0.9}"
MAXENT_LENGTH_DUAL_LR="${OAT_ZERO_MAXENT_LENGTH_DUAL_LR:-0.0002}"
DIAYN_NUM_OPTIONS="${OAT_ZERO_DIAYN_NUM_OPTIONS:-0}"
DIAYN_MI_BETA="${OAT_ZERO_DIAYN_MI_BETA:-0.0}"
DIAYN_MI_EMA_DECAY="${OAT_ZERO_DIAYN_MI_EMA_DECAY:-0.9}"
DIAYN_MI_SMOOTHING="${OAT_ZERO_DIAYN_MI_SMOOTHING:-1.0}"
DIAYN_MI_BONUS_CLIP="${OAT_ZERO_DIAYN_MI_BONUS_CLIP:-5.0}"
DIAYN_MI_CORRECT_ONLY="${OAT_ZERO_DIAYN_MI_CORRECT_ONLY:-1}"
DIAYN_MI_LEAVE_ONE_OUT="${OAT_ZERO_DIAYN_MI_LEAVE_ONE_OUT:-0}"
OUTCOME_COLLISION_COEF="${OAT_ZERO_OUTCOME_COLLISION_COEF:-0.0}"
OUTCOME_COLLISION_OUTSIDE_CENTERING="${OAT_ZERO_OUTCOME_COLLISION_OUTSIDE_CENTERING:-0}"
SEMANTIC_SHANNON_COEF="${OAT_ZERO_SEMANTIC_SHANNON_COEF:-0.0}"
SEMANTIC_SHANNON_ALLOW_ZERO_COEFFICIENT_CONTROL="${OAT_ZERO_SEMANTIC_SHANNON_ALLOW_ZERO_COEFFICIENT_CONTROL:-0}"
SEMANTIC_SHANNON_SURPRISAL_CLIP="${OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP:-5.0}"
SEMANTIC_SHANNON_PSEUDOCOUNT="${OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT:-1.0}"
SEMANTIC_SHANNON_SEPARATE_ADVANTAGE="${OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE:-0}"
SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE="${OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE:-0}"
SEMANTIC_SHANNON_QUALITY_GATED_CAP="${OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_CAP:-0.05}"
SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE="${OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE:-0}"
SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP="${OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP:-0.05}"
SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE="${OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE:-0}"
SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE="${OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE:-0}"
SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK="${OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK:-0}"
SEMANTIC_RMS_CONTROL="${OAT_ZERO_SEMANTIC_RMS_CONTROL:-0}"
SEMANTIC_RMS_TARGET_RATIO="${OAT_ZERO_SEMANTIC_RMS_TARGET_RATIO:-0.05}"
SEMANTIC_RMS_MIN_COEFFICIENT="${OAT_ZERO_SEMANTIC_RMS_MIN_COEFFICIENT:-0.02}"
SEMANTIC_RMS_MAX_COEFFICIENT="${OAT_ZERO_SEMANTIC_RMS_MAX_COEFFICIENT:-0.40}"
SEMANTIC_RMS_EMA_DECAY="${OAT_ZERO_SEMANTIC_RMS_EMA_DECAY:-0.98}"
SEMANTIC_RMS_GAIN="${OAT_ZERO_SEMANTIC_RMS_GAIN:-0.5}"
SEMANTIC_RMS_MAX_STEP_RATIO="${OAT_ZERO_SEMANTIC_RMS_MAX_STEP_RATIO:-1.1}"
SEMANTIC_RMS_WARMUP_STEPS="${OAT_ZERO_SEMANTIC_RMS_WARMUP_STEPS:-64}"
SEMANTIC_RMS_MIN_ELIGIBLE_FRACTION="${OAT_ZERO_SEMANTIC_RMS_MIN_ELIGIBLE_FRACTION:-0.05}"
VERIFIED_DISCOVERY_TRACKING="${OAT_ZERO_VERIFIED_DISCOVERY_TRACKING:-1}"
ONLINE_CANONICAL_BANK_ALPHA="${OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA:-0.0}"
ONLINE_CANONICAL_BANK_PSEUDOCOUNT="${OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT:-1.0}"
ONLINE_CANONICAL_BANK_SURPRISAL_CLIP="${OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP:-5.0}"
ONLINE_CANONICAL_DUAL_TARGET_RATIO="${OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO:-0.0}"
ONLINE_CANONICAL_DUAL_MIN_ALPHA="${OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA:-0.005}"
ONLINE_CANONICAL_DUAL_MAX_ALPHA="${OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA:-0.5}"
ONLINE_CANONICAL_DUAL_ALPHA_LR="${OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR:-0.003}"
ONLINE_CANONICAL_DUAL_EMA_DECAY="${OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY:-0.9}"
ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION="${OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION:-0}"
ONLINE_CANONICAL_POLICY_ENTROPY_WARMUP_STEPS="${OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_WARMUP_STEPS:-64}"
ONLINE_CANONICAL_POLICY_ENTROPY_EMA_DECAY="${OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_EMA_DECAY:-0.9}"
ONLINE_CANONICAL_REPLAY="${OAT_ZERO_ONLINE_CANONICAL_REPLAY:-0}"
ONLINE_CANONICAL_REPLAY_ALPHA="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA:-0.1}"
ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED:-0}"
ONLINE_CANONICAL_REPLAY_PER_MODE_COEFFICIENT="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_PER_MODE_COEFFICIENT:-0.0325}"
ONLINE_CANONICAL_REPLAY_OBJECTIVE="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE:-bank_balance}"
ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING:-uniform}"
ONLINE_CANONICAL_REPLAY_CAPACITY="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY:-16}"
ONLINE_CANONICAL_REPLAY_BANK_FREEZE_STEP="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_FREEZE_STEP:-0}"
ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP:-0}"
ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS:-0}"
ONLINE_CANONICAL_REPLAY_MASS_ALPHA="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA:-0.1}"
ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE:-0}"
ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY:-0}"
ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS:-0}"
ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY:-0}"
ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT:-0}"
ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY:-0}"
ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS:-1}"
ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS:-0}"
ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS:-256}"
ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS:-3}"
ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE:-1.0}"
ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK:-0}"
ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_PATIENCE_UPDATES="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_PATIENCE_UPDATES:-64}"
ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK_MAX_ATTEMPTS="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK_MAX_ATTEMPTS:-4}"
ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_BURST_UPDATES="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_BURST_UPDATES:-16}"
ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_COOLDOWN_UPDATES="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_COOLDOWN_UPDATES:-48}"
ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS:-0}"
ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS="${OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS:-0}"
ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER="${OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER:-1.0}"
ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING="${OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING:-0}"
ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY="${OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY:-0}"
ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MISSED_ROLLOUT_OPPORTUNITIES="${OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MISSED_ROLLOUT_OPPORTUNITIES:-2}"
ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MEAN_LOGPROB_DROP="${OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MEAN_LOGPROB_DROP:-0.5}"
ONLINE_CANONICAL_PROPOSAL_RETENTION_REFRESH_VISITS="${OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_REFRESH_VISITS:-4}"
ONLINE_CANONICAL_PROPOSAL_RETENTION_SCORE_COOLDOWN_OBSERVATIONS="${OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_SCORE_COOLDOWN_OBSERVATIONS:-2}"
ONLINE_CANONICAL_KEY_MODE="${OAT_ZERO_ONLINE_CANONICAL_KEY_MODE:-modebench_outcome}"
VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE="${OAT_ZERO_VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE:-16}"
VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS="${OAT_ZERO_VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS:-2}"
VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP="${OAT_ZERO_VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP:-2.0}"
if [[ "$ONLINE_CANONICAL_KEY_MODE" == "verified_route" ]]; then
  COUNTERFACTUAL_TEMPERATURE_STEP=0
else
  COUNTERFACTUAL_TEMPERATURE_STEP=0.2
fi
MATH_STRATEGY_ENDPOINT="${OAT_ZERO_MATH_STRATEGY_ENDPOINT:-}"
MATH_STRATEGY_MODEL="${OAT_ZERO_MATH_STRATEGY_MODEL:-qwen2.5-72b}"
MATH_STRATEGY_TIMEOUT_SECONDS="${OAT_ZERO_MATH_STRATEGY_TIMEOUT_SECONDS:-600}"
MATH_STRATEGY_WORKERS="${OAT_ZERO_MATH_STRATEGY_WORKERS:-4}"
MATH_STRATEGY_MAX_ITEM_CHARS="${OAT_ZERO_MATH_STRATEGY_MAX_ITEM_CHARS:-4000}"
MATH_STRATEGY_GATE_TASK_REWARD="${OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD:-0}"
MATH_STRATEGY_ALLOW_UNSTRUCTURED_INFERENCE="${OAT_ZERO_MATH_STRATEGY_ALLOW_UNSTRUCTURED_INFERENCE:-0}"
CANONICAL_GRAPH_ACTIONS="${OAT_ZERO_CANONICAL_GRAPH_ACTIONS:-0}"
CANONICAL_ACTION_TASK="${OAT_ZERO_CANONICAL_ACTION_TASK:-none}"
CANONICAL_GRAPH_ACTION_COUNT="${OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT:-3}"
CANONICAL_GRAPH_LEARNER_SAMPLING="${OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING:-0}"
CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING="${OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING:-0}"
SEED_ALPHA="${OAT_ZERO_SEED_ENTROPY_ALPHA:-0.0}"
ENTROPY_COEF="${OAT_ZERO_POLICY_ENTROPY_COEF:-0.0}"
SEED="${OAT_ZERO_SEED:-42}"
PROMPT_MAX_LENGTH="${OAT_ZERO_PROMPT_MAX_LENGTH:-256}"
GENERATE_MAX_LENGTH="${OAT_ZERO_GENERATE_MAX_LENGTH:-192}"
EVAL_GENERATE_MAX_LENGTH="${OAT_ZERO_EVAL_GENERATE_MAX_LENGTH:-192}"
if (( GENERATE_MAX_LENGTH > EVAL_GENERATE_MAX_LENGTH )); then
  MAX_GENERATE_LENGTH="$GENERATE_MAX_LENGTH"
else
  MAX_GENERATE_LENGTH="$EVAL_GENERATE_MAX_LENGTH"
fi
# Qwen advertises a 32k context, so vLLM otherwise reserves KV cache for 32k
# tokens even though this campaign truncates prompts and completions below 512
# tokens. On 48 GB L40/A40 cards that needless reservation can fail engine
# initialization. Bound the engine to the longest sequence the run can create.
MODEL_CONTEXT_LENGTH="${OAT_ZERO_MAX_MODEL_LEN:-$((PROMPT_MAX_LENGTH + MAX_GENERATE_LENGTH))}"

# Most paper cohorts are standardized to at most five complete passes over the
# prompt pool. A prospectively frozen extension may raise the ceiling
# explicitly; keeping the default at five prevents inherited Slurm
# environments and old restart manifests from silently restoring older
# 16/24-pass budgets.
MAX_PROMPT_EPOCHS="${OAT_ZERO_MAX_PROMPT_EPOCHS:-5}"
if [[ ! "$MAX_PROMPT_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "OAT_ZERO_MAX_PROMPT_EPOCHS must be a positive integer; got ${MAX_PROMPT_EPOCHS}" >&2
  exit 1
fi
REQUESTED_PROMPT_EPOCHS="${OAT_ZERO_NUM_PROMPT_EPOCH:-$MAX_PROMPT_EPOCHS}"
if [[ ! "$REQUESTED_PROMPT_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "OAT_ZERO_NUM_PROMPT_EPOCH must be a positive integer; got ${REQUESTED_PROMPT_EPOCHS}" >&2
  exit 1
fi
if (( 10#$REQUESTED_PROMPT_EPOCHS > MAX_PROMPT_EPOCHS )); then
  echo "[train] capping prompt epochs ${REQUESTED_PROMPT_EPOCHS} -> ${MAX_PROMPT_EPOCHS}" >&2
  PROMPT_EPOCHS="$MAX_PROMPT_EPOCHS"
else
  PROMPT_EPOCHS="$((10#$REQUESTED_PROMPT_EPOCHS))"
fi
export OAT_ZERO_NUM_PROMPT_EPOCH="$PROMPT_EPOCHS"

for path in "$PROMPT_DATA" "$EVAL_DATA"; do
  if [[ ! -d "$path" ]]; then
    echo "Missing dataset: $path" >&2
    exit 1
  fi
done

if [[ ( "$CANONICAL_GRAPH_ACTIONS" == "1" || "$CANONICAL_ACTION_TASK" != "none" ) && "${VLLM_USE_V1:-1}" != "0" ]]; then
  echo "Canonical actions require VLLM_USE_V1=0 for audited rollout log probabilities." >&2
  exit 1
fi

bool_flag() {
  local value="$1" enabled="$2" disabled="$3"
  if [[ "$value" == "1" ]]; then
    printf '%s\n' "$enabled"
  else
    printf '%s\n' "$disabled"
  fi
}

flash_flag="$(bool_flag "${OAT_ZERO_ENABLE_FLASH_ATTN:-0}" --flash-attn --no-flash-attn)"
eos_flag="$(bool_flag "${OAT_ZERO_IGNORE_NO_EOS:-0}" --ignore-no-eos --no-ignore-no-eos)"
rlep_sparse_flag="$(
  bool_flag "${OAT_ZERO_RLEP_SPARSE_FALLBACK:-0}" \
    --rlep-sparse-fallback --no-rlep-sparse-fallback
)"
rlep_online_flag="$(
  bool_flag "${OAT_ZERO_RLEP_ONLINE_POOL:-0}" \
    --rlep-online-pool --no-rlep-online-pool
)"
maxrl_flag="$(
  bool_flag "${OAT_ZERO_MAXRL_TASK_OBJECTIVE:-0}" \
    --maxrl-task-objective --no-maxrl-task-objective
)"

cmd=(
  "$PYTHON_BIN" -m remax.train_zero_math
  --critic_type "${OAT_ZERO_CRITIC_TYPE:-drgrpo}"
  "$maxrl_flag"
  --gpus "${OAT_ZERO_N_GPU:-1}"
  --num_gpus_per_actor "${OAT_ZERO_NUM_GPUS_PER_ACTOR:-1}"
  --enable_prefix_caching
  --vllm_gpu_ratio "${OAT_ZERO_VLLM_GPU_RATIO:-0.25}"
  --max_model_len "$MODEL_CONTEXT_LENGTH"
  "$flash_flag"
  --shm_size_mb "${OAT_ZERO_SHM_SIZE_MB:-8000}"
  --gradient-checkpointing
  --bf16
  --no-rnd-seed --seed "$SEED"
  --learning_rate "${OAT_ZERO_LEARNING_RATE:-0.0000002}"
  --lr_scheduler "${OAT_ZERO_LR_SCHEDULER:-constant}"
  --lr_warmup_ratio "${OAT_ZERO_LR_WARMUP_RATIO:-0.0}"
  --max_step_adjustment "${OAT_ZERO_MAX_STEP_ADJUSTMENT:-1.0}"
  --adam_beta_1 "${OAT_ZERO_ADAM_BETA_1:-0.9}"
  --adam_beta_2 "${OAT_ZERO_ADAM_BETA_2:-0.95}"
  --l2 "${OAT_ZERO_L2:-0.0}"
  --num_ppo_epochs "${OAT_ZERO_NUM_PPO_EPOCHS:-1}"
  --sync_params_every "${OAT_ZERO_SYNC_PARAMS_EVERY:-1}"
  --max_norm "${OAT_ZERO_MAX_NORM:-1.0}"
  "$eos_flag"
  --beta "${OAT_ZERO_BETA:-0}"
  --policy-entropy-coef "$ENTROPY_COEF"
  --xdr-tau "$XDR_TAU"
  --xdr-tau-control-target-ratio "$XDR_TAU_CONTROL_TARGET_RATIO"
  --xdr-tau-control-warmup-steps "$XDR_TAU_CONTROL_WARMUP_STEPS"
  --xdr-tau-control-min "$XDR_TAU_CONTROL_MIN"
  --xdr-tau-control-ema-decay "$XDR_TAU_CONTROL_EMA_DECAY"
  --xdr-tau-control-gain "$XDR_TAU_CONTROL_GAIN"
  --xdr-sac-dual-target-ratio "$XDR_SAC_DUAL_TARGET_RATIO"
  --xdr-sac-dual-warmup-steps "$XDR_SAC_DUAL_WARMUP_STEPS"
  --xdr-sac-dual-min-tau "$XDR_SAC_DUAL_MIN_TAU"
  --xdr-sac-dual-max-tau "$XDR_SAC_DUAL_MAX_TAU"
  --xdr-sac-dual-alpha-lr "$XDR_SAC_DUAL_ALPHA_LR"
  --maxent-alpha "$MAXENT_ALPHA"
  --ucpo-tau "${OAT_ZERO_UCPO_TAU:-0.0}"
  --rlep-experience-root "${OAT_ZERO_RLEP_EXPERIENCE_ROOT:-}"
  --rlep-replay-count "${OAT_ZERO_RLEP_REPLAY_COUNT:-0}"
  "$rlep_sparse_flag"
  --maxent-control-target-ratio "$MAXENT_CONTROL_TARGET_RATIO"
  --maxent-control-target-entropy "$MAXENT_CONTROL_TARGET_ENTROPY"
  --maxent-control-warmup-steps "$MAXENT_CONTROL_WARMUP_STEPS"
  --maxent-control-max-alpha "$MAXENT_CONTROL_MAX_ALPHA"
  --maxent-control-ema-decay "$MAXENT_CONTROL_EMA_DECAY"
  --maxent-control-gain "$MAXENT_CONTROL_GAIN"
  --maxent-dual-target-ratio "$MAXENT_DUAL_TARGET_RATIO"
  --maxent-dual-target-entropy "$MAXENT_DUAL_TARGET_ENTROPY"
  --maxent-dual-warmup-steps "$MAXENT_DUAL_WARMUP_STEPS"
  --maxent-dual-min-alpha "$MAXENT_DUAL_MIN_ALPHA"
  --maxent-dual-max-alpha "$MAXENT_DUAL_MAX_ALPHA"
  --maxent-dual-alpha-lr "$MAXENT_DUAL_ALPHA_LR"
  --maxent-dual-ema-decay "$MAXENT_DUAL_EMA_DECAY"
  --seed-entropy-alpha "$SEED_ALPHA"
  --oracle_type reward --oracle math
  --pretrain "$PRETRAIN"
  --prompt_template "${OAT_ZERO_PROMPT_TEMPLATE:-qwen_boxed}"
  --verifier_version "${OAT_ZERO_VERIFIER_VERSION:-fast}"
  --modebench-domain "${OAT_ZERO_MODEBENCH_DOMAIN:-none}"
  --modebench-syntax-profile "${OAT_ZERO_MODEBENCH_SYNTAX_PROFILE:-none}"
  --zero-stage "${OAT_ZERO_ZERO_STAGE:-2}"
  --ref_offload
  --prompt_data "$PROMPT_DATA"
  --train_split train
  --input_key "${OAT_ZERO_INPUT_KEY:-problem}"
  --output_key "${OAT_ZERO_OUTPUT_KEY:-answer}"
  --max-train "${OAT_ZERO_MAX_TRAIN:-9216}"
  --max_queries "${OAT_ZERO_MAX_QUERIES:-100000}"
  --num_prompt_epoch "$PROMPT_EPOCHS"
  --prompt_max_length "$PROMPT_MAX_LENGTH"
  --num_samples "${OAT_ZERO_NUM_SAMPLES:-16}"
  --temperature "${OAT_ZERO_TEMPERATURE:-1.0}"
  --top_p "${OAT_ZERO_TOP_P:-1.0}"
  --generate_max_length "$GENERATE_MAX_LENGTH"
  --save_path "$SAVE_PATH"
  --save_steps "${OAT_ZERO_SAVE_STEPS:-64}"
  --save_from "${OAT_ZERO_SAVE_FROM:-64}"
  --max_save_num "${OAT_ZERO_MAX_SAVE_NUM:-4}"
  --max_save_mem "${OAT_ZERO_MAX_SAVE_MEM:-2000}"
  --train_batch_size "${OAT_ZERO_TRAIN_BATCH_SIZE:-16}"
  --train_batch_size_per_device "${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE:-16}"
  --rollout_batch_size "${OAT_ZERO_ROLLOUT_BATCH_SIZE:-1}"
  --rollout_batch_size_per_device "${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE:-1}"
  --pi_buffer_maxlen_per_device "${OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE:-16}"
  --eval_batch_size "${OAT_ZERO_EVAL_BATCH_SIZE:-64}"
  --eval_steps "${OAT_ZERO_EVAL_STEPS:-64}"
  --eval_temperature "${OAT_ZERO_EVAL_TEMPERATURE:-0}"
  --eval_generate_max_length "$EVAL_GENERATE_MAX_LENGTH"
  --eval_mode_coverage_k "${OAT_ZERO_EVAL_MODE_COVERAGE_K:-0}"
  --eval_mode_coverage_temperature "${OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE:-1.0}"
  --eval_mode_coverage_draws "${OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS:-4}"
  --eval_mode_coverage_seed "${OAT_ZERO_EVAL_MODE_COVERAGE_SEED:-1001}"
  --eval_data "$EVAL_DATA"
  --eval_input_key "${OAT_ZERO_EVAL_INPUT_KEY:-${OAT_ZERO_INPUT_KEY:-problem}}"
  --eval_output_key "${OAT_ZERO_EVAL_OUTPUT_KEY:-${OAT_ZERO_OUTPUT_KEY:-answer}}"
  --test_split "${OAT_ZERO_TEST_SPLIT:-all}"
)

# Python is snapshotted when a job is submitted, while this shell entrypoint
# may be newer than that snapshot. The MaxEnt objective selector was added
# after the original E4 xDr source was frozen. Omit this inert MaxEnt-only flag
# for a non-MaxEnt run against that older source; fail closed if a MaxEnt run
# ever tries to use a source tree that cannot represent its requested
# objective.
ARG_SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:-$ROOT_DIR/src}"
if grep -q 'maxent_objective' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(--maxent-objective "$MAXENT_OBJECTIVE")
  if [[ "$MAXENT_OBSERVE_MASKED_MEAN" == "1" ]]; then
    cmd+=(--maxent-observe-masked-mean-entropy)
  else
    cmd+=(--no-maxent-observe-masked-mean-entropy)
  fi
elif [[ "$MAXENT_ALPHA" != "0" && "$MAXENT_ALPHA" != "0.0" ]]; then
  echo "Frozen source lacks maxent_objective for an active MaxEnt run: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates maxent_objective; omitting inert flag"
fi
# E72 decoding frontier. Both flags are inert at their defaults, so a frozen
# source snapshot that predates them still runs every earlier protocol
# unchanged; only a run that actually requests them fails closed.
EVAL_ONLY="${OAT_ZERO_EVAL_ONLY:-0}"
EVAL_MODE_COVERAGE_TOP_P="${OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P:-1.0}"
if grep -q 'eval_mode_coverage_top_p' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(--eval-mode-coverage-top-p "$EVAL_MODE_COVERAGE_TOP_P")
elif [[ "$EVAL_MODE_COVERAGE_TOP_P" != "1.0" && "$EVAL_MODE_COVERAGE_TOP_P" != "1" ]]; then
  echo "Frozen source lacks eval_mode_coverage_top_p: $ARG_SOURCE_ROOT" >&2
  exit 1
fi
# Every terminal evaluation this repo has published was recorded before
# eval_mode_coverage_disjoint_draws existed, when consecutive draws seeded at
# base+draw_index shared vLLM child streams. That seeding is now off by default
# because it is wrong for a fresh run, but a sweep that re-measures a published
# checkpoint has to reproduce the draws the published number came from, or its
# T=1 column will not match the paper and its reproduction gate will fail for a
# reason that has nothing to do with decoding. Unset leaves the source default
# alone, so no existing protocol changes.
EVAL_MODE_COVERAGE_DISJOINT_DRAWS="${OAT_ZERO_EVAL_MODE_COVERAGE_DISJOINT_DRAWS:-}"
if [[ -n "$EVAL_MODE_COVERAGE_DISJOINT_DRAWS" ]]; then
  if grep -q 'eval_mode_coverage_disjoint_draws' "$ARG_SOURCE_ROOT/remax/args.py"; then
    if [[ "$EVAL_MODE_COVERAGE_DISJOINT_DRAWS" == "1" ]]; then
      cmd+=(--eval-mode-coverage-disjoint-draws)
    else
      cmd+=(--no-eval-mode-coverage-disjoint-draws)
    fi
  elif [[ "$EVAL_MODE_COVERAGE_DISJOINT_DRAWS" == "1" ]]; then
    # A source that predates the flag always shares streams, so it can satisfy
    # a request for 0 but never one for 1.
    echo "Frozen source lacks eval_mode_coverage_disjoint_draws: $ARG_SOURCE_ROOT" >&2
    exit 1
  fi
fi
if grep -q 'eval_only:' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$EVAL_ONLY" == "1" ]]; then
    cmd+=(--eval-only)
  else
    cmd+=(--no-eval-only)
  fi
elif [[ "$EVAL_ONLY" == "1" ]]; then
  echo "Frozen source lacks eval_only: $ARG_SOURCE_ROOT" >&2
  exit 1
fi

# DAPO is a default-off direct baseline. Preserve compatibility with every
# earlier immutable source snapshot, but fail closed if a run requests DAPO
# against code that cannot represent all four components of its recipe.
DAPO_ENABLED="${OAT_ZERO_DAPO_ENABLED:-0}"
if grep -q 'dapo_enabled:' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$DAPO_ENABLED" == "1" ]]; then
    cmd+=(--dapo-enabled)
  else
    cmd+=(--no-dapo-enabled)
  fi
  cmd+=(
    --dapo-clip-low "${OAT_ZERO_DAPO_CLIP_LOW:-0.20}"
    --dapo-clip-high "${OAT_ZERO_DAPO_CLIP_HIGH:-0.28}"
    --dapo-max-num-gen-batches "${OAT_ZERO_DAPO_MAX_NUM_GEN_BATCHES:-10}"
    --dapo-overlong-buffer-ratio \
      "${OAT_ZERO_DAPO_OVERLONG_BUFFER_RATIO:-0.20}"
    --dapo-overlong-penalty-factor \
      "${OAT_ZERO_DAPO_OVERLONG_PENALTY_FACTOR:-1.0}"
  )
elif [[ "$DAPO_ENABLED" == "1" ]]; then
  echo "Frozen source lacks DAPO support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates DAPO; omitting inert flags"
fi

# GAPO and SetPO are default-off diversity-preserving comparators. They follow
# DAPO's compatibility contract: older frozen snapshots stay runnable, but a run
# that asks for either against code that cannot represent it fails closed rather
# than training a silently different objective.
GAPO_ENABLED="${OAT_ZERO_GAPO_ENABLED:-0}"
if grep -q 'gapo_enabled:' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$GAPO_ENABLED" == "1" ]]; then
    cmd+=(--gapo-enabled)
  else
    cmd+=(--no-gapo-enabled)
  fi
  cmd+=(
    --gapo-support-index "${OAT_ZERO_GAPO_SUPPORT_INDEX:-}"
    --gapo-reward-scale "${OAT_ZERO_GAPO_REWARD_SCALE:-unit}"
  )
elif [[ "$GAPO_ENABLED" == "1" ]]; then
  echo "Frozen source lacks GAPO support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates GAPO; omitting inert flags"
fi

SETPO_COEFFICIENT="${OAT_ZERO_SETPO_COEFFICIENT:-0.0}"
if grep -q 'setpo_coefficient:' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(
    --setpo-coefficient "$SETPO_COEFFICIENT"
    --setpo-embedder-path "${OAT_ZERO_SETPO_EMBEDDER_PATH:-}"
    --setpo-embed-batch-size "${OAT_ZERO_SETPO_EMBED_BATCH_SIZE:-64}"
  )
elif [[ "$SETPO_COEFFICIENT" != "0.0" && "$SETPO_COEFFICIENT" != "0" ]]; then
  echo "Frozen source lacks SetPO support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates SetPO; omitting inert flags"
fi

if grep -q 'maxent_inverse_adaptation' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$MAXENT_INVERSE_ADAPTATION" == "1" ]]; then
    cmd+=(--maxent-inverse-adaptation)
  else
    cmd+=(--no-maxent-inverse-adaptation)
  fi
  cmd+=(
    --maxent-inverse-warmup-steps "$MAXENT_INVERSE_WARMUP_STEPS"
    --maxent-inverse-ema-decay "$MAXENT_INVERSE_EMA_DECAY"
  )
elif [[ "$MAXENT_INVERSE_ADAPTATION" == "1" ]]; then
  echo "Frozen source lacks direct inverse MaxEnt adaptation: $ARG_SOURCE_ROOT" >&2
  exit 1
fi

if grep -q 'diayn_num_options' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(
    --diayn-num-options "$DIAYN_NUM_OPTIONS"
    --diayn-mi-beta "$DIAYN_MI_BETA"
    --diayn-mi-ema-decay "$DIAYN_MI_EMA_DECAY"
    --diayn-mi-smoothing "$DIAYN_MI_SMOOTHING"
    --diayn-mi-bonus-clip "$DIAYN_MI_BONUS_CLIP"
  )
  if [[ "$DIAYN_MI_CORRECT_ONLY" == "1" ]]; then
    cmd+=(--diayn-mi-correct-only)
  else
    cmd+=(--no-diayn-mi-correct-only)
  fi
  if [[ "$DIAYN_MI_LEAVE_ONE_OUT" == "1" ]]; then
    cmd+=(--diayn-mi-leave-one-out)
  else
    cmd+=(--no-diayn-mi-leave-one-out)
  fi
elif [[ "$DIAYN_NUM_OPTIONS" != "0" || ( "$DIAYN_MI_BETA" != "0" && "$DIAYN_MI_BETA" != "0.0" ) ]]; then
  echo "Frozen source lacks DIAYN answer-option MI support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates DIAYN option MI; omitting inert flags"
fi

if grep -q 'outcome_collision_coef' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(--outcome-collision-coef "$OUTCOME_COLLISION_COEF")
elif [[ "$OUTCOME_COLLISION_COEF" != "0" && "$OUTCOME_COLLISION_COEF" != "0.0" ]]; then
  echo "Frozen source lacks outcome-collision support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates outcome collision; omitting inert flag"
fi

if grep -q 'outcome_collision_outside_centering' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$OUTCOME_COLLISION_OUTSIDE_CENTERING" == "1" ]]; then
    cmd+=(--outcome-collision-outside-centering)
  else
    cmd+=(--no-outcome-collision-outside-centering)
  fi
elif [[ "$OUTCOME_COLLISION_OUTSIDE_CENTERING" == "1" ]]; then
  echo "Frozen source lacks outside-centered outcome-collision support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates outside-centered outcome collision; omitting inert flag"
fi

if grep -q 'semantic_shannon_coef' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(
    --semantic-shannon-coef "$SEMANTIC_SHANNON_COEF"
    --semantic-shannon-surprisal-clip "$SEMANTIC_SHANNON_SURPRISAL_CLIP"
    --semantic-shannon-pseudocount "$SEMANTIC_SHANNON_PSEUDOCOUNT"
  )
elif [[ "$SEMANTIC_SHANNON_COEF" != "0" && "$SEMANTIC_SHANNON_COEF" != "0.0" ]]; then
  echo "Frozen source lacks semantic-Shannon support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates semantic Shannon; omitting inert flags"
fi

if grep -q 'semantic_shannon_separate_advantage' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$SEMANTIC_SHANNON_SEPARATE_ADVANTAGE" == "1" ]]; then
    cmd+=(--semantic-shannon-separate-advantage)
  else
    cmd+=(--no-semantic-shannon-separate-advantage)
  fi
elif [[ "$SEMANTIC_SHANNON_SEPARATE_ADVANTAGE" == "1" ]]; then
  echo "Frozen source lacks separate semantic-Shannon advantage support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates separate semantic-Shannon advantage; omitting inert flag"
fi

if grep -q 'semantic_shannon_allow_zero_coefficient_control:' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$SEMANTIC_SHANNON_ALLOW_ZERO_COEFFICIENT_CONTROL" == "1" ]]; then
    cmd+=(--semantic-shannon-allow-zero-coefficient-control)
  else
    cmd+=(--no-semantic-shannon-allow-zero-coefficient-control)
  fi
elif [[ "$SEMANTIC_SHANNON_ALLOW_ZERO_COEFFICIENT_CONTROL" == "1" ]]; then
  echo "Frozen source lacks semantic zero-coefficient controls: $ARG_SOURCE_ROOT" >&2
  exit 1
fi

if grep -q 'semantic_shannon_quality_gated_advantage' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(--semantic-shannon-quality-gated-cap "$SEMANTIC_SHANNON_QUALITY_GATED_CAP")
  if [[ "$SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE" == "1" ]]; then
    cmd+=(--semantic-shannon-quality-gated-advantage)
  else
    cmd+=(--no-semantic-shannon-quality-gated-advantage)
  fi
elif [[ "$SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE" == "1" ]]; then
  echo "Frozen source lacks quality-gated semantic-Shannon advantage support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates quality-gated semantic-Shannon advantage; omitting inert flags"
fi

if grep -q 'semantic_rms_control' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(
    --semantic-rms-target-ratio "$SEMANTIC_RMS_TARGET_RATIO"
    --semantic-rms-min-coefficient "$SEMANTIC_RMS_MIN_COEFFICIENT"
    --semantic-rms-max-coefficient "$SEMANTIC_RMS_MAX_COEFFICIENT"
    --semantic-rms-ema-decay "$SEMANTIC_RMS_EMA_DECAY"
    --semantic-rms-gain "$SEMANTIC_RMS_GAIN"
    --semantic-rms-max-step-ratio "$SEMANTIC_RMS_MAX_STEP_RATIO"
    --semantic-rms-warmup-steps "$SEMANTIC_RMS_WARMUP_STEPS"
    --semantic-rms-min-eligible-fraction "$SEMANTIC_RMS_MIN_ELIGIBLE_FRACTION"
  )
  if [[ "$SEMANTIC_RMS_CONTROL" == "1" ]]; then
    cmd+=(--semantic-rms-control)
  else
    cmd+=(--no-semantic-rms-control)
  fi
elif [[ "$SEMANTIC_RMS_CONTROL" == "1" ]]; then
  echo "Frozen source lacks adaptive semantic MaxEnt support: $ARG_SOURCE_ROOT" >&2
  exit 1
fi

if grep -q 'semantic_shannon_success_conditioned_signed_advantage' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(--semantic-shannon-success-conditioned-signed-cap "$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP")
  if [[ "$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE" == "1" ]]; then
    cmd+=(--semantic-shannon-success-conditioned-signed-advantage)
  else
    cmd+=(--no-semantic-shannon-success-conditioned-signed-advantage)
  fi
elif [[ "$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE" == "1" ]]; then
  echo "Frozen source lacks success-conditioned signed semantic-Shannon advantage support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates success-conditioned signed semantic-Shannon advantage; omitting inert flags"
fi

if grep -q 'semantic_shannon_success_conditioned_group_centered_advantage' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE" == "1" ]]; then
    cmd+=(--semantic-shannon-success-conditioned-group-centered-advantage)
  else
    cmd+=(--no-semantic-shannon-success-conditioned-group-centered-advantage)
  fi
elif [[ "$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE" == "1" ]]; then
  echo "Frozen source lacks group-centered semantic-Shannon advantage support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates group-centered semantic-Shannon advantage; omitting inert flag"
fi

if grep -q 'semantic_shannon_success_conditioned_verified_support_advantage' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE" == "1" ]]; then
    cmd+=(--semantic-shannon-success-conditioned-verified-support-advantage)
  else
    cmd+=(--no-semantic-shannon-success-conditioned-verified-support-advantage)
  fi
elif [[ "$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE" == "1" ]]; then
  echo "Frozen source lacks verified-support semantic-Shannon advantage support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates verified-support semantic-Shannon advantage; omitting inert flag"
fi

if grep -q 'semantic_shannon_verified_support_include_replay_bank' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK" == "1" ]]; then
    cmd+=(--semantic-shannon-verified-support-include-replay-bank)
  else
    cmd+=(--no-semantic-shannon-verified-support-include-replay-bank)
  fi
elif [[ "$SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK" == "1" ]]; then
  echo "Frozen source lacks replay-bank semantic-support coupling: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates replay-bank semantic-support coupling; omitting inert flag"
fi

if grep -q 'verified_discovery_tracking' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$VERIFIED_DISCOVERY_TRACKING" == "1" ]]; then
    cmd+=(--verified-discovery-tracking)
  else
    cmd+=(--no-verified-discovery-tracking)
  fi
else
  echo "[train] compatibility: frozen source predates passive verified-discovery tracking; omitting flag"
fi

if grep -q 'online_canonical_bank_alpha' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(
    --online-canonical-bank-alpha "$ONLINE_CANONICAL_BANK_ALPHA"
    --online-canonical-bank-pseudocount "$ONLINE_CANONICAL_BANK_PSEUDOCOUNT"
    --online-canonical-bank-surprisal-clip "$ONLINE_CANONICAL_BANK_SURPRISAL_CLIP"
    --online-canonical-key-mode "$ONLINE_CANONICAL_KEY_MODE"
  )
  if grep -q 'verified_route_replay_capacity_per_route:' "$ARG_SOURCE_ROOT/remax/args.py"; then
    cmd+=(
      --verified-route-replay-capacity-per-route \
        "$VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE"
      --verified-route-recurring-min-neutral-prompts \
        "$VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS"
      --verified-route-proposal-max-mean-logprob-drop \
        "$VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP"
    )
  elif [[ "$ONLINE_CANONICAL_KEY_MODE" == "verified_route" ]]; then
    echo "Frozen source lacks verified-route replay support: $ARG_SOURCE_ROOT" >&2
    exit 1
  fi
  if grep -q 'online_canonical_dual_target_ratio' "$ARG_SOURCE_ROOT/remax/args.py"; then
    cmd+=(
      --online-canonical-dual-target-ratio "$ONLINE_CANONICAL_DUAL_TARGET_RATIO"
      --online-canonical-dual-min-alpha "$ONLINE_CANONICAL_DUAL_MIN_ALPHA"
      --online-canonical-dual-max-alpha "$ONLINE_CANONICAL_DUAL_MAX_ALPHA"
      --online-canonical-dual-alpha-lr "$ONLINE_CANONICAL_DUAL_ALPHA_LR"
      --online-canonical-dual-ema-decay "$ONLINE_CANONICAL_DUAL_EMA_DECAY"
    )
  elif [[ "$ONLINE_CANONICAL_DUAL_TARGET_RATIO" != "0" && "$ONLINE_CANONICAL_DUAL_TARGET_RATIO" != "0.0" ]]; then
    echo "Frozen source lacks normalized online-canonical dual support: $ARG_SOURCE_ROOT" >&2
    exit 1
  fi
  if grep -q 'online_canonical_policy_entropy_adaptation' "$ARG_SOURCE_ROOT/remax/args.py"; then
    if [[ "$ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION" == "1" ]]; then
      cmd+=(--online-canonical-policy-entropy-adaptation)
    else
      cmd+=(--no-online-canonical-policy-entropy-adaptation)
    fi
    cmd+=(
      --online-canonical-policy-entropy-warmup-steps "$ONLINE_CANONICAL_POLICY_ENTROPY_WARMUP_STEPS"
      --online-canonical-policy-entropy-ema-decay "$ONLINE_CANONICAL_POLICY_ENTROPY_EMA_DECAY"
    )
  elif [[ "$ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION" == "1" ]]; then
    echo "Frozen source lacks online-canonical policy-entropy adaptation: $ARG_SOURCE_ROOT" >&2
    exit 1
  fi
  if grep -q 'online_canonical_replay:' "$ARG_SOURCE_ROOT/remax/args.py"; then
    if [[ "$ONLINE_CANONICAL_REPLAY" == "1" ]]; then
      cmd+=(--online-canonical-replay)
    else
      cmd+=(--no-online-canonical-replay)
    fi
    cmd+=(--online-canonical-replay-alpha "$ONLINE_CANONICAL_REPLAY_ALPHA")
    # Guarded the same way as the compute-only control: a content-addressed
    # source snapshot taken before this arm existed has no such field, and
    # passing the flag to it would fail the run rather than fall back.
    if grep -q 'online_canonical_replay_bank_normalized:' \
      "$ARG_SOURCE_ROOT/remax/args.py"; then
      cmd+=(--online-canonical-replay-per-mode-coefficient \
        "$ONLINE_CANONICAL_REPLAY_PER_MODE_COEFFICIENT")
      if [[ "$ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED" == "1" ]]; then
        cmd+=(--online-canonical-replay-bank-normalized)
      else
        cmd+=(--no-online-canonical-replay-bank-normalized)
      fi
    elif [[ "$ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED" == "1" ]]; then
      echo "Frozen source lacks bank-normalized replay: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
    # The online RLEP pool (E135) is newer than most frozen runtimes; only a
    # source tree that declares the field is handed the flag.
    if grep -q 'rlep_online_pool:' "$ARG_SOURCE_ROOT/remax/args.py"; then
      cmd+=("$rlep_online_flag")
    elif [[ "${OAT_ZERO_RLEP_ONLINE_POOL:-0}" == "1" ]]; then
      echo "OAT_ZERO_RLEP_ONLINE_POOL=1 but $ARG_SOURCE_ROOT has no rlep_online_pool" >&2
      exit 1
    fi
    if grep -q 'online_canonical_replay_objective:' "$ARG_SOURCE_ROOT/remax/args.py"; then
      cmd+=(--online-canonical-replay-objective "$ONLINE_CANONICAL_REPLAY_OBJECTIVE")
    elif [[ "$ONLINE_CANONICAL_REPLAY_OBJECTIVE" != "bank_balance" ]]; then
      echo "Frozen source lacks the requested canonical replay objective: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
    if grep -q 'online_canonical_replay_key_weighting:' "$ARG_SOURCE_ROOT/remax/args.py"; then
      cmd+=(--online-canonical-replay-key-weighting "$ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING")
    elif [[ "$ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING" != "uniform" ]]; then
      echo "Frozen source lacks requested canonical replay key weighting: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
    cmd+=(
      --online-canonical-replay-capacity "$ONLINE_CANONICAL_REPLAY_CAPACITY"
    )
    if grep -q 'online_canonical_replay_bank_freeze_step:' "$ARG_SOURCE_ROOT/remax/args.py"; then
      cmd+=(--online-canonical-replay-bank-freeze-step "$ONLINE_CANONICAL_REPLAY_BANK_FREEZE_STEP")
    elif [[ "$ONLINE_CANONICAL_REPLAY_BANK_FREEZE_STEP" != "0" ]]; then
      echo "Frozen source lacks canonical replay bank freezing: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
    if grep -q 'online_canonical_replay_global_groups_per_step:' "$ARG_SOURCE_ROOT/remax/args.py"; then
      cmd+=(
        --online-canonical-replay-global-groups-per-step \
          "$ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP"
      )
    elif [[ "$ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP" != "0" ]]; then
      echo "Frozen source lacks global canonical replay scheduling: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
    if grep -q 'online_canonical_replay_global_bootstrap_steps:' "$ARG_SOURCE_ROOT/remax/args.py"; then
      cmd+=(
        --online-canonical-replay-global-bootstrap-steps \
          "$ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS"
      )
    elif [[ "$ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS" != "0" ]]; then
      echo "Frozen source lacks finite global replay bootstrap: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
    if grep -q 'online_canonical_replay_mass_alpha:' "$ARG_SOURCE_ROOT/remax/args.py"; then
      cmd+=(
        --online-canonical-replay-mass-alpha "$ONLINE_CANONICAL_REPLAY_MASS_ALPHA"
      )
    elif [[ "$ONLINE_CANONICAL_REPLAY_OBJECTIVE" == "split_mass_balance_per_rollout" ]]; then
      echo "Frozen source lacks split canonical replay mass control: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
    if grep -q 'online_canonical_replay_retention_safe_balance:' "$ARG_SOURCE_ROOT/remax/args.py"; then
      if [[ "$ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE" == "1" ]]; then
        cmd+=(--online-canonical-replay-retention-safe-balance)
      else
        cmd+=(--no-online-canonical-replay-retention-safe-balance)
      fi
    elif [[ "$ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE" == "1" ]]; then
      echo "Frozen source lacks retention-safe replay balance: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
    if grep -q 'online_canonical_replay_compute_only:' "$ARG_SOURCE_ROOT/remax/args.py"; then
      if [[ "$ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY" == "1" ]]; then
        cmd+=(--online-canonical-replay-compute-only)
      else
        cmd+=(--no-online-canonical-replay-compute-only)
      fi
    elif [[ "$ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY" == "1" ]]; then
      echo "Frozen source lacks replay compute-only controls: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
    if grep -q 'online_canonical_counterfactual_proposals:' "$ARG_SOURCE_ROOT/remax/args.py"; then
      if [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS" == "1" ]]; then
        cmd+=(--online-canonical-counterfactual-proposals)
      else
        cmd+=(--no-online-canonical-counterfactual-proposals)
      fi
      if grep -q 'online_canonical_counterfactual_admission_compute_only:' "$ARG_SOURCE_ROOT/remax/args.py"; then
        if [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY" == "1" ]]; then
          cmd+=(--online-canonical-counterfactual-admission-compute-only)
        else
          cmd+=(--no-online-canonical-counterfactual-admission-compute-only)
        fi
      elif [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY" == "1" ]]; then
        echo "Frozen source lacks proposal-admission compute-only controls: $ARG_SOURCE_ROOT" >&2
        exit 1
      fi
      if grep -q 'online_canonical_counterfactual_separate_objective_support:' "$ARG_SOURCE_ROOT/remax/args.py"; then
        if [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT" == "1" ]]; then
          cmd+=(--online-canonical-counterfactual-separate-objective-support)
        else
          cmd+=(--no-online-canonical-counterfactual-separate-objective-support)
        fi
      elif [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT" == "1" ]]; then
        echo "Frozen source lacks separated proposal/objective support: $ARG_SOURCE_ROOT" >&2
        exit 1
      fi
      if grep -q 'online_canonical_counterfactual_transform_proposals:' "$ARG_SOURCE_ROOT/remax/args.py"; then
        if [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS" == "1" ]]; then
          cmd+=(--online-canonical-counterfactual-transform-proposals)
        else
          cmd+=(--no-online-canonical-counterfactual-transform-proposals)
        fi
      elif [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS" != "1" ]]; then
        echo "Frozen source lacks transform-proposal controls: $ARG_SOURCE_ROOT" >&2
        exit 1
      fi
      if grep -q 'online_canonical_counterfactual_exact_grammar_transforms:' "$ARG_SOURCE_ROOT/remax/args.py"; then
        if [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS" == "1" ]]; then
          cmd+=(--online-canonical-counterfactual-exact-grammar-transforms)
        else
          cmd+=(--no-online-canonical-counterfactual-exact-grammar-transforms)
        fi
      elif [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS" != "0" ]]; then
        echo "Frozen source lacks exact-grammar proposal transforms: $ARG_SOURCE_ROOT" >&2
        exit 1
      fi
      cmd+=(
        --online-canonical-counterfactual-anchor-max-tokens \
          "$ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS"
        --online-canonical-counterfactual-max-attempts \
          "$ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS"
        --online-canonical-counterfactual-sampling-temperature \
          "$ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE"
      )
      if grep -q 'online_canonical_counterfactual_starvation_fallback:' "$ARG_SOURCE_ROOT/remax/args.py"; then
        if [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK" == "1" ]]; then
          cmd+=(--online-canonical-counterfactual-starvation-fallback)
        else
          cmd+=(--no-online-canonical-counterfactual-starvation-fallback)
        fi
        cmd+=(
          --online-canonical-counterfactual-starvation-patience-updates \
            "$ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_PATIENCE_UPDATES"
          --online-canonical-counterfactual-starvation-fallback-max-attempts \
            "$ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK_MAX_ATTEMPTS"
          --online-canonical-counterfactual-starvation-burst-updates \
            "$ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_BURST_UPDATES"
          --online-canonical-counterfactual-starvation-cooldown-updates \
            "$ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_COOLDOWN_UPDATES"
        )
      elif [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK" == "1" ]]; then
        echo "Frozen source lacks proposal starvation fallback: $ARG_SOURCE_ROOT" >&2
        exit 1
      fi
      if grep -q 'online_canonical_counterfactual_fixed_control_groups:' "$ARG_SOURCE_ROOT/remax/args.py"; then
        cmd+=(
          --online-canonical-counterfactual-fixed-control-groups \
            "$ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS"
        )
      elif [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS" != "0" ]]; then
        echo "Frozen source lacks fixed counterfactual compute controls: $ARG_SOURCE_ROOT" >&2
        exit 1
      fi
      if grep -q 'online_canonical_proposal_replay_priority_visits:' "$ARG_SOURCE_ROOT/remax/args.py"; then
        cmd+=(
          --online-canonical-proposal-replay-priority-visits "$ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS"
          --online-canonical-proposal-replay-priority-multiplier "$ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER"
        )
      elif [[ "$ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS" != "0" ]]; then
        echo "Frozen source lacks proposal replay priority: $ARG_SOURCE_ROOT" >&2
        exit 1
      fi
      if grep -q 'online_canonical_proposal_retention_tracking:' "$ARG_SOURCE_ROOT/remax/args.py"; then
        if [[ "$ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING" == "1" ]]; then
          cmd+=(--online-canonical-proposal-retention-tracking)
        else
          cmd+=(--no-online-canonical-proposal-retention-tracking)
        fi
        if [[ "$ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY" == "1" ]]; then
          cmd+=(--online-canonical-proposal-adaptive-retention-priority)
        else
          cmd+=(--no-online-canonical-proposal-adaptive-retention-priority)
        fi
        cmd+=(
          --online-canonical-proposal-retention-max-missed-rollout-opportunities "$ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MISSED_ROLLOUT_OPPORTUNITIES"
          --online-canonical-proposal-retention-max-mean-logprob-drop "$ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MEAN_LOGPROB_DROP"
          --online-canonical-proposal-retention-refresh-visits "$ONLINE_CANONICAL_PROPOSAL_RETENTION_REFRESH_VISITS"
          --online-canonical-proposal-retention-score-cooldown-observations "$ONLINE_CANONICAL_PROPOSAL_RETENTION_SCORE_COOLDOWN_OBSERVATIONS"
        )
      elif [[ "$ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING" == "1" || "$ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY" == "1" ]]; then
        echo "Frozen source lacks proposal admission-retention tracking: $ARG_SOURCE_ROOT" >&2
        exit 1
      fi
      if grep -q 'online_canonical_counterfactual_singleton_only:' "$ARG_SOURCE_ROOT/remax/args.py"; then
        if [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY" == "1" ]]; then
          cmd+=(--online-canonical-counterfactual-singleton-only)
        else
          cmd+=(--no-online-canonical-counterfactual-singleton-only)
        fi
      elif [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY" == "1" ]]; then
        echo "Frozen source lacks singleton-only proposals: $ARG_SOURCE_ROOT" >&2
        exit 1
      fi
    elif [[ "$ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS" == "1" ]]; then
      echo "Frozen source lacks counterfactual canonical proposals: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
  elif [[ "$ONLINE_CANONICAL_REPLAY" == "1" ]]; then
    echo "Frozen source lacks verified canonical replay support: $ARG_SOURCE_ROOT" >&2
    exit 1
  fi
  if grep -q 'math_strategy_endpoint' "$ARG_SOURCE_ROOT/remax/args.py"; then
    cmd+=(
      --math-strategy-endpoint "$MATH_STRATEGY_ENDPOINT"
      --math-strategy-model "$MATH_STRATEGY_MODEL"
      --math-strategy-timeout-seconds "$MATH_STRATEGY_TIMEOUT_SECONDS"
      --math-strategy-workers "$MATH_STRATEGY_WORKERS"
      --math-strategy-max-item-chars "$MATH_STRATEGY_MAX_ITEM_CHARS"
    )
    if grep -q 'math_strategy_gate_task_reward' "$ARG_SOURCE_ROOT/remax/args.py"; then
      if [[ "$MATH_STRATEGY_GATE_TASK_REWARD" == "1" ]]; then
        cmd+=(--math-strategy-gate-task-reward)
      else
        cmd+=(--no-math-strategy-gate-task-reward)
      fi
    elif [[ "$MATH_STRATEGY_GATE_TASK_REWARD" == "1" ]]; then
      echo "Frozen source lacks MATH strategy task-reward gating: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
    if grep -q 'math_strategy_allow_unstructured_inference' "$ARG_SOURCE_ROOT/remax/args.py"; then
      if [[ "$MATH_STRATEGY_ALLOW_UNSTRUCTURED_INFERENCE" == "1" ]]; then
        cmd+=(--math-strategy-allow-unstructured-inference)
      else
        cmd+=(--no-math-strategy-allow-unstructured-inference)
      fi
    elif [[ "$MATH_STRATEGY_ALLOW_UNSTRUCTURED_INFERENCE" == "1" ]]; then
      echo "Frozen source lacks unstructured MATH menu inference: $ARG_SOURCE_ROOT" >&2
      exit 1
    fi
  elif [[ "$ONLINE_CANONICAL_KEY_MODE" == "math_strategy_qwen72" ]]; then
    echo "Frozen source lacks MATH strategy canonicalization: $ARG_SOURCE_ROOT" >&2
    exit 1
  fi
elif [[ "$ONLINE_CANONICAL_BANK_ALPHA" != "0" && "$ONLINE_CANONICAL_BANK_ALPHA" != "0.0" ]]; then
  echo "Frozen source lacks online canonical bank support: $ARG_SOURCE_ROOT" >&2
  exit 1
else
  echo "[train] compatibility: frozen source predates online canonical banks; omitting inert flags"
fi

if grep -q 'resume_steps:' "$ARG_SOURCE_ROOT/remax/args.py"; then
  cmd+=(
    --export-steps "${OAT_ZERO_EXPORT_STEPS:-0}"
    --export-from "${OAT_ZERO_EXPORT_FROM:-0}"
    --resume-steps "${OAT_ZERO_RESUME_STEPS:--1}"
    --resume-from "${OAT_ZERO_RESUME_FROM:-0}"
    --max-export-num "${OAT_ZERO_MAX_EXPORT_NUM:-1}"
    --max-resume-num "${OAT_ZERO_MAX_RESUME_NUM:-1}"
    --max-export-mem "${OAT_ZERO_MAX_EXPORT_MEM:-64}"
    --max-resume-mem "${OAT_ZERO_MAX_RESUME_MEM:-256}"
  )
  if [[ "${OAT_ZERO_PRUNE_RESUME_ON_SUCCESS:-1}" == "0" ]]; then
    cmd+=(--no-prune-resume-on-success)
  fi
else
  echo "[train] compatibility: frozen source uses coupled save/export retention"
fi

# Add length-control arguments only for explicitly constrained runs so older
# snapshots without that later CLI surface remain valid.
if [[ "$MAXENT_LENGTH_TARGET" != "0" && "$MAXENT_LENGTH_TARGET" != "0.0" ]]; then
  cmd+=(
    --maxent-length-target "$MAXENT_LENGTH_TARGET"
    --maxent-length-lambda-init "$MAXENT_LENGTH_LAMBDA_INIT"
    --maxent-length-lambda-max "$MAXENT_LENGTH_LAMBDA_MAX"
    --maxent-length-ema-decay "$MAXENT_LENGTH_EMA_DECAY"
    --maxent-length-dual-lr "$MAXENT_LENGTH_DUAL_LR"
  )
fi

if [[ "$CANONICAL_GRAPH_ACTIONS" == "1" || "$CANONICAL_ACTION_TASK" != "none" ]]; then
  if [[ "$CANONICAL_ACTION_TASK" != "none" ]]; then
    cmd+=(--canonical-action-task "$CANONICAL_ACTION_TASK")
  fi
  if [[ "$CANONICAL_GRAPH_ACTIONS" == "1" ]]; then
    cmd+=(--canonical-graph-actions)
  fi
  cmd+=(
    --canonical-graph-action-count "$CANONICAL_GRAPH_ACTION_COUNT"
  )
  if [[ "$CANONICAL_GRAPH_LEARNER_SAMPLING" == "1" ]]; then
    cmd+=(--canonical-graph-learner-sampling)
  fi
  if [[ "$CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING" == "1" ]]; then
    cmd+=(--canonical-graph-fixed-shape-sampling)
  fi
fi

if [[ "${OAT_ZERO_REPLICATED_FREEFORM_SAMPLING:-0}" == "1" ]]; then
  cmd+=(--replicated-freeform-sampling)
fi
if [[ "${OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC:-0}" == "1" ]]; then
  cmd+=(--local-actor-weight-sync)
fi
cmd+=(--vllm-sleep-level "${OAT_ZERO_VLLM_SLEEP_LEVEL:-1}")

if [[ "${OAT_ZERO_XDR_MODE_ADAPTIVE:-0}" == "1" ]]; then
  cmd+=(--xdr-mode-adaptive)
fi
if grep -q 'xdr_task_advantage_weights' "$ARG_SOURCE_ROOT/remax/args.py"; then
  if [[ "$XDR_TASK_ADVANTAGE_WEIGHTS" == "1" ]]; then
    cmd+=(--xdr-task-advantage-weights)
  else
    cmd+=(--no-xdr-task-advantage-weights)
  fi
elif [[ "$XDR_TASK_ADVANTAGE_WEIGHTS" == "1" ]]; then
  echo "Frozen source lacks task-only xDr weighting support: $ARG_SOURCE_ROOT" >&2
  exit 1
fi
if [[ "${OAT_ZERO_VLLM_SLEEP:-1}" == "1" ]]; then
  cmd+=(--vllm_sleep)
fi
if [[ "${OAT_ZERO_COLLOCATE:-1}" == "1" ]]; then
  cmd+=(--collocate)
fi
if [[ "${OAT_ZERO_ADAM_OFFLOAD:-0}" == "1" ]]; then
  cmd+=(--adam_offload)
fi
if [[ "${OAT_ZERO_ACTIVATION_OFFLOADING:-0}" == "1" ]]; then
  cmd+=(--activation_offloading)
fi
if [[ "${OAT_ZERO_DISABLE_TRACE_CACHE:-0}" == "1" ]]; then
  cmd+=(--disable_trace_cache)
fi
if [[ "${OAT_ZERO_SAVE_CKPT:-0}" == "1" ]]; then
  cmd+=(--save-ckpt)
fi
if [[ -n "${OAT_ZERO_RESUME_DIR:-}" ]]; then
  cmd+=(--resume_dir "$OAT_ZERO_RESUME_DIR")
  if [[ -n "${OAT_ZERO_RESUME_TAG:-}" ]]; then
    cmd+=(--resume_tag "$OAT_ZERO_RESUME_TAG")
  fi
fi
if [[ "${OAT_ZERO_USE_WB:-0}" == "1" ]]; then
  cmd+=(
    --use-wb
    --wb_project "${OAT_ZERO_WB_PROJECT:-maxent-grpo}"
    --wb-run-name "${OAT_ZERO_WB_RUN_NAME:-xdr-run}"
  )
fi

echo "[train] model=$PRETRAIN tau=$XDR_TAU seed_alpha=$SEED_ALPHA entropy_coef=$ENTROPY_COEF"
echo "[train] outcome_collision_coef=$OUTCOME_COLLISION_COEF outside_centering=$OUTCOME_COLLISION_OUTSIDE_CENTERING"
echo "[train] semantic_shannon_coef=$SEMANTIC_SHANNON_COEF allow_zero_coefficient_control=$SEMANTIC_SHANNON_ALLOW_ZERO_COEFFICIENT_CONTROL surprisal_clip=$SEMANTIC_SHANNON_SURPRISAL_CLIP pseudocount=$SEMANTIC_SHANNON_PSEUDOCOUNT separate_advantage=$SEMANTIC_SHANNON_SEPARATE_ADVANTAGE success_conditioned_signed_advantage=$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE success_conditioned_group_centered_advantage=$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE success_conditioned_verified_support_advantage=$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE verified_support_include_replay_bank=$SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK semantic_estimator=mode_selected coefficient_control=fixed"
echo "[train] online_canonical_replay=$ONLINE_CANONICAL_REPLAY balance_alpha=$ONLINE_CANONICAL_REPLAY_ALPHA mass_alpha=$ONLINE_CANONICAL_REPLAY_MASS_ALPHA retention_safe_balance=$ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE replay_objective=$ONLINE_CANONICAL_REPLAY_OBJECTIVE replay_key_weighting=$ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING replay_capacity=$ONLINE_CANONICAL_REPLAY_CAPACITY replay_bank_freeze_step=$ONLINE_CANONICAL_REPLAY_BANK_FREEZE_STEP replay_bank_normalized=$ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED replay_per_mode_coefficient=$ONLINE_CANONICAL_REPLAY_PER_MODE_COEFFICIENT replay_compute_only=$ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY global_groups_per_step=$ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP global_bootstrap_steps=$ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS coefficient_control=fixed counterfactual_proposals=$ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS counterfactual_admission_compute_only=$ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY counterfactual_singleton_only=$ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY counterfactual_transform_proposals=$ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS counterfactual_exact_grammar_transforms=$ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS proposal_priority_visits=$ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS proposal_priority_multiplier=$ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER proposal_retention_tracking=$ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING adaptive_retention_priority=$ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY retention_max_misses=$ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MISSED_ROLLOUT_OPPORTUNITIES retention_max_mean_logprob_drop=$ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MEAN_LOGPROB_DROP retention_refresh_visits=$ONLINE_CANONICAL_PROPOSAL_RETENTION_REFRESH_VISITS gold_support_feedback=none proposal_rows_to_ppo=0 control_rows_to_ppo=0"
echo "[train] online_canonical_bank_alpha=$ONLINE_CANONICAL_BANK_ALPHA pseudocount=$ONLINE_CANONICAL_BANK_PSEUDOCOUNT surprisal_clip=$ONLINE_CANONICAL_BANK_SURPRISAL_CLIP key_mode=$ONLINE_CANONICAL_KEY_MODE"
echo "[train] verified_route_capacity_per_route=$VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE recurring_min_neutral_prompts=$VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS proposal_max_mean_logprob_drop=$VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP"
echo "[train] verified_discovery_tracking=$VERIFIED_DISCOVERY_TRACKING objective_influence=zero_for_plain_drgrpo"
echo "[train] online_canonical_dual_target_ratio=$ONLINE_CANONICAL_DUAL_TARGET_RATIO alpha_bounds=[$ONLINE_CANONICAL_DUAL_MIN_ALPHA,$ONLINE_CANONICAL_DUAL_MAX_ALPHA] alpha_lr=$ONLINE_CANONICAL_DUAL_ALPHA_LR ema_decay=$ONLINE_CANONICAL_DUAL_EMA_DECAY sensor=postupdate_H_over_log_support"
echo "[train] tau_control_target_ratio=$XDR_TAU_CONTROL_TARGET_RATIO warmup=$XDR_TAU_CONTROL_WARMUP_STEPS min_tau=$XDR_TAU_CONTROL_MIN ema_decay=$XDR_TAU_CONTROL_EMA_DECAY gain=$XDR_TAU_CONTROL_GAIN"
echo "[train] sac_dual_target_ratio=$XDR_SAC_DUAL_TARGET_RATIO warmup=$XDR_SAC_DUAL_WARMUP_STEPS tau_bounds=[$XDR_SAC_DUAL_MIN_TAU,$XDR_SAC_DUAL_MAX_TAU] alpha_lr=$XDR_SAC_DUAL_ALPHA_LR"
echo "[train] maxent_alpha=$MAXENT_ALPHA control_ratio=$MAXENT_CONTROL_TARGET_RATIO control_target=$MAXENT_CONTROL_TARGET_ENTROPY control_max=$MAXENT_CONTROL_MAX_ALPHA dual_ratio=$MAXENT_DUAL_TARGET_RATIO dual_target=$MAXENT_DUAL_TARGET_ENTROPY dual_bounds=[$MAXENT_DUAL_MIN_ALPHA,$MAXENT_DUAL_MAX_ALPHA] dual_ema_decay=$MAXENT_DUAL_EMA_DECAY inverse=$MAXENT_INVERSE_ADAPTATION inverse_warmup=$MAXENT_INVERSE_WARMUP_STEPS inverse_ema_decay=$MAXENT_INVERSE_EMA_DECAY inverse_projection=none"
if [[ "$DIAYN_NUM_OPTIONS" != "0" || ( "$DIAYN_MI_BETA" != "0" && "$DIAYN_MI_BETA" != "0.0" ) ]]; then
  echo "[train] diayn_num_options=$DIAYN_NUM_OPTIONS mi_beta=$DIAYN_MI_BETA mi_ema_decay=$DIAYN_MI_EMA_DECAY mi_smoothing=$DIAYN_MI_SMOOTHING mi_bonus_clip=$DIAYN_MI_BONUS_CLIP correct_only=$DIAYN_MI_CORRECT_ONLY leave_one_out=$DIAYN_MI_LEAVE_ONE_OUT"
fi
if [[ "$MAXENT_LENGTH_TARGET" != "0" && "$MAXENT_LENGTH_TARGET" != "0.0" ]]; then
  echo "[train] maxent_length_target=$MAXENT_LENGTH_TARGET lambda_init=$MAXENT_LENGTH_LAMBDA_INIT lambda_max=$MAXENT_LENGTH_LAMBDA_MAX ema_decay=$MAXENT_LENGTH_EMA_DECAY dual_lr=$MAXENT_LENGTH_DUAL_LR"
fi
if [[ "$CANONICAL_GRAPH_ACTIONS" == "1" ]]; then
  echo "[train] canonical_graph_actions=1 action_count=$CANONICAL_GRAPH_ACTION_COUNT learner_sampling=$CANONICAL_GRAPH_LEARNER_SAMPLING fixed_shape_sampling=$CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING max_entropy=3.295836866004329 vllm_use_v1=${VLLM_USE_V1:-unset}"
fi
if [[ "$CANONICAL_ACTION_TASK" != "none" ]]; then
  echo "[train] canonical_action_task=$CANONICAL_ACTION_TASK action_count=$CANONICAL_GRAPH_ACTION_COUNT learner_sampling=$CANONICAL_GRAPH_LEARNER_SAMPLING fixed_shape_sampling=$CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING vllm_use_v1=${VLLM_USE_V1:-unset}"
fi
echo "[train] data=$PROMPT_DATA save=$SAVE_PATH seed=$SEED"
if [[ -n "${OAT_ZERO_PROTOCOL_IDENTITY:-}" ]]; then
  echo "[train] protocol_identity=${OAT_ZERO_PROTOCOL_IDENTITY} sha256=$(sha256sum "${OAT_ZERO_PROTOCOL_IDENTITY}" | cut -d' ' -f1)"
fi
if [[ -n "${OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES:-}" ]]; then
  echo "[train] target_optimizer_updates=${OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES}"
fi
echo "[train] prompt_epochs=${PROMPT_EPOCHS} ceiling=${MAX_PROMPT_EPOCHS}"
echo "[train] max_model_len=${MODEL_CONTEXT_LENGTH} prompt_max=${PROMPT_MAX_LENGTH} train_generate_max=${GENERATE_MAX_LENGTH} eval_generate_max=${EVAL_GENERATE_MAX_LENGTH}"
if [[ "${OAT_ZERO_DRY_RUN:-0}" == "1" ]]; then
  printf '[train] command:'
  printf ' %q' "${cmd[@]}" "$@"
  printf '\n'
  exit 0
fi


exec "${cmd[@]}" "$@"
