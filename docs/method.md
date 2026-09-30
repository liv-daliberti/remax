# Method and implementation

Re:Max and Re:Dr add verified replay to a fresh-sample RL objective. The maintained recipes do not require an exhaustive catalogue of correct solutions, a desired mode count, or evaluation feedback for bank admission.

## Verify, retain, revisit

1. Generate responses to a training prompt using the current policy.
2. Execute the domain validator. A retained exemplar must be active, validator-positive, and associated with an admitted canonical outcome key.
3. Retain an exemplar under that prompt and mode key, subject to the configured capacity and admission rules. Invalid or inactive responses cannot enter the bank.
4. Visit eligible prompt banks through the deterministic recurrent schedule. Its cursor is included in the checkpoint state.
5. Teacher-force the retained exemplars and apply the configured replay derivative alongside the fresh-sample objective.

The exported recipes configure one global replay prompt group per optimizer step, capacity 16, no finite bootstrap cutoff, and `verified_likelihood_per_rollout`. They disable the historical bank-entropy bonus and counterfactual-proposal intervention. The bank implementation also contains other experimental modes; inspect the recipe before interpreting a run.

ModeBench supplies executable correctness and identity. Canonical keys are prompt-local. Two programs can share a mode if they produce the same verified output; a different string is not automatically a new solution mode.

## Replay objective

Let `s[g, m]` be the current model's length-normalized teacher-forced log score for the retained exemplar of mode m in scheduled prompt bank g. The uniform verified-likelihood primitive minimizes:

```text
L_replay = mean over scheduled banks g [ mean over retained modes m (-s[g, m]) ]
```

With G banks and M_g retained modes in bank g, the derivative with respect to each score is `-1 / (G * M_g)`. Gradient descent therefore raises every retained exemplar's score. Singleton banks are valid for this likelihood objective even though within-bank entropy is undefined.

The learner applies the configured coefficient and its registered per-rollout/gradient-accumulation scaling. Reproducing training requires those settings as well as the primitive above; copying this equation alone does not reproduce the integrated update.

The core recipes use uniform key weighting and replay coefficient 0.1. A frequency-weighted option and historical bank-balance objective exist in the module for retained comparisons. `canonical_replay_uniform_loss` is the historical bank-balance loss; the maintained verified-likelihood primitive is **`canonical_replay_uniform_verified_likelihood_loss`**.

Teacher-forced exemplar scores are not exact canonical-mode sampling probabilities. The method rehearses observed modes; it does not certify discovery or coverage of every correct mode, and the included implementation is not itself a proof of a neural-network convergence guarantee.

### Runnable gradient example

```python
import torch
from remax.core import canonical_replay_uniform_verified_likelihood_loss

scores = torch.tensor([-1.0, -3.0, -2.0], requires_grad=True)
# Prompt A has two retained modes; prompt B has one.
result = canonical_replay_uniform_verified_likelihood_loss(scores, [2, 1])
result.loss.backward()
assert result.loss.item() == 2.0
assert torch.allclose(scores.grad, torch.tensor([-0.25, -0.25, -0.5]))
```

This example checks the score-space derivative. It does not tokenize a prompt, run a model, or replace the learner's scaling and optimization code. The same smoke example is available as `python examples/replay_loss.py`.

## Binary MaxRL

For a prompt group of N responses with binary rewards r_i and K successes, the implemented advantage is:

```text
A_i = N * r_i / K - 1    when K > 0
A_i = 0                 when K = 0
```

`binary_maxrl_advantages` requires a finite, binary tensor of shape `[prompts, samples_per_prompt]`, with at least two samples per prompt.

```python
from remax.core import binary_maxrl_advantages

advantages = binary_maxrl_advantages(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
assert advantages.tolist() == [[1.0, -1.0], [0.0, 0.0]]
```

Re:Max selects this fresh-sample objective plus verified replay. Re:Dr uses Dr.GRPO plus the same replay intervention.

## Compute-matched controls

The control arms retain bank state, scheduling, exemplar scoring, and backward traversal. In the learner, `online_canonical_replay_compute_only` replaces the applied replay score gradients with exact zeros. The auxiliary derivative is therefore zero even though the replay path still consumes computation.

Disabling bank bookkeeping or skipping the replay pass is a different control. The recipe regression tests check objective selection and the compute-only flag across all 20 exported configurations.

## Training conformance

```sh
make conformance
# Before reviewing a refactor, also compare to the branch base:
python ops/check_training_reference.py --base-ref origin/main
```

The suite executes the production `_grpo_learning_step_with_progress` path, including the installed ModeBench validator, bank admission/update, global scheduler, replay materialization, teacher-forced scoring, fresh advantages, replay backward passes, and a real PyTorch SGD update. It covers all four maintained methods. No model download, generation, or GPU is needed.

[Version 1 inputs](../tests/fixtures/training_v1/inputs.json) and [expected traces](../tests/fixtures/training_v1/expected.json) freeze **12 sequential updates** at source commit `e4c7b50980e1a2a33c67de613b68a1e013ce5639`. Each method runs three steps:

1. Admit two Countdown modes and a duplicate canonical alias. Select the deterministic exemplar despite different response lengths; reject an incorrect response.
2. Observe another prompt with a valid singleton, an actor-negative valid answer, a verifier-negative actor-positive answer, and an inactive valid answer. Only the singleton enters the bank. Global scheduling still replays the earlier prompt.
3. Observe an all-failure group. Replay the second prompt's singleton from memory, leaving fresh discovery counts unchanged.

The toy tokenizer maps fixed token sequences to real Countdown responses. The model is an eight-token causal bigram table, initialized by `((7*i) % 23 - 11) / 13` for flattened parameter index `i`. EOS is token 7 and padding is token 0. Unequal prompt/response lengths and unused trailing padding expose mask and normalization mistakes. This isolates the training contract from tokenizer downloads and stochastic generation; it is not a test of a pretrained tokenizer.

Two references protect the result. Frozen traces compare bank counts, exemplars, schedule state, input labels/masks, scores, losses, all 64 parameter gradients, and updated parameters. A separate [scalar oracle](../tests/training_oracle.py), with no PyTorch or ReMax imports, differentiates categorical probabilities directly. For these one-epoch same-snapshot cases, fresh PPO ratios are one and clipping is inactive:

```text
fresh gradient = mean_i [-A_i * active_i / Tmax * sum_t grad log p(y_it)]
replay gradient = -alpha * (N-1)/N² * mean_banks mean_modes mean_response_tokens grad log p(y_t)
SGD update     = parameters - learning_rate * (fresh gradient + replay gradient)
```

The checks vary microbatch size (1, 2, 4), accumulation width, rollout count (4 and 16), temperature, and replay coefficient (including zero). They also cover empty/singleton banks, replay capacity versus discovery counts, candidate permutation, bank/scheduler restoration, and the deterministic replicated-layout branch with a real one-rank Gloo group. Restoring this small SGD state does not qualify the full training checkpoint/resume mechanism.

Controls must perform both detached and differentiable replay scoring and every replay backward call. Instrumentation checks actual calls, their order, model eval/train mode, and gradient increments. Each control replay increment must be exactly zero, and its resulting parameters must be **bitwise equal** to a fresh-only update. Treatment gradients must still be nonzero when fresh rewards are all zero but a singleton is available.

Discrete identities and masks compare exactly. Float32 scores, gradients, and parameters use `rtol=2e-6`, `atol=2e-7`; the control's zero-gradient and bitwise checks have no tolerance. Baseline gradients were checked against the scalar oracle before freezing. These are implementation references, not independent human scientific approval or evidence that the paper's specification is correct.

The [CPU adapter](../tests/training_harness.py) substitutes only external infrastructure: OAT's argument base, two statistical helpers, completion-mask/reduction interfaces, CUDA device selection, and the DeepSpeed accumulation/step contract. It imports and executes the production ReMax methods; it does not copy their implementations. Its strategy divides every backward call by the accumulation width and steps only at a microbatch boundary. Actual autograd and SGD run in PyTorch. Multi-rank collectives, ZeRO/offload, mixed precision, Adam state, actor generation, multiple PPO epochs, and historical comparator branches require separate integration qualification.

CI runs conformance with the ordinary regression suite on Python 3.10–3.12. A separate PR-base comparison rejects changing or deleting existing reference files, **even if their local hashes are refreshed**. Keep version 1 immutable. An intentional method change needs an explicitly versioned contract, new fixtures/tests, and scientific review explaining the difference; changing expected numbers to make a refactor pass is not acceptable.

## ModeBench boundary and failures

The maintained methods call `modebench.api.grade(Task(...), response)` through [benchmark.py](../src/remax/benchmark.py), pinned to ModeBench **0.4.0**, commit `33cfc3fd1732bd500fe13f13555419557aa248e2`. They derive correctness and canonical identity from the same structured verdict. Private grading helpers are confined to historical proposal/route compatibility; they do not decide rewards or bank admission in the 20 maintained recipes.

| Verifier status | Reward / action |
| --- | --- |
| `correct` | Reward 1; retain the returned canonical key |
| `incorrect`, `malformed` | Reward 0; no canonical key |
| `timeout`, `invalid_reference`, `worker_failure`, `resource_limit` | Raise `EvaluationFailure` with the original diagnostic; no numeric reward |
| Unknown or inconsistent reply, missing rows/diagnostics | Treat as worker failure and abort |

`EvaluationFailure.diagnostic` survives pickle/RPC and JSON transport, including status, detail, and row/context where available. Neither the thread collector nor the full-MATH worker substitutes zero after failure. The full worker bounds writes and partial reads, reaps failed processes, and allows a subsequent request to start a new worker. It does not silently retry a failed request. ModeBench owns its own bounded worker; its calls bypass the historical one-second ordinary-MATH thread deadline. Legacy full-MATH internal deadlines are also surfaced if `math_verify` catches its own timeout.

Successful diagnostics accompany actor trajectories and sampled-evaluation rows. The learner checks transported failures before bank mutation or optimization, and independently revalidates bank candidates through the supported API. Old trajectories without diagnostic fields remain readable and still receive independent admission validation. A failed admission batch does not commit its discoveries or apply an optimizer update.

Sampled evaluation requires complete diagnostic/reward/key rows before computing metrics. Rank-local failures are exchanged before score broadcasts. The top-level evaluator raises and appends `evaluation_failures.jsonl` with `status: "failed"`, the step, diagnostic, and `metrics: null`. There is no completed aggregate for that evaluation. Earlier successful draw records may already exist; a failure record invalidates the **complete evaluation at that step**, and those partial draws must not be presented as a completed result. Restart or rerun explicitly after fixing the environment; never replace failures with incorrect answers.

Historical recipes supply reference-only **decoded** responses. For Countdown, Graph, Python, and MathIR, the public validators have level-independent response semantics. Historical Pantry actors convert Level 1 support masks into allocations before grading. The compatibility adapter uses the public allocation contract (the Level 2+ Pantry grading surface) for these decoded strings; the dataset and experiment remain Level 1. New callers grading raw registered masks should supply their actual level/domain through `grade_task(Task(...), response)`. The adapter never guesses a surface from the generated answer.

The [175 historical cases](../tests/fixtures/benchmark_boundary_v1.json) were captured **before** upgrading, with the old ModeBench commit and fixture-source digest recorded. They cover accepted/rejected answers and formatting across all 25 level/domain cells, including Pantry decoding. Every frozen reward and admission key remains identical under the new API, as do the 12 frozen training updates. Legacy R1/ORZ tag gates still produce genuine malformed-answer verdicts when formatting fails. Invalid references and environment failures intentionally change from ambiguous failure behavior to explicit exceptions. The `formatted` diagnostic now follows the structured malformed status; it does not change the binary reward. Compatibility fixtures retain their ModeBench dataset provenance; see the pinned [dataset license and attribution](https://github.com/liv-daliberti/modeBench/blob/33cfc3fd1732bd500fe13f13555419557aa248e2/DATA_LICENSE).

```sh
make boundary
python ops/check_training_reference.py --base-ref origin/main
```

The PR-base guard protects both training and boundary fixtures from rewriting, including attempts to refresh their checksums. CPU tests exercise real worker crashes, deadlines, blocked writes, partial/malformed replies, cleanup/restart, actor reward calls, failed bank admission, transported diagnostics, and evaluation failure records. OAT/vLLM are represented by test adapters; multi-rank failure exchange is simulated. A fresh GPU/DeepSpeed or multi-process distributed training/evaluation run remains a separate qualification.

## Source map

Start with [`integrations/oat/grpo.py`](../src/remax/integrations/oat/grpo.py). Its maintained step checks verifier diagnostics, admits/schedules exemplars, scores the rollout policy, computes fresh advantages, and calls the optimizer adapter. It contains no comparator reward or objective branches.

```text
ModeBench verdicts + sampled trajectories
                 ↓
OAT admission → core bank → core scheduler
                 ↓
OAT scoring → core replay objective → OAT backward/optimizer
                 ↓
OAT checkpoint adapter → core atomic checkpoint protocol
```

The `remax.core` API imports without OAT, vLLM, DeepSpeed, Transformers, or comparator implementations. PyTorch is required for tensor objectives and scoring masks. It accepts verified identities and token sequences; a caller must supply trusted verification results before admission. The OAT adapter performs that independent ModeBench check, preserves fatal diagnostics, and intersects verifier-positive identities with active, positive-reward rows.

| Responsibility | Maintained implementation |
| --- | --- |
| Fresh MaxRL advantage and verified-likelihood objective | [`core/objectives.py`](../src/remax/core/objectives.py) |
| Admission and deterministic exemplars | [`core/admission.py`](../src/remax/core/admission.py) |
| Persistent bank and typed replay groups | [`core/bank.py`](../src/remax/core/bank.py), [`core/bank_types.py`](../src/remax/core/bank_types.py) |
| Prompt-local/global replay selection and cursor | [`core/scheduling.py`](../src/remax/core/scheduling.py) |
| Causal teacher-forcing masks and typed tensor results | [`core/scoring.py`](../src/remax/core/scoring.py), [`core/replay_types.py`](../src/remax/core/replay_types.py) |
| Bank schema and atomic checkpoint protocol | [`core/bank_state.py`](../src/remax/core/bank_state.py), [`core/checkpoints.py`](../src/remax/core/checkpoints.py) |
| Trajectory admission and replay selection | [`integrations/oat/admission.py`](../src/remax/integrations/oat/admission.py) |
| Behavior-policy checks and mean response-token scores | [`integrations/oat/scoring.py`](../src/remax/integrations/oat/scoring.py) |
| Fresh PPO loss, accumulation boundary and optimizer step | [`integrations/oat/update.py`](../src/remax/integrations/oat/update.py) |
| Detached/live replay scoring and backward scaling | [`integrations/oat/replay.py`](../src/remax/integrations/oat/replay.py) |
| Historical metric names, detached from method arithmetic | [`integrations/oat/telemetry.py`](../src/remax/integrations/oat/telemetry.py) |
| Lifecycle, data, generation and actor synchronization | `integrations/oat/{lifecycle,data,sampling,sync}.py` |
| Evaluation, progress and framework checkpoint state | `integrations/oat/{evaluation,progress,checkpoints}.py` |
| Supported ModeBench API and failure policy | [`benchmark.py`](../src/remax/benchmark.py) |
| Authenticated launch and complete effective configuration | [`ops/run_recipe.py`](../ops/run_recipe.py) |

Replay runs once at each optimizer boundary, **after fresh backward and before optimizer step**. Both replay score passes temporarily use eval mode and restore the previous model mode. The detached pass obtains the score derivative; the live pass applies it in bounded chunks. OAT divides every backward by the accumulation width, so the replay adapter compensates by that width after applying `alpha * (N-1)/N²`. Compute-only controls traverse the same scoring/backward calls with exactly zero score derivatives. These details are covered by frozen conformance and exact historical/extracted-path comparisons.

### Historical implementations and compatibility

[`experiments/`](../src/remax/experiments) owns comparator objectives and [`experiments/oat/`](../src/remax/experiments/oat) owns their historical integration, proposal generation and controller initialization. The maintained OAT update does not import them. [`selection.py`](../src/remax/integrations/oat/selection.py) lists the settings/state that require the historical adapter; dispatch records its selection and reasons on the learner. Strict identity-bound recipes reject a historical fallback. Legacy direct invocations can still select the retained implementation.

| Retained implementation | Dependency boundary checked before moving |
| --- | --- |
| DAPO, UCPO, xDr, SEED, SetPO, on-policy MaxEnt | Tensor arithmetic and standard library; shared aggregation diagnostics now live in `core/metrics.py` |
| GAPO and semantic-Shannon tracking | Historical outcome-collision identity sentinel; GAPO additionally consumes its support-index data |
| Outcome-collision shaping | Standard-library identity/counting helpers |
| RLEP | Its own frequency-preserving trajectory pool and state; loaded for historical pool updates/restores |
| SetPO embedder | Transformers/model loading remains lazy and confined to the historical embedder |
| Alternative replay objectives | Depend on core tensor result types; the core never imports these alternatives |
| Proposal/route/controller integration | Kept in the historical OAT adapter, with shared verification and checkpoint compatibility surfaces |

The bank retains historical schema fields and optional admission/retention features so old state dictionaries remain readable; the maintained method uses zero bank-entropy shaping and disables proposal interventions. Shared verification, route compatibility helpers, actor interfaces and argument definitions remain available at existing paths. This extraction does not qualify every historical experiment as a maintained method.

Old imports such as `remax.online_canonical_bank`, `remax.canonical_replay`, `remax.maxrl`, comparator root modules, and `remax.learner.*` remain compatibility surfaces. New integrations should import `remax.core`; contributors should edit the implementation module, not the compatibility file. Strict resume identities still include source hashes: a checkpoint from before this refactor is not an authorized continuation under the new source, even though its bank schema is readable.

`tests/test_replay_architecture.py` blocks training-framework/comparator imports from the core, runs all four maintained methods while rejecting comparator imports, checks dispatch guards, and compares complete update traces against the retained historical learner. Existing frozen numerical references are unchanged. `make check` runs these checks together with full-run resume and verifier-failure conformance.

For a GPU refactor audit, run the bounded `ops/resume_gpu.py` workflow at both revisions and compare the whole-run artifacts without restoring across revisions:

```sh
python tests/compare_training_runs.py outputs/before outputs/after
# A single-domain subset can use --methods remax.
```

This requires identical non-source run identities, exact response/replay/evaluation traces, exact bank/RNG/scheduler/progress state, and the declared model/optimizer tensor tolerances at update six. Each checkpoint's own manifest is verified. It is an offline comparison, not permission to bypass source-bound resume checks.


The package is `remax`, while the distribution name is `remax-rl`. The benchmark is the installed `modebench` package; runtime imports do not reach into a parent or sibling repository.
