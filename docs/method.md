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
from remax.canonical_replay import canonical_replay_uniform_verified_likelihood_loss

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
from remax.maxrl import binary_maxrl_advantages

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

The suite executes the unchanged `_grpo_learning_step_with_progress` path, including the installed ModeBench validator, bank admission/update, global scheduler, replay materialization, teacher-forced scoring, fresh advantages, replay backward passes, and a real PyTorch SGD update. It covers all four maintained methods. No model download, generation, or GPU is needed.

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

## Source map

| Module | Responsibility |
| --- | --- |
| `src/remax/online_canonical_bank.py` | Prompt-local verified banks, exemplar state, scheduling and restore |
| `src/remax/canonical_replay.py` | Replay materialization, weighting and score-space losses |
| `src/remax/maxrl.py` | Binary MaxRL advantages |
| `src/remax/actor.py` | Sampling and reward integration |
| `src/remax/learner/grpo.py` | Integrated optimization and applied replay/control gradients |
| `src/remax/learner/run.py` | Training loop and evaluation plumbing |
| `src/remax/math_grader.py` | Trainer grading adapter; executable ModeBench identity delegates to `modebench.grading` |
| `ops/run_recipe.py` | Translate an exported recipe into a portable command |
| `ops/train.sh` | OAT CLI construction and local execution |

The package is `remax`, while the distribution name is `remax-rl`. The shared benchmark is imported as the installed `modebench` package; runtime imports do not reach into a parent or sibling repository.
