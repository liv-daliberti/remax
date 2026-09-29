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
