# Re:Max / Re:Dr

**Retain and rehearse the correct solution modes a policy discovers.** Verified replay keeps a prompt-local bank of verified exemplars and revisits them with a uniform teacher-forced likelihood objective. **Re:Max** combines replay with MaxRL; **Re:Dr** combines it with Dr.GRPO.

[![Verified replay: verify model outputs, store one exemplar per discovered mode, revisit prompt-local banks recurrently, and replay their exemplars uniformly.](docs/assets/verified-replay.png)](docs/assets/verified-replay.png)

*The verified-replay mechanism from the paper. Click to view the full-resolution figure.*

This repository contains replay objectives, bank state and scheduling, OAT learner integration, **20 Level 1 training recipes**, and a frozen analysis covering **473 seed records across 95 arms**. [ModeBench](https://github.com/liv-daliberti/modeBench) separately owns the benchmark, datasets, validators, and canonical mode identities.

## Quick start: CPU checks

Use Linux with Python 3.10 or later; the historical GPU training stack uses Python 3.10. Check your interpreter version before creating the environment.

```sh
git clone https://github.com/liv-daliberti/remax.git
cd remax
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.6.0+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python examples/replay_loss.py
make check
```

ModeBench is installed from the immutable Git commit declared in `pyproject.toml`; no local ModeBench checkout is needed for these checks. Installation requires network access. The example and checks themselves make no model calls and launch no training jobs.

The example prints a replay loss of `2.0` and score gradients `[-0.25, -0.25, -0.5]` for two illustrative prompt banks. `make check` runs regression tests, verifies the retained training analysis, and checks shell syntax.

Run `make conformance` for the training contract alone: frozen admission-to-optimizer cases for Re:Max, Re:Dr, and both compute-matched controls. These execute the production learner on a tiny CPU model and compare every update with an independent gradient calculation. See [training conformance](docs/method.md#training-conformance) for coverage and limits.

A GPU training environment needs the separate `train` extra and a compatible CUDA/PyTorch stack. Follow the [training guide](docs/training.md) rather than using the CPU wheel above for training.

## Four matched methods

| Recipe prefix | Fresh-sample objective | Applied replay derivative |
| --- | --- | --- |
| `drgrpo` | Dr.GRPO | Exactly zero; compute-matched control |
| `redr` | Dr.GRPO | Uniform verified replay |
| `maxrl` | Binary MaxRL | Exactly zero; compute-matched control |
| `remax` | Binary MaxRL | Uniform verified replay |

Both controls retain bank bookkeeping, recurrent traversal, and replay computation. The treatment changes the applied replay derivative. These controls are not equivalent to disabling the entire replay path.

The public recipes cover graph coloring, Countdown, Python factors, MathIR, and PantryPlan at Level 1 with Qwen2.5-0.5B-Instruct. They record the model revision, registered seeds, domain interfaces, and launch settings. See the [recipe inventory](docs/training.md#recipe-inventory).

## Reproduce the retained analysis

```sh
python ops/reproduce_training.py
```

This checks the unchanged verified-key archive against the frozen summary, including seed records, arm aggregates, support eligibility, and missing-checkpoint decisions. Expected coverage is 473 records and 95 arms, of which 84 are terminal-reportable under the retained rules.

This is saved-key numerical reproduction. It does not retrain a policy or regrade every original response. Newer summaries under `evidence/snapshots/` are separately labeled provenance snapshots; their complete raw-source reproduction chains are not included. See [reproducibility](docs/reproducibility.md).

## Training entry point

After preparing the exact data and model snapshot described in the [training guide](docs/training.md), render a command:

```sh
python ops/run_recipe.py configs/remax_countdown_05b.json \
  --data-root /path/to/materialized/countdown \
  --model /path/to/pinned/model/snapshot \
  --output outputs/remax-countdown-s43
```

The default prints the command. Add `--execute` to train locally or `--seed 44` to select another registered seed. The launcher does not require Slurm. A newly installed GPU training run has not been validated for this extracted release; [release scope](RELEASE_STATUS.md) records that boundary.

## Documentation

| Guide | Contents |
| --- | --- |
| [Method and implementation](docs/method.md) | Admission, replay scheduling, objectives, gradient example, module map |
| [Training](docs/training.md) | Installation, exact data/model preparation, recipes, command rendering, resume behavior |
| [Reproducibility](docs/reproducibility.md) | Frozen evidence, provenance, environment versions, reporting |
| [Contributing](CONTRIBUTING.md) | Tests and scientific compatibility expectations |
| [Release scope](RELEASE_STATUS.md) | Included work and known limitations |
| [Validation](VALIDATION.md) | Checks actually performed |

The integrated learner retains comparator and historical branches needed for compatibility. The 20 public recipes select the four methods above; a class or module name alone is not evidence that every historical objective is part of the maintained release.

## License and reference

Code is licensed under [Apache 2.0](LICENSE), with original copyright notices retained. ModeBench documents dataset source terms separately.

Cite this repository URL and the exact Git commit used, together with the ModeBench commit, recipe, model revision, and evidence identities. Scientific citation metadata and additional experiment packages remain tracked in [release scope](RELEASE_STATUS.md).
