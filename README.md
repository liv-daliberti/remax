# Re:Max / Re:Dr

**Retain and rehearse the correct solution modes a policy discovers.** Verified replay keeps a prompt-local bank of verified exemplars and revisits them with a uniform teacher-forced likelihood objective. **Re:Max** combines replay with MaxRL; **Re:Dr** combines it with Dr.GRPO.

[![Verified replay: verify model outputs, store one exemplar per discovered mode, revisit prompt-local banks recurrently, and replay their exemplars uniformly.](docs/assets/verified-replay.png)](docs/assets/verified-replay.png)

*The verified-replay mechanism from the paper. Click to view the full-resolution figure.*

This repository contains replay objectives, bank state and scheduling, OAT learner integration, **20 Level 1 training recipes**, and a frozen analysis covering **473 seed records across 95 arms**. [ModeBench](https://github.com/liv-daliberti/modeBench) separately owns the benchmark, datasets, validators, and canonical mode identities.

## Quick start: installed CPU workflow

Use Linux x86_64 with CPython 3.10–3.12. This installs a regular package, then runs it outside the source directory:

```sh
git clone https://github.com/liv-daliberti/remax.git
cd remax
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
cd "$(mktemp -d)"
remax demo
remax recipes
remax recipes remax_countdown_05b > recipe.json
remax-run recipe.json --data-root ./data --model ./model \
  --output ./run --render-only
```

`remax demo` prints `{"maxrl_advantages": [[1.0, -1.0]], "replay_loss": 2.0, "score_gradients": [-0.25, -0.25, -0.5]}`. `remax recipes` lists 20 bundled recipes. The final command previews training arguments and labels them **UNVERIFIED**; it does not download inputs or create a run. `remax environment` reports installed versions. Python module equivalents are `python -m remax` and `python -m remax.launcher`.

You can instead install a built wheel or source distribution with `python -m pip install /path/to/remax_rl-0.1.0-py3-none-any.whl` or `python -m pip install /path/to/remax_rl-0.1.0.tar.gz`. [CI builds both artifacts](https://github.com/liv-daliberti/remax/actions/workflows/check.yml) and tests them in clean environments outside the checkout. No PyPI release is claimed. The [installation guide](docs/training.md#package-artifacts-and-supported-environments) explains artifact builds and supported dependency combinations.

The core install includes PyTorch, NumPy, and ModeBench 0.4.0 from an immutable Git commit. Git and network access are required during installation; no sibling ModeBench checkout is used at runtime. Core commands make no model calls and require no OAT, vLLM, DeepSpeed, Transformers, or Datasets installation.

For contributor checks, return to the checkout, install the development extra with `python -m pip install '.[dev]'`, and run `make check`. `make conformance`, `make boundary`, and `make resume` select training, verifier-failure, and continuation contracts. Reinstall after source edits so tests exercise the new package. See [contributing](CONTRIBUTING.md).

GPU training uses a separate, qualified Python 3.10 / CUDA 12.4 environment. Follow the [training guide](docs/training.md); the CPU wheel above cannot train on a GPU.

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

After preparing the exact data and model snapshot described in the [training guide](docs/training.md), preview a command:

```sh
remax-run remax_countdown_05b \
  --data-root /path/to/materialized/countdown \
  --model /path/to/pinned/model/snapshot \
  --output outputs/remax-countdown-s43 \
  --render-only
```

The preview is explicitly unverified. Replace `--render-only` with `--execute` to authenticate inputs and train locally, or `--validate-only` to save the complete effective configuration and exit before training. Use a fresh output directory for either; `--seed 44` selects another registered seed. Unknown fields, incompatible settings, wrong input identities, and conflicting inherited settings fail before training. The launcher does not require Slurm. [Explicit resume](docs/training.md#explicit-identity-bound-resume) binds checkpoints to the configuration and input identities; `make resume` checks full-run continuation. A clean installation completed the [four-method GPU walkthrough](docs/training.md#complete-gpu-smoke-workflow) on one 48 GB A6000, including checkpoint reload and identical evaluation records. See [measured results](VALIDATED_GPU_RUN.json) and [release scope](RELEASE_STATUS.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Method and implementation](docs/method.md) | Admission, replay scheduling, objectives, gradient example, module map |
| [Training](docs/training.md) | Installation, exact data/model preparation, recipes, command rendering, resume behavior |
| [Reproducibility](docs/reproducibility.md) | Frozen evidence, provenance, environment versions, reporting |
| [Contributing](CONTRIBUTING.md) | Tests and scientific compatibility expectations |
| [Release scope](RELEASE_STATUS.md) | Included work and known limitations |
| [Validation](VALIDATION.md) | Checks actually performed |

The maintained method lives in `remax.core`, with training plumbing in `remax.integrations.oat`. Historical comparator implementations live in `remax.experiments` and load only when selected. Start with the [source map and reading guide](docs/method.md#source-map); the 20 public recipes select the four methods above.

## License and reference

Code is licensed under [Apache 2.0](LICENSE), with original copyright notices retained. ModeBench documents dataset source terms separately.

Cite this repository URL and the exact Git commit used, together with the ModeBench commit, recipe, model revision, and evidence identities. Scientific citation metadata and additional experiment packages remain tracked in [release scope](RELEASE_STATUS.md).
