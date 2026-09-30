# Contributing

Use the CPU environment in the [README](README.md#quick-start-installed-cpu-workflow), then run:

```sh
make check
make conformance
make boundary
make resume
make scaling
python examples/replay_loss.py
```

These commands test replay objectives, bank admission and scheduling, method/control flags, retained numerical evidence, and shell syntax. GPU training is a separate validation step.

## Installed-package checks

CI builds a wheel from the sdist, then installs wheel and sdist independently on Python 3.10–3.12. The wheel jobs use NumPy 1.26.4; sdist jobs resolve NumPy 2.x. Each clean venv runs the core-only workflow before installing `dev`, then runs the regression suite from a temporary directory containing tests and evidence, with no `src/` tree. No test configuration adds the source directory to `sys.path`.

For a local artifact check, build with `python -m build`, install one artifact into a fresh venv with the CPU PyTorch wheel, copy `tests/installed/smoke.py` to a directory outside the checkout, unset `PYTHONPATH`/`PYTHONHOME`, and run `python -I smoke.py` there. This suite needs no pytest or training extras. It checks real installation provenance, commands, all 20 recipe previews, verification, bank admission, replay gradients and checkpoint restoration. See [pytest's installed-package guidance](https://docs.pytest.org/en/stable/explanation/goodpractices.html).

Use regular installs (`python -m pip install '.[dev]'`) while changing launcher assets: the wheel build copies the canonical `configs/` recipes and `ops/` scripts into package resources. Reinstall after edits. Editable installations are not the release validation path. The build hook is `src/build_support.py`; changing assets must preserve recipe command parity and strict launch identities.

## Scientific compatibility

Explain how a change affects fresh-sample advantages, bank admission, exemplar identity, replay weights, schedule/resume state, loss scaling, or compute-matched controls. Add focused behavioral tests for changes to those contracts.

Before refactoring training, run `make conformance` and `python ops/check_training_reference.py --base-ref origin/main`. The frozen integration traces and independent scalar oracle cover admission through parameter updates. See [the contract and CPU adapter limits](docs/method.md#training-conformance). PR CI rejects edits to existing training fixture bytes even if hashes are refreshed. Add explicitly versioned cases and a scientific compatibility explanation for deliberate method changes; preserve the old references.

Run `make boundary` for dependency or grading changes. Preserve the historical reward/key fixtures and propagate `EvaluationFailure`; catching it and returning zero or `None` is a scientific correctness bug.

Preserve the ModeBench admission boundary: correctness and canonical mode identity must derive from the executable benchmark validator. Training support must not be initialized from evaluation answers or a certified solution catalogue.

The strict recipe contract is in `src/remax/recipes.py` and `recipe_types.py`. Run `python -m pytest -q tests/test_strict_recipes.py tests/test_release_recipes.py` for configuration/input changes. Preserve existing `input_registry_v*.json` bytes: derive a new version from hash-verified frozen Parquet and immutable model Git/LFS identities, explain prompt/method compatibility, and update the consumer explicitly. Never authenticate a new input by copying the hash of an unverified local file into the registry. The PR-base guard protects these registries alongside training fixtures.

Keep frozen evidence immutable. New analyses, recipes, exclusions, or protocol revisions need their own identities. Do not update expected result files simply to make a numerical check pass. Preserve original copyright notices and source attribution.

The public package should support CPU objective testing without installing OAT, vLLM, or DeepSpeed. Keep GPU integration dependencies in the `train` extra. Treat removed historical learner branches as scientific refactors requiring equivalence checks, not automatic cleanup.

Resume changes must preserve model/optimizer/scheduler state, RNG streams, data and replay cursors, and evaluation cadence. Run `make resume`; changes to supported GPU recovery additionally need the real-runtime comparison in `ops/resume_gpu.py`. Declare tolerances before running, and keep decision/bank comparisons exact. Changing sampling protocol, identity normalization, or checkpoint boundaries needs an explicit compatibility explanation.

Changes to gradient accumulation, rank sharding or replay selection must preserve the configured logical batch and pass `make scaling` (real Gloo/DDP at 1/2/4 ranks). Report replay scoring separately from fresh training, and distinguish discovered identities from bounded exemplars. Use `ops/profile_replay.py` for measured memory/scoring costs; do not infer a leak from CUDA reserved memory alone. This CPU distributed proof does not qualify a new GPU backend or distributed resume protocol.

## Changes and issues

Use [GitHub issues](https://github.com/liv-daliberti/remax/issues) for bugs and protocol questions. Include the commit, recipe, Python/package versions, minimal reproduction, and expected versus observed behavior. For numerical reports, include checkpoint/evaluation identities and relevant support counts.

A pull request should describe the final behavior, why it changes, and checks performed. State explicitly whether GPU training was tested.

## Release identities

After reviewed edits, maintainers can refresh the final release hashes:

```sh
python ops/verify_release.py --refresh
python ops/verify_release.py
```

Review the manifest and provenance diff before committing. Hash refresh records changed files; it does not replace regression tests, numerical reproduction, or a scientific review of altered behavior.
