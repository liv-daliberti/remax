# Training guide

The core package and saved-key reproduction work on CPU. Actual policy training uses OAT, vLLM, DeepSpeed, and a compatible GPU environment. A clean Python 3.10 environment has completed a small GPU workflow for all four methods; the commands below reproduce it.

All commands below assume the Re:Max checkout is the current directory, unless a command explicitly changes it. Use Bash on Linux.

## Package artifacts and supported environments

| Use | Platform / Python | PyTorch / CUDA | Other constraints |
| --- | --- | --- | --- |
| Core API, commands and CPU conformance | Linux x86_64, CPython 3.10, 3.11, 3.12 | PyTorch 2.6.x CPU; CUDA not required | NumPy >=1.26.4,<3; immutable ModeBench 0.4.0 pin |
| Maintained GPU training | Linux x86_64, CPython 3.10 | PyTorch 2.6.0+cu124, CUDA runtime 12.4 | Exact [GPU dependency lock](../requirements-gpu-py310-cu124.txt); one qualified 48 GB RTX A6000 |

The package metadata bounds Python to `>=3.10,<3.13` and PyTorch to `>=2.6,<2.7`. CPU CI exercises the NumPy floor and the newest available NumPy 2.x compatible with each Python version. New PyTorch minor versions, other CUDA runtimes, Python 3.11/3.12 GPU training, Windows, macOS and ARM are not qualified. Installing the `train` extra does not broaden this support matrix or install an NVIDIA driver.

The base install supplies the replay core, ModeBench boundary, CPU example, recipe inspection and command previews. `train` adds OAT, Transformers, vLLM, DeepSpeed, Datasets and legacy MATH grading dependencies; its NumPy/PyArrow pins match the qualified GPU stack. `dev` adds pytest and legacy MATH verifier tests. Use the complete GPU lock for reproducible training rather than treating the extra as a lockfile.

Build both distribution formats in a build environment:

```sh
python -m pip install build
python -m build
```

The default build produces a source distribution and builds the wheel **from that distribution**. Both include the authenticated input registry, 20 recipes and required shell assets. The sdist also carries the GPU lock and repository reproduction utilities. Installation does not need the original build directory afterward.

In a fresh CPU environment, install the CPU torch wheel first, then either `dist/remax_rl-0.1.0-py3-none-any.whl` or `dist/remax_rl-0.1.0.tar.gz` using pip. Run `remax demo`, `remax recipes`, and `remax-run --help` from any writable working directory. An unverified preview also accepts a bundled name:

```sh
remax-run remax_countdown_05b --data-root ./data --model ./model \
  --output ./run --render-only
```

With the exact GPU lock installed and the model/data prepared below, the same **installed** command works outside a Re:Max checkout. Replace the paths with the authenticated local inputs and `--render-only` with `--validate-only` for a preflight or `--execute` to train. `remax environment --training` checks platform, dependency versions, the ModeBench source pin and CUDA runtime; actual GPU availability is checked by the training preflight. The launcher reads its bundled assets and hashes the installed runtime, including those assets, into the effective configuration. It never injects a checkout into Python's import path. Checkpoints remain bound to those source bytes, so older source-based identities require a new run.

The `ops/` audit and frozen-analysis commands below remain repository/sdist maintenance tools. `ops/run_recipe.py` is a compatibility entry point to the installed launcher. These tools require an installation, and their source directory is not a substitute for installing `remax-rl`. Caches default to `$XDG_CACHE_HOME/remax` (or `~/.cache/remax`), not the installation directory. Set `XDG_CACHE_HOME` to writable scratch storage if your home quota is limited. The [installed-package validation](../VALIDATION.md#installed-distribution-qualification) includes both artifact formats and an actual GPU run outside the checkout.

## CPU development environment

Follow the [README quick start](../README.md#quick-start-installed-cpu-workflow). Install a CPU PyTorch build before the development extra to avoid pulling a GPU stack for tests. Git is needed to fetch the pinned ModeBench source dependency.

After selecting the CPU PyTorch build, you can use the exact tested Linux/Python 3.10 dependencies:

```sh
python -m pip install -c constraints-cpu-py310-tested.txt '.[dev]'
```

The [CPU constraints](../constraints-cpu-py310-tested.txt) are separate from the historical GPU training constraints.

`python examples/replay_loss.py` tests an illustrative score-space update. `make check` runs objective/bank tests, renders all 20 recipes, reproduces the frozen numerical summary, and checks shell syntax. These checks do not require model weights.

## GPU training environment

Use a separate Python 3.10 environment for training; the CPU wheel in the quick start is not a training runtime. The recorded major versions are:

| Component | Recorded version |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.6.0 |
| Transformers | 4.51.3 |
| vLLM | 0.8.4 |
| OAT (`oat-llm`) | 0.1.3.post1 |
| DeepSpeed | 0.16.8 |
| Datasets | 2.16.1 |
| NumPy | 1.26.4 |
| PyArrow | 11.0.0 |

The pinned Linux x86_64 / CPython 3.10 / CUDA 12.4 runtime dependency set is in [requirements-gpu-py310-cu124.txt](../requirements-gpu-py310-cu124.txt). Create a new environment with:

```sh
PYTHON=python3.10 bash ops/setup_gpu_environment.sh .venv
source .venv/bin/activate
```

This installs PyTorch's CUDA runtime wheels, the matching official FlashAttention wheel, and the pinned ModeBench dependency. An NVIDIA driver must already be installed. Launchpad also needs a shared Python library; the launcher discovers its directory from the selected interpreter, including Conda-backed virtual environments. The supported smoke workflow uses PyTorch's precompiled fused AdamW implementation through the existing DeepSpeed adapter; it does not need a local CUDA compiler. Python 3.11/3.12 remain supported for CPU checks, but OAT 0.1.3 requires Python 3.10 for this GPU environment. macOS, Windows, AMD GPUs, and multiple learner GPUs are not qualified by this run.

Allow about 60 GB of free disk for the environment, temporary downloads, a 1 GB model, and four model/optimizer checkpoints. `outputs/` is ignored by Git. Keep the recovery checkpoints until the audit passes. The earlier [recorded constraints](../constraints-train-recorded.txt) remain historical metadata rather than the installation entry point.

## Complete GPU smoke workflow

Validated hardware: **one NVIDIA RTX A6000 (48 GB), eight allocated CPU cores, and 64 GB host RAM**, on Linux x86_64 (glibc 2.34), with NVIDIA driver **610.57.04** and the PyTorch CUDA **12.4** runtime. The four training runs took approximately **2.1–2.2 minutes each**, excluding installation and model preparation. Observed peak GPU use was at most **21,139 MiB**. This qualifies the tested 48 GB GPU configuration, not a minimum-memory claim. See [the measured run](../VALIDATED_GPU_RUN.json) for the audit and [validation](../VALIDATION.md) for scope.

The smoke uses Qwen2.5-0.5B-Instruct at the registered revision, the first four frozen Level 1 PantryPlan training rows, and the first two evaluation rows. It runs Dr.GRPO, Re:Dr, MaxRL, and Re:Max sequentially. Each method keeps the registered 16 fresh samples, microbatch size, learning rate, canonical action space, and replay coefficients; only the training budget and checkpoint/evaluation cadence change. No reference solutions are seeded into the bank.

First obtain the pinned ModeBench checkout as shown below, then prepare the inputs on a machine with internet access:

```sh
python ops/gpu_smoke.py prepare \
  --modebench-data ../modeBench/data --workdir outputs/gpu-smoke
```

Preparation downloads the pinned model (or accepts `--model /path/to/snapshot`), verifies its Git/LFS hashes against that revision, verifies the frozen dataset registry and Parquet hashes, and saves exact row selections and local input hashes. Training uses these local inputs offline and rejects files changed after preparation.

Inside a one-GPU allocation, run:

```sh
python ops/gpu_smoke.py run --workdir outputs/gpu-smoke
python ops/audit_gpu_smoke.py --workdir outputs/gpu-smoke
python ops/compare_gpu_recipe.py
```

The runner prints per-method log locations and limits each subprocess group to 15 minutes. Each run performs generation, structured verification, online bank admission, teacher-forced replay, four optimizer updates, recovery checkpoint writes, a terminal model export, and greedy/sampled evaluation. It then reloads Re:Max's last recovery checkpoint and repeats evaluation. Launch environments, source hashes, dependency versions, GPU memory samples, raw metrics, verifier diagnostics, and process exit records stay under the work directory.

The audit must exit successfully and write `report.json`. Expect `optimizer_updates: 4` for each method, nonzero saved bank counts, zero applied replay gradients in Dr.GRPO/MaxRL, and `exact_draw_records_match: true` for the Re:Max reload. The terminal export uses bookkeeping step 5 after four updates; its most recent evaluation and recovery checkpoint are at step 4. It checks nonempty restored banks, changed model parameters, executed replay in every method, exactly zero applied replay score gradient for the two controls, positive applied replay gradients for Re:Dr/Re:Max, complete successful evaluator diagnostics, and identical checkpoint evaluation records after reload. These are execution checks; two evaluation prompts and four updates cannot estimate paper-level performance. Frozen CPU conformance tests independently cover method gradients.

Use a fresh work directory for another run. An existing run is never overwritten or silently resumed. Missing CUDA means the command is outside its GPU allocation; a failed verifier or worker means the run failed, regardless of any partial metrics. Inspect `<method>.log` before retrying. A dependency import error usually means the command uses a different interpreter from the environment above.

The historical comparison checks the pantry Re:Max recipe against the retained launch ledger and actual runtime log. All 85 explicit registered environment settings match. Two runtime differences are recorded: the historical wrapper capped evaluation to **96 steps**, while the portable recipe declares **192 steps**; historical recovery checkpoint writes were enabled, while the portable launcher requires explicit `OAT_ZERO_SAVE_CKPT=1`. The smoke sets this flag. This comparison does not assert identical historical trajectories, sampling defaults across runtime revisions, or final scores.

## Prepare the exact dataset

The installed ModeBench Python package supplies validators. Its **Git checkout** supplies frozen Parquet data and the materialization script. Obtain the same commit pinned by this release:

```sh
git clone https://github.com/liv-daliberti/modeBench.git ../modeBench
git -C ../modeBench checkout 33cfc3fd1732bd500fe13f13555419557aa248e2
python ../modeBench/ops/verify_data.py
python ../modeBench/ops/materialize_training_data.py \
  --config level1_countdown --output outputs/data/countdown
```

The frozen source files are organized as `data/level1/<domain>/<split>.parquet` in ModeBench; configuration names remain unchanged.

The result has `train/` (DatasetDict subset `train`, 384 rows) and `eval/` (subset `multi_answer`, 128 rows). The materializer verifies source hashes and refuses an existing destination. The strict Re:Max launcher independently authenticates every ordered row, column, prompt, and reference against its release-owned registry, derived from the pinned frozen Parquet files. Renaming a directory or rewriting a local manifest cannot satisfy that check. All 384/128 rows must be present, even when a copied recipe selects fewer training rows. Use the config matching the selected recipe domain; do not feed evaluation rows into training.

For another domain, replace `level1_countdown` with `level1_graph_coloring`, `level1_python_factors`, `level1_mathir`, or `level1_pantry_plan`. The single-answer graph diagnostic is not a training recipe input.

## Prepare the exact model

Every recipe contains `model_id` and `model_revision`. Download that immutable snapshot or supply an existing local copy:

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download

recipe = json.loads(Path("configs/remax_countdown_05b.json").read_text())
model_path = snapshot_download(
    repo_id=recipe["model_id"],
    revision=recipe["model_revision"],
)
print(model_path)
```

This step downloads model weights and requires network/storage. Pass the returned directory to `--model`. The launcher checks all seven model/configuration/tokenizer files against release-owned SHA-256 identities, independently verified against the immutable Hub commit's Git/LFS metadata. Missing, modified, or unexpected model inputs (including adapter files) fail. Authentication works offline; a snapshot directory name is not evidence of identity.

## Validate, then execute

For a configuration-only preview that needs no weights or training dependencies:

```sh
python ops/run_recipe.py configs/remax_countdown_05b.json \
  --data-root outputs/data/countdown \
  --model /path/returned/by/snapshot_download \
  --output outputs/remax-countdown-s43 \
  --render-only
```

This preview is explicitly **UNVERIFIED**. The `train/` and `eval/` directories must exist for the shell renderer. It validates recipe types and compatibility but does not authenticate inputs or write run records. This option cannot be combined with execution.

Omitting `--render-only` authenticates both inputs, prints the command, and writes `launch_request.json` under a fresh output directory. To also resolve every runtime default and validate the actual training environment, run this inside a GPU allocation:

```sh
python ops/run_recipe.py configs/remax_countdown_05b.json \
  --data-root outputs/data/countdown \
  --model /path/returned/by/snapshot_download \
  --output outputs/validate-countdown-s43 \
  --validate-only
```

Expect `validated complete effective configuration: .../effective_config.json`. This uses the real training parser and both argument validators, including available-GPU checks, then exits **before models or workers are built**. For training, use a separate fresh output directory and replace the last option with `--execute`:

```sh
python ops/run_recipe.py configs/remax_countdown_05b.json \
  --data-root outputs/data/countdown \
  --model /path/returned/by/snapshot_download \
  --output outputs/remax-countdown-s43 \
  --execute
```

Use `--seed 44` for another registered seed from `[43, 44, 45, 46, 47]`. Seeds outside that set fail. No Slurm submission, automatic requeue, or external message is performed by this command.

Recipes use a frozen typed configuration. Existing JSON strings remain supported, but unknown top-level or environment fields, duplicate JSON keys, missing required fields, malformed numbers/booleans, incompatible batch/context budgets, and mismatched method/prompt/domain controls fail. The maintained contract supports the five Level 1 domains and the pinned Qwen model. To change supported numerical settings, edit a copied recipe explicitly; the resulting settings are recorded. New models, prompt interfaces, levels, or methods require a reviewed contract/identity-registry update.

Inherited `OAT_ZERO_*`, `SAVE_PATH`, Python import overrides, shell startup hooks, and conflicting evaluator/backend settings fail with the offending variable's name. Unset that variable rather than relying on silent precedence. The child receives an allowlist of infrastructure variables (paths, visible GPUs, caches, compiler locations), fixed runtime flags, and explicit recipe settings; unrelated variables and credentials are not copied into launch records. The launcher uses the calling interpreter and checks the maintained Linux x86_64/Python 3.10 runtime's principal dependency versions and ModeBench commit before execution.

Historical `AUTO_RESUME` is explicitly recorded as **unimplemented metadata**. `EVAL_PROMPT_INTERVAL` is checked against effective optimizer-step cadence and batch size. Missing historical objective/critic flags get explicit typed defaults; missing required scientific settings cannot silently fall back to shell defaults.

## Recipe inventory

Each domain has `drgrpo`, `redr`, `maxrl`, and `remax` JSON recipes, giving 20 configurations in total.

| Domain suffix | Example recipe | Prompt template | Generation cap | Action interface |
| --- | --- | --- | ---: | --- |
| `countdown` | `configs/remax_countdown_05b.json` | `qwen_boxed` | 192 | Expression |
| `graph_coloring` | `configs/remax_graph_coloring_05b.json` | `qwen_boxed` | 192 | Coloring |
| `python_factors` | `configs/remax_python_factors_05b.json` | `qwen_boxed` | 192 | Restricted lambda |
| `mathir` | `configs/remax_mathir_05b.json` | `qwen_boxed` | 64 | Equation action menu |
| `pantry_plan` | `configs/remax_pantry_plan_05b.json` | `qwen_pantry_support_mask` | 8 | Six-bit support mask |

All exported recipes use Qwen2.5-0.5B-Instruct and Level 1 data. Their registered protocol includes 384 training prompts, eight passes, 16 fresh samples per prompt, and a half-pass 192-step evaluation cadence. Mode coverage uses four groups of eight; evaluation seeds and decoding settings are recorded per recipe.

Pantry's action interface requires `VLLM_USE_V1=0`; the wrapper sets it when canonical actions are enabled. The benchmark performs the registered deterministic quantity projection for the submitted support.

The exact environment in each JSON file is authoritative. The fresh-sample objective and compute-only setting differ by method; see [method definitions](method.md#compute-matched-controls). The recipe tests check these differences, the selected seed, the eight-pass budget, and evaluation cadence.

## Outputs and resume

`--output` sets the training save root. OAT may create a run-specific subdirectory beneath it. The configs retain historical resume settings and a terminal model export. **The portable launcher does not implement automatic checkpoint discovery**: `OAT_ZERO_AUTO_RESUME` is retained metadata. Recovery writes require `OAT_ZERO_SAVE_CKPT=1` in a copied recipe. The strict recipe launcher requires a fresh output directory and accepts an explicit, authenticated `--resume` step directory. See the protocol below. The separate bounded GPU smoke runner uses the low-level shell interface with explicit `OAT_ZERO_RESUME_DIR`/`OAT_ZERO_RESUME_TAG` and retains checkpoints for its reload audit. Direct shell launches and smoke overrides are outside the strict registered-recipe input contract.

Use a new output root for every attempt, including resume attempts. Preserve config, code, dataset, model, bank state, and schedule identity across a resume. Never infer cohort membership from a directory name alone; match the recorded seed and protocol.

`launch_request.json` records the original recipe and its hash, typed settings, selected seed, ordered dataset identities, local file hashes, pinned model identity, prompt interface, exact command, controlled launch environment, and source hashes. Immediately before entering the training runtime, source/input bytes and resolved recipe settings are checked again. `effective_config.json` adds **every validated dataclass argument**, including inherited OAT defaults, installed dependency versions, and the runtime environment. Infinite numerical settings use the JSON string `"inf"`; JSON NaN is rejected. Both files are written atomically. A validation failure prevents training; existing records are never overwritten. A record with `status: validated` proves configuration validation, not training completion. Preserve it alongside metrics, checkpoints, and evaluation diagnostics.

The trust anchors are the versioned registry shipped with Re:Max and its pinned ModeBench source, not a user-editable dataset receipt. Model hashes were checked against upstream Git/LFS identities; materialized-row hashes were derived only after verifying frozen Parquet bytes. The PR-base guard rejects rewrites of an existing input registry. This protects against accidental input substitution and configuration drift; it does not isolate a run from someone who can modify installed code or files during execution.

## Explicit, identity-bound resume

Strict launches use the versioned `remax-resume-v1` protocol. They support one collocated learner GPU, synchronous generation, one full candidate group per optimizer boundary, and the four maintained methods. Unsupported layouts fail before training. Historical direct-shell checkpoints lack this contract and cannot be loaded by the strict launcher.

Enable recovery writes in a **copied recipe before the first launch**:

```sh
python - <<'PY'
import json
from pathlib import Path
source = Path("configs/remax_pantry_plan_05b.json")
recipe = json.loads(source.read_text())
recipe["environment"].update({
    "OAT_ZERO_SAVE_CKPT": "1",
    "OAT_ZERO_MAX_RESUME_NUM": "2",
    "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS": "0",
})
Path("outputs").mkdir(exist_ok=True)
Path("outputs/remax-resumable.json").write_text(json.dumps(recipe, indent=2) + "\n")
PY
```

Launch that recipe with `--execute` as above. After an interruption, select an explicit committed step and use a fresh output directory:

```sh
python ops/run_recipe.py outputs/remax-resumable.json \
  --data-root outputs/data/pantry_plan \
  --model /path/to/pinned/model/snapshot \
  --output outputs/remax-s43-restarted \
  --resume outputs/remax-s43/debug_TIMESTAMP/checkpoints/step_00192 \
  --execute
```

Keep the **original total training horizon**, seed, evaluation schedule, optimizer/replay settings, source, dependencies, and hardware. Changing `NUM_PROMPT_EPOCH` to the number of remaining epochs changes the scheduler and is rejected. Input directories may move if their contents still authenticate. The output and explicit checkpoint selector may change. `--validate-only --resume ...` checks restore compatibility before starting workers; old checkpoints, missing manifests, changed files, and incompatible identities fail. There is no automatic latest-checkpoint discovery.

Checkpoints use DeepSpeed’s Python serialization and must come from a trusted run; manifest hashes check integrity, not publisher identity.

Each checkpoint saves model and optimizer state, scheduler state, DeepSpeed microstep position, Python/NumPy/Torch CPU/CUDA random-number states, the retained rollout buffer, data position, bank/exemplars/replay cursor, progress counters, and the last evaluated policy step. Restore skips consumed rows and the already-completed boundary evaluation. DataLoader construction uses an isolated generator so creating a replacement iterator cannot advance the policy RNG.

Free-form actor requests now have seeds derived from the run seed and consumed prompt position. Each strict free-form training request also clears the actor prefix cache: a warm cache after evaluation can change prefill batch shapes and sampled tokens compared with a newly started actor. Canonical learner sampling already has position-bound seeds. This is an explicit sampling-protocol change: strict free-form trajectories should not be equated with historical unseeded actor streams. The method's objective, admission rules, replay coefficients, and frozen numerical conformance cases are unchanged.

Writes go into a `.pending-*` directory. After all state files are flushed, `resume_manifest.json` binds their hashes to the run identity and step; a same-filesystem rename publishes the complete directory. Only then are `latest` and retention updated. Interrupted writes cannot replace the previous committed checkpoint. Ignore abandoned staging directories after a killed process; never rename one into a committed step. If interruption happens after commit but before updating `latest`, the new explicit step is still valid. Training resumes from a completed optimizer boundary, not from the middle of an unfinished backward pass.

`resume_decisions.jsonl` records each prompt position, sampled response tokens/rewards, bank identity, and selected replay groups. When combining attempts, retain the original prefix **through the selected checkpoint** and the resumed suffix after it. Discard any original attempt's later uncommitted metrics or evaluations. A `latest` pointer is a convenience, not an identity check.

## Reproduce resume equivalence

`make resume` executes the production run loop and real GRPO/replay gradients with a tiny CPU actor and a DeepSpeed transport adapter. AdamW and a changing learning-rate schedule are real. Across all four methods, interruption positions cover within-epoch, epoch-boundary, evaluation-boundary, and final-update cases, with both cleared and retained rollout buffers. CPU model/optimizer/scheduler/RNG states and all bank/scheduling/evaluation decisions must match **exactly**. Tests also kill a writer subprocess, simulate a failed latest-pointer publication, corrupt state, and reject incompatible identities.

For actual GPU training, use the pinned environment and full authenticated PantryPlan dataset:

```sh
python ops/resume_gpu.py run \
  --workdir outputs/resume-equivalence \
  --data-root outputs/data/pantry_plan \
  --model /path/to/pinned/model/snapshot
```

The `run` command performs the state audit automatically. To check saved artifacts again later without rerunning training, use `python ops/resume_gpu.py audit --workdir outputs/resume-equivalence`.

This runs six updates over the first three training rows and two epochs for each method, then launches a fresh process from its step-2 checkpoint to finish the same six-update horizon. The full 128-row evaluation split is used, with two samples and one coverage draw to bound validation cost. The audit compares model/optimizer states at steps 4 and 6, exact scheduler and RNG state, data and replay decisions, banks, and the complete evaluation suffix. This is a bounded full-run equivalence test, not a reproduction of paper-level training scores.

GPU tolerances are declared before running: model tensors `atol=1e-6, rtol=1e-6`; optimizer tensors `atol=1e-8, rtol=1e-5`. Bank contents, replay/sample/data decisions, scheduler/RNG state, and evaluation records require exact equality. Differences cannot be hidden by a matching aggregate score. The completed measurements and their limits are recorded in [validation](../VALIDATION.md#full-run-resume-conformance) and [the resume report](../VALIDATED_RESUME_RUN.json). Use `--domain countdown --methods remax` with the matching dataset to exercise free-form generation. Raw logs, manifests, configurations, checkpoints, and the resulting `report.json` remain in the work directory. Allow additional disk space for both trajectories' model/optimizer checkpoints and exports; preserve these until the audit succeeds.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing `torch` or CUDA libraries | Use the CPU environment for checks or activate the separate GPU runtime for training. |
| `No module named modebench` | Install this repository's declared dependencies; ModeBench is pinned to a public Git commit. |
| Missing dataset directory | Materialize the matching frozen config first and point `--data-root` to its parent directory. |
| Dataset/PyArrow import errors | Check the recorded Datasets, NumPy, and PyArrow versions together. |
| Canonical actions require vLLM V0 | Use the recipe wrapper, which sets the required runtime flag for Pantry. |
| Inherited-setting rejection | Unset the named variable; put supported scientific settings in a copied recipe. |
| Model/dataset identity mismatch | Restore the pinned snapshot and matching full frozen splits. Do not regenerate expected hashes from the mismatched input. |
| Unexpected resume behavior | Inspect the selected save root and retained state; use a new root for a new run. |
| Frozen numerical reproduction fails | Restore the matching evidence and code revision; do not overwrite expected results to silence a mismatch. |

CPU CI does not exercise GPU sampling or distributed optimization. The separate manual GPU workflows validate small PantryPlan runs and resume comparisons for all four methods, plus a free-form Countdown resume comparison. Full historical training reproduction remains separate.

Verifier failures are fatal, with structured diagnostics rather than zero rewards. Inspect `evaluation_failures.jsonl` for failed evaluation steps; do not combine partial draw records from those steps into a completed score. See [the boundary and compatibility policy](method.md#modebench-boundary-and-failures).
