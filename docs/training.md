# Install, train, resume, evaluate

This is the primary user workflow. It runs a six-update Re:Max PantryPlan example with online verification, replay, checkpointing, and evaluation, then repeats its suffix from a checkpoint. It uses the same bounded configuration as the qualified resume comparison. It is an execution walkthrough, not a paper-score estimate. The [CPU API example](method.md#public-api-walkthrough) needs no GPU.

Use Bash. Start in the directory where you want to clone Re:Max; subsequent commands run from its checkout. Training needs Linux x86_64, Python 3.10, one qualified **48 GB NVIDIA RTX A6000**, eight CPU cores, 64 GB RAM, the CUDA 12.4 runtime, and an installed NVIDIA driver. Allow roughly 60 GB of disk for dependencies, weights, and retained optimizer checkpoints. Other GPU configurations are not qualified by the existing measurements. Pick fresh output directories throughout.

## GPU training environment

```sh
git clone https://github.com/liv-daliberti/remax.git
cd remax
PYTHON=python3.10 bash ops/setup_gpu_environment.sh .venv-train
source .venv-train/bin/activate
remax environment --training
```

The setup script installs the [qualified dependency lock](../requirements-gpu-py310-cu124.txt), including PyTorch 2.6.0+cu124, OAT 0.1.3.post1, vLLM 0.8.4, Transformers 4.51.3 and DeepSpeed 0.16.8, then installs this package normally. Do not reuse a CPU PyTorch environment. If your home quota is limited, set `XDG_CACHE_HOME` to writable scratch space. The environment check validates dependency/platform identities; launch preflight also checks GPU availability.

## Prepare the exact dataset

```sh
git clone https://github.com/liv-daliberti/modeBench.git ../modeBench
git -C ../modeBench checkout 33cfc3fd1732bd500fe13f13555419557aa248e2
python ../modeBench/ops/verify_data.py
python ../modeBench/ops/materialize_training_data.py \
  --config level1_pantry_plan --output outputs/data/pantry_plan
```

The checkout supplies frozen Parquet data; the installed ModeBench package supplies validators. Materialization verifies source hashes and creates `train/` with 384 rows and `eval/` with 128 rows. Re:Max independently authenticates ordered rows and columns against its release-owned registry. Keep all rows even though the short recipe selects only three for training. Changing a local receipt or renaming a directory cannot authenticate different data. Evaluation answers never initialize the bank.

## Prepare the exact model

Download the pinned model once; training uses the local copy offline:

```sh
python - <<'PYMODEL'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
recipe = json.loads(Path("configs/remax_pantry_plan_05b.json").read_text())
model = snapshot_download(repo_id=recipe["model_id"], revision=recipe["model_revision"])
Path("outputs").mkdir(exist_ok=True)
Path("outputs/model-path.txt").write_text(model + "\n")
PYMODEL
model_path="$(cat outputs/model-path.txt)"
```

This is Qwen2.5-0.5B-Instruct at `7ae557604adf67be50417f59c2c2f167def9a775`. The launcher checks all seven model/tokenizer/configuration files against immutable Git/LFS-derived SHA-256 identities, including unexpected loading files. A directory named after the revision is insufficient.

## Validate, then execute

Prepare the bounded recipe and preflight inside your GPU allocation:

```sh
python examples/prepare_training_walkthrough.py outputs/walkthrough.json
remax-run outputs/walkthrough.json --data-root outputs/data/pantry_plan \
  --model "$model_path" --output outputs/walkthrough-preflight --validate-only
remax-run outputs/walkthrough.json --data-root outputs/data/pantry_plan \
  --model "$model_path" --output outputs/walkthrough-first --execute
```

Expect `validated complete effective configuration: .../effective_config.json` from preflight. Training uses three prompts for two passes (six optimizer updates), retains the registered 16 fresh samples and replay coefficient, evaluates every two updates, and keeps checkpoints at steps 2, 4, and 6. It uses the full evaluation split with K=2 and one sampled draw to bound cost. The first run can take several minutes including cold imports, compilation, evaluation, and checkpoint writes; these are not throughput estimates for the historical recipes.

`--render-only` is an explicitly **UNVERIFIED** preview available in a CPU installation. `--validate-only` authenticates inputs and resolves every runtime default, then exits before models/workers are created. Preflight and training need separate output roots. Unknown fields, incompatible method/prompt settings, changed inputs, unsupported seeds, and conflicting inherited settings fail before training. Put supported changes in a copied JSON recipe, not arbitrary `OAT_ZERO_*` environment variables.

## Resume the same run

A normal completed first run is sufficient to demonstrate restore; there is no need to kill it. Select its committed step-2 checkpoint and repeat the remaining four updates in a fresh process:

```sh
checkpoint_path="$(python - <<'PYCKPT'
from pathlib import Path
matches = list(Path("outputs/walkthrough-first").glob("*/checkpoints/step_00002"))
assert len(matches) == 1, matches
print(matches[0])
PYCKPT
)"
remax-run outputs/walkthrough.json --data-root outputs/data/pantry_plan \
  --model "$model_path" --output outputs/walkthrough-resumed \
  --resume "$checkpoint_path" --execute
```

Keep the original recipe, total training horizon, seed, installed source, dependency set, hardware, and input bytes. Input locations and output directories may move. Do not shorten the recipe to the number of remaining updates or use a `latest` symlink. The resumed suffix starts after the committed boundary and skips its already-completed evaluation. Formatting source files also changes resume identity; retain the original environment for older runs. See [the full checkpoint contract](#explicit-identity-bound-resume).

## Evaluate and interpret the run

Evaluation is part of both training commands: greedy and fixed-seed sampled evaluation run on the held-out split at the declared cadence. This walkthrough does not require a separate generation script. For a saved model's standalone evaluation, the maintained qualification is checkpoint reload within the [GPU smoke workflow](#complete-gpu-smoke-workflow); there is no general `remax evaluate` command for arbitrary checkpoints.

Inspect the first run's evaluation summaries without loading model weights:

```sh
python - <<'PYREPORT'
import json
from pathlib import Path
root = Path("outputs/walkthrough-first")
failures = [p for p in root.rglob("evaluation_failures.jsonl") if p.stat().st_size]
assert not failures, f"Evaluation failed: {failures}"
logs = list(root.rglob("train_metrics.jsonl"))
assert len(logs) == 1, logs
fields = ["misc/policy_sgd_step", "eval/multi_answer/accuracy",
          "eval/multi_answer/sampled_any_correct_at_2",
          "eval/multi_answer/sampled_distinct_correct_at_2"]
with logs[0].open() as stream:
    for line in stream:
        row = json.loads(line)
        if "eval/multi_answer/accuracy" in row:
            print({key: row.get(key) for key in fields})
PYREPORT
```

Read these fields as policy update step, greedy correctness, sampled pass@2, and the mean number of distinct verified modes among two samples. Correctness and breadth answer different questions. Values are seed/runtime dependent; missing or failed evaluations are not zero scores. The short run is too small to support a scientific performance claim. PCMD in the frozen result catalog uses a different registered analysis and budget; do not compare it directly to this smoke's distinct@2.

| Artifact | Meaning |
| --- | --- |
| `launch_request.json`, `effective_config.json` | Authenticated recipe, inputs, source, environment and all resolved arguments; validation is not proof of completion |
| `debug_*/train_metrics.jsonl` | Update, objective, replay, and evaluation summaries |
| `debug_*/eval_mode_coverage_draws.jsonl` | Saved responses/keys, verifier diagnostics, draw seeds and evaluated step |
| `debug_*/evaluation_failures.jsonl` | Fatal evaluator faults; reject partial scores from affected steps |
| `debug_*/checkpoints/step_*/resume_manifest.json` | Complete identity-bound recovery state and file hashes |
| `debug_*/resume_decisions.jsonl` | Data position, sampled responses, bank and replay decisions |

Check the process exit status as well as artifacts. For a resumed scientific run, join the original prefix **through the selected checkpoint** with the resumed suffix; do not count repeated suffix evaluations twice. Timeouts or broken verifier workers raise `EvaluationFailure`, not zero reward.

To independently compare full model/optimizer/RNG/bank states and evaluation decisions for all four methods, run `python ops/resume_gpu.py run --workdir outputs/resume-proof --data-root outputs/data/pantry_plan --model "$model_path"`. It creates both trajectories and writes `report.json` after auditing automatically. Expected decision/bank equality is exact; GPU model/optimizer tolerances are [declared here](#reproduce-resume-equivalence). Retain outputs until the audit succeeds.

## Package artifacts and supported environments

| Use | Platform / Python | Dependencies |
| --- | --- | --- |
| Core API and CPU conformance | Linux x86_64, CPython 3.10–3.12 | PyTorch 2.6.x; NumPy >=1.26.4,<3; immutable ModeBench 0.4.0 pin |
| Maintained GPU training | Linux x86_64, CPython 3.10; one qualified A6000 | Exact CUDA 12.4 [GPU lock](../requirements-gpu-py310-cu124.txt) |

Windows, macOS, ARM, other CUDA/PyTorch versions, multiple learner GPUs, and Python 3.11/3.12 GPU training are not qualified. The `train` extra declares dependencies but does not replace the lockfile or install an NVIDIA driver. The `dev` extra adds conformance tests and [quality tools](../CONTRIBUTING.md#formatting-linting-and-typing).

`python -m pip install build && python -m build` builds an sdist and then a wheel from that sdist. Install either artifact normally with pip; the base package contains the input registry, recipes, shell assets and result-analysis inputs. Installed `remax`/`remax-run` commands work outside the checkout without `PYTHONPATH`. Repository examples and `ops/` audit tools still need the checkout or extracted sdist. See [artifact validation](../VALIDATION.md#installed-distribution-qualification).

## CPU development environment

Use the [README CPU quick start](../README.md#quick-start-installed-cpu-workflow), return to the checkout root, then install `.[dev]` and run `make quality` and `make check`. Reinstall after code/asset edits. Python 3.10 users can additionally select `-c constraints-cpu-py310-tested.txt`. Contributor checks and GPU qualification are separate.

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
Run identities and restore behavior are described in the workflow and [checkpoint contract](#explicit-identity-bound-resume). The launcher writes both identity records atomically and rechecks source/input bytes before entering training. Existing outputs are never overwritten. Historical `AUTO_RESUME` is metadata; there is no automatic latest-checkpoint discovery.

## Explicit, identity-bound resume

Strict launches use the versioned `remax-resume-v1` protocol. They support one collocated learner GPU, synchronous generation, one full candidate group per optimizer boundary, and the four maintained methods. Unsupported layouts fail before training. Historical direct-shell checkpoints lack this contract and cannot be loaded by the strict launcher.

The walkthrough above enables recovery writes before the first launch and demonstrates explicit restore. For a full registered recipe, set `OAT_ZERO_SAVE_CKPT=1` in a copied JSON recipe and retain the checkpoints you intend to resume.

Keep the **original total training horizon**, seed, evaluation schedule, optimizer/replay settings, source, dependencies, and hardware. Changing `NUM_PROMPT_EPOCH` to the number of remaining epochs changes the scheduler and is rejected. Input directories may move if their contents still authenticate. The output and explicit checkpoint selector may change. `--validate-only --resume ...` checks restore compatibility before starting workers; old checkpoints, missing manifests, changed files, and incompatible identities fail. There is no automatic latest-checkpoint discovery.

Checkpoints use DeepSpeed’s Python serialization and must come from a trusted run; manifest hashes check integrity, not publisher identity.

Each checkpoint saves model and optimizer state, scheduler state, DeepSpeed microstep position, Python/NumPy/Torch CPU/CUDA random-number states, the retained rollout buffer, data position, bank/exemplars/replay cursor, progress counters, and the last evaluated policy step. Restore skips consumed rows and the already-completed boundary evaluation. DataLoader construction uses an isolated generator so creating a replacement iterator cannot advance the policy RNG.

Free-form actor requests now have seeds derived from the run seed and consumed prompt position. Each strict free-form training request also clears the actor prefix cache: a warm cache after evaluation can change prefill batch shapes and sampled tokens compared with a newly started actor. Canonical learner sampling already has position-bound seeds. This is an explicit sampling-protocol change: strict free-form trajectories should not be equated with historical unseeded actor streams. The method's objective, admission rules, replay coefficients, and frozen numerical conformance cases are unchanged.

Writes go into a `.pending-*` directory. After all state files are flushed, `resume_manifest.json` binds their hashes to the run identity and step; a same-filesystem rename publishes the complete directory. Only then are `latest` and retention updated. Interrupted writes cannot replace the previous committed checkpoint. Ignore abandoned staging directories after a killed process; never rename one into a committed step. If interruption happens after commit but before updating `latest`, the new explicit step is still valid. Training resumes from a completed optimizer boundary, not from the middle of an unfinished backward pass.

`resume_decisions.jsonl` records each prompt position, sampled response tokens/rewards, bank identity, and selected replay groups. When combining attempts, retain the original prefix **through the selected checkpoint** and the resumed suffix after it. Discard any original attempt's later uncommitted metrics or evaluations. A `latest` pointer is a convenience, not an identity check.
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
