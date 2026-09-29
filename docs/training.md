# Training guide

The core package and saved-key reproduction work on CPU. Actual policy training uses OAT, vLLM, DeepSpeed, and a compatible GPU environment. This release preserves recorded settings but has not completed a fresh end-to-end GPU training run.

All commands below assume the Re:Max checkout is the current directory, unless a command explicitly changes it. Use Bash on Linux.

## CPU development environment

Follow the [README quick start](../README.md#quick-start-cpu-checks). Install a CPU PyTorch build before the development extra to avoid pulling a GPU stack for tests. Git is needed to fetch the pinned ModeBench source dependency.

After selecting the CPU PyTorch build, you can use the exact tested Linux/Python 3.10 dependencies:

```sh
python -m pip install -c constraints-cpu-py310-tested.txt -e '.[dev]'
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

Install a CUDA-compatible PyTorch 2.6.0 build for your machine, then install the remaining stack:

```sh
python -m pip install --upgrade pip
python -m pip install -c constraints-train-recorded.txt -e '.[train,dev]'
```

[constraints-train-recorded.txt](../constraints-train-recorded.txt) records known versions, including NumPy/PyArrow compatibility for the older dataset stack. It is not a container or a clean-install guarantee. CUDA drivers, compilers, and extension builds remain environment-specific. [VALIDATED_ENVIRONMENT.json](../VALIDATED_ENVIRONMENT.json) records the environment used for the original extraction checks.

The public configs default to one GPU with collocated actor/learner, subject to the underlying launcher settings. Memory use depends on the domain, response length, model, and runtime. This release does not claim a tested minimum VRAM configuration.

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

The result has `train/` (DatasetDict subset `train`, 384 rows) and `eval/` (subset `multi_answer`, 128 rows). The materializer verifies the frozen source hashes and refuses an existing destination. Use the config matching the selected recipe domain; do not feed evaluation rows into training.

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

This step downloads model weights and requires network/storage. Pass the returned local directory to `--model`. The launcher checks for `config.json` before executing but does not authenticate that the directory matches the recipe revision; preserve the snapshot identity in your run records.

## Render, then execute

```sh
python ops/run_recipe.py configs/remax_countdown_05b.json \
  --data-root outputs/data/countdown \
  --model /path/returned/by/snapshot_download \
  --output outputs/remax-countdown-s43
```

The default only prints the OAT command. Data `train/` and `eval/` directories must exist even for command rendering. `--execute` additionally requires both DatasetDict metadata files and the model's `config.json`, then starts local training:

```sh
python ops/run_recipe.py configs/remax_countdown_05b.json \
  --data-root outputs/data/countdown \
  --model /path/returned/by/snapshot_download \
  --output outputs/remax-countdown-s43 \
  --execute
```

Use `--seed 44` for another seed from the registered set `[43, 44, 45, 46, 47]`. Other seeds are accepted by the launcher but are new experimental runs, not members of the retained cohort.

The wrapper replaces inherited `OAT_ZERO_*` configuration with the recipe and explicit path/seed arguments. Change a copied recipe to make a deliberate scientific override. It selects the calling Python interpreter, so activate the intended GPU environment before execution. No Slurm submission, automatic scheduler requeue, or external message is performed.

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

`--output` sets the training save root. OAT may create a run-specific subdirectory beneath it. The configs enable auto-resume and retain resume/checkpoint settings, including 192-step resume cadence and terminal model export. Reusing an output root can therefore select previous training state.

Use a new output root for every method/domain/seed unless you intend to resume that same run. Preserve config, code, dataset, model, bank state, and schedule identity across a resume. Never infer cohort membership from a directory name alone; match the recorded seed and protocol.

The launcher validates basic file existence, not complete dataset/model provenance. Keep the ModeBench file hashes, materialization record, model revision, rendered command, actual environment versions, and evaluation draw identities with the run.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing `torch` or CUDA libraries | Use the CPU environment for checks or activate the separate GPU runtime for training. |
| `No module named modebench` | Install this repository's declared dependencies; ModeBench is pinned to a public Git commit. |
| Missing dataset directory | Materialize the matching frozen config first and point `--data-root` to its parent directory. |
| Dataset/PyArrow import errors | Check the recorded Datasets, NumPy, and PyArrow versions together. |
| Canonical actions require vLLM V0 | Use the recipe wrapper, which sets the required runtime flag for Pantry. |
| Unexpected resume behavior | Inspect the selected save root and retained state; use a new root for a new run. |
| Frozen numerical reproduction fails | Restore the matching evidence and code revision; do not overwrite expected results to silence a mismatch. |

CPU CI does not exercise GPU sampling, distributed optimization, or checkpoint equivalence. Those remain explicit validation requirements for a full training reproduction.

Verifier failures are fatal, with structured diagnostics rather than zero rewards. Inspect `evaluation_failures.jsonl` for failed evaluation steps; do not combine partial draw records from those steps into a completed score. See [the boundary and compatibility policy](method.md#modebench-boundary-and-failures).
