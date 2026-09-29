# Validation — 2026-09-29

## Standalone public repository

A fresh Linux/Python 3.10.19 virtual environment installed PyTorch 2.6.0+cpu, the development extra, and ModeBench from public commit `94c45d1f66d3eb181fac2225dc04d62e08568b8e`. This environment was subsequently upgraded to ModeBench 0.4.0 at `33cfc3fd1732bd500fe13f13555419557aa248e2`. The installed package's VCS metadata and import location were checked; no sibling ModeBench checkout was used.

- `make check`: **443 tests passed**, with no skipped tests. Coverage includes objective gradients, bank state, the supported ModeBench API boundary, and all 20 exported recipe commands.
- `ops/reproduce_training.py`: **473 seed records / 95 arms** match the retained verified-key archive and summary, including support and missing-checkpoint rules. There are 47 reportable before/after arms and 84 terminal-reportable arms.
- `examples/replay_loss.py`: the MaxRL advantage and verified-replay loss/gradient example passed on CPU.
- Python examples in the method guide executed successfully.
- `pip check` reports no broken requirements. A standalone wheel builds from this checkout.
- Shell syntax checks pass. Documentation links and anchors were reviewed. The README schematic is an unchanged copy of the paper figure with recorded source SHA-256.
- `ops/verify_release.py` verifies the final file inventory and hashes.

Exact CPU dependency versions and ModeBench source metadata are in [VALIDATED_CPU_ENVIRONMENT.json](VALIDATED_CPU_ENVIRONMENT.json) and [constraints-cpu-py310-tested.txt](constraints-cpu-py310-tested.txt).

## Training conformance

`make conformance` passes **45 tests**. Twelve frozen updates cover Re:Max, Re:Dr, MaxRL, and Dr.GRPO from admission through real autograd/SGD updates on an eight-token CPU model. A separate scalar oracle checks every gradient and parameter update. Tests include microbatch/accumulation invariance, 4/16-rollout scaling, temperature, zero/nonzero coefficients, exact masks and EOS handling, singleton/empty banks, scheduling/restore, capacity, and a real single-rank Gloo layout. Controls execute replay scoring/backward with exactly zero replay gradient and parameters bitwise equal to fresh-only training.

An in-memory mutation audit confirmed that frozen comparisons detect eight deliberate errors: replacing mean response scores with sums; dropping accumulation compensation; doubling replay alpha; dropping per-rollout scaling; dropping reward-estimator scaling; applying replay gradients in controls; excluding singleton banks; and scoring prompt labels. Production source files were not modified by the conformance-only commit `1482f67` or its audit. The subsequent boundary upgrade preserves all frozen updates.

PR-base guard tests reject fixture deletion and changed expectations even after rehashing, allow unchanged references and added versions, and fail on a missing base ref. The adapter substitutes external OAT/DeepSpeed interfaces and CPU device selection; see [method coverage and limitations](docs/method.md#training-conformance). This does not validate GPU execution, real DeepSpeed/Adam, multi-rank behavior, or historical checkpoint equivalence.

## ModeBench boundary upgrade

The installed ModeBench VCS metadata matches the new immutable pin. All **175 historical reward/key pairs** across **25 cells** match the pre-upgrade dependency, and all frozen training updates still pass. Tests exercise structured failure propagation through the actual actor oracle, learner admission, trajectory collation, full-verifier JSON transport, and sampled-evaluation methods using CPU infrastructure adapters.

Real subprocess tests cover timeout, worker exit, malformed/partial replies, blocked writes, cleanup, and explicit recovery on a subsequent request. Internal legacy MATH deadline suppression, missing diagnostics, zero fallbacks, mismatched keys, and partial evaluation payloads are rejected. Failed evaluations produce a failure record with null metrics. Simulated rank exchange verifies that a remote verifier failure prevents score broadcasts; no multi-process/GPU claim is made.

The PR guard also protects the historical boundary fixture and its checksum. `make boundary` runs the focused checks; `make check` runs the complete suite and the unchanged 473-record/95-arm reproduction.

## Earlier integration checks

The original combined ModeBench/Re:Max extraction suite passed 240 tests in the research Python 3.10 environment. The actual OAT training module imported and completed `--help` there, and isolated installed-package checks exercised the shared benchmark boundary and MaxRL objective. [VALIDATED_ENVIRONMENT.json](VALIDATED_ENVIRONMENT.json) records that environment's versions.

## Complete GPU workflow

A fresh Python 3.10.19 virtual environment with **system site packages disabled** installed the public ModeBench pin and the GPU training dependencies. `pip check` passed. The exact runtime dependency set is [requirements-gpu-py310-cu124.txt](requirements-gpu-py310-cu124.txt); [the walkthrough](docs/training.md#complete-gpu-smoke-workflow) supplies installation, verified preparation, training, and artifact-audit commands.

All four methods ran Qwen2.5-0.5B-Instruct at the registered revision on the first four frozen Level 1 PantryPlan training rows and first two evaluation rows, using one **48 GB RTX A6000**, eight CPU cores, and 64 GB host RAM. NVIDIA driver: **610.57.04**; PyTorch CUDA runtime: **12.4**. Model Git/LFS hashes and frozen dataset hashes passed before execution. Every method performed 64 fresh generations, four DeepSpeed/AdamW optimizer updates, bank admission, replay scoring, checkpointing, model export, and greedy/sampled evaluation.

| Method | Saved bank outcomes | Applied replay score gradient | Run time |
| --- | ---: | --- | ---: |
| Dr.GRPO | 20 | Exactly zero | 133 s |
| Re:Dr | 18 | Nonzero | 132 s |
| MaxRL | 20 | Exactly zero | 126 s |
| Re:Max | 19 | Nonzero | 128 s |

Every exported model contains over 2.4 million numerically changed parameter elements; each saved optimizer records four updates. All four bank states load successfully. **792 evaluation responses** carry complete scorable verifier diagnostics across the four runs. Reloading Re:Max's step-4 checkpoint produced **identical five draw records / 66 responses**, including rewards, canonical identities, and diagnostics. The reload took 69 seconds. Peak sampled GPU memory was **21,139 MiB**; total allocation time including artifact auditing was **11 minutes 27 seconds**. The terminal export's step-5 label is the loop cursor after four updates; checkpoint and final evaluation are at step 4.

The clean run exposed and fixed shared-Python-library discovery for Launchpad and a canonical Pantry decoding boundary bug. Legal but infeasible masks remain scorable rejections, while verifier failures still abort. Regression checks cover all 64 masks against the supported raw-mask API and every fatal verifier status. Frozen numerical/training references remain unchanged.

[VALIDATED_GPU_RUN.json](VALIDATED_GPU_RUN.json) records the measurements, hashes, and configuration comparison. All 85 explicit pantry Re:Max ledger settings match the public recipe. The historical runtime log reveals two launcher differences: its wrapper capped evaluation to 96 steps versus the portable recipe's 192, and enabled recovery writes that the portable launcher makes opt-in. The smoke explicitly enables checkpoint writes and supplies a checkpoint path for reload; it does not depend on automatic discovery. The historical fixture is hash-checked and protected against rewrites by the PR-base guard.

## What remains untested

Full-budget historical trajectories and scores, other GPU types, multiple learner GPUs, and training in the other four domains remain unqualified by this small pantry workflow. CPU CI checks the software and frozen numerical contracts; the GPU validation above was a separate manual run. Newer result snapshots remain outside the frozen-core numerical reproduction check.
