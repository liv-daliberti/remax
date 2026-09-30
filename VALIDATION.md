# Validation — 2026-09-30

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

## Strict recipes and authenticated inputs

The expanded `make check` suite passes **537 tests**, with no skips, on a CPU host using the pinned Python 3.10 GPU environment. This includes 94 additional checks for strict field/type validation, domain/prompt/method compatibility, inherited-setting rejection, changed model/tokenizer bytes, dataset row order/prompts/references, rewritten local manifests, runtime configuration drift, atomic records, and immutable input registries. The 473-record/95-arm numerical reproduction and existing frozen training/reference fixtures remain unchanged.

All **20 recipes** passed `--validate-only` in the actual pinned runtime on one RTX A6000, using the full **384 train / 128 evaluation rows** for each of the five Level 1 domains and the pinned Qwen snapshot. Every saved `effective_config.json` contains **332 validated runtime arguments**, the explicit seed override **47**, correct method/control flags, authenticated input identities, current source hashes, and installed dependency versions. Each output contains only the request and effective configuration: no model, worker, checkpoint, or optimizer update was created. The final preflight allocation completed successfully in **4 minutes 43 seconds**. These are configuration/hardware checks, not new training measurements.

Separate real CLI invocations rejected a wrong-domain dataset, an incorrect model configuration, the wrong Pantry prompt interface, an unknown field, and an inherited setting. Each exited with code 2 before creating an output directory. The model's seven input files were independently rechecked against the immutable Hub commit's Git/LFS metadata; all ten frozen train/eval Parquet splits were hash-verified before deriving the release registry's ordered materialized-row hashes.

A wheel built and installed successfully. An isolated `python -I` smoke from outside the checkout loaded the installed recipe API and packaged registry (ten dataset splits and seven model files). `pip check`, the historical recipe comparison, CPU example, shell syntax, local documentation links, formatting, and the PR-base reference guard passed. Existing GPU training/reload measurements above remain scoped to the earlier Pantry smoke; historical direct shell launches are outside the strict recipe launch contract.

## Full-run resume conformance

`make check` passes **627 tests**, without skips; `make resume` selects **82** resume and audit checks. The new production-loop CPU suite covers **40 uninterrupted/interrupted comparisons**: four methods, five interruption positions (1, 2, 3, 6, and 9 of nine updates across three shuffled epochs), and two rollout-buffer policies. Each continuation starts after different random initialization. Real AdamW, a changing LambdaLR schedule, GRPO/replay autograd, banks, and production checkpoint/progress code are used; the actor and DeepSpeed transport are CPU adapters. Model, optimizer, scheduler, RNG state, data/sample/replay decisions, banks, evaluation cadence, and accumulation position agree **exactly**. Every baseline performs nine real optimizer updates and executes replay.

Fault tests cover process death inside a staged write, write failure, interruption after commit but before updating `latest`, retention, corrupted/missing/extra state, invalid cursors, incompatible identities, and explicit-selector validation before workers. The GPU audit has regression tests for tensor tolerances, NaNs, loss-scaler state, exact discrete state, and incomplete experiment records. A wheel installation outside the checkout successfully executed the checkpoint transaction and identity-validation API without a source-path override.

The strict protocol commits after evaluation/logging and captures retained rollout buffers as well as Python/NumPy/Torch CPU/CUDA RNG state. Free-form request seeds are bound to prompt position; the actor prefix cache is cleared before each such training request. These sampling changes are explicit and are not a claim of historical unseeded-stream equivalence. The frozen training fixtures and 473-record/95-arm numerical reproduction remain unchanged.

On the pinned Python 3.10 / CUDA 12.4 runtime and one **48 GB RTX A6000**, all four methods completed six updates across two epochs of the first three authenticated PantryPlan training rows. Each then continued in a fresh process from the whole run's committed step-2 checkpoint. Evaluation used all **128** frozen rows, with two samples and one coverage draw. At steps **4 and 6**, model tensors meet `atol=1e-6, rtol=1e-6` and optimizer tensors meet `atol=1e-8, rtol=1e-5`; those tolerances were declared before execution. Banks, replay/sample/data decisions, scheduler/RNG state, progress counters, and evaluation records match exactly.

| Method | Accepted fresh responses / 96 | Replay groups selected | Whole / resumed time |
| --- | ---: | ---: | ---: |
| Dr.GRPO | 32 | 6 | 266 / 194 s |
| Re:Dr | 30 | 6 | 231 / 221 s |
| MaxRL | 34 | 6 | 253 / 213 s |
| Re:Max | 30 | 6 | 233 / 192 s |

A separate six-update Re:Max Countdown pair passes the same state and decision audit (205 / 173 seconds). The initial free-form comparison exposed different sampled tokens after restart despite fixed request seeds. Clearing the prefix cache before each training request removed that discrepancy; the corrected learner is also exercised by the CPU cache regression. This tiny Countdown sample had **zero accepted training answers**, so it qualifies free-form sampling/state continuity rather than learning from successful answers. Pantry supplies the nonempty banks and replay updates.

[VALIDATED_RESUME_RUN.json](VALIDATED_RESUME_RUN.json) records input/source identities, configuration and checkpoint-manifest hashes, timings, audit outcomes, and limits. The GPU comparison forks a saved checkpoint into a fresh process; CPU tests separately exercise actual writer process death. These are complete bounded runs, not full-budget paper-score reproductions. Historical direct-shell checkpoints, other GPU types, asynchronous execution, multiple learner ranks, and mid-backward recovery are outside this qualification. The [training guide](docs/training.md#reproduce-resume-equivalence) gives commands to reproduce the comparisons.

## Maintained replay-core extraction

`make check` passes **687 tests**, without skips, including **54 architecture/parity checks** and **six rejection checks for the GPU comparison tool**. The unchanged frozen reproduction still matches **473 seed records / 95 arms**, and the PR-base guard preserves all eight reference/registry files from the previous commit.

The maintained learner now orchestrates admission, scoring, fresh advantages and optimization in **172 lines**, with separate OAT modules for replay backward, telemetry, sampling, data, synchronization, evaluation and checkpoints. Framework-independent bank/admission/scheduling/scoring/objective/checkpoint code lives in `remax.core`; historical comparator objectives and integrations live in `remax.experiments`. Old import paths remain compatibility surfaces. The retained historical GRPO class is syntax-tree identical to its pre-refactor implementation. Forty-four extracted run-loop methods are also syntax-tree identical; two checkpoint/progress methods defer historical type imports until those states are present.

Twelve direct historical/extracted comparisons cover four methods and three microbatch widths, each over the three frozen trajectory groups. Model calls, complete bank/schedule/mask traces, model parameters, gradients, and all non-timing metrics agree **exactly**. Separate dependency tests block OAT/vLLM/DeepSpeed/Transformers and comparator imports from the core, and execute all four maintained update paths with comparator imports forbidden. Experimental switches are routed explicitly; an identity-bound maintained run cannot silently fall back to the historical adapter. All prior verifier-failure and full-run resume tests remain active.

A wheel installed in a separate Python 3.10 environment with PyTorch **2.6.0+cpu**, NumPy **1.26.4**, and the public ModeBench pin passes an isolated `python -I` smoke from `/tmp`: verified-bank admission, deterministic replay selection, causal masks, replay loss/gradients, and exact bank-state restoration. The import guard rejects training frameworks and comparator packages throughout that smoke. `pip check` reports no broken requirements.


The extracted implementation also completed all four six-update PantryPlan GPU runs and fresh-process continuations from step 2 on one **48 GB RTX A6000**, using the same authenticated inputs and pinned runtime as the pre-refactor resume qualification. The automatic resume audit passes at steps **4 and 6**. An independent, read-only comparison against the pre-refactor whole runs passes at step **6**: non-source run identities, full response/replay/evaluation traces, banks, RNG, scheduler and progress state agree exactly. Model tensors meet `atol=1e-6, rtol=1e-6`; optimizer tensors meet `atol=1e-8, rtol=1e-5`. No checkpoint was restored across source revisions.

| Method | Accepted fresh responses / 96 | Replay groups selected |
| --- | ---: | ---: |
| Dr.GRPO | 32 | 6 |
| Re:Dr | 30 | 6 |
| MaxRL | 34 | 6 |
| Re:Max | 30 | 6 |


A separate six-update Re:Max Countdown run, its step-2 continuation, and its independent pre-refactor comparison also pass. This sample admits no correct training answers, so it qualifies free-form sampling and state continuity; Pantry supplies verified banks and replay updates. The Countdown allocation and independent comparison both completed successfully.

[VALIDATED_REPLAY_CORE_RUN.json](VALIDATED_REPLAY_CORE_RUN.json) records runtime source hashes, configuration/checkpoint hashes, all ten training processes, resume and pre-refactor comparisons, and the qualification limits. The Pantry allocation was cancelled only after its automatic audit had succeeded, while an accidentally repeated audit was running; the completed automatic report is retained. These bounded runs qualify the extraction on one GPU, not full-budget scores or historical comparators.


## Installed distribution qualification

The default isolated `python -m build` produced an sdist and built the wheel from that sdist. The wheel includes all 20 recipe JSON files, both launcher shell assets and the authenticated input registry. Clean virtual environments with system site packages disabled installed the wheel with **PyTorch 2.6.0+cpu / NumPy 1.26.4** and the sdist with **PyTorch 2.6.0+cpu / NumPy 2.2.6**, on CPython 3.10.19. Both pass `pip check` and the four standalone core-workflow checks using `python -I` outside the checkout, without pytest or any training extra installed. The checks cover console/module commands, all recipe previews, structured verification, bank admission, replay gradients and checkpoint restoration.

After installing development dependencies, the sdist installation passes **694 regression tests** from a temporary directory containing tests, audit helpers and frozen evidence, with **no `src/` directory**. The wheel installation passed the preceding 693-test suite and the final 128-test launcher/package subset after the cache-location regression was added. Numerical reproduction remains **473 records / 95 arms** and all eight pre-change reference/registry files are preserved. Pytest uses importlib mode; neither tests nor verifier children add a checkout to Python's import path.

CI now builds release artifacts once, then independently installs wheel and sdist on **Python 3.10, 3.11 and 3.12**. Wheel jobs select the NumPy floor; sdist jobs select compatible NumPy 2.x. Each job runs the core-only workflow before installing development dependencies and running the full suite outside the checkout. Training dependencies remain separate: their declared extra resolves against the existing qualified GPU lock without dependency changes, and the runtime guard rejects unsupported CUDA versions. See the [support matrix and installed workflow](docs/training.md#package-artifacts-and-supported-environments).


All six artifact CI jobs passed on the implementation commit. A wheel installed into the qualified GPU environment also completed a **three-update Re:Max PantryPlan run** on one 48 GB A6000, launched from a temporary working directory with no `PYTHONPATH`. The audit confirms **14 retained verified outcomes**, nonzero applied replay gradients, **768 evaluation responses** with complete scorable diagnostics, a valid step-3 checkpoint and a terminal model export. All **133 installed runtime/asset files** match both the tested wheel and the implementation checkout. The allocation completed successfully in **5 minutes 54 seconds**, including cold imports and initialization.

The first GPU attempt stopped before training because the home cache quota was full; the successful run uses the tested `XDG_CACHE_HOME` override on writable scratch storage. [VALIDATED_PACKAGE_RUN.json](VALIDATED_PACKAGE_RUN.json) records artifact hashes, CI coverage, runtime identities, measurements and scope. This qualifies installed execution on the existing pinned training stack, not additional hardware/CUDA versions or full-budget scores.


## Distributed weighting and resource audit

The expanded suite passes **703 tests**, including real Gloo/DDP execution at **1, 2 and 4 learner ranks**. Four methods × three microbatch sizes × three rank counts give **36 configurations / 108 logical updates**, each compared with the independent scalar gradient oracle. Accumulation widths span 1–16. Gradients and parameters meet `atol=3e-7, rtol=3e-6`; banks/scheduling match exactly, and compute-only replay gradients are exactly zero. Invalid accumulation/global-batch combinations and inconsistent rank coefficients or selected membership fail before backward. The frozen 473-record/95-arm reproduction and all eight protected references/registries remain unchanged.

The GPU profiling allocation completed successfully on one **48 GB A6000**. Six shapes/control cases use the production replay scorer/backward, frozen BF16 Qwen2.5-0.5B-Instruct weights, two warmups and six measurements each. All cases return to exactly the same live GPU allocation after each measured repetition. Larger shapes raise allocator reservation, which persists for later smaller cases; the audit reports allocated and reserved bytes separately. The bounded exemplar bank is also separated from the exact discovery ledger: at 4,096 discovered modes and capacity 16, the bank still retains only 16 exemplars while its serialized state reaches roughly 74 KiB. After warmup, 256 fixed-support admission/scheduling repetitions add less than 1 KiB of traced live Python allocation in each measured case.

[VALIDATED_SCALING_RUN.json](VALIDATED_SCALING_RUN.json) records the full matrix, timings, memory, throughput, source hashes and measurements from the prior 6.44 GiB full training checkpoint. The [method guide](docs/method.md#rank-count-accumulation-and-resource-costs) explains the coefficient cancellation, replicated compute cost and storage boundaries. This audit does **not** newly qualify multi-GPU DeepSpeed/NCCL, distributed checkpoint recovery or full-budget trajectories. It adds execution guards and observability; it does not evict discoveries or change replay coefficients.

## Installed result packages

The result-package change passes the 719-test full regression suite plus the additional frozen-catalog PR-guard test (720 cases in total). The original 473-cell / 95-arm numerical analysis and all eight pre-existing protected reference files remain unchanged. New conformance cases cover complete 475-cell accounting, seed exclusions, Level-2 numerical reproduction, Qwen-3B model/prompt records, E122 versus E123 qualification, changed runtime hashes, and rejection of unsupported reproduction claims or altered artifacts.

The wheel passes all five core-only installed smoke cases from `/tmp`, without source-path overrides. A separately installed sdist runs the documented `remax results reproduce` command outside the checkout using NumPy 2.x; summary-only reproduction returns a nonzero exit status. Both distribution formats include the analysis archive and authenticated result catalog. These checks qualify installed saved-key analysis, not fresh historical training. The audit records six missing runtime snapshot identities and two changed snapshot trees; no additional historical training recipe is presented as verified.

## What remains untested

Full-budget historical trajectories and scores, other GPU types, multiple learner GPUs, and nontrivial learning in the other four domains remain unqualified. The bounded Countdown run checks free-form sampling and restore continuity but admitted no correct training responses. CPU CI checks the software and frozen numerical contracts; the GPU validation above was a separate manual run. Newer result snapshots remain outside the frozen-core numerical reproduction check.
