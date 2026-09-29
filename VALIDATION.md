# Validation — 2026-09-29

## Standalone public repository

A fresh Linux/Python 3.10.19 virtual environment installed PyTorch 2.6.0+cpu, the development extra, and ModeBench from public commit `94c45d1f66d3eb181fac2225dc04d62e08568b8e`. This environment was subsequently upgraded to ModeBench 0.4.0 at `33cfc3fd1732bd500fe13f13555419557aa248e2`. The installed package's VCS metadata and import location were checked; no sibling ModeBench checkout was used.

- `make check`: **428 tests passed**, with no skipped tests. Coverage includes objective gradients, bank state, the supported ModeBench API boundary, and all 20 exported recipe commands.
- `ops/reproduce_training.py`: **473 seed records / 95 arms** match the retained verified-key archive and summary, including support and missing-checkpoint rules. There are 47 reportable before/after arms and 84 terminal-reportable arms.
- `examples/replay_loss.py`: the MaxRL advantage and verified-replay loss/gradient example passed on CPU.
- Python examples in the method guide executed successfully.
- `pip check` reports no broken requirements. A standalone wheel builds from this checkout.
- Shell syntax checks pass. Documentation links and anchors were reviewed. The README schematic is an unchanged copy of the paper figure with recorded source SHA-256.
- `ops/verify_release.py` verifies the final file inventory and hashes.

Exact CPU dependency versions and ModeBench source metadata are in [VALIDATED_CPU_ENVIRONMENT.json](VALIDATED_CPU_ENVIRONMENT.json) and [constraints-cpu-py310-tested.txt](constraints-cpu-py310-tested.txt).

## Training conformance

`make conformance` passes **44 tests**. Twelve frozen updates cover Re:Max, Re:Dr, MaxRL, and Dr.GRPO from admission through real autograd/SGD updates on an eight-token CPU model. A separate scalar oracle checks every gradient and parameter update. Tests include microbatch/accumulation invariance, 4/16-rollout scaling, temperature, zero/nonzero coefficients, exact masks and EOS handling, singleton/empty banks, scheduling/restore, capacity, and a real single-rank Gloo layout. Controls execute replay scoring/backward with exactly zero replay gradient and parameters bitwise equal to fresh-only training.

An in-memory mutation audit confirmed that frozen comparisons detect eight deliberate errors: replacing mean response scores with sums; dropping accumulation compensation; doubling replay alpha; dropping per-rollout scaling; dropping reward-estimator scaling; applying replay gradients in controls; excluding singleton banks; and scoring prompt labels. Production source files were not modified by the conformance-only commit `1482f67` or its audit. The subsequent boundary upgrade preserves all frozen updates.

PR-base guard tests reject fixture deletion and changed expectations even after rehashing, allow unchanged references and added versions, and fail on a missing base ref. The adapter substitutes external OAT/DeepSpeed interfaces and CPU device selection; see [method coverage and limitations](docs/method.md#training-conformance). This does not validate GPU execution, real DeepSpeed/Adam, multi-rank behavior, or historical checkpoint equivalence.

## ModeBench boundary upgrade

The installed ModeBench VCS metadata matches the new immutable pin. All **175 historical reward/key pairs** across **25 cells** match the pre-upgrade dependency, and all frozen training updates still pass. Tests exercise structured failure propagation through the actual actor oracle, learner admission, trajectory collation, full-verifier JSON transport, and sampled-evaluation methods using CPU infrastructure adapters.

Real subprocess tests cover timeout, worker exit, malformed/partial replies, blocked writes, cleanup, and explicit recovery on a subsequent request. Internal legacy MATH deadline suppression, missing diagnostics, zero fallbacks, mismatched keys, and partial evaluation payloads are rejected. Failed evaluations produce a failure record with null metrics. Simulated rank exchange verifies that a remote verifier failure prevents score broadcasts; no multi-process/GPU claim is made.

The PR guard also protects the historical boundary fixture and its checksum. `make boundary` runs the focused checks; `make check` runs the complete suite and the unchanged 473-record/95-arm reproduction.

## Earlier integration checks

The original combined ModeBench/Re:Max extraction suite passed 240 tests in the research Python 3.10 environment. The actual OAT training module imported and completed `--help` there, and isolated installed-package checks exercised the shared benchmark boundary and MaxRL objective. [VALIDATED_ENVIRONMENT.json](VALIDATED_ENVIRONMENT.json) records that environment's versions.

## What remains untested

No fresh GPU training, distributed optimization, model generation, or historical-checkpoint equivalence run was performed for this publication. The recorded GPU constraints are not a complete clean-install lock. Newer result snapshots are outside the frozen-core numerical reproduction check.

The GitHub workflow defines CPU checks on Python 3.10, 3.11, and 3.12. This local standalone validation record covers Python 3.10; GPU qualification remains separate.
