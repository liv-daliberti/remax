# Changelog

## Unreleased

Public source version remains 0.1.0; no PyPI or tagged release is implied. Use a Git commit to identify a checkout. Historical protocols and numerical references retain their own versions.

### Added

- Approved citation for *Measuring and Mitigating Solution Mode Collapse in RLVR*: accepted at MATH-AI 2026; under review at ICLR 2027.
- One install/train/resume/evaluate workflow, an installed CPU API walkthrough, repository ownership, and a scientific compatibility policy.
- Ruff formatting/linting across `remax.core` and an initial strict mypy gate on numerical objectives, scoring, execution checks, and tensor/group types.
- Installed result discovery, provenance inspection, integrity checks, and saved-key reproduction: 95 arms, 475 bound cells, 473 included records, six explicitly labeled summary snapshots.
- Strict recipes with authenticated model/data inputs and complete effective configurations; structured verifier failures; installed wheel/sdist checks on Python 3.10–3.12.
- Full-run resume conformance, atomic identity-bound checkpoints, distributed weighting tests, and measured replay resource costs.

### Changed

- Maintained replay lives in `remax.core`, OAT integration in `remax.integrations.oat`, and historical comparators in `remax.experiments`. Compatibility imports remain available.
- Strict free-form generation uses position-bound seeds and clears the actor prefix cache for resume continuity. These trajectories are not equivalent to historical unseeded actor streams.
- Recovery is explicit (`--resume`) and requires a committed manifest; historical automatic-resume settings are metadata. Old direct-shell checkpoints cannot be upgraded by renaming directories.

### Compatibility and evidence

Core formatting and annotations preserve executable function bodies and the frozen numerical contracts. They still change source bytes: existing strict checkpoints require their original installed runtime. Keep the old environment for resuming an old run.

The four-method GPU smoke and bounded resume comparisons qualify one A6000 / Python 3.10 / CUDA 12.4 stack. No full-budget historical scores, additional GPU platforms, or distributed recovery are newly qualified. Level-2/3 and larger-model historical training exports remain incomplete because runtime/input/configuration evidence is missing or changed. See [release scope](RELEASE_STATUS.md) and [validation](VALIDATION.md).
