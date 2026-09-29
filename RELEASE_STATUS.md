# Release scope — 0.1.0

This initial standalone release includes verified replay losses and bank state, OAT learner integration, 20 portable Qwen2.5-0.5B Level 1 recipes, core regression tests, frozen training conformance cases through CPU optimizer updates, a runnable CPU gradient example, and saved-key reproduction of 473 seed records across 95 arms.

ModeBench is a separate public dependency pinned to commit `94c45d1f66d3eb181fac2225dc04d62e08568b8e`. See [validation](VALIDATION.md) for checks actually completed.

## Known gaps and follow-up work

- A fresh end-to-end GPU training/evaluation run, followed by comparison to the corresponding historical runtime and configuration, remains required for a full training reproduction.
- The public recipes preserve recorded settings and explicitly resolve half-pass evaluation, resume cadence, and the base objective. A complete effective-launch and resumed-run audit across historical wrappers remains open.
- Comparator and historical branches remain in the learner's integration graph. Removing them requires numerical equivalence checks.
- Falcon, 3B, Level 2, newer tuned-control, RLEP, mechanism, and additional application reproduction chains are not fully exported as runnable recipes. Newer summary snapshots alone are not complete reproduction packages.
- Each additional headline result still needs a complete binding to source runtime, model, dataset/prompt, evaluation seeds, exclusion policy, and raw evidence.
- The recorded GPU version constraints are not a validated fresh-install lock or container. CUDA and distributed execution have not been qualified through CPU CI.
- Final scientific citation metadata remain to be added when available.

The extraction included working-tree changes and does not claim exact equivalence to every historical experiment. Original source hashes and frozen core evidence are retained to support that audit. No new GPU runs or model-provider calls were launched during publication preparation.
