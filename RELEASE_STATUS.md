# Release scope — 0.1.0

This initial standalone release includes verified replay losses and bank state, OAT learner integration, 20 portable Qwen2.5-0.5B Level 1 recipes, core regression tests, frozen training conformance cases through CPU optimizer updates, a runnable CPU gradient example, and saved-key reproduction of 473 seed records across 95 arms.

ModeBench is a separate public dependency pinned to commit `33cfc3fd1732bd500fe13f13555419557aa248e2`. See [validation](VALIDATION.md) for checks actually completed.

## Known gaps and follow-up work

- A clean one-GPU PantryPlan workflow now covers all four methods through checkpoint reload and evaluation. Full-budget historical score/trajectory reproduction and GPU qualification of the other domains remain open.
- The selected pantry recipe matches 85 recorded settings. Its historical wrapper actually evaluated every 96 steps, versus the public recipe's explicit 192; the checked comparison records this difference and the portable launcher's opt-in recovery writes. Automatic recovery discovery is not implemented by the portable launcher. A complete audit across all historical wrappers remains open.
- Comparator and historical branches remain in the learner's integration graph. Removing them requires numerical equivalence checks.
- Falcon, 3B, Level 2, newer tuned-control, RLEP, mechanism, and additional application reproduction chains are not fully exported as runnable recipes. Newer summary snapshots alone are not complete reproduction packages.
- Each additional headline result still needs a complete binding to source runtime, model, dataset/prompt, evaluation seeds, exclusion policy, and raw evidence.
- The GPU runtime now has a clean-install dependency set and measured A6000 workflow. Multi-GPU operation, other hardware, and containers remain unqualified; CPU CI does not provide GPU validation.
- Final scientific citation metadata remain to be added when available.

The extraction included working-tree changes and does not claim exact equivalence to every historical experiment. Original source hashes and frozen core evidence are retained to support that audit. The original extraction launched no new GPU runs. The subsequent GPU validation launched small local-model training runs; no hosted model-provider calls were used.
