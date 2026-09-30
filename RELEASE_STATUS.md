# Release scope — 0.1.0

This initial standalone release includes verified replay losses and bank state, OAT learner integration, 20 portable Qwen2.5-0.5B Level 1 recipes, core regression tests, frozen training conformance cases through CPU optimizer updates, a runnable CPU gradient example, and saved-key reproduction of 473 seed records across 95 arms.

ModeBench is a separate public dependency pinned to commit `33cfc3fd1732bd500fe13f13555419557aa248e2`. See [validation](VALIDATION.md) for checks actually completed.

The maintained recipe launcher now validates typed configuration and method/domain/prompt compatibility, authenticates local model and full frozen dataset contents offline, rejects conflicting inherited settings, and records every validated runtime argument before training. See [the strict launch workflow](docs/training.md#validate-then-execute). Historical shell entry points remain available outside that strict contract.

## Known gaps and follow-up work

- A clean one-GPU PantryPlan workflow covers all four methods through training, resume equivalence, and evaluation. A bounded Countdown comparison additionally checks free-form generation continuity. Full-budget historical score/trajectory reproduction and nontrivial learning qualification in the other domains remain open.
- The selected pantry recipe matches 85 recorded settings. Its historical wrapper actually evaluated every 96 steps, versus the public recipe's explicit 192; the checked comparison records this difference and the portable launcher's opt-in recovery writes. Automatic recovery discovery is not implemented by the portable launcher. A complete audit across all historical wrappers remains open.
- Strict resume is explicit and identity-bound, with atomic commits and full learner state. The supported contract is one synchronous collocated learner GPU at completed optimizer boundaries; asynchronous, multi-rank, and mid-backward recovery are not qualified. Historical direct-shell checkpoints cannot claim this contract.
- Maintained replay now lives in `remax.core` and `remax.integrations.oat`; historical comparators are isolated in `remax.experiments`. Historical training-result qualification still requires evidence beyond this code separation.
- Falcon, 3B, Level 2, newer tuned-control, RLEP, mechanism, and additional application reproduction chains are not fully exported as runnable recipes. Newer summary snapshots alone are not complete reproduction packages.
- Each additional headline result still needs a complete binding to source runtime, model, dataset/prompt, evaluation seeds, exclusion policy, and raw evidence.
- The GPU runtime now has a clean-install dependency set and measured A6000 workflow. Multi-GPU operation, other hardware, and containers remain unqualified; CPU CI does not provide GPU validation.
- Approved authorship and paper status are in [CITATION.cff](CITATION.cff). Repository ownership and scientific change policy are in [Contributing](CONTRIBUTING.md).

The extraction included working-tree changes and does not claim exact equivalence to every historical experiment. Original source hashes and frozen core evidence are retained to support that audit. The original extraction launched no new GPU runs. The subsequent GPU validation launched small local-model training runs; no hosted model-provider calls were used.


## Result package qualification

The installed `remax results` commands expose 95 runnable saved-key analysis arms and six retained summary snapshots. All 475 archived cells are bound, including the two excluded from the 473-cell numerical analysis. The Level-3 E122 snapshot is included with its admission decisions and recorded launches; the unlaunched Qwen-3B E123 plan is distinguished explicitly.

No new Level-2/3 or larger-model training recipe is promoted as verified by this audit. Six historical runtime identity records are missing and two surviving snapshot trees fail their original identity hashes. Registered launches also lack complete effective configuration and dependency authentication. These gaps do not invalidate saved-key numerical reproduction. See the [audit, commands, and promotion requirements](docs/reproducibility.md#result-packages-and-training-export-audit).
