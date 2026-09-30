# Reproducibility and evidence

## What the included check reproduces

```sh
python ops/reproduce_training.py
```

The check reads `evidence/verified_samples_completed_cohort.jsonl.gz`, verifies its SHA-256 against the retained summary, and recomputes:

- every included seed record and its before/after checkpoint summaries;
- per-arm summaries and support eligibility;
- missing-step and terminal-only admission decisions;
- coverage counts and the archived protocol metadata.

The expected coverage is:

```json
{"cells":473,"arms":95,"reportable_arms":47,"terminal_reportable_arms":84}
```

Here `cells` counts retained seed records. `reportable_arms` refers to before/after change support, while `terminal_reportable_arms` requires support at the terminal checkpoint. These are distinct populations, not interchangeable denominators.

The builder compares steps 0 and 3072 in the frozen archive. Selected step-0-only failures can retain an admissible terminal record under the original rules; issues affecting other steps are treated differently. The source builder preserves those distinctions. It does not silently turn missing observations into zeros.

PCMD is computed from prompt-local verified-mode counts, pooling a prompt's recorded groups and averaging defined prompts equally. Keep correctness and support alongside PCMD; see [ModeBench's metric guide](https://github.com/liv-daliberti/modeBench/blob/33cfc3fd1732bd500fe13f13555419557aa248e2/README.md#metrics).

## Evidence boundaries

| Included artifact | Role |
| --- | --- |
| `evidence/verified_samples_completed_cohort.jsonl.gz` | Unchanged verified-key source archive for the core reproduction |
| `evidence/mode_diversity_training.json` | Frozen expected core analysis |
| `evidence/snapshots/matched_redr_20260924.json` | Newer matched Re:Dr summary snapshot |
| `evidence/snapshots/replay_mechanism_ladder_20260924.json` | Mechanism-study summary snapshot |
| `evidence/snapshots/tuned_control_sweep_20260927.json` | Tuned-control sweep summary snapshot |
| `evidence/snapshots/tuned_control_stage2_20260927.json` | Tuned-control follow-up summary snapshot |
| `evidence/snapshots/online_rlep_20260927.json` | Online-RLEP summary snapshot |
| `protocols/` | Selected original preregistration documents |

The snapshot files preserve numerical records and provenance. Their full raw-source reproduction chains have not been migrated, so passing the core reproduction check does not validate every number in those snapshots. Likewise, the archive covers more experimental arms than the 20 exported training recipes.

The included archive is a saved-key analysis input, not a complete collection of original model responses or trained weights. The check does not regenerate responses, re-execute every historical verifier, or rerun training.

## Identities and environment

The package depends on ModeBench commit `33cfc3fd1732bd500fe13f13555419557aa248e2`, rather than an unversioned sibling checkout or an assumed PyPI release.

- `PROVENANCE.json` maps extracted files to research source paths and hashes. The extraction included working-tree changes, so the original research commit alone is insufficient to reconstruct it.
- `RELEASE_MANIFEST.json` binds final release files, including evidence and documentation. Check it with `python ops/verify_release.py`.
- `VALIDATED_CPU_ENVIRONMENT.json` and `constraints-cpu-py310-tested.txt` record the fresh CPU development environment tested for publication.
- `VALIDATED_ENVIRONMENT.json` records the environment used during the initial extraction.
- `constraints-train-recorded.txt` records compatibility-sensitive historical training versions; it is not a complete fresh-install lock or container.
- `VALIDATION.md` distinguishes actual local checks from unperformed GPU reproduction.

Historical paths in source manifests, archive metadata, and preregistration documents identify original research artifacts. The maintained test and reproduction commands do not require those paths.

The source figure in the README is copied unchanged from the paper. Its original path and SHA-256 are in `PROVENANCE.json`.

## Report a reproduced or new result

Retain the Re:Max commit, ModeBench commit, recipe JSON and any overrides, selected model revision, dataset config/split hashes, exact prompts, seed, actual dependency versions, and evaluation draw identities. Preserve failure/retry records, checkpoint step selection, exclusions, and the paired seed intersection used by a comparison.

Separate these claims:

1. **Saved-key analysis matches:** the included frozen numerical check succeeds.
2. **Response regrading matches:** supplied raw responses produce the original keys under a specified verifier version.
3. **Fresh training reproduces results:** a new run uses the intended runtime/protocol and is compared under the same evidence rules.

The first claim is exercised by this release. The second and third require additional artifacts and execution. Floating-point optimization, sampling, GPU behavior, and hosted-model deployment changes can affect fresh outputs; exact source and protocol identities remain necessary even when bit-identical checkpoints are not claimed.

## Citing the implementation

Use the approved paper citation in the [README](../README.md#citation) or [CITATION.cff](../CITATION.cff), plus the exact Re:Max and ModeBench commits, recipe, model revision, and dataset identities. The paper is accepted at MATH-AI 2026 and under review at ICLR 2027; the citation does not claim ICLR acceptance. No DOI has been assigned here.

## Result packages and training export audit

Installed users can inspect and reproduce individual frozen result arms without the research checkout, training dependencies, a model download, or a GPU:

```sh
remax results list
remax results show level1/qwen3b/countdown/replay_maxrl
remax results reproduce level2/qwen05b/countdown/replay_maxrl
remax results verify
```

`list` returns 101 entries: 95 `saved_key_reproducible` arms and six `summary_snapshot` files. `show` emits JSON with the result and available bindings. `reproduce` validates the catalog, recomputes the entire frozen archive against every expected arm and seed, then returns the selected arm with its seed results and excluded cells. Its status is `saved_key_reproduced`, with `training_reproduction_verified: false`. Expected whole-archive coverage remains 473 cells, 95 arms, 47 before/after-reportable arms, and 84 terminal-reportable arms. `verify` authenticates artifact bytes and checks catalog coverage; it explicitly reports `numerical_reproduction_performed: false`.

| Frozen result stratum | Runnable saved-key arms | Training recipe relationship |
| --- | ---: | --- |
| Level 1, Qwen2.5-0.5B | 25 | 20 related maintained recipes; five plain-GRPO arms have none |
| Level 1, Qwen2.5-3B | 25 | No qualified historical training export |
| Level 1, Falcon3-1B | 25 | No qualified historical training export |
| Level 2, Qwen2.5-0.5B | 20 | No qualified historical training export |
| Level 3, Qwen2.5-0.5B | 0 | E122 summary snapshot, with 100 registered launches and explicit admission gaps |

The historical method names `replay_drgrpo` and `replay_maxrl` mean Re:Dr and Re:Max. Plain `grpo` is a separate comparator. Neither a matching method name nor a related maintained recipe implies that historical scores have been reproduced. In particular, the previously checked Pantry recipe differs from historical evaluation cadence and checkpoint defaults.

### What is bound

[`catalog_v1.json`](../evidence/experiments/catalog_v1.json) pins the analysis archive, expected numerical results, original analysis code, all related recipes, six snapshots, and [`historical_bindings_v1.json`](../evidence/experiments/historical_bindings_v1.json) by SHA-256. Its 95 arm entries account for all **475 archived cells**, including the two omitted by the frozen analysis rules. Per-cell records preserve analysis inclusion, terminal-only decisions, support/reportability, paired-cohort membership, sample issues, approved exclusions, evaluation certificates, draw seeds, and original source locations/hashes. Missing values remain unknown; they are never replaced with zero scores or fabricated seeds.

Where retained ledgers exist, each cell also records the ledger digest, registered run/job identity, full registered environment, model cache identity, dataset paths, prompt/syntax condition, and runtime snapshot reference. Ledger metadata retains available protocol and dataset identity pins. Model IDs/revisions extracted from historical paths are **recorded claims, not byte authentication**. The 75 plain-GRPO cells have saved-key provenance but no launch-ledger binding in this archive; their `launch` field is null. Source paths are provenance identifiers, never paths readers need to recreate, and recorded shell strings are never executed.

### Why no additional training recipe is called verified

The audit used the original E76 tree-hash algorithm against the runtime snapshots referenced by the frozen launch records:

- Six referenced snapshots have no surviving `SNAPSHOT_IDENTITY.json` at the recorded location.
- The surviving `50d36295558a8958` (Level 2) and `089bcea44b44cc70` snapshots differ from their recorded full tree hashes. Both expected and observed hashes are retained. A mismatch establishes that the tree changed; it does not establish whether a numerical method changed.
- Registered scheduler exports omit resolved wrapper defaults and do not authenticate every recovery attempt. The surviving Level-2 wrapper contains campaign-specific recovery overrides, including Pantry checkpoint settings. Larger-model launch records also include optimizer/offloading settings beyond the current strict recipe surface.
- This export has no per-attempt historical dependency lock or complete model/dataset byte authentication. The current qualified GPU lock describes the maintained runtime, not every historical run.

The Level-3 snapshot is **E122 / Qwen2.5-0.5B**, with one excluded terminal cell (MathIR / MaxRL / seed 45: conflicted or invalid terminal draws). Baseline admission is separate and has its own gaps. Its snapshot retains all seed decisions and source hashes; `show snapshot/level3_comparison_20260917` also exposes the current retained E122 launch ledger and whether its digest matches the snapshot's source pin. The separate **E123 / Qwen2.5-3B** ledger contains zero launched runs. Dataset admission and a planned campaign are not training results.

Consequently this release adds runnable **analysis packages**, not falsely qualified training recipes. To promote a training package, recover an authenticated runtime for every relevant attempt (including recovery changes), its dependency environment, exact model/dataset bytes, effective prompt/optimizer/evaluation configuration, seed and exclusion policy, and raw evaluation evidence. Then export a strict portable recipe and validate the installed training workflow. Matching old numerical scores is a further claim and must be checked separately.

### Snapshots and maintenance

`remax results reproduce snapshot/level3_comparison_20260917` fails with an explicit “summary snapshot only” error. The same rule applies to matched Re:Dr, tuned-control sweep/stage 2, online RLEP, and the replay-mechanism ladder. Their integrity checks preserve the retained summaries and their source metadata; they do not validate the numerical analysis behind them.

Release authors can re-run the read-only extraction from a retained research checkout with `python ops/export_experiment_packages.py --source-root /path/to/research-checkout`. It reads historical scripts as data and recomputes available runtime identities; it does not launch jobs. The versioned catalog and its hash are frozen references. A new audit should receive a new catalog version rather than overwrite the released evidence. CI checks artifact installation outside the checkout, numerical reproduction, complete cell coverage, and rejection of altered artifacts, removed seed bindings, unsafe paths, or unsupported qualification claims.
