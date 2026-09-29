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

Record the repository URL and exact commit in software citations:

```text
Re:Max / Re:Dr. https://github.com/liv-daliberti/remax
Software version 0.1.0; Git commit: <the commit used for the experiment>.
```

Cite the associated scientific work when its final bibliographic metadata are available, and cite ModeBench separately when reporting benchmark results. This repository does not assign an unverified DOI or paper author list.
