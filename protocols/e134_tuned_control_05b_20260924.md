# E134 preregistration: Re:Dr against a tuned control

Date frozen: 2026-09-24, before submission of any E134 cell.

## The question this answers

A reviewer objects that the Dr.GRPO control is fragile: one prompt group of 16
per update, no KL, no entropy term and an unswept constant learning rate of
2e-7. The control loses `pass@8` against its own initialization in 6 of 15
comparisons (App. "Control behavior at fixed learning rates"), so part of
replay's gain may be rescue of a degenerate baseline rather than preservation
of diversity. The appendix already shows the gain survives restriction to the
controls that train (+.247 `pass@8`, +.136 PCMD over 6 comparisons, positive in
all six), and E129 already sweeps a reference-KL anchor. What no cohort does is
tune the control itself. E134 does, and then runs Re:Dr at the setting the
tuning picked.

## Stage 1: the control sweep (selection)

Nine control settings on the full Level-1 panel at Qwen2.5-0.5B. The baseline
is the completed E128 control and is not re-run.

| setting        | learning rate | prompts per update | entropy coef |
|----------------|---------------|--------------------|--------------|
| baseline (E128)| 2e-7          | 1                  | 0            |
| `lr5e8_b1`     | 5e-8          | 1                  | 0            |
| `lr1e7_b1`     | 1e-7          | 1                  | 0            |
| `lr5e7_b1`     | 5e-7          | 1                  | 0            |
| `lr1e6_b1`     | 1e-6          | 1                  | 0            |
| `lr2e7_b4`     | 2e-7          | 4                  | 0            |
| `lr5e7_b4`     | 5e-7          | 4                  | 0            |
| `lr1e6_b4`     | 1e-6          | 4                  | 0            |
| `ent1e3_b1`    | 2e-7          | 1                  | 1e-3         |

- **Learning rate** is `OAT_ZERO_LEARNING_RATE`, constant schedule as in E128.
- **Four prompts per update** holds the fresh-rollout budget fixed: 384 prompts,
  8 passes, 16 rollouts per prompt, as in E128. One update consumes four
  prompts' groups (64 rollouts), so there are 768 updates rather than 3,072.
  Five keys move together: `ROLLOUT_BATCH_SIZE` and
  `ROLLOUT_BATCH_SIZE_PER_DEVICE` to 4, `TRAIN_BATCH_SIZE` and
  `PI_BUFFER_MAXLEN_PER_DEVICE` to 64, and the save/resume cadence to 48
  updates, so checkpoints still fall every 192 prompts. The evaluation interval
  is counted in prompts and is unchanged. Low learning rates are not crossed
  with four prompts per update: a quarter of the updates at a rate already
  below the baseline cannot plausibly beat it.
- **Entropy bonus** is `OAT_ZERO_POLICY_ENTROPY_COEF`, which subtracts
  coef x (masked mean token entropy) from the loss. A cell is trusted only if
  `policy_entropy_loss` is nonzero in its training log.

Seeds 43 and 44 on all five domains: 8 new settings x 5 domains x 2 seeds = 80
cells, plus one smoke job per mechanical path (four prompts per update; entropy).

### Selection rule (registered)

For each domain, the tuned control is the setting with the highest terminal
`pass@8`, averaged over seeds 43 and 44, among the nine settings above,
measured by the same terminal evaluator as every other Level-1 cell. Ties
within .005 go to the higher terminal `mean@8`. A cell that diverges or
collapses keeps its measured value; it is not re-run or excluded. The rule
selects on correctness only, never on PCMD, so the tuning cannot favour
replay's metric.

Selection uses the same evaluation prompts the comparison reports, and the two
selection seeds reappear in stage 2. Both choices inflate the tuned control
(winner's curse), so they bias the comparison against replay. That is
intended.

## Stage 2: the comparison

For each domain whose selected setting is not the baseline:

- the tuned control on seeds 45, 46 and 47 (seeds 43 and 44 come from stage 1);
- Re:Dr at the identical setting on seeds 43 to 47: the E132 objective with the
  selected learning rate, batch and entropy keys layered on.

Where the baseline wins, the comparison is E128 against E132, which already
exist. At four prompts per update Re:Dr's code still admits exactly one replay
group per update, so replay rehearses a quarter as often per fresh prompt as
at the baseline. This is a dose reduction against replay and is kept rather
than patched.

## Registered predictions

1. **The gap persists over the tuned control.** The pooled paired Re:Dr minus
   tuned-control effect over 25 cells (5 domains x 5 seeds) is positive with a
   95% CI excluding zero, for both `pass@8` and PCMD.
2. **Tuning closes part of the `pass@8` gap, less of the PCMD gap.** The tuned
   control's `pass@8` exceeds E128 on the five-domain mean. Its PCMD stays
   below half of E132's PCMD effect over E128.

If prediction 1 fails on either metric, the paper reports the replay effect as
conditional on the untuned regime for that metric and scopes the claim to it.

## Reporting

Effects are reported as paired 95% CIs over seeds, not as counts of domains
passing a threshold. All stage-1 settings are reported, not only the winners.

## Frozen cohort

- Model: Qwen2.5-0.5B-Instruct, exactly as E128.
- Objective: `e78.fixed_objective("control")` with only the keys above moved;
  the replay traversal stays inert (`COMPUTE_ONLY=1`) in every control arm.
- Domains: Graph Coloring, Countdown, PythonFactors, MathIR, PantryPlan.
- Runtime: the snapshot E132 and E133 ran under,
  `diversity_comparators_40628bb13a366baa`.
- Placement: `cs` partition, `allcs` account, A5000, node203 or node204, as
  E128, E132 and E133.

## No source change

E134 declares no source change. Learning rate, prompts per update and the
entropy coefficient are existing training parameters.

## Addendum, 2026-09-26: stage-1 outcome, two defects, and the stage-2 cells

Written after stage 1 landed and before any stage-2 or rerun cell was
submitted. Nothing above is altered; this section records what the sweep
found and what the two submissions of this date consist of.

### Stage-1 outcome

74 of 80 cells completed at the full horizon (3,072 updates at one prompt per
update; 768 at four). Measured with the frozen comparator extractor
(`ops/build_paper_tuned_control_sweep.py`), the registered rule selects:

| domain     | winner       | winner `pass@8` | baseline `pass@8` |
|------------|--------------|-----------------|-------------------|
| Graph      | `lr5e8_b1`   | .429            | .325              |
| Countdown  | `lr5e7_b4`   | .622            | .493              |
| Python     | baseline     | .586            | .586              |
| MathIR     | baseline     | .460            | .460              |
| PantryPlan | `lr5e8_b1`   | .600            | .531              |

Countdown is a tie within .005 between `lr1e7_b1` and `lr5e7_b4` (.622 each)
and goes to `lr5e7_b4` on `mean@8` (.605 against .590), as registered. Every
Python setting sits at the .172 floor; the baseline's .586 is the published
E128 seed-44 cell at 1.0. High learning rates collapse MathIR (.06 at 1e-6).

### Defect 1: the entropy arm ran without an entropy term

The `grpo_compute_matched` branch of the frozen `ops/run_experiment.sh`
assigns `OAT_ZERO_POLICY_ENTROPY_COEF=0.0` unconditionally after the
environment is read, so the launcher's 1e-3 never reached the learner. The
scheduler record carries 1e-3; the learner's argument dump says 0.0 and no
`policy_entropy_loss` row was logged, which is exactly the trust condition
the protocol states. The ten `ent1e3_b1` cells are therefore a second
measurement of the baseline setting (2e-7, one prompt, no entropy) and are
reported under the arm name `baseline_replicate`. They are not selection
candidates. They read .316 / .599 / .172 / .421 / .531 on Graph / Countdown /
Python / MathIR / PantryPlan against the E128 seeds-43--44 means of .325 /
.493 / .586 / .460 / .531, so part of the Countdown tuning gain over E128 is
the E128 seed-43 cell (.369) sitting low.

The entropy setting is resubmitted as arm `ent1e3_b1r` on the selection
seeds, under a runtime whose control branch passes the coefficient through
(`ops/exp_scaling/launch_e134_entropy_rerun_05b.py`, ledger
`var/artifacts/e134_tuned_control_05b_stage1_entropy_rerun_jobs.json`). A
rerun cell is trusted only if its log carries a nonzero
`policy_entropy_loss`; the builder refuses otherwise. **Selection rule for
the rerun:** the registered rule is re-applied over the nine settings once
the rerun lands. If `ent1e3_b1r` wins a domain, that domain's stage 2 is run
at the entropy setting and the stage-2 cells submitted today for that domain
are reported as a non-selected tuned arm, not discarded. If it does not, the
stage-2 cells submitted today are the comparison.

### Defect 2: PantryPlan cannot take four prompts per update

PantryPlan's canonical learner-side sampler requires `rollout_batch_size=1`
(`validate_zero_math_args`), so the six `*_b4` PantryPlan cells failed at
argument validation on every attempt and are recorded as unrunnable. The
setting is excluded from PantryPlan's selection. This is a property of the
domain's policy surface, not of the sweep, and is not patched.

### Stage 2 as submitted

Three domains have a tuned winner. Per the stage-2 section above:

- tuned control, seeds 45--47: Graph and PantryPlan at `lr5e8_b1`, Countdown
  at `lr5e7_b4` (9 cells);
- Re:Dr at the identical setting, seeds 43--47: the E132 objective with the
  selected learning-rate and batch keys layered on (15 cells).

Python and MathIR are compared as E128 against E132. Launcher
`ops/exp_scaling/launch_e134_stage2_05b.py`, which reads the stage-1 payload
and refuses to submit if its winners differ from the ones listed here; ledger
`var/artifacts/e134_tuned_control_05b_stage2_jobs.json`. One smoke per
(variant, batch) path on Graph s43. Countdown's Re:Dr runs at four prompts
per update with one replay group per update, the registered dose reduction.

### Runtime

Both submissions run under a snapshot derived from the same base with the
E126--E133 patch set plus `src/oat_drgrpo/rlep.py` (E135's online pool) and
`src/oat_drgrpo/canonical_actions.py` (the token-to-code inverse E135's
PantryPlan repair uses); `ops/run_experiment.sh` in that set now passes the
entropy coefficient through in the control branch. None of the three hunks is
reachable for an arm that requests no entropy coefficient and is not an
online-pool arm, so every stage-2 cell is runtime-identical to E128/E132. The
"No source change" section above therefore still holds for stage 2; the
entropy rerun's only effective change is that the requested coefficient is
applied.

## Addendum, 2026-09-27: the entropy rerun re-selects two domains

The rerun `ent1e3_b1r` landed with a nonzero `policy_entropy_loss` in every
cell. Re-applying the registered rule over all nine settings:

| domain     | winner       | winner `pass@8` | baseline `pass@8` |
|------------|--------------|-----------------|-------------------|
| Graph      | `lr5e8_b1`   | .429            | .325              |
| Countdown  | `ent1e3_b1r` | .635            | .493              |
| Python     | baseline     | .586            | .586              |
| MathIR     | `ent1e3_b1r` | .490            | .460              |
| PantryPlan | `lr5e8_b1`   | .600            | .531              |

Per the 2026-09-26 rule, Countdown and MathIR are re-run at the entropy
setting and the Countdown `lr5e7_b4` stage-2 cells are reported as a
non-selected tuned arm.

**Source decision (user, 2026-09-27).** Re:Dr at a setting with an entropy
term was refused by `validate_zero_math_args` ("online canonical bank and
token entropy are separate treatments"), and the Re:Dr branch of
`ops/run_experiment.sh` zeroed the coefficient as the control branch had. Both
now admit it: the entropy term acts on the fresh-rollout loss only, and the
verified-likelihood replay term is a separate loss on stored rows, so the two
compose additively. A live bank coefficient or the split mass/balance
objective is still refused.

**Stage 2b.** `ops/exp_scaling/launch_e134_stage2b_05b.py`, ledger
`var/artifacts/e134_tuned_control_05b_stage2b_jobs.json`: tuned control at
`ent1e3_b1r` on seeds 45--47 and Re:Dr at the same setting on seeds 43--47,
Countdown and MathIR (16 cells), behind one smoke per arm on Countdown s43.
Prediction 1 is scored over the 25 cells formed by Graph and PantryPlan
(stage 2), Countdown and MathIR (stage 2b) and Python (E128 against E132).
