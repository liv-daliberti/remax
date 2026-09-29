# E78: verified replay only across all five ModeBench domains

**Frozen before submission on 2026-08-04.**

## Question

Does recurrent replay of the policy's own validator-positive exemplars retain
verified execution modes better than compute-matched Dr.GRPO when replay is the
only intervention?

E78 replaces the superseded semantic-MaxEnt, verified-mass, and explicit
known-mode-balance stack with one objective: uniform teacher-forced likelihood
over one stored exemplar per observed verified mode. There is no semantic
shaping term and no separately normalized balance loss.

## Design

- Model: the pinned Qwen2.5-0.5B-Instruct revision used by the existing
  0.5B main-body cohort.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- Data: each domain's released 384-prompt training pool and fixed 128-prompt
  evaluation split.
- Seeds: 43, 44, 45, 46, and 47, paired within domain and GPU model.
- Training: exactly eight passes, hence 3,072 optimizer updates per run;
  group size 16, one PPO epoch, learning rate 2e-7, rollout temperature 1,
  top-p 1, and beta_KL = 0.
- Evaluation and resumable model checkpoints: every 192 updates, corresponding
  to passes 0, 0.5, 1.0, ..., 8.0. Pass 8 is the terminal endpoint.

The complete cohort has 5 domains x 2 arms x 5 seeds = 50 runs.

## Arms

### Compute-matched Dr.GRPO (`control`)

The control maintains the same passive verified bank, global round-robin
scheduler, replay batch materialization, teacher-forced score traversal, and
backward-compute envelope as the treatment, but the applied replay derivative
is exactly zero.

### Verified replay (`replay`)

For each scheduled prompt-local bank B_x, retain at most one policy-generated,
validator-positive exemplar per observed canonical mode and minimize

    L_replay(x) = mean_{b in B_x} -s_theta(b | x),

where `s_theta` is the implementation's length-normalized teacher-forced log
score. One bank is scheduled per optimizer update in deterministic global
round-robin order, including singleton banks. Capacity is 16 observed modes
per prompt.

The fixed replay loss weight is 0.10. This is a reproducibility setting, not an
adaptive controller or a theoretically privileged value; the categorical
result requires only positive recurrent replay dose. E78 does not tune this
weight or select it using evaluation behavior.

## Exact exclusions

Both arms hard-disable semantic Shannon shaping, semantic separate advantages,
quality-gated or signed semantic pressure, conditioned-bank balance KL,
canonical-bank entropy shaping, token entropy bonuses, adaptive coefficients,
counterfactual proposals, singleton escape, novelty reward, and reference KL.
No exhaustive support, evaluation outcome, desired mode count, or desired
entropy is available to training, scheduling, stopping, or checkpoint choice.

## Outcomes and estimands

At every registered half-pass checkpoint report, separately by domain:

- greedy pass@1;
- sampled mean correctness@8;
- sampled pass@8;
- mean distinct correct modes@8; and
- excess multiplicity, `distinct@8 - pass@8`.

The primary comparison is the paired seed difference `replay - control` at
pass 8 for `distinct@8` and `pass@8`. The secondary trajectory summary is
trapezoidal AUC over the complete pass-0 through pass-8 half-pass grid. Show all
five paired seed differences and their mean/range; do not pool domains into one
effect and do not select a best checkpoint.

Mechanism telemetry reports banked-mode survival, verified bank size, replay
actuator opportunities, applied replay gradient, and replayed verified score.
The control must have exact-zero applied replay gradient whenever a replay
opportunity exists; the treatment must have a finite nonzero applied gradient
on at least one eligible update.

## Integrity and failure policy

All 50 cells are submitted held from one hash-bound runtime snapshot and are
released only after their scheduler environments pass an exact audit. A
malformed environment, source mismatch, duplicate run directory, non-finite
loss, proposal leakage, traceback, or missing pass-8 endpoint fails closed.
Infrastructure interruption may resume only from the same run's hash-bound
checkpoint and exact bank, optimizer, data-cursor, and request-stream state.
No failed scientific run is silently replaced or excluded.
