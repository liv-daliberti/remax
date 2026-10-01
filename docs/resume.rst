Resume and interpret a run
==========================

Resume from a committed checkpoint
----------------------------------

Continue the :doc:`training` example from its step-2 checkpoint in a new process
and a fresh output directory. This also works after the original run finishes:

.. code-block:: bash

   checkpoint_path="$(python - <<'PYTHON'
   from pathlib import Path
   matches = list(Path("outputs/train").glob("*/checkpoints/step_00002"))
   assert len(matches) == 1, matches
   print(matches[0])
   PYTHON
   )"
   remax-run outputs/walkthrough.json --data-root outputs/data/pantry_plan \
     --model "$model_path" --output outputs/resumed \
     --resume "$checkpoint_path" --execute

The checkpoint restores model, optimizer, scheduler, random-number state, data
position, bank contents, replay cursor, and evaluation cadence. Keep the original
recipe, total horizon, seed, installed source, dependencies, hardware and input bytes.

.. important::
   Resume is explicit and identity-bound. There is no automatic checkpoint discovery.
   Never restore an incomplete ``.pending-*`` directory, a ``latest`` symlink,
   or an untrusted checkpoint. Even source formatting changes can alter the resume
   identity; preserve the original environment for an existing run.

Read evaluation output
----------------------

Greedy and sampled evaluations run during training. This prints the relevant
metrics after checking that no evaluation-failure journal contains diagnostics:

.. code-block:: python

   import json
   from pathlib import Path

   root = Path("outputs/train")
   assert not any(p.stat().st_size for p in root.rglob("evaluation_failures.jsonl")), "Evaluation failed"
   logs = list(root.rglob("train_metrics.jsonl"))
   assert len(logs) == 1, logs
   fields = [
       "misc/policy_sgd_step",
       "eval/multi_answer/accuracy",
       "eval/multi_answer/sampled_any_correct_at_2",
       "eval/multi_answer/sampled_distinct_correct_at_2",
   ]
   with logs[0].open() as stream:
       for line in stream:
           row = json.loads(line)
           if "eval/multi_answer/accuracy" in row:
               print({key: row.get(key) for key in fields})

The fields mean optimizer step, greedy correctness, pass@2, and mean distinct
verified modes among two samples. Missing evaluations are not zero scores.
These K=2 walkthrough scores are not the pass@8 historical comparison.

Keep ``launch_request.json``, ``effective_config.json``, committed checkpoints,
and ``eval_mode_coverage_draws.jsonl`` alongside the summaries. When joining resumed
attempts, take the original prefix through the checkpoint and the resumed suffix;
do not count duplicate evaluations.

Verify recovery equivalence
---------------------------

The full-state comparison runs uninterrupted and resumed trajectories, then audits
them automatically:

.. code-block:: console

   python ops/resume_gpu.py run --workdir outputs/resume-proof \
     --data-root outputs/data/pantry_plan --model "$model_path"

The declared tolerances are ``atol=1e-6, rtol=1e-6`` for model tensors and
``atol=1e-8, rtol=1e-5`` for optimizer tensors, with exact bank, RNG, scheduling,
and evaluation decisions. Qualification covers bounded Pantry runs for all four
methods and Countdown sampling continuity; it is not a distributed-GPU or
full-budget historical-score qualification.
