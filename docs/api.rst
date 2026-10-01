Replay API
==========

The maintained interfaces live in ``remax.core`` and ``remax.benchmark``.
References below are generated from the installed 0.1.1 release. Training framework
imports remain opt-in.

Verify before bank admission
----------------------------

``grade_task`` accepts a public ModeBench ``Task`` and returns structured results
for ordinary candidate outcomes. Evaluator failures raise ``EvaluationFailure``;
never catch that exception and substitute a zero reward or rejected discovery.

.. autofunction:: remax.benchmark.grade_task

.. autoclass:: remax.benchmark.EvaluationFailure

Maintain and schedule a bank
----------------------------

.. autoclass:: remax.core.OnlineCanonicalBank

.. automethod:: remax.core.OnlineCanonicalBank.score_and_update

.. automethod:: remax.core.OnlineCanonicalBank.scheduled_global_replay_groups

.. automethod:: remax.core.OnlineCanonicalBank.state_dict

.. automethod:: remax.core.OnlineCanonicalBank.load_state_dict

Bank state is one part of a full training checkpoint. Restoring it alone does not
restore the policy, optimizer, RNG, sampler, or evaluation cadence; see :doc:`resume`.

Materialize and score replay
----------------------------

.. autofunction:: remax.core.materialize_canonical_replay_batch

.. autofunction:: remax.core.canonical_replay_uniform_verified_likelihood_loss

.. autofunction:: remax.core.binary_maxrl_advantages

The materialized batch supplies input IDs, causal response masks, and group sizes.
Teacher-forced scoring must exclude prompt and padding tokens and preserve response
length normalization. The OAT adapter owns the combination with fresh RL loss and
the coefficient/gradient-accumulation scaling.

Complete executable example
---------------------------

This is the source of ``remax walkthrough``. It uses a tiny character-level model
to make verification, admission, masking, scoring, updating, and restoring visible
without downloading weights. Its output is tested in :doc:`quickstart`.

.. literalinclude:: ../src/remax/walkthrough.py
   :language: python
   :start-at: def walkthrough()

Inspect result packages
-----------------------

For programmatic comparison, ``remax.results.compare(level="level1",
scale="qwen05b", domain=None, recompute=False)`` returns terminal scores and exact
seed populations. ``recompute=True`` first verifies the saved-key analysis.
``inspect(identifier)`` returns a package's bindings, while ``reproduce(identifier)``
refuses summary-only snapshots. See :doc:`results` for IDs and scope.
