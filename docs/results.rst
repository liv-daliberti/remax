Performance and other comparisons
=================================

ModeBench performance
---------------------

**Qwen2.5-0.5B-Instruct, Level 1**, final recorded step 3072, seeds 43–47,
128 held-out prompts per domain, and four groups of eight responses.
Each entry is **pass@8 / PCMD [eligible PCMD seeds]**. This table is rendered from
the evidence-checked README, not maintained as a separate copy.

.. readme-table:: remax-performance

All methods have five terminal seeds in this preview. PCMD uses only seeds with
at least 30 eligible prompts. A dash means insufficient support, not zero diversity.
Seed populations can differ, so these endpoints are descriptive comparisons, not
paired treatment-effect estimates. See the
`ModeBench metric guide <https://liv-daliberti.github.io/modeBench/metrics.html>`_
for the conditional diversity definition.

Recompute comparisons from saved keys
-------------------------------------

These commands work in an installed CPU package:

.. code-block:: console

   remax results compare --level level1 --scale qwen05b --reproduce
   remax results compare --level level1 --scale falcon1b --reproduce
   remax results compare --level level1 --scale qwen3b --reproduce
   remax results compare --level level2 --scale qwen05b --reproduce --json

``--reproduce`` checks the entire retained canonical-key archive before presenting
the comparison. ``--json`` includes exact metric values and the seed IDs used for
each metric. ``--domain countdown`` narrows the display. Missing observations and
excluded seeds are not imputed.

.. code-block:: console

   remax results list
   remax results show level2/qwen05b/countdown/replay_maxrl
   remax results reproduce level2/qwen05b/countdown/replay_maxrl
   remax results verify

There are 95 saved-key analysis arms and six labeled summary snapshots. Numerical
reproduction checks 473 included records; two additional records are explicitly
excluded. ``verify`` checks integrity and bindings only. Neither command retrains
models or independently regrades their original response strings.

Which comparisons can be run?
-----------------------------

* **Dr.GRPO, Re:Dr, MaxRL, Re:Max:** :doc:`training` gives maintained Level-1
  Qwen-0.5B launch recipes across five domains. These are not claimed to reproduce
  full-budget historical scores exactly.
* **GRPO, Falcon-1B, Qwen-3B, and Level-2 Qwen-0.5B:** the saved-key commands above
  reproduce retained numerical analyses and expose their historical bindings.
* **The comparisons below:** retained summaries can be inspected, but complete
  verified training reproduction is unavailable. ``reproduce`` refuses these IDs.

.. code-block:: console

   remax results show snapshot/matched_redr_20260924
   remax results show snapshot/online_rlep_20260927
   remax results show snapshot/tuned_control_stage2_20260927
   remax results show snapshot/tuned_control_sweep_20260927
   remax results show snapshot/replay_mechanism_ladder_20260924
   remax results show snapshot/level3_comparison_20260917

These cover matched Re:Dr, online RLEP, tuned controls, replay-mechanism ablations,
and Level 3 respectively. Their records preserve the gaps rather than presenting
summary snapshots as runnable experiment packages.

From a checkout, ``python ops/summarize_performance.py --check`` verifies the preview
table. The `ModeBench results guide <https://liv-daliberti.github.io/modeBench/results.html>`_
also covers untrained-model measurements and level calibration under their own
explicitly recorded protocols.
