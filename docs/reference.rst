CLI reference and troubleshooting
=================================

These command definitions are captured from the installed release at site-build time.

Core commands
-------------

.. command-help:: remax

Result comparisons
------------------

.. command-help:: remax results compare

Training launcher
-----------------

.. command-help:: remax-run

Common problems
---------------

Import or dependency mismatch
   Check ``remax environment`` in the active environment. Use a separate CPU
   environment for the walkthrough and the locked setup in :doc:`training` for GPU work.

Wrong model or dataset identity
   Restore the exact recipe revision and complete materialized splits. Do not edit
   expected hashes to make a different input pass preflight.

Unknown recipe field or inherited setting
   Put intended changes in the strict recipe JSON. Remove conflicting inherited
   experiment environment variables; preflight records the complete effective configuration.

Cache or disk quota errors
   Set ``XDG_CACHE_HOME`` to writable scratch storage with enough room for model
   weights and dependencies. Use fresh output directories for each attempt.

Verifier timeout or broken worker
   Inspect the structured diagnostic and fix the environment or reference. These
   failures raise ``EvaluationFailure`` and must not become zero rewards, rejected
   discoveries, or partial evaluation scores.

Resume identity mismatch
   Restore the original source, runtime, inputs, recipe, seed, and hardware, or
   begin a new run. A resume is not an opportunity to change the training horizon.

Historical comparison cannot be reproduced
   ``results list`` distinguishes saved-key reproducible arms from summary snapshots.
   The latter do not become runnable experiments merely because their summaries
   are retained; see :doc:`results`.

For help, open a `GitHub issue <https://github.com/liv-daliberti/remax/issues>`_
with the version, recipe, command, environment, input identities, and a minimal
reproduction. Preserve the launch and failure records with your run.
