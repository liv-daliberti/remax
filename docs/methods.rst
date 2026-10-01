Methods and replay semantics
============================

The four maintained methods
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Recipe method
     - Fresh objective
     - Replay
   * - ``drgrpo``
     - Dr.GRPO
     - Executes replay with exactly zero applied replay gradient.
   * - ``redr``
     - Dr.GRPO
     - Uniform verified-mode replay.
   * - ``maxrl``
     - Binary MaxRL
     - Executes replay with exactly zero applied replay gradient.
   * - ``remax``
     - Binary MaxRL
     - Uniform verified-mode replay.

The controls preserve bank bookkeeping and replay computation. This makes the
replay-gradient comparison compute-matched; it does not imply equivalence to an
arbitrary implementation that skips replay entirely.

From discovery to update
------------------------

1. Generate fresh responses from a training prompt.
2. Verify them with ModeBench and obtain canonical identities. Wrong answers and
   malformed responses remain model outcomes; evaluator failures stop evaluation.
3. Admit eligible, verified discoveries to a prompt-local bank. Evaluation answers
   never initialize the bank.
4. Schedule retained banks and modes deterministically. Singleton banks participate.
5. Score stored responses under the current policy, excluding prompt and padding
   tokens. Normalize response log scores by their scored token lengths.
6. Average the negative scores over modes, then banks. The training adapter applies
   the registered replay coefficient and accumulation scaling beside the fresh loss.

An exemplar is a stored response, while a mode is a canonical outcome identity.
The bounded exemplar bank and cumulative discovery counts answer different
questions; capacity-limited storage is not the total number of modes ever found.

What belongs in each package
----------------------------

`ModeBench <https://liv-daliberti.github.io/modeBench/>`_ owns task data,
prompt construction, verifier behavior, canonical identities, and benchmark metrics.
ReMax owns admission, replay banks, scheduling, scoring, objectives, checkpoints,
and the training integration.

The maintained replay primitives live in ``remax.core``. The OAT adapter supplies
training plumbing; historical comparator integrations are isolated separately.
The :doc:`api` page shows how to use the core without importing those integrations.

Scientific compatibility
------------------------

Changing reward handling, canonical identity, admission, normalization, replay
weighting, sampling, evaluation, or exclusions can change the implemented method.
Such changes need explicit compatibility reasoning and new versioned identities;
a green software test suite alone does not establish scientific equivalence.
