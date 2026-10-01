Install and run the CPU walkthrough
===================================

Install the released package
----------------------------

Use Linux x86_64 and Python 3.10–3.12. A separate environment keeps the CPU API
example apart from the more tightly constrained GPU training runtime.

.. code-block:: console

   python3.10 -m venv .venv-cpu
   source .venv-cpu/bin/activate
   python -m pip install --upgrade pip
   python -m pip install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
   python -m pip install remax-rl==0.1.1
   remax environment
   remax walkthrough

The package installs the qualified ``modebench==0.4.0`` dependency. No repository
checkout, model weights, GPU, or training framework is needed for this example.

What the walkthrough does
-------------------------

It grades three saved Countdown responses, admits two correct modes to a bank,
schedules replay, creates causal response masks, applies a teacher-forced replay
update to a tiny character model, and restores the bank from saved state.

These are the important output fields:

.. code-block:: json

   {
     "statuses": ["correct", "correct", "incorrect"],
     "retained_modes": 2,
     "prompt_tokens_scored": 0,
     "parameters_updated": true,
     "bank_restored": true
   }

The same sequence is checked directly while building this guide:

.. doctest::

   >>> from remax.walkthrough import walkthrough
   >>> result = walkthrough()
   >>> result["statuses"]
   ['correct', 'correct', 'incorrect']
   >>> result["retained_modes"], result["prompt_tokens_scored"]
   (2, 0)
   >>> result["parameters_updated"], result["bank_restored"]
   (True, True)

.. important::
   This demonstrates the replay primitive. A complete Re:Dr/Re:Max training run
   also needs the fresh RL objective, generation, evaluation, and the adapter's
   coefficient/gradient-accumulation scaling. Use :doc:`training` for that workflow.

Inspect recipes and results
---------------------------

.. code-block:: console

   remax recipes
   remax recipes remax_pantry_plan_05b
   remax results compare --level level1 --scale qwen05b --reproduce

The package bundles 20 maintained Level-1 recipes: four methods across five domains.
The comparison command recomputes retained canonical-key results on CPU; it does
not retrain the original models. See :doc:`results` for all available comparisons
and their evidence limits.
