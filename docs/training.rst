Run a small GPU experiment
==========================

Qualified environment
---------------------

The documented training environment is **Linux x86_64, Python 3.10, CUDA 12.4,
one 48 GB RTX A6000, eight CPU cores, and 64 GB RAM**. Allow about 60 GB of disk
space for dependencies, weights, and checkpoints. Run training commands inside
a GPU allocation. Other hardware and distributed GPU recovery remain unqualified.

The workflow below uses six updates and full held-out evaluation. It demonstrates
execution, not paper-level performance.

Install the training environment
--------------------------------

Use the versioned source checkout for setup utilities and examples:

.. code-block:: console

   git clone --branch v0.1.1 --depth 1 https://github.com/liv-daliberti/remax.git remax-training
   cd remax-training
   PYTHON=python3.10 bash ops/setup_gpu_environment.sh .venv-train
   source .venv-train/bin/activate
   remax environment --training

The setup script installs the qualified GPU dependency lock and checks the
installation. Core Python 3.11/3.12 support does not imply GPU training support
on those Python versions. For training, use this locked setup rather than assuming
an arbitrary resolver-selected ``[train]`` environment is qualified.

Prepare authenticated data
--------------------------

.. code-block:: console

   git clone --branch v0.4.0 --depth 1 https://github.com/liv-daliberti/modeBench.git ../modebench-data
   python ../modebench-data/ops/verify_data.py
   python ../modebench-data/ops/materialize_training_data.py \
     --config level1_pantry_plan --output outputs/data/pantry_plan

Keep the complete 384-row training and 128-row evaluation splits. The short recipe
selects fewer training rows itself; physically truncating the data fails authentication.

Download the pinned model
-------------------------

The recipe names the immutable Qwen2.5-0.5B-Instruct revision:

.. code-block:: bash

   python - <<'PYTHON'
   import json
   from pathlib import Path
   from huggingface_hub import snapshot_download
   recipe = json.loads(Path("configs/remax_pantry_plan_05b.json").read_text())
   model = snapshot_download(repo_id=recipe["model_id"], revision=recipe["model_revision"])
   Path("outputs/model-path.txt").write_text(model + "\n")
   PYTHON
   model_path="$(cat outputs/model-path.txt)"

The launcher checks model file identities, dataset contents, the ModeBench package,
recipe compatibility, and inherited environment settings before starting workers.
A similarly named model directory is not sufficient.

Preflight and execute
---------------------

.. code-block:: console

   python examples/prepare_training_walkthrough.py outputs/walkthrough.json
   remax-run outputs/walkthrough.json --data-root outputs/data/pantry_plan \
     --model "$model_path" --output outputs/preflight --validate-only
   remax-run outputs/walkthrough.json --data-root outputs/data/pantry_plan \
     --model "$model_path" --output outputs/train --execute

The example uses three training prompts for two passes, checkpoints every two
updates, and evaluates the full held-out split with K=2 and one sampled draw.
Preflight and execution each need a fresh output directory.

``--render-only`` can preview commands on CPU, but it explicitly does not qualify
the inputs or runtime. Make configuration changes in the strict recipe JSON;
unknown fields and conflicting inherited settings fail before training.

Run the four methods
--------------------

After preparing the same data and model, this runs **full-budget** recipes
sequentially, substantially longer than the six-update walkthrough:

.. code-block:: bash

   for method in drgrpo redr maxrl remax; do
     remax-run "${method}_pantry_plan_05b" --data-root outputs/data/pantry_plan \
       --model "$model_path" --seed 43 --output "outputs/comparison/${method}-s43" --execute
   done

Repeat with seeds 44–47 for five runs per method. Other domains use their matching
materialized data directory and recipe name from ``remax recipes``. These are new
runs of the maintained methods; historical score equivalence is not established.

Next, :doc:`resume` explains recovery and the evaluation files produced by the run.
