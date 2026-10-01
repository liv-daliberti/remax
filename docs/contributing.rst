Citation and maintenance
========================

Citation
--------

The paper is accepted at `MATH-AI 2026 <https://mathai-2026.github.io/>`_ and under
review at ICLR 2027. Cite it and record ReMax/ModeBench versions, commits, recipe,
input hashes, seeds, and exclusions:

.. literalinclude:: ../README.md
   :language: bibtex
   :start-after: ```bibtex
   :end-before: ```

`CITATION.cff <https://github.com/liv-daliberti/remax/blob/main/CITATION.cff>`_
contains machine-readable metadata. Liv G. d'Aliberti is the repository steward.
The code is Apache-2.0; ModeBench supplies its own benchmark data terms.

Contribute a change
-------------------

From a checkout in a CPU environment, install ``.[dev]``, then run ``make quality``
and ``make check``. Focused gates are ``make conformance boundary resume scaling``.
Reinstall after changing code or bundled assets so tests exercise the installed
package. Wheel and source-distribution CI runs outside the checkout.

Preserve frozen fixtures, input registries, and result identities. Changes to
rewards, canonicalization, admission, normalization, weighting, sampling,
evaluation, or exclusions need a scientific compatibility explanation and new
versioned identities. Benchmark semantics belong in ModeBench first.

Build these guides
------------------

In a Linux Python 3.10–3.12 environment, from the repository root:

.. code-block:: console

   python -m pip install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
   python -m pip install 'remax-rl==0.1.1' -r docs/requirements.txt
   python -m sphinx -W --keep-going -b doctest docs outputs/docs-doctest
   python -m sphinx -n -W --keep-going -b html docs outputs/docs
   python ops/check_docs.py outputs/docs
   python -m http.server --directory outputs/docs 8000

Open ``http://localhost:8000``. The site imports the released package without source
path overrides. Pull requests check examples and links; main-branch updates deploy
to GitHub Pages. Guide sources live in ``docs/`` as reStructuredText, leaving README
as the only Markdown file in the repository.
