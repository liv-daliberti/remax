PYTHON ?= python
.PHONY: check test reproduce
check: test reproduce
	$(PYTHON) ops/summarize_performance.py --check
	$(PYTHON) ops/check_training_reference.py
	bash -n ops/train.sh ops/repo_env.sh ops/setup_gpu_environment.sh
test:
	$(PYTHON) -m pytest -q
reproduce:
	$(PYTHON) ops/reproduce_training.py

.PHONY: conformance
conformance:
	$(PYTHON) ops/check_training_reference.py
	$(PYTHON) -m pytest -q tests/test_training_conformance.py tests/test_training_reference_guard.py

.PHONY: boundary
boundary:
	$(PYTHON) ops/check_training_reference.py
	$(PYTHON) -m pytest -q tests/test_benchmark_boundary.py tests/test_verifier_workers.py tests/test_evaluation_failure_boundary.py

.PHONY: resume
resume:
	$(PYTHON) -m pytest -q tests/test_resume_equivalence.py tests/test_resume_audit.py

.PHONY: scaling
scaling:
	$(PYTHON) -m pytest -q tests/test_execution_resources.py

.PHONY: quality format
quality:
	$(PYTHON) -m ruff check src/remax/core
	$(PYTHON) -m ruff format --check src/remax/core
	$(PYTHON) -m mypy
format:
	$(PYTHON) -m ruff check --fix src/remax/core
	$(PYTHON) -m ruff format src/remax/core
