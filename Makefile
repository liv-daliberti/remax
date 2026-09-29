PYTHON ?= python
.PHONY: check test reproduce
check: test reproduce
	$(PYTHON) ops/check_training_reference.py
	bash -n ops/train.sh ops/repo_env.sh
test:
	$(PYTHON) -m pytest -q
reproduce:
	$(PYTHON) ops/reproduce_training.py

.PHONY: conformance
conformance:
	$(PYTHON) ops/check_training_reference.py
	$(PYTHON) -m pytest -q tests/test_training_conformance.py tests/test_training_reference_guard.py
