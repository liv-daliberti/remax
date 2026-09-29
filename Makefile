PYTHON ?= python
.PHONY: check test reproduce
check: test reproduce
	bash -n ops/train.sh ops/repo_env.sh
test:
	$(PYTHON) -m pytest -q
reproduce:
	$(PYTHON) ops/reproduce_training.py
