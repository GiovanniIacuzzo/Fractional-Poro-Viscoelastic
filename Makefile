# reteco-v1/Makefile
SHELL := /bin/bash
PYTHON := ./.conda_env/bin/python
PYTEST := ./.conda_env/bin/pytest

.PHONY: setup build test verify-forward identifiability twin-robustness sota-benchmark clean

setup:
	chmod +x prepare.sh
	./prepare.sh

build:
	source ./activate_reteco.sh && cmake --build build --parallel

test: build
	source ./activate_reteco.sh && $(PYTEST) tests/

verify-forward: build
	source ./activate_reteco.sh && $(PYTHON) experiments/01_run_forward_verification.py

identifiability: build
	source ./activate_reteco.sh && $(PYTHON) experiments/02_run_identifiability_study.py

twin-robustness: build
	source ./activate_reteco.sh && $(PYTHON) experiments/03_run_twin_robustness.py

sota-benchmark: build
	source ./activate_reteco.sh && $(PYTHON) experiments/04_run_sota_benchmark.py

clean:
	rm -rf build/ python/reteco/_core*.so artifacts/*.tmp