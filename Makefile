.PHONY: all install test train eval run clean help

VENV = backend/.venv
PYTHON = $(VENV)/bin/python
PYTEST = $(VENV)/bin/pytest

all: test eval

install:
	@echo "Installing python requirements in $(VENV)..."
	$(PYTHON) -m pip install -r backend/requirements.txt

test:
	@echo "Running anti-leakage invariance and feature verification tests..."
	$(PYTEST) backend/tests -v --durations=5

train:
	@echo "Training walk-forward ensemble models..."
	$(PYTHON) -m backend.cli train --ticker RELIANCE.NS --splits 3 --horizon 5

eval:
	@echo "Running comprehensive evaluation and generating EVALUATION_REPORT.md..."
	$(PYTHON) -m backend.cli eval --ticker RELIANCE.NS --splits 3 --horizon 5 --output EVALUATION_REPORT.md

run:
	@echo "Starting AARC Quant Terminal services (FastAPI on 8000, Next.js on 3000)..."
	./start.sh

clean:
	@echo "Cleaning caches and artifacts..."
	rm -rf .cache backend/data_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "AARC Quant Terminal Make Commands:"
	@echo "  make test   - Run anti-leakage invariance and feature unit tests"
	@echo "  make train  - Train purged walk-forward ensemble and save artifact bundle"
	@echo "  make eval   - Run comprehensive walk-forward evaluation and output report"
	@echo "  make run    - Start FastAPI backend and Next.js frontend"
	@echo "  make clean  - Clean cache directories and bytecode files"
