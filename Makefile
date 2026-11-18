.PHONY: install run test lint
install:
	pip install -r requirements-dev.txt && pip install -e .
run:
	python -m fraud.experiment
test:
	pytest -q
lint:
	ruff check src tests
