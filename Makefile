.PHONY: check check-lock check-ruff check-ty check-test benchmark build

check: check-lock check-ruff check-ty check-test

check-lock:
	uv lock --check

check-ruff:
	uv run --extra cpu --extra dev ruff check src tests benchmarks scripts
	uv run --extra cpu --extra dev ruff format --check src tests benchmarks scripts

check-ty:
	uv run --extra cpu --extra dev ty check

check-test:
	uv run --extra cpu --extra dev pytest --cov=src --cov-report=term-missing

benchmark:
	uv run --extra cpu --extra dev python -m benchmarks.run_synthetic

build:
	docker --context desktop-linux build --build-arg ACCELERATOR=cpu -t pii-engine:local-cpu .
