.PHONY: check check-lock check-ruff check-ty check-test benchmark build

check: check-lock check-ruff check-ty check-test

check-lock:
	uv lock --check

check-ruff:
	uv run --frozen --extra dev ruff check src tests benchmarks scripts
	uv run --frozen --extra dev ruff format --check src tests benchmarks scripts

check-ty:
	uv run --frozen --extra dev ty check

check-test:
	uv run --frozen --extra dev pytest --cov=src --cov-report=term-missing

benchmark:
	uv run --frozen --extra dev python -m benchmarks.run_synthetic

build:
	docker --context desktop-linux build --platform linux/amd64 -t pii-engine:local .
