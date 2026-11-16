.PHONY: install-gpu install-cpu lint test test-all smoke
install-gpu:  ## WSL2/Linux with an NVIDIA GPU
	uv sync --extra cu128 --extra dev --extra deploy
install-cpu:  ## CPU-only (CI, laptops)
	uv sync --extra cpu --extra dev --extra deploy
lint:
	uv run ruff check . && uv run ruff format --check .
test:         ## fast tests (no env build)
	uv run pytest -q -m "not slow"
test-all:     ## + CPU env construction and terrain measurement (~1 min)
	uv run pytest -q
smoke:        ## build both envs on CPU and step them
	uv run pytest -q tests/test_env_cpu.py
