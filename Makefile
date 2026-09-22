.PHONY: setup test report demo format

setup:
	uv sync --group dev

test:
	uv run pytest -q

report:
	uv run python -m robovision report

demo:
	uv run python -m robovision demo

format:
	uv run ruff format robovision scripts tests
