.PHONY: setup test

setup:
	uv sync --group dev

test:
	uv run pytest -q
