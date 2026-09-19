.PHONY: setup test demo

setup:
	uv sync --group dev

test:
	uv run pytest -q

demo:
	uv run python -m robovision demo
