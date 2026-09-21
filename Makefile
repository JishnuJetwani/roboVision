.PHONY: setup test demo report

setup:
	uv sync --group dev

test:
	uv run pytest -q

demo:
	uv run python -m robovision demo

report:
	uv run python -m robovision report
