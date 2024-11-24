PHONY: install_uv setup activate hello

install_uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

setup:
	uv sync

shell:
	uv run ipython

activate:
	${HOME}/.cargo/env .

