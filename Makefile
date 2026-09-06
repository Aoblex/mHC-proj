UV ?= uv
SYNC = $(UV) sync --frozen --no-install-project
INSTALL_PROJECT = $(UV) pip install --no-build-isolation --no-deps --reinstall-package mhc-proj --editable .

.PHONY: build test

build:
	$(SYNC)
	$(INSTALL_PROJECT)

test: build
	$(UV) run --no-sync pytest

