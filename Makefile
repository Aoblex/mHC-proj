UV ?= uv
ARGS ?=
SYNC = $(UV) sync --frozen --no-install-project
INSTALL_PROJECT = $(UV) pip install --no-build-isolation --no-deps --reinstall-package mhc-proj --editable .

.PHONY: build test benchmark plot

build:
	$(SYNC)
	$(INSTALL_PROJECT)

test: build
	$(UV) run --no-sync pytest

benchmark:
	$(SYNC) --group benchmark
	$(INSTALL_PROJECT)
	$(UV) run --no-sync python benchmark/run.py $(ARGS)

plot:
	$(SYNC) --group benchmark
	$(UV) run --no-sync python benchmark/plot.py $(ARGS)
