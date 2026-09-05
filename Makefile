UV ?= uv
ARGS ?=

.PHONY: build test benchmark plot

build:
	$(UV) sync --reinstall-package mhc-proj

test: build
	$(UV) run --no-sync pytest

benchmark:
	$(UV) sync --group benchmark --reinstall-package mhc-proj
	$(UV) run --no-sync python benchmark/run.py $(ARGS)

plot:
	$(UV) run --group benchmark python benchmark/plot.py $(ARGS)
