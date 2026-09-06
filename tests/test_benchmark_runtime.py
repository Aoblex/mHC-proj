import argparse
from collections.abc import Callable
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    # Test the runner without importing optional benchmark backends.
    monkeypatch.setitem(
        sys.modules, "backends", SimpleNamespace(BACKENDS={"mHC-proj": None}, Forward=Callable)
    )
    path = Path(__file__).resolve().parents[1] / "benchmark" / "run.py"
    spec = importlib.util.spec_from_file_location("benchmark_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backward", [False, True])
def test_measure_reuses_original_input(runtime, monkeypatch, backward):
    event = Mock()
    event.elapsed_time.return_value = 3.0
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    logits = torch.ones(2, 4, 4, requires_grad=backward)
    calls = []

    def forward(x):
        calls.append(x.detach().clone())
        return 2 * x

    elapsed = runtime.measure(
        forward, logits, torch.ones_like(logits),
        backward=backward, warmup=2, iterations=3,
    )
    assert elapsed == 1000.0
    assert len(calls) == 5
    for value in calls:
        torch.testing.assert_close(value, logits)


def test_run_uses_paper_inputs_and_per_round_ratios(runtime, monkeypatch):
    randn = torch.randn
    monkeypatch.setattr(
        torch, "randn", lambda *shape, device: randn(*shape, device="cpu")
    )
    torch.manual_seed(123)
    expected = 10 * randn(128, 4, 4)
    forwards = {name: (lambda x: x) for name in ("other", "mHC-proj")}
    runtime.BACKENDS = {
        name: SimpleNamespace(build=lambda n, f=f: f)
        for name, f in forwards.items()
    }
    timings = {
        (name, backward): iter(values)
        for backward in (False, True)
        for name, values in (("other", [2, 30, 40]), ("mHC-proj", [1, 10, 100]))
    }
    calls = []

    def measure(forward, logits, gradient, *, backward, **kwargs):
        torch.testing.assert_close(logits, expected)
        assert logits.requires_grad == backward
        name = next(name for name, f in forwards.items() if f is forward)
        calls.append((name, backward))
        return next(timings[name, backward])

    monkeypatch.setattr(runtime, "measure", measure)
    rows = runtime.run(argparse.Namespace(
        seed=123, n=[4], batch_size=[128], backends=list(forwards),
        runs=3, warmup=2, iterations=3,
    ))
    assert calls == [
        (name, backward)
        for backward in (False, True)
        for _ in range(3)
        for name in forwards
    ]
    assert len(rows) == 4
    for row in rows:
        assert row["time_us"] == (30 if row["backend"] == "other" else 10)
        assert row["relative_to_projection"] == (2 if row["backend"] == "other" else 1)


def test_paper_defaults(runtime, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark/run.py"])
    args = runtime.parse_args()
    assert args.n == (4,)
    assert args.batch_size == (512, 2048, 8192, 32768, 131072)
    assert (args.warmup, args.iterations, args.runs) == (100, 100, 10)
    assert not hasattr(args, "output")
