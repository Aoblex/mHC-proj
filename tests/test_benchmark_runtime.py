import argparse
from collections.abc import Callable
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, "backends",
        SimpleNamespace(BACKENDS={"mHC-proj": SimpleNamespace(build=lambda n: lambda x: x)}, Forward=Callable),
    )
    path = Path(__file__).resolve().parents[1] / "benchmark" / "run.py"
    spec = importlib.util.spec_from_file_location("benchmark_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(module, "environment", lambda: {"device": {"architecture": "sm89"}})
    return module


@pytest.fixture
def cpu_random(monkeypatch):
    randn, rand = torch.randn, torch.rand
    monkeypatch.setattr(torch, "randn", lambda *shape, device: randn(*shape))
    monkeypatch.setattr(torch, "rand", lambda *shape, device: rand(*shape))


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


@pytest.mark.parametrize("scale", [1, 10])
def test_runtime_records_each_round_with_fixed_scaled_inputs(runtime, cpu_random, monkeypatch, tmp_path, scale):
    torch.manual_seed(123)
    expected = scale * torch.randn(128, 4, 4, device="cpu")
    forwards = {name: (lambda x: x) for name in ("other", "mHC-proj")}
    timings = {
        (name, backward): iter(values)
        for backward in (False, True)
        for name, values in (("other", [2, 30, 40]), ("mHC-proj", [1, 10, 100]))
    }

    def measure(forward, logits, gradient, *, backward, **kwargs):
        torch.testing.assert_close(logits, expected)
        assert logits.requires_grad == backward
        name = next(name for name, f in forwards.items() if f is forward)
        return next(timings[name, backward])

    monkeypatch.setattr(runtime, "measure", measure)
    path = tmp_path / "runtime.csv"
    runtime.record_runtime(path, forwards, n=4, scale=scale, args=argparse.Namespace(
        seed=123, batch_size=[128], runs=3, warmup=2, iterations=3,
    ))
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 12
    for mode in ("forward", "forward_backward"):
        for name, expected_times in (("other", [2, 30, 40]), ("mHC-proj", [1, 10, 100])):
            selected = [r for r in rows if r["mode"] == mode and r["backend"] == name]
            assert [int(r["round"]) for r in selected] == [0, 1, 2]
            assert [float(r["time_us"]) for r in selected] == expected_times


def test_accuracy_records_each_matrix(runtime, cpu_random, tmp_path):
    path = tmp_path / "accuracy.csv"
    runtime.record_accuracy(path, {"mHC-proj": lambda x: x}, n=4, scale=10, batch_size=4, seed=123)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 8
    for distribution in ("normal", "uniform"):
        torch.manual_seed(123)
        x = torch.randn(4, 4, 4, device="cpu") if distribution == "normal" else 2 * torch.rand(4, 4, 4, device="cpu") - 1
        expected = runtime.marginal_errors(10 * x).tolist()
        selected = [r for r in rows if r["distribution"] == distribution]
        assert [int(r["matrix_index"]) for r in selected] == list(range(4))
        assert [float(r["error"]) for r in selected] == expected


def arguments(**kwargs):
    defaults = dict(n=[4], scale=[1], backends=["mHC-proj"], seed=123,
                    accuracy_batch_size=4, batch_size=[4], warmup=0, iterations=1, runs=2,
                    overwrite=False)
    return argparse.Namespace(**(defaults | kwargs))


def test_run_writes_config_and_raw_files(runtime, cpu_random, monkeypatch):
    monkeypatch.setattr(runtime, "measure", lambda *a, **kw: 1.0)
    paths = runtime.run(arguments(scale=[1, 10]))
    assert [p.name for p in paths] == ["sm89-n4-scale1", "sm89-n4-scale10"]
    for path, scale in zip(paths, [1, 10]):
        assert {p.name for p in path.iterdir()} == {"config.json", "accuracy.csv", "runtime.csv"}
        config = json.loads((path / "config.json").read_text())
        assert (config["n"], config["scale"], config["seed"]) == (4, scale, 123)
        assert config["runtime"]["runs"] == 2
        assert config["completed_at"] >= config["started_at"]


def test_preflight_refuses_overwrite_before_any_changes(runtime):
    existing = runtime.RESULTS_DIR / "sm89-n4-scale10"
    existing.mkdir()
    (existing / "config.json").write_text("original")
    with pytest.raises(FileExistsError, match="--overwrite"):
        runtime.run(arguments(scale=[1, 10]))
    assert not (runtime.RESULTS_DIR / "sm89-n4-scale1").exists()
    assert (existing / "config.json").read_text() == "original"


def test_explicit_overwrite_invalidates_old_plots(runtime, cpu_random, monkeypatch):
    path = runtime.RESULTS_DIR / "sm89-n4-scale1"
    path.mkdir()
    for name in runtime.RUN_FILES:
        (path / name).write_text("old")
    (path / "notes.txt").write_text("keep")
    monkeypatch.setattr(runtime, "measure", lambda *a, **kw: 1.0)
    runtime.run(arguments(overwrite=True))
    assert not (path / "accuracy.png").exists()
    assert not (path / "runtime.png").exists()
    assert (path / "notes.txt").read_text() == "keep"
    assert json.loads((path / "config.json").read_text())["n"] == 4


def test_failed_run_has_no_completion_config(runtime, cpu_random, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("measurement failed")
    monkeypatch.setattr(runtime, "record_runtime", fail)
    with pytest.raises(RuntimeError, match="measurement failed"):
        runtime.run(arguments())
    assert not (runtime.RESULTS_DIR / "sm89-n4-scale1" / "config.json").exists()


@pytest.mark.parametrize("flags", [
    ["--scale", "0"], ["--scale", "-1"], ["--runs", "0"], ["--iterations", "0"],
    ["--warmup", "-1"], ["--batch-size", "0"], ["--accuracy-batch-size", "1"],
    ["--n", "4", "4"], ["--scale", "1", "1"],
])
def test_invalid_parameters(runtime, monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["run.py", *flags])
    with pytest.raises(SystemExit):
        runtime.parse_args()


def test_defaults(runtime, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark/run.py"])
    args = runtime.parse_args()
    assert args.n == (4,)
    assert args.scale == (1, 10)
    assert args.accuracy_batch_size == 10000
    assert args.batch_size == (512, 2048, 8192, 32768, 65536, 131072)
    assert (args.warmup, args.iterations, args.runs) == (100, 100, 10)
    assert not args.overwrite
