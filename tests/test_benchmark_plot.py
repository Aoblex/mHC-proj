import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import pytest


@pytest.fixture
def plotting(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    path = Path(__file__).resolve().parents[1] / "benchmark" / "plot.py"
    spec = importlib.util.spec_from_file_location("benchmark_plot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ASSETS_DIR", tmp_path / "assets")
    return module


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def recorded(tmp_path):
    directory = tmp_path / "results" / "sm89-n8-scale10"
    directory.mkdir(parents=True)
    config = {
        "n": 8, "scale": 10, "device": {"architecture": "sm89", "name": "Test GPU"},
        "backends": ["other", "mHC-proj"],
        "accuracy": {"batch_size": 4, "distributions": ["normal", "uniform"]},
        "runtime": {"batch_sizes": [512], "runs": 3, "modes": ["forward", "forward_backward"]},
    }
    (directory / "config.json").write_text(json.dumps(config))
    accuracy = [
        dict(distribution=distribution, backend=backend, matrix_index=i, error=v * scale)
        for distribution in ("normal", "uniform")
        for backend, scale in (("other", 2), ("mHC-proj", 1))
        for i, v in enumerate((4, 1, 3, 2))
    ]
    timing = [
        dict(mode=mode, batch_size=512, round=i, backend=backend, time_us=v)
        for mode in config["runtime"]["modes"]
        for backend, values in (("other", [2, 30, 40]), ("mHC-proj", [1, 10, 100]))
        for i, v in enumerate(values)
    ]
    write_csv(directory / "accuracy.csv", accuracy)
    write_csv(directory / "runtime.csv", timing)
    return directory, config


def test_accuracy_statistics_use_sample_std_and_lower_median(plotting, recorded):
    directory, config = recorded
    values = plotting.accuracy_statistics(directory / "accuracy.csv", config)
    assert values["normal", "mHC-proj"] == (2.5, statistics.stdev([1, 2, 3, 4]), 2, 4)
    assert values["uniform", "other"] == (5, statistics.stdev([2, 4, 6, 8]), 4, 8)


def test_runtime_uses_median_of_paired_ratios(plotting, recorded):
    directory, config = recorded
    values = plotting.runtime_ratios(directory / "runtime.csv", config)
    for mode in config["runtime"]["modes"]:
        assert values[mode, 512, "other"] == 2  # Not median(times)/median(baseline) == 3.
        assert values[mode, 512, "mHC-proj"] == 1


@pytest.mark.parametrize("kind", ["accuracy", "runtime"])
@pytest.mark.parametrize("damage", ["missing", "duplicate", "nonfinite"])
def test_reject_incomplete_or_invalid_samples(plotting, recorded, kind, damage):
    directory, config = recorded
    path = directory / f"{kind}.csv"
    rows = list(csv.DictReader(path.open()))
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["error" if kind == "accuracy" else "time_us"] = "nan"
    write_csv(path, rows)
    fn = plotting.accuracy_statistics if kind == "accuracy" else plotting.runtime_ratios
    with pytest.raises(ValueError):
        fn(path, config)


def test_plots_and_selected_publication_are_reproducible(plotting, recorded):
    directory, _ = recorded
    plotting.plot(directory)
    assert not plotting.ASSETS_DIR.exists()
    assert {p.name for p in directory.iterdir()} == {
        "config.json", "accuracy.csv", "runtime.csv", "accuracy.png", "runtime.png",
    }
    image = directory / "accuracy.png"
    before = hashlib.sha256(image.read_bytes()).digest()
    plotting.plot(directory, publish=["accuracy"])
    assert hashlib.sha256(image.read_bytes()).digest() == before
    assert [p.name for p in plotting.ASSETS_DIR.iterdir()] == ["sm89-n8-accuracy-scale10.png"]
    assert (plotting.ASSETS_DIR / "sm89-n8-accuracy-scale10.png").read_bytes() == image.read_bytes()
