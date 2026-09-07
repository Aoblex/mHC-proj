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
    config["accuracy"]["colors"] = {
        distribution: {
            "other": ["Salmon", "Goldenrod", "lime", "Salmon"],
            "mHC-proj": ["lime"] * 4,
        }
        for distribution in config["accuracy"]["distributions"]
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


@pytest.mark.parametrize("annotated", [True, False])
def test_accuracy_uses_explicit_cell_colors(plotting, recorded, monkeypatch, annotated):
    directory, config = recorded
    if not annotated:
        del config["accuracy"]["colors"]
    values = plotting.accuracy_statistics(directory / "accuracy.csv", config)
    figures = []
    monkeypatch.setattr(plotting, "save", lambda fig, path: figures.append(fig))
    plotting.draw_accuracy(directory / "accuracy.png", config, values)
    try:
        colors = [plotting.matplotlib.colors.to_hex(p.get_facecolor())
                  for p in figures[0].axes[0].patches]
        expected = (["#f49289", "#ffde44", "#bfff00", "#f49289"] + ["#bfff00"] * 4) * 2
        assert colors == (expected if annotated else [])
    finally:
        plotting.plt.close(figures[0])


@pytest.mark.parametrize(("n", "scale", "table", "magnitude"), [
    (4, 1, 1, "small"), (8, 10, 2, "large"),
])
def test_original_captions_and_runtime_columns(plotting, recorded, monkeypatch, n, scale, table, magnitude):
    directory, config = recorded
    config.update(n=n, scale=scale)
    captions, figures = [], []
    canvas = plotting.canvas

    def capture(count, caption, **kwargs):
        captions.append(caption)
        return canvas(count, caption, **kwargs)

    def save(fig, path):
        figures.append(fig)
        plotting.plt.close(fig)

    monkeypatch.setattr(plotting, "canvas", capture)
    monkeypatch.setattr(plotting, "save", save)
    values = plotting.accuracy_statistics(directory / "accuracy.csv", config)
    plotting.draw_accuracy(directory / "accuracy.png", config, values)
    assert captions[0] == (
        f"Table {table}: Accuracy of different projection methods "
        f"for {magnitude}-magnitude inputs."
    )
    assert rf'$n={n}$' in [t.get_text() for t in figures[0].axes[0].texts]
    assert len(figures[0].axes[0].patches) == 16
    config["backends"] = list(plotting.SHORT_NAMES)
    ratios = {(mode, 512, name): 1 for mode in config["runtime"]["modes"] for name in config["backends"]}
    plotting.draw_runtime(directory / "runtime.png", config, ratios)
    assert captions[1].startswith(
        f"Table 3: Median normalized computational time of different projection methods (n={n})."
    )
    assert config["device"]["name"] not in " ".join(captions)
    headings = {t.get_text(): t.get_position()[0] for t in figures[1].axes[0].texts}
    assert headings["Vanilla"] < headings["Proj-TL"] < headings["Proj"]
    assert "mHC.cu" in headings
    assert not figures[1].axes[0].patches
    top_rule = figures[1].axes[0].collections[0].get_segments()[0][0][1]
    assert all(t.get_ha() == "left" for t in figures[1].axes[0].texts if t.get_position()[1] < top_rule)
