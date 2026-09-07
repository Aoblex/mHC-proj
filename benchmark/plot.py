"""Draw saved benchmark measurements; no CUDA or benchmark execution required."""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
NORMAL = FontProperties(fname=FONT_DIR / "cmr10.ttf", size=19)
BOLD = FontProperties(fname=FONT_DIR / "cmb10.ttf", size=19)
NORMAL.set_math_fontfamily("cm")
BOLD.set_math_fontfamily("cm")
SHORT_NAMES = {
    "Vanilla": "Vanilla", "Triton-Sinkhorn": "Triton", "mHC.cu": "mHC.cu",
    "TileLangExamples": "TLE", "TileKernels": "TK",
    "mHC-proj-TL": "Proj-TL", "mHC-proj": "Proj",
}


def accuracy_statistics(path: Path, config: dict) -> dict:
    count = config["accuracy"]["batch_size"]
    groups = {
        key: [] for key in itertools.product(
            config["accuracy"]["distributions"], config["backends"]
        )
    }
    seen = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            group = row["distribution"], row["backend"]
            index, error = int(row["matrix_index"]), float(row["error"])
            key = (*group, index)
            if (group not in groups or not 0 <= index < count or key in seen
                    or not math.isfinite(error) or error < 0):
                raise ValueError(f"Invalid or duplicate accuracy sample: {row}")
            seen.add(key)
            groups[group].append(error)
    if any(len(values) != count for values in groups.values()):
        raise ValueError(f"Incomplete accuracy samples: {path}")
    return {
        key: (statistics.mean(v), statistics.stdev(v), statistics.median_low(v), max(v))
        for key, v in groups.items()
    }


def runtime_ratios(path: Path, config: dict) -> dict:
    runtime = config["runtime"]
    expected = set(itertools.product(
        runtime["modes"], runtime["batch_sizes"], range(runtime["runs"]), config["backends"]
    ))
    times = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["mode"], int(row["batch_size"]), int(row["round"]), row["backend"]
            value = float(row["time_us"])
            if key not in expected or key in times or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid or duplicate timing sample: {row}")
            times[key] = value
    if times.keys() != expected:
        raise ValueError(f"Incomplete timing samples: {path}")
    return {
        (mode, batch, backend): statistics.median(
            times[mode, batch, r, backend] / times[mode, batch, r, "mHC-proj"]
            for r in range(runtime["runs"])
        )
        for mode, batch, backend in itertools.product(
            runtime["modes"], runtime["batch_sizes"], config["backends"]
        )
    }


def text(ax, x, y, value, *, bold=False, size=19):
    font = (BOLD if bold else NORMAL).copy()
    font.set_size(size)
    ax.text(x, y, value, fontproperties=font, ha="center", va="center")


def canvas(count, caption):
    lines = textwrap.wrap(caption, width=97)
    offset = (len(lines) - 1) * .85
    fig, ax = plt.subplots(figsize=(13, (count + 2.4 + offset) * .44))
    ax.set(xlim=(0, 1), ylim=(count + 2.4 + offset, 0))
    ax.axis("off")
    for i, line in enumerate(lines):
        text(ax, .5, .25 + i * .85, line)
    ax.hlines([.8 + offset, 1.8 + offset], 0, 1, colors="black", linewidths=[1.5, 1])
    return fig, ax, offset


def save(fig, path):
    fig.tight_layout(pad=.4)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def number(value):
    if value < .01:
        return f"{value:.4f}"
    if value < 10:
        return f"{value:.3f}"
    if value < 100:
        return f"{value:.2f}"
    if value < 1000:
        return f"{value:.1f}"
    return f"{value:.0f}"


def draw_accuracy(path, config, values):
    methods = config["backends"]
    distributions = config["accuracy"]["distributions"]
    n, scale = config["n"], config["scale"]
    caption = (
        f'Marginal errors (n={n}, scale={scale}, N={config["accuracy"]["batch_size"]}). '
        f'{config["device"]["name"]}, FP32 inputs. '
        'Lowest displayed values within each distribution and statistic are bold and green.'
    )
    fig, ax, offset = canvas(len(distributions) * len(methods), caption)
    centers = (.08, .315, .535, .645, .755, .865)
    for x, label in zip(centers, ("Entries", "Method", "Mean", "Std.", "Median", "Max")):
        text(ax, x, 1.3 + offset, label)
    ax.hlines(1.9 + offset, 0, 1, color="black", linewidth=.6)
    exponent = -6 if scale == 1 else -3
    factor = 10 ** -exponent
    labels = {
        "normal": rf'$N(0,{scale ** 2})$',
        "uniform": rf'$\mathrm{{Unif}}(-{scale},{scale})$',
    }
    for group, distribution in enumerate(distributions):
        rows = [[v * factor for v in values[distribution, method]] for method in methods]
        minima = [min(row[j] for row in rows) for j in range(4)]
        middle = 2.45 + offset + group * len(methods) + (len(methods) - 1) / 2
        text(ax, .08, middle, labels[distribution])
        text(ax, .973, middle, rf'$(\times10^{{{exponent}}})$', size=15)
        for i, method in enumerate(methods):
            y = 2.45 + offset + group * len(methods) + i
            label = "CUDA-Sinkhorn" if method == "mHC.cu" and n == 8 else method
            if method.startswith("mHC-proj"):
                label += " (ours)"
            text(ax, .315, y, label)
            for j, value in enumerate(rows[i]):
                x = centers[j + 2]
                best = number(value) == number(minima[j])
                if best:
                    ax.add_patch(Rectangle((x - .055, y - .37), .11, .74, facecolor="#bfff00"))
                text(ax, x, y, number(value), bold=best)
            last = i == len(methods) - 1
            ax.hlines(y + .5, 0 if last else .165, 1 if last else .92,
                      color="black", linewidth=1.4 if last else .5)
    save(fig, path)


def draw_runtime(path, config, ratios):
    runtime = config["runtime"]
    methods, batches, modes = config["backends"], runtime["batch_sizes"], runtime["modes"]
    caption = (
        f'Normalized runtime (n={config["n"]}, entries from N(0, {config["scale"] ** 2})). '
        f'{config["device"]["name"]}. Median per-round ratios relative to mHC-proj. '
        'Triton: Triton-Sinkhorn; TLE: TileLangExamples; TK: TileKernels; '
        'Proj-TL: mHC-proj-TL; Proj: mHC-proj.'
    )
    if config["n"] == 8:
        caption += " CUDA-SK: CUDA-Sinkhorn."
    fig, ax, offset = canvas(len(modes) * len(batches), caption)
    centers = [.32 + .625 * i / max(1, len(methods) - 1) for i in range(len(methods))]
    text(ax, .08, 1.3 + offset, "Feature")
    text(ax, .215, 1.3 + offset, "Batch")
    for x, method in zip(centers, methods):
        label = "CUDA-SK" if method == "mHC.cu" and config["n"] == 8 else SHORT_NAMES.get(method, method)
        text(ax, x, 1.3 + offset, label)
    for group, mode in enumerate(modes):
        middle = 2.45 + offset + group * len(batches) + (len(batches) - 1) / 2
        text(ax, .08, middle, "Fwd." if mode == "forward" else "Fwd.+Bwd.")
        for i, batch in enumerate(batches):
            y = 2.45 + offset + group * len(batches) + i
            text(ax, .215, y, rf'${batch / 1024:g}K$')
            for x, method in zip(centers, methods):
                text(ax, x, y, f"{ratios[mode, batch, method]:.3f}", bold=method == "mHC-proj")
        ax.hlines(y + .5, 0, 1, color="black", linewidth=1.5 if group else 1)
    save(fig, path)


def plot(directory: Path, publish=()):
    config = json.loads((directory / "config.json").read_text())
    accuracy = accuracy_statistics(directory / "accuracy.csv", config)
    runtime = runtime_ratios(directory / "runtime.csv", config)
    draw_accuracy(directory / "accuracy.png", config, accuracy)
    draw_runtime(directory / "runtime.png", config, runtime)
    if publish:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        for kind in publish:
            name = f'{config["device"]["architecture"]}-n{config["n"]}-{kind}-scale{config["scale"]}.png'
            shutil.copyfile(directory / f"{kind}.png", ASSETS_DIR / name)
    print(f"Plotted {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--publish", choices=("accuracy", "runtime"), nargs="+", default=())
    args = parser.parse_args()
    for directory in args.directories:
        plot(directory, args.publish)
