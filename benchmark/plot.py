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


def text(ax, x, y, value, *, bold=False, ha="center"):
    ax.text(x, y, value, fontproperties=BOLD if bold else NORMAL, ha=ha, va="center")


def canvas(count, caption, *, kind="accuracy"):
    lines = textwrap.wrap(caption, width=97)
    offset = (len(lines) - 1) * .85
    height = count + 2.4 + offset
    fig, ax = plt.subplots(figsize=(13, height * (.55 if kind == "accuracy" else .44)))
    ax.set(xlim=(0, 1), ylim=(height, 0))
    ax.axis("off")
    for i, line in enumerate(lines):
        text(ax, 0 if len(lines) > 1 else .5, .25 + i * .85, line,
             ha="left" if len(lines) > 1 else "center")
    ax.hlines([.8 + offset, 1.8 + offset], 0, 1, colors="black", linewidths=[1.5, 1])
    if kind == "accuracy":
        ax.hlines(1.9 + offset, 0, 1, color="black", linewidth=.6)
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


# Like the paper's \cellcolor annotations, colors are specified per saved table.
ACCURACY_COLORS = {"lime": "#bfff00", "Goldenrod": "#ffde44", "Salmon": "#f49289"}


def draw_accuracy(path, config, values):
    methods = config["backends"]
    distributions = config["accuracy"]["distributions"]
    n, scale = config["n"], config["scale"]
    colors = config["accuracy"].get("colors", {})
    caption = (
        f'Table {1 if scale == 1 else 2}: Accuracy of different projection methods '
        f'for {"small" if scale == 1 else "large"}-magnitude inputs.'
    )
    fig, ax, offset = canvas(len(distributions) * len(methods), caption)
    centers = (.09, .32, .50, .605, .71, .815)
    for x, label in zip(centers, ("Entries", "Method", "Mean", "Std.", "Median", "Max")):
        text(ax, x, 1.3 + offset, label)
    text(ax, .935, 1.3 + offset, rf'$n={n}$')
    exponent = -6 if scale == 1 else -3
    factor = 10 ** -exponent
    labels = {
        "normal": r'$N(0,1)$' if scale == 1 else rf'$N(0,{scale}^2)$',
        "uniform": rf'$\mathrm{{Unif}}(-{scale},{scale})$',
    }
    for group, distribution in enumerate(distributions):
        rows = [[v * factor for v in values[distribution, method]] for method in methods]
        minima = [min(row[j] for row in rows) for j in range(4)]
        start = 2.45 + offset + group * len(methods)
        middle = start + (len(methods) - 1) / 2
        text(ax, centers[0], middle, labels[distribution])
        text(ax, .935, middle, rf'$(\times10^{{{exponent}}})$')
        for i, method in enumerate(methods):
            y = start + i
            label = method
            if method.startswith("mHC-proj"):
                label += " (ours)"
            text(ax, centers[1], y, label)
            for j, value in enumerate(rows[i]):
                x = centers[j + 2]
                best = number(value) == number(minima[j])
                cell_colors = colors.get(distribution, {}).get(method)
                if cell_colors is not None:
                    ax.add_patch(Rectangle(
                        (x - .0525, y - .37), .105, .74,
                        facecolor=ACCURACY_COLORS[cell_colors[j]], edgecolor="none",
                    ))
                text(ax, x, y, number(value), bold=best)
            last = i == len(methods) - 1
            ax.hlines(y + .5, 0 if last else .18, 1 if last else .8675,
                      color="black", linewidth=1.4 if last else .5)
    save(fig, path)


def draw_runtime(path, config, ratios):
    runtime = config["runtime"]
    methods, batches, modes = config["backends"], runtime["batch_sizes"], runtime["modes"]
    caption = (
        'Table 3: Median normalized computational time of different projection methods '
        f'(n={config["n"]}). Short column labels denote Triton-Sinkhorn (Triton), '
        'TileLangExamples (TLE), TileKernels (TK), mHC-proj-TL (Proj-TL), '
        'and mHC-proj (Proj); Fwd. and Fwd.+Bwd. denote the forward pass and '
        'forward-backward computation, respectively.'
    )
    fig, ax, offset = canvas(len(modes) * len(batches), caption, kind="runtime")
    centers = [.32 + .625 * i / max(1, len(methods) - 1) for i in range(len(methods))]
    text(ax, .08, 1.3 + offset, "Feature")
    text(ax, .215, 1.3 + offset, "Batch")
    for x, method in zip(centers, methods):
        label = SHORT_NAMES.get(method, method)
        text(ax, x, 1.3 + offset, label)
    for group, mode in enumerate(modes):
        start = 2.45 + offset + group * len(batches)
        middle = start + (len(batches) - 1) / 2
        text(ax, .08, middle, "Fwd." if mode == "forward" else "Fwd.+Bwd.")
        for i, batch in enumerate(batches):
            y = start + i
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
