import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
DEFAULT_INPUT = ROOT / "results" / "benchmarks" / "results.csv"
DEFAULT_OUTPUT = ROOT / "results" / "benchmarks" / "benchmark.png"


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["n"] = int(row["n"])
        row["batch_size"] = int(row["batch_size"])
        row["time_us"] = float(row["time_us"])
    return rows


def plot_results(rows: list[dict], output: Path) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["n"], row["mode"], row["backend"])].append(row)
    backends = list(dict.fromkeys(row["backend"] for row in rows))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for row_index, n in enumerate((4, 8)):
        for column_index, mode in enumerate(("forward", "forward_backward")):
            axis = axes[row_index, column_index]
            for backend in backends:
                if (n, mode, backend) not in groups:
                    continue
                values = sorted(
                    groups[(n, mode, backend)], key=lambda row: row["batch_size"]
                )
                axis.plot(
                    [row["batch_size"] for row in values],
                    [row["time_us"] for row in values],
                    marker="o",
                    label=backend,
                )
            axis.set(
                title=f"n={n}, {mode.replace('_', ' + ')}",
                xlabel="Batch size",
                ylabel="Time (us)",
                xscale="log",
                yscale="log",
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize="small")

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mHC benchmark results")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    plot_results(load_results(arguments.input), arguments.output)
    print(f"Saved plot to {arguments.output}")
