import argparse
import csv
import statistics
from pathlib import Path

import torch
from backends import BACKENDS, Forward

DEFAULT_OUTPUT = Path(__file__).parents[1] / "results" / "benchmarks" / "results.csv"


def marginal_error(output: torch.Tensor) -> tuple[float, float]:
    row_error = (output.double().sum(dim=-1) - 1.0).abs().sum(dim=-1)
    col_error = (output.double().sum(dim=-2) - 1.0).abs().sum(dim=-1)
    error = row_error + col_error
    return error.mean().item(), error.max().item()


def _step(
    forward: Forward,
    logits: torch.Tensor,
    gradient: torch.Tensor,
    backward: bool,
) -> torch.Tensor:
    output = forward(logits)
    if backward:
        torch.autograd.grad(output, logits, gradient)
    return output


def measure(
    forward: Forward,
    logits: torch.Tensor,
    gradient: torch.Tensor,
    *,
    backward: bool,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        _step(forward, logits, gradient, backward)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _step(forward, logits, gradient, backward)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def run(args: argparse.Namespace) -> list[dict]:
    torch.manual_seed(args.seed)
    rows = []
    for n in args.n:
        forwards = {name: BACKENDS[name].build(n) for name in args.backends}
        for batch_size in args.batch_size:
            base = torch.randn(batch_size, n, n, device="cuda")
            gradient = torch.randn_like(base)
            errors = {}
            with torch.no_grad():
                for name, forward in forwards.items():
                    errors[name] = marginal_error(forward(base))

            for mode in ("forward", "forward_backward"):
                times = {}
                for name, forward in forwards.items():
                    samples = []
                    for _ in range(args.runs):
                        logits = (
                            base.detach()
                            .clone()
                            .requires_grad_(mode == "forward_backward")
                        )
                        samples.append(
                            measure(
                                forward,
                                logits,
                                gradient,
                                backward=mode == "forward_backward",
                                warmup=args.warmup,
                                iterations=args.iterations,
                            )
                        )
                    times[name] = statistics.median(samples)

                baseline = times["cuda-projection"]
                for name in args.backends:
                    mean_error, max_error = errors[name]
                    rows.append(
                        {
                            "mode": mode,
                            "n": n,
                            "batch_size": batch_size,
                            "backend": name,
                            "time_us": times[name],
                            "relative_to_projection": times[name] / baseline,
                            "mean_error": mean_error,
                            "max_error": max_error,
                        }
                    )
    return rows


def write_results(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mHC projection backends")
    parser.add_argument("--n", type=int, nargs="+", choices=(4, 8), default=(4, 8))
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=(128, 512, 2048, 8192, 32768, 131072),
    )
    parser.add_argument(
        "--backends", nargs="+", choices=tuple(BACKENDS), default=tuple(BACKENDS)
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if "cuda-projection" not in args.backends:
        parser.error("--backends must include cuda-projection")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    results = run(arguments)
    write_results(results, arguments.output)
    print(f"Saved {len(results)} rows to {arguments.output}")
