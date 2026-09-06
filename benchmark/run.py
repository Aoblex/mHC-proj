import argparse
import csv
import statistics
import sys

import torch
from backends import BACKENDS, Forward


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
    # Every call uses the original logits, never the previous projection.
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
            # The paper uses N(0, 10^2) for both timing modes.
            base = 10 * torch.randn(batch_size, n, n, device="cuda")
            gradient = torch.randn_like(base)
            errors = {}
            with torch.no_grad():
                for name, forward in forwards.items():
                    errors[name] = marginal_error(forward(base))

            for mode in ("forward", "forward_backward"):
                samples = {name: [] for name in forwards}
                ratios = {name: [] for name in forwards}
                for _ in range(args.runs):
                    times = {}
                    for name, forward in forwards.items():
                        logits = (
                            base.detach()
                            .clone()
                            .requires_grad_(mode == "forward_backward")
                        )
                        times[name] = measure(
                            forward,
                            logits,
                            gradient,
                            backward=mode == "forward_backward",
                            warmup=args.warmup,
                            iterations=args.iterations,
                        )
                    for name, time in times.items():
                        samples[name].append(time)
                        ratios[name].append(time / times["mHC-proj"])
                for name in args.backends:
                    mean_error, max_error = errors[name]
                    rows.append(
                        {
                            "mode": mode,
                            "n": n,
                            "batch_size": batch_size,
                            "backend": name,
                            "time_us": statistics.median(samples[name]),
                            "relative_to_projection": statistics.median(ratios[name]),
                            "mean_error": mean_error,
                            "max_error": max_error,
                        }
                    )
    return rows


def print_results(rows: list[dict]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mHC projection backends")
    parser.add_argument("--n", type=int, nargs="+", choices=(4, 8), default=(4,))
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=(512, 2048, 8192, 32768, 131072),
    )
    parser.add_argument(
        "--backends", nargs="+", choices=tuple(BACKENDS), default=tuple(BACKENDS)
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if "mHC-proj" not in args.backends:
        parser.error("--backends must include mHC-proj")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    results = run(arguments)
    print_results(results)
