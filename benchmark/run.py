import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import torch
from backends import BACKENDS, Forward

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark" / "results"
RUN_FILES = ("config.json", "accuracy.csv", "runtime.csv", "accuracy.png", "runtime.png")


def marginal_errors(output: torch.Tensor) -> torch.Tensor:
    output = output.double()
    return (output.sum(dim=-1) - 1).abs().sum(dim=-1) + (
        output.sum(dim=-2) - 1
    ).abs().sum(dim=-1)


def marginal_error(output: torch.Tensor) -> tuple[float, float]:
    error = marginal_errors(output)
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


def environment() -> dict:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    major, minor = torch.cuda.get_device_capability()
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "device": {
            "architecture": f"sm{major}{minor}",
            "name": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_home": os.environ.get("CUDA_HOME"),
            **{name: version(name) for name in ("tilelang", "triton", "apache-tvm-ffi", "matplotlib")},
        },
    }


def record_accuracy(path, forwards, *, n, scale, batch_size, seed):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("distribution", "backend", "matrix_index", "error")
        )
        writer.writeheader()
        for distribution in ("normal", "uniform"):
            torch.manual_seed(seed)
            if distribution == "normal":
                logits = scale * torch.randn(batch_size, n, n, device="cuda")
            else:
                logits = scale * (2 * torch.rand(batch_size, n, n, device="cuda") - 1)
            with torch.no_grad():
                for name, forward in forwards.items():
                    output = forward(logits)
                    if not torch.isfinite(output).all():
                        raise RuntimeError(f"Non-finite output: {name}, {distribution}")
                    errors = marginal_errors(output).cpu().tolist()
                    writer.writerows(
                        dict(distribution=distribution, backend=name, matrix_index=i, error=error)
                        for i, error in enumerate(errors)
                    )


def record_runtime(path, forwards, *, n, scale, args):
    torch.manual_seed(args.seed)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("mode", "batch_size", "round", "backend", "time_us")
        )
        writer.writeheader()
        for batch_size in args.batch_size:
            base = scale * torch.randn(batch_size, n, n, device="cuda")
            gradient = torch.randn_like(base)
            for mode in ("forward", "forward_backward"):
                backward = mode == "forward_backward"
                # Probe outside the timed region, including gradient finiteness.
                for name, forward in forwards.items():
                    logits = base.detach().clone().requires_grad_(backward)
                    output = forward(logits)
                    finite = bool(torch.isfinite(output).all())
                    if backward:
                        grad, = torch.autograd.grad(output, logits, gradient)
                        finite = finite and bool(torch.isfinite(grad).all())
                    if not finite:
                        raise RuntimeError(f"Non-finite {mode}: {name}, batch={batch_size}")
                for round_index in range(args.runs):
                    for name, forward in forwards.items():
                        logits = base.detach().clone().requires_grad_(backward)
                        time_us = measure(
                            forward, logits, gradient, backward=backward,
                            warmup=args.warmup, iterations=args.iterations,
                        )
                        if not math.isfinite(time_us) or time_us <= 0:
                            raise RuntimeError(f"Invalid timing: {name}, {time_us}")
                        writer.writerow(dict(
                            mode=mode, batch_size=batch_size, round=round_index,
                            backend=name, time_us=time_us,
                        ))
                    stream.flush()


def run(args: argparse.Namespace) -> list[Path]:
    info = environment()
    architecture = info["device"]["architecture"]
    destinations = {
        (n, scale): RESULTS_DIR / f"{architecture}-n{n}-scale{scale}"
        for n in args.n for scale in args.scale
    }
    # Check every destination before modifying any existing run.
    for path in destinations.values():
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists; use --overwrite to replace it")
    for (n, scale), path in destinations.items():
        path.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for name in RUN_FILES:
                (path / name).unlink(missing_ok=True)
        forwards = {name: BACKENDS[name].build(n) for name in args.backends}
        started_at = datetime.now(timezone.utc).isoformat()
        record_accuracy(
            path / "accuracy.csv", forwards, n=n, scale=scale,
            batch_size=args.accuracy_batch_size, seed=args.seed,
        )
        record_runtime(path / "runtime.csv", forwards, n=n, scale=scale, args=args)
        config = {
            **info, "n": n, "scale": scale, "seed": args.seed,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "backends": args.backends,
            "accuracy": {
                "batch_size": args.accuracy_batch_size,
                "distributions": ["normal", "uniform"],
                "input_dtype": "float32", "error_dtype": "float64",
            },
            "runtime": {
                "batch_sizes": args.batch_size, "distribution": "normal",
                "input_dtype": "float32", "modes": ["forward", "forward_backward"],
                "warmup": args.warmup, "iterations": args.iterations, "runs": args.runs,
                "timing": "eager CUDA Events; microseconds per call",
            },
        }
        # A config is written only after both measurements finish successfully.
        (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"Saved {path}", flush=True)
    return list(destinations.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record raw projection benchmark results")
    parser.add_argument("--n", type=int, nargs="+", choices=(4, 8), default=(4,))
    parser.add_argument("--scale", type=int, nargs="+", default=(1, 10))
    parser.add_argument("--accuracy-batch-size", type=int, default=10000)
    parser.add_argument(
        "--batch-size", type=int, nargs="+", default=(512, 2048, 8192, 32768, 65536, 131072)
    )
    parser.add_argument(
        "--backends", nargs="+", choices=tuple(BACKENDS), default=tuple(BACKENDS)
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if "mHC-proj" not in args.backends:
        parser.error("--backends must include mHC-proj")
    if min(args.scale) <= 0:
        parser.error("--scale values must be positive")
    if args.accuracy_batch_size < 2 or min(args.batch_size) < 1:
        parser.error("accuracy batch size must be at least 2; runtime batches must be positive")
    if args.warmup < 0 or args.iterations < 1 or args.runs < 1:
        parser.error("warmup must be nonnegative; iterations and runs must be positive")
    for name in ("n", "scale", "batch_size", "backends"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"duplicate values in --{name.replace('_', '-')}")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error
