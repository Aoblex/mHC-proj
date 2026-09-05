# Project Guidance

## Priorities

- Prefer the smallest direct implementation that satisfies the current requirement.
- Preserve mathematical behavior, numerical accuracy, supported public behavior, and performance
  unless the task explicitly changes them.
- Keep changes focused and preserve unrelated user work.
- Do not add fallback paths, compatibility layers, dependencies, or speculative abstractions unless
  explicitly requested.
- Ask before making an unclear API, numerical, architectural, or performance trade-off.

## Architecture

- Keep the public Python API, autograd integration, PyTorch binding, native tensor handling, CUDA
  launchers, and CUDA kernels as distinct responsibilities.
- Keep the projection solver and Sinkhorn comparison implementation separate.
- Preserve the public distinction between the n=4 and n=8 APIs; share code only when it simplifies
  the implementation without obscuring their different CUDA schedules.
- Keep benchmark-only backends and dependencies out of the library implementation.
- Add a new file or layer only when it creates a clear ownership boundary.

Before changing a subsystem, trace its public entry point, tests, binding path, launcher, and kernel.
Treat the current source and tests as the authority for exact interfaces and supported behavior.

## Python and Native Bindings

- Run project Python commands through `uv run`; include the benchmark dependency group when needed.
- Test behavior through the public `mhc_proj` API rather than private extension entry points unless
  the task specifically concerns the low-level API.
- Keep validation errors clear and preserve arbitrary leading batch dimensions, non-contiguous input
  handling, empty batches, dtype/device behavior, and current CUDA stream semantics.
- Keep bindings and launchers thin; do not move algorithmic logic into them.

## CUDA

- Optimize the implementation as general CUDA code, not specifically for the GPUs currently
  available for testing.
- Do not hard-code architecture-specific assumptions, instructions, or launch choices unless there
  is an explicit dispatch with a sound general path.
- The currently available sm89 and sm120 devices are validation platforms, not the only intended
  targets. Report other architectures as untested rather than unsupported or assumed equivalent.
- Preserve the numerical contract of the Birkhoff projection, including finite outputs, convergence,
  marginal accuracy, and correct forward and backward behavior.
- Treat changes to scheduling, memory layout, synchronization, solver steps, and compilation
  boundaries as performance-sensitive.

## Performance Work

- Do not optimize from intuition alone. Establish a correct baseline, identify a concrete limitation,
  change one material variable, verify correctness, and measure again.
- Use representative n=4 and n=8 workloads, relevant batch sizes, and both forward and
  forward-plus-backward modes.
- Record numerical accuracy alongside latency; a faster result that violates the current tolerances
  is not an improvement.
- Use repeated uninstrumented benchmark measurements to make keep-or-revert decisions. Profiler
  timings and counters are diagnostic evidence, not proof of a speedup.
- Keep a performance change only when reproducible gains exceed noise on the tested devices without
  unacceptable regressions. Do not claim that gains transfer to untested architectures.

## Testing and Verification

- Choose verification according to the affected behavior and risk.
- Run focused checks while iterating, then the relevant complete checks before finishing.
- For shared CUDA changes, verify on both available sm89 and sm120 devices when practical.
- Report checks that could not be run instead of silently changing environments or substituting a
  different test.
- Update README only when user-facing installation, API, or usage behavior changes.

## Environment and Artifacts

- Use the Python, CUDA, build, and device configuration selected by the caller.
- Do not search for alternate CUDA installations, install missing tools, alter GPU settings, or add
  machine-specific configuration without explicit approval.
- Keep temporary builds, benchmark runs, profiler reports, and experiment records untracked and out
  of source directories. Add published benchmark results only when explicitly requested.
- Do not create development logs or commit generated binaries.
