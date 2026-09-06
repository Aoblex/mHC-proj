---
name: kernel-optimization
description: Optimize the mHC-proj CUDA solvers end to end using correctness gates, reproducible benchmarks, focused profiling, and keep-or-revert decisions. Use for CUDA performance campaigns, implementation experiments, and candidate comparisons; use kernel-profiling or kernel-knowledge for standalone diagnosis or research.
license: MIT
compatibility: Requires the caller-selected Python and CUDA environment to build, test, and benchmark mHC-proj.
---

# Kernel Optimization

Improve projection performance without sacrificing correctness, numerical accuracy, or general CUDA support.

## Core loop

**Baseline → Profile → Diagnose → Research → Hypothesize → Change one variable → Verify → Measure → Keep or revert**

Do not optimize from intuition alone.

## Start from the checkout

1. Read `AGENTS.md` and inspect the Git state.
2. Trace the target from the public `mhc_proj` API through autograd, bindings, launcher, and kernel.
3. Inspect the current Makefile, tests, benchmark CLI, and nearby implementation patterns.
4. Identify the affected solver, n=4/n=8 path, forward/backward mode, and representative batch sizes.
5. Ask when the performance objective or permitted numerical trade-off is unclear.

Use the caller-selected Python, CUDA, device, and build configuration. Do not install tools, search for another toolkit, or alter GPU settings without approval.

## Record the run

Create a unique ignored directory under `.agents/runs/kernel-optimization/` and copy [`assets/run.md`](assets/run.md) into it. Record the revision, pre-existing changes, toolchain, devices, exact commands, raw measurements, and decisions.

The implementation target remains general CUDA. The available sm89 and sm120 GPUs are validation platforms only; do not assume results transfer to untested architectures.

## Workflow

### 1. Define the experiment

Before editing, record:

- target path and user-relevant performance quantity;
- unchanged baseline revision or binary;
- correctness oracle, tolerances, and marginal-error requirements;
- representative n, batch sizes, input distributions, and forward/backward modes;
- API, dtype/device, stream, memory, and portability constraints;
- evidence required to retain the candidate.

### 2. Establish the baseline

Build through the Makefile and verify the relevant public behavior before measuring. Use the existing benchmark path where possible. Keep input preparation, compilation, allocation warmup, and unrelated setup outside the timed region.

Collect repeated uninstrumented measurements. Prefer paired or alternating baseline/candidate runs when drift, thermals, or clocks could affect the result. Save raw samples, not only the best value.

### 3. Diagnose with evidence

State one concrete question. Use [`kernel-profiling`](../kernel-profiling/SKILL.md) when timeline or hardware-counter evidence is needed. Reduce the evidence to a falsifiable statement:

> The target is limited by **X**, supported by **Y**; changing **Z** should improve **W**.

Profiler duration is not benchmark evidence. Higher occupancy or lower register use is not automatically better.

### 4. Research applicable prior art

Use [`kernel-knowledge`](../kernel-knowledge/SKILL.md) when official CUDA documentation, upstream implementations, or algorithmic prior art can inform the diagnosis. Check architecture, dtype, shape, layout, and workload assumptions before applying a technique.

### 5. Build one candidate

Make the smallest coherent change that tests the hypothesis. Change one material variable at a time and keep cleanup out of the experiment. Prefer architecture-neutral improvements. Architecture-specific code requires an explicit need, supported dispatch, and a sound general path.

### 6. Promote or reject

Evaluate in this order:

1. Build.
2. Focused public-API correctness and numerical checks.
3. Relevant complete tests.
4. Repeated uninstrumented benchmark against the baseline.
5. Additional profiling only if needed to explain the result.

Check n=4 and n=8 when shared code changes, and both sm89 and sm120 when practical. Retain a candidate only when correctness passes and representative gains exceed noise without unacceptable regressions. Otherwise remove only the candidate changes and preserve pre-existing work.

### 7. Finish

Rebuild and re-run the relevant tests and benchmark for the retained implementation. Report the objective, workload, baseline, diagnosis, retained change, correctness results, final measurements, rejected candidates when informative, and untested architectures. Keep generated evidence under the ignored run directory.
