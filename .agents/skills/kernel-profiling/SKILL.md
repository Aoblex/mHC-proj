---
name: kernel-profiling
description: Profile and diagnose mHC-proj CUDA execution with the caller-selected NVIDIA tools and hardware. Use for Nsight Systems or Nsight Compute collection, launch and synchronization analysis, occupancy or stall investigation, source attribution, and read-only performance diagnosis.
license: MIT
compatibility: Requires a runnable mHC-proj workload and any requested profiler in the caller-selected environment.
---

# Kernel Profiling

Collect the least profiler evidence needed to answer one concrete performance question. This skill diagnoses behavior; it does not edit kernels or decide whether to promote a candidate.

## Start from the project

1. Read `AGENTS.md` and inspect the Git state.
2. Trace the selected public workload to the launched kernel.
3. Confirm the normal build, relevant correctness check, and uninstrumented benchmark work.
4. Record the revision, build configuration, Python and CUDA versions, device, GPU, compute capability, and profiler version.

Use the selected environment. If a profiler or capability is missing, report it rather than installing tools, changing environments, or substituting another device.

## Store evidence

For a standalone investigation, create `.agents/runs/kernel-profiling/<experiment>/` and copy [`assets/report.md`](assets/report.md) into it. When profiling inside an optimization run, use that run's `profiles/` directory. Keep exact commands, original reports, exports, and analysis together.

## Workflow

### 1. State the question

Name the solver and dispatch path, n, mode, batch size, input distribution, observed symptom, and alternatives the profile must distinguish. Avoid generic requests such as “why is it slow?”

### 2. Choose the observation layer

Start at the highest uncertain layer:

- Use the normal benchmark for user-visible latency.
- Use Nsight Systems when host overhead, launch order, synchronization, or many short kernels may dominate.
- Use Nsight Compute when the dominant kernel is known and the question concerns instructions, dependencies, resources, memory behavior, or work distribution.
- Collect focused counters or source attribution only after an overview leaves a specific question unanswered.

### 3. Inspect the active tool

Read the installed profiler's help and available sections or metrics before constructing a command. Names and capabilities vary by profiler version and architecture. Do not copy a metric list from sm89 to sm120, or vice versa, without confirming availability.

### 4. Isolate and collect

Use the repository's public benchmark path with one backend, device, n, and batch size whenever it can isolate the target. Warm up before capture and keep the profiled region short. Confirm the captured kernel identity from output instead of relying on a guessed name or skip count.

Build a temporary standalone harness only when the project path cannot answer the question, and ask before adding one. Never use profiler-reported duration as the promotion benchmark.

### 5. Analyze and report

Connect a small set of observations to one primary mechanism. Consider launch overhead, available parallelism, resource limits, dependencies, synchronization, memory traffic, and work balance as relevant. State alternatives ruled out, remaining uncertainty, and the next measurement or falsifiable candidate.

Treat sm89 and sm120 as observed validation platforms, not as the implementation's support boundary. Do not generalize measured counters or conclusions to untested architectures.
