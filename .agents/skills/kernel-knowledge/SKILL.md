---
name: kernel-knowledge
description: Research evidence-backed CUDA and Birkhoff-projection techniques applicable to mHC-proj. Use for official hardware documentation, upstream implementations or pull requests, architecture applicability checks, known failed approaches, and prior art supporting a measured optimization hypothesis.
license: MIT
compatibility: Uses repository and available read-only web sources; no external corpus is required.
---

# Kernel Knowledge

Find prior art for a concrete measured problem without treating another kernel or GPU architecture as a recipe. This is a read-only research skill: it does not edit code, profile kernels, or make promotion decisions.

## Define the question

Record:

- the projection or Sinkhorn path and current implementation;
- target dtype, n, batch size, layout, and forward/backward mode;
- the active GPU and compute capability;
- benchmark or profiler evidence;
- approaches already attempted;
- evidence that would make a technique applicable or rule it out.

Do not start with a catalogue of generic CUDA optimization tips.

## Source order

Search progressively and stop when enough evidence supports a testable direction:

1. Current source, tests, benchmark results, local experiment records, README, and linked paper.
2. Current official CUDA, architecture, instruction, and profiler documentation.
3. Primary upstream code, commits, pull requests, issues with reproductions, and release notes.
4. Technical reports with enough hardware, shape, dtype, and measurement context to evaluate the claim.

Prefer primary sources and stable links.

## Check applicability

For each useful source, identify its assumptions:

- GPU product, compute capability, and required instructions;
- CUDA and compiler generation;
- dtype, accumulation precision, matrix size, layout, and alignment;
- launch geometry, workload size, and surrounding operations;
- numerical requirements and backward formulation.

The implementation target is general CUDA. A technique observed on sm89 or sm120 is architecture-specific evidence until its requirements and a general implementation path are established. Never convert an upstream speedup into an expected local speedup.

## Produce a shortlist

For each retained direction, report:

- source and stable URL, commit, pull request, or document section;
- evidence level: official, upstream-verified, source-reported, or inferred;
- original hardware, workload, metric, and result when performance is claimed;
- why the mechanism may apply here;
- prerequisites and numerical, portability, or performance risks;
- the local benchmark or profiler signal that would support or falsify it.

Rank only a small number of evidence-backed directions. Clearly label uncertainty and rejected prior art. Local correctness and repeated benchmark measurements remain required.
