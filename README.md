# main

The **stable baseline** branch: the BBC VC-2 HQ codec (intra-only wavelet +
entropy codec) ported to Windows / MSVC, together with a repeatable profiling
and benchmarking harness.

## Why this branch exists

- It is the single source of truth for the upstream code, left untouched by
  optimization experiments.
- It is the reference against which every change on the other branches is
  validated. The correctness contract for any optimisation is **byte-identical
  output** (SHA-256 of the coded stream and decoded pixels must match).

## What is here

- Windows/MSVC build via CMake + vcpkg (`build-windows.ps1`, `vcpkg.json`,
  `BUILDING-WINDOWS.md`).
- The `bench/` harness: process pinned to one logical CPU, `AboveNormal`
  priority, median of N runs, output writes disabled.
- Stage-level profiling via the `VC2HQ_PROFILE=1` environment variable
  (CSV `VC2HQ_STAGE_PROFILE` lines at exit).
- `profiles/INITIAL-PROFILE.md`: the first full profiling pass that identified
  the hot paths targeted on later branches.

## History

- `Initial VC-2 Windows MSVC port and profiling baseline`
- `chore: ignore local build and profiler artifacts`
- `chore: cover generic CMake and Nsight outputs`
