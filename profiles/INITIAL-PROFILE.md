# Initial VC-2 HQ fast-path profile

Date: 2026-08-12

## System and workload

- CPU: Intel Xeon E5-1650 v4, 6 physical cores / 12 logical processors
- Compiler: MSVC 19.44, optimized Release build
- SIMD detected: SSE4.2, AVX, AVX2
- Picture: 1920x1080, planar 10-bit 4:2:2
- Compression ratio: 2:1
- Fastest encoder configuration found: `fastest`, Haar0, depth 3,
  32x8 slices, 12 workers
- Runs: 30 frames, output writes disabled, three throughput repetitions and
  five stage-timing repetitions

The 2:1 ratio was retained so that the result represents useful coding work.
Increasing the ratio without bound could make the benchmark artificially fast
by producing progressively less coded data.

## Throughput and scaling

Median uninstrumented measurements:

| Workers | Encoder fps | Encoder ms/frame | Decoder fps | Decoder ms/frame |
|---:|---:|---:|---:|---:|
| 1 | 24.021 | 41.63 | crash | - |
| 2 | 45.854 | 21.81 | crash | - |
| 3 | - | - | 113.208 | 8.83 |
| 4 | 79.936 | 12.51 | 131.004 | 7.63 |
| 6 | 98.931 | 10.11 | 156.251 | 6.40 |
| 12 | 118.864 | 8.41 | 201.340 | 4.97 |

Encoder speedup at 12 workers is 4.95x over one worker (41% parallel
efficiency). Decoder speedup from 3 to 12 workers is 1.78x versus an ideal 4x.
Both codecs obtain diminishing returns beyond the six physical cores.

## Aggregate worker-stage timing

These percentages are medians of five runs. They are aggregate worker-thread
time, so parallel stages overlap in wall-clock time.

### Encoder

| Stage | Worker time |
|---|---:|
| Quantizer selection + entropy coding | 78.11% |
| Slice serialization | 11.69% |
| Input conversion + horizontal level 0 | 9.08% |
| Horizontal wavelet levels 1+ | 0.89% |
| Vertical wavelet | 0.44% |

The fastest preset already selects `QUANTISER_SELECTION_EIGHTHSEARCH` and a
single allocation pass. The transform dispatcher selects AVX2, but quantizer
selection and slice entropy encoding are scalar/template code. This makes the
quantizer-length estimation and entropy component loop the primary encoder
optimization target, not the wavelet transform.

### Decoder

| Stage | Worker time |
|---|---:|
| Entropy decode + dequantization | 65.07% |
| Inverse vertical transform | 16.50% |
| Final horizontal transform + output store | 14.63% |
| Earlier inverse horizontal levels | 3.89% |

The decoder reports AVX2 support, but its runtime dispatcher has only C and
SSE4.2 implementations. Entropy decoding, dequantization, inverse transforms,
and the final output path therefore use the SSE4.2 family. An AVX2 entropy /
dequantization path is the highest-value decoder experiment.

## Transform sweep

Single-run encoder screening at otherwise identical fastest settings:

| Wavelet | fps |
|---|---:|
| Haar0 | 121.724 |
| Haar1 | 120.983 |
| LeGall 5/3 | 104.992 |
| Deslauriers-Dubuc 9/7 | 61.201 |
| Deslauriers-Dubuc 13/7 | 41.380 |

Depth 3 was faster than depth 1 or 2 for Haar0 in this implementation, and the
default 32x8 slices were faster than the other valid slice sizes tested.

## Limitations and next profiling step

Windows Performance Recorder, Analyzer, Xperf, and the Intel PMU sources are
installed. Kernel stack/PMU collection was rejected because the current Codex
process is not elevated. The source-level stage profiler is enabled only when
the `VC2HQ_PROFILE` environment variable is present.

An elevated WPR/VTune run should split the combined encoder and decoder hot
stages into concrete functions and provide IPC, LLC misses, branch
mispredictions, and Intel top-down pipeline metrics.

The decoder also has an independent correctness defect: one- and two-worker
configurations terminate with integer divide-by-zero (`0xC0000094`).
