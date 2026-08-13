# VC-2 HQ fast-path optimization profile

Date: 2026-08-12

## Workload

- Intel Xeon E5-1650 v4, 6 physical cores / 12 logical processors
- MSVC 19.44 Release build
- 1920x1080 planar 10-bit 4:2:2
- 2:1 compression, Haar0, depth 3, 32x8 slices, 12 workers
- Nine alternating baseline/optimized runs of 120 frames
- Output writes disabled for timing
- Baseline built from commit `3d24abf` in an isolated worktree

The single-thread follow-up used nine alternating runs of 30 frames. The
baseline decoder included only the `JobConfig` correctness fix described
below, since the untouched optimized MSVC build divides by zero when
configured with one worker.

## Results

| Codec | Baseline median | Optimized median | Latency per frame | Improvement |
|---|---:|---:|---:|---:|
| Encoder | 124.220 fps | 159.814 fps | 8.05 ms -> 6.26 ms | 28.6% |
| Decoder | 242.921 fps | 255.325 fps | 4.12 ms -> 3.92 ms | 5.1% |

### Single-thread result

| Codec | Baseline median | Optimized median | Latency per frame | Improvement |
|---|---:|---:|---:|---:|
| Encoder | 19.698 fps | 29.320 fps | 50.77 ms -> 34.11 ms | 48.8% |
| Decoder | 40.268 fps | 50.673 fps | 24.83 ms -> 19.73 ms | 25.8% |

These figures use one codec worker throughout; no frame, slice, or stage was
split across additional threads.

The encoder output was byte-for-byte identical to the untouched baseline
(`765917f6d4f21e7d7588ad581e25fc94c8dd35992a6f1ce1022553f4ac7243c5`).
The decoder output was also byte-for-byte identical
(`48a27492179d8c7181fd99508cc1b9fd902d6859b9f39bbae10e3a0346e85d46`).

## Bottlenecks and changes

### Encoder

The initial stage profile attributed 78.11% of aggregate worker time to
quantizer selection and entropy coding. The fastest search estimated the
coded length at the selected quantizer and then immediately repeated the same
quantization and VLC work to produce the codewords.

The eighth-search path now emits codewords during its final size trial and the
encoder reuses that result. The coefficient safety scan and all size checks
remain intact. This removes one full quantize/VLC pass per slice without
changing the selected quantizer or bitstream.

After the change, the combined quantize/entropy stage accounted for 70.58% of
aggregate worker time. It remains the main encoder target; slice serialization
is next at 15.68%.

### Decoder

The initial profile attributed 65.07% of worker time to entropy decoding and
dequantization. The SSE4.2 VLC loop extracted six scalar fields from its SIMD
register for every input byte. It now reads control fields directly from the
already-hot LUT entry while retaining SIMD coefficient stores.

The decoder also used to issue 256 prefetch instructions for the VLC table at
the start of every small decode job. With 64 jobs per frame this repeatedly
prefetched a table that was already resident; those job-level prefetch sweeps
were removed. A dropped `_mm_insert_epi32` return value in truncated-input
handling was corrected at the same time.

Entropy/dequantization remains the largest decoder stage at 66.42%. A future
larger gain requires a true AVX2 VLC/dequantization implementation or a revised
job pipeline, rather than more wavelet work.

The original 15-argument inline `JobData` constructor was also refactored to
accept a `JobConfig` structure. Optimized MSVC passed its final `sample_size`
argument as zero in the one-worker configuration, leading to an integer
divide-by-zero while constructing the last job. Passing one configuration
object fixes the ABI/code-generation edge and makes single-thread decoding
reliable.

## Single-thread stage profile

With the retained changes, a 30-frame run attributed encoder worker time as
follows: quantization/entropy 75.94%, serialization 16.33%, input plus first
horizontal transform 5.24%, and remaining transforms 2.48%. Decoder worker
time was entropy/dequantization 64.12%, inverse vertical transform 17.79%,
final horizontal/output 13.74%, and intermediate horizontal transform 4.34%.

## Rejected experiments

- Removing the remaining per-component decoder prefetches was neutral
  (52.265 vs 52.448 fps, +0.35%, within run-to-run noise).
- MSVC whole-program optimization regressed encoder throughput by 7.7% and
  decoder throughput by 3.4%.
- Replacing the generated 32x8 encoder specialization with the compact generic
  loop regressed encoder throughput by 8.2%.
- MSVC `/favor:INTEL64` was neutral for the encoder and regressed the decoder
  by 6.0% on the Xeon E5-1650 v4.

The remaining hotspots are serial, data-dependent VLC/quantizer work. Further
material gains would require a substantially new fused or AVX2 entropy path,
with a much larger correctness and maintenance burden than the retained
changes.

## Verification

- Six of six native CTest targets passed.
- The VC-2 conformance bitstream validator reported no errors.
- Optimized encoder bitstream matched the baseline SHA-256 exactly.
- Optimized decoder pixels matched the baseline SHA-256 exactly.
- Single-thread encoder stream matched the corrected baseline SHA-256 exactly
  (`4f98b0449d27c18a8cbd5c45e66e8091a270e9e0fd7855cf54d53686b2bfbf9b`).
- Single-thread decoder pixels matched the corrected baseline SHA-256 exactly
  (`5be3ee7ae2f03cb5739f8f3342c3cdf41dd7e2f060eb04a8f9c3044b355c07e7`).

---

# Round 2: single-thread focus — profiling tooling + decoder SIMD

Date: 2026-08-13

## Workload and methodology

- Same CPU/compiler as round 1, but single-thread focus throughout
  (`--threads=1`).
- Benchmark harness `bench/bench.ps1` pins the process to one logical CPU
  (affinity `0x100`) at AboveNormal priority and reports medians of 7-9
  interleaved runs, which cut run-to-run variance from ~25% to ~2%.
- Reference baseline = committed `9b374b5` built in an isolated worktree.

## Results (single thread, medians of interleaved runs)

| Codec | Baseline median | Optimized median | Improvement |
|---|---:|---:|---:|
| Encoder | 34.98 fps | 34.78 fps | ~0% (neutral) |
| Decoder | 65.44 fps | 76.72 fps | **+17.2%** |

Decoder latency per frame fell from 15.28 ms to 13.03 ms. The encoder changes
are retained only where they are byte-identical and non-harmful; two
experiments regressed and were reverted (see below).

## Profiling tooling

- Fine-grained `VC2HQ_PROFILE` stage scopes: the encoder `quantise+entropy`
  stage now splits into `quantiser-search` vs `quantise+encode`, and the
  decoder `entropy+dequantise` splits into `vlc-decode` vs `dequantise`
  (scopes moved inside `encode_slices` / `decode_slices_*`).
- `stage_profile::enabled()` now reads a plain namespace-scope flag instead of
  a function-local static. MSVC's thread-safe static-init guard
  (`_Init_thread_header`) was showing up as ~2% of VTune samples on millions
  of hot-path calls.
- Intel VTune software sampling (`sampling-mode=sw`) works and produced
  symbolised function hotspots. Hardware sampling is unavailable: the current
  VTune cannot recognise the Xeon E5-1650 v4 (Broadwell) PMU.

## Decoder changes (retained)

The SSE4.2 inverse-transform dispatcher had **no Haar kernels for 16-bit
samples** (10-bit streams), so the benchmark's inverse-vertical and
inverse-horizontal stages ran fully scalar. Added:

- `Haar_invtransform_V_inplace_sse4_2_int16_t<skip>` for skip 1/2/4/8:
  elementwise 8-wide processing with lane blending to preserve the untouched
  horizontal subband columns (`0x55`/`0x11`/`0x01` masks).
- `Haar_invtransform_H_inplace_1_sse4_2_int16_t<shift>` (skip 1): even/odd
  de-interleave, filter, interleave back.
- `Haar_invtransform_H_inplace_sse4_2_int16_t<skip, shift>` for skip 2/4:
  shuffle-extract the strided pairs, filter, spread back, lane-blend.
  (skip 8 pairs span two 128-bit vectors, left on the C path.)
- Dispatch entries in `invtransform_sse4_2.cpp` for both `HAAR_NO_SHIFT` and
  `HAAR_SINGLE_SHIFT` at sample size 2.

Stage shift (30-frame profile, `VC2HQ_PROFILE=1`):

| Stage | Before | After |
|---|---:|---:|
| vlc-decode | 57.0% (314 ms) | 64.8% (299 ms) |
| dequantise | 11.7% (65 ms) | 13.7% (63 ms) |
| inverse-vertical | 18.3% (101 ms) | 6.9% (32 ms) |
| inverse-horizontal | 4.7% (26 ms) | 4.9% (22 ms) |
| final-horizontal+output | 8.3% (46 ms) | 9.8% (45 ms) |

The inverse-vertical stage is ~3x faster; VLC is now the sole dominant stage.

## Encoder changes (retained)

- Merged `CWLUT` + `WLLUT` into one 32-bit table (`CLWLUT` in `lut.hpp`);
  `encode_sample` does one LUT load instead of two. Byte-identical, ~neutral.

## Rejected experiments

- **EIGHTHSEARCH length-only trials** (est. length during search, one final
  encode): regressed ~30% because the common case is a single trial, so the
  old "emit during the final trial and reuse" scheme already does exactly one
  full encode; the new scheme added a second pass.
- **Batched byte-flush bit-packer** in `serialise_slices`: regressed ~7%.
  The data-dependent `if (bits >= 8)` branches cost more than the 4-byte
  per-codeword stores save (the store buffer absorbs the overwrites).
- **Samples-scan** (separate last-nonzero pass to break the serial max-chain):
  reverted — measurements were contradictory (stage profile ±3%, fps A/B
  noisy); the added pass did not robustly pay for itself.

## Verification (against `9b374b5` worktree build)

- Encoder stream SHA-256 identical (`4e45ed1c...`) across haar0, haar1,
  LeGall 5/3, and DD 9-7.
- Decoder pixels SHA-256 identical (`56e5c487...`) for Haar and LeGall
  streams.
- Six of six native CTest targets passed; conformance validator clean.

