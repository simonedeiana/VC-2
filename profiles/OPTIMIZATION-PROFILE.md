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

---

# Round 3 — interleaved multi-stream VLC decode

Date: 2026-08-13

## Result

| Codec | Baseline median | Optimized median | Improvement |
|---|---:|---:|---:|
| Decoder (single-thread, 120f) | 66.04 fps | 73.45 fps | **+11.2%** |

Baseline = `bench/baseline/vc2decode.exe` (the round-2 state, saved before
the VLC changes), pinned to one CPU, AboveNormal priority, nine alternating
120-frame runs, medians. Byte-identical output (SHA-256 `56e5c487...`).

## What was done

The VLC stage is ~66% of decode time and is **chain-latency bound**: a
discriminator that stripped the SIMD payload from the decode loop showed the
serial LUT-latency chain + scalar control is ~76% of the loop, the SIMD
payload ~24%. The fix hides the serial chain by decoding independent streams
in one interleaved loop.

- `vlc_step_ex` / `vlc_tail_ex` in `vlc_sse4_2.cpp`: one decode step with the
  per-stream state passed by reference as scalars (`ic`, `oc`, `V`, `next`) so
  the compiler keeps it in registers.
- `decode_sse4_2_x3`: decodes a slice's three components (Y/C1/C2) together —
  a 3-way interleaved loop (then 2-way, then 1-way drains, then tails).
- The `decode_slices_sse4_2<T>` driver pairs adjacent slices and keeps each
  decode→dequant pair adjacent so the three scratch buffers are never reused
  before their dequant (fixes an earlier clobbering bug).
- Stage profile (30-frame): vlc-decode 355 ms → ~320 ms at the same total
  frame count, now 67% of a faster total.

## Rejected experiments

- **Struct-based interleaved state** (`VLCState`): only +3.3%. The 8-field
  struct spilled to the stack; passing state by reference as scalars kept it
  in GPRs and delivered the +11.2% above.
- **AVX2 dequantise** (8-wide `abs/mul/add/shift/sign` for the 32x8x3 and
  16x8x3 paths): **-5.2% regression** in a same-session interleaved A/B
  (67.95 vs 71.68 fps), despite the dequant stage itself profiling faster
  (57.5 vs 72.5 ms). Likely AVX downclocking on this Broadwell-EP Xeon hurting
  the latency-bound VLC loop. Reverted.
- **Single-job decode grid** (`n_jobs = 1` for `--threads=1`): +4.6% in
  same-session A/B, but it changed decoded pixels for the
  deslauriers-debuc-9-7 wavelet (~9 kbytes near the 4-job grid boundary).
  The 1-job path is only compiled with `DEBUG_ONE_JOB` (the `debug.hpp`
  include sits inside `#ifdef DEBUG`, so it never reaches Release builds);
  it could not be validated byte-exact, so it was reverted.
- **Matched-length stream grouping** (decode three adjacent slices' same
  component together — `x3(Y_a,Y_b,Y_c)`, `x3(C1_a,C1_b,C1_c)`, ...): the
  literature-driven idea that grouping same-length streams maximises the
  interleaved window. The per-slice `x3(Y,C1,C2)` interleaves Y (256 coeffs)
  with the shorter chroma (128 each), so its 3-way loop covers only the first
  half and the Y tail decodes solo. Grouping by component removes that drain,
  but it scatters the input reads: the bitstream is slice-major, so three
  slices' Y streams are a full slice apart and defeat the sequential
  prefetcher. **-3.5% regression** in a same-session A/B (63.8 vs 61.5 fps).
  Reverted; the per-slice grouping keeps the contiguous Y/C1/C2 input, which
  wins over the drain elimination. (Required sizing all scratch buffers to
  the largest component; also reverted.)

## Verification

- Decoder pixels SHA-256 identical (`56e5c487...`) for haar0, haar1, LeGall,
  and deslauriers-debuc-9-7 streams vs `bench/baseline`.
- Six of six native CTest targets passed; conformance validator clean.
- Encoder untouched (still `4e45ed1c...` at the same config).

---

# Round 4 — single-thread pointer-alias refinement

Date: 2026-08-13

## Result

The retained change adds `__restrict` qualifiers to the input and output
buffers of the inlined `vlc_step_ex` SSE4.2 decoder step. This gives MSVC
permission to assume that the compressed-byte input and coefficient output do
not alias while it schedules the LUT/state chain and SIMD stores.

In the fixed 1920x1080, Haar0, 30-frame, one-worker benchmark, the decoder
measured 68.962 fps over 15 runs with the change. The no-change comparison
measured 67.874 fps over 9 runs. The samples are from separate runs, so this
is recorded as a small observed improvement rather than a strict alternating
A/B claim. The encoder annotation trial was removed because it did not produce
a stable gain.

The final stage profile remains VLC-dominated:

| Stage | Worker time |
|---|---:|
| vlc-decode | 60.38% |
| dequantise | 12.60% |
| inverse-vertical | 6.62% |
| inverse-horizontal | 4.57% |
| final-horizontal+output | 15.83% |

## Verification

- Six of six native CTest targets passed after the final build.
- A fresh one-frame stream passed the installed VC-2 bitstream validator with
  `No errors found in bitstream`.
- The encoder stream remained byte-identical to the pre-change stream
  (`D5DD1542CAC95E70AC273321B6CE685C76EEC9B0F5F6904B1DD40794B1D08709`).

---

# Round 5 — DD9/7 and DD13/7 transform SIMD

Date: 2026-08-13

## Coverage

The supported encoder wavelets were swept at one worker on the fixed
1920x1080, 10-bit 4:2:2, Haar0-style 30-frame workload. Fidelity is exposed
by the command-line help but is rejected by the current encoder dispatch as an
invalid wavelet, so it has no valid encoder benchmark. The scalar DD9/7 and
DD13/7 inverse paths were the clear transform-specific bottleneck.

Their initial stage profiles put inverse vertical lifting at 43.01% and
43.27% of decoder worker time respectively. The existing implementation was
scalar C++ for these filters, unlike the already-vectorized Haar and LeGall
paths.

## Retained change

- Added SSE4.2 four-column kernels for the finest-level DD9/7 and DD13/7
  inverse vertical transforms. Each SIMD lane is an independent image column;
  the scalar lifting recurrence is preserved within each lane.
- Kept the existing scalar implementation as the fallback for short or
  non-four-column geometries.
- Left final horizontal/output scalar. A row-interleaved DD9 trial was
  byte-identical but regressed the decoder from 38.3 fps to 35.6 fps because
  its strided row loads and stores outweighed the vector arithmetic.

The corresponding DD13/7 encoder vertical-kernel trial was rejected. Nine
alternating runs measured 19.019 fps for the scalar dispatch and 18.642 fps
for the SIMD dispatch, a 1.98% regression. The trial stream was nevertheless
byte-identical to the scalar stream, so no encoder dispatch change was kept.

## Same-session A/B results

Nine alternating 30-frame runs, scalar dispatch versus SIMD dispatch:

| Transform | Scalar | SIMD | Improvement |
|---|---:|---:|---:|
| DD9/7 decoder | 27.985 fps | 33.861 fps | **+21.00%** |
| DD13/7 decoder | 24.834 fps | 29.970 fps | **+20.68%** |

The final profile still shows final horizontal/output as the next major
transform cost (about 34–35%), followed by inverse vertical at about 29–30%.

## Verification

- Six of six native CTest targets passed after the retained kernels were
  restored.
- Scalar and SIMD DD9/7 decoder output hashes matched:
  `45B4DA1EBC8559B47223DF2084433BFBAEC53A1C96E1D9377D8F9680BC9B5277`.
- Scalar and SIMD DD13/7 decoder output hashes matched:
  `C08EB0D68953FDD1AE4DB36362E538981B77695D73BE2DBBF834737CABEB3AC0`.
- Fresh DD9/7 and DD13/7 one-frame streams were accepted by the installed
  validator; no bitstream errors were reported.

## Research follow-up

The implementation follows the established vertical-lane strategy described
in SIMD lifting literature: vectorize independent columns while keeping the
lifting dependency chain within each lane. The 2-D lifting literature also
emphasizes cache-aware treatment of vertical versus horizontal filtering; that
matches the current profile, where horizontal output is now the limiting
stage. See [Vectorization of the 2D Wavelet Lifting Transform Using SIMD
Extensions](https://www.researchgate.net/publication/220951124_Vectorization_of_the_2D_Wavelet_Lifting_Transform_Using_SIMD_Extensions),
[A Single-Loop Approach to SIMD Parallelization of 2-D Wavelet
Lifting](https://www.researchgate.net/publication/221392398_A_Single-Loop_Approach_to_SIMD_Parallelization_of_2-D_Wavelet_Lifting),
and the [JPEG 2000 lifting-transform specification](https://www.itu.int/epublications/publication/itu-t-t-801-v3-2023-08-08-jpeg-2000-image-coding-system-extensions).

