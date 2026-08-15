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


---

# Round 6 � DD9/7 and DD13/7 final horizontal SIMD (rejected)

Date: 2026-08-13

## Coverage

Continuation of the Round 5 transform work. Stage profiling put the
final-horizontal+output stage at 34.49% (DD9/7) and ~35% (DD13/7) of decoder
worker time, making it the largest transform cost. The goal was to vectorize
the scalar final horizontal recurrence.

## Analysis

The scalar final-H looks like a serial recurrence along x (state carried
between iterations), but every output depends only on loaded samples:

- DD9/7: even `out[p] = D[p] = X[p] - ((X[p-1]+X[p+1]+2)>>2)`;
  odd `out[p] = X[p] + ((-D[p-3]+9*D[p-1]+9*D[p+1]-D[p+3]+8)>>4)`.
- DD13/7: same odd formula, with predict
  `D[p] = X[p] - ((-X[p-3]+9*X[p-1]+9*X[p+1]-X[p+3]+16)>>5)`.

It is therefore a parallel column stencil, vectorizable with contiguous
loads/stores (unlike the rejected Round 5 row-interleaved trial). SSE4.2
kernels were written that compute four output columns per block from two
(DD9) or three (DD13) 16-byte window loads, extracting the strided even/odd
operand lanes with byte shuffles, and keeping D in int32. Boundary columns
reproduced the exact scalar mirrors (traced from the C tail: left
X[-1]->X[1], X[-3]->X[3]; right DD9 D[W]->D[W-2], D[W+2]->D[W-4]; DD13
X[W+1]->X[W-1], D[W]->D[W-2], D[W+2]->D[W-4]).

## Result (rejected)

The kernels were byte-identical (SHA-256 matched the recorded hashes; the
full inverse-transform test suite passed for active_bits 10 and 12 across all
crop offsets), but they were SLOWER than the existing scalar code:

| Kernel | Scalar | SSE4.2 (new) |
|---|---:|---:|
| DD9/7 final-H | 6.91 ms/frame | 8.15 ms/frame |
| DD13/7 final-H | 7.41 ms/frame | 10.55 ms/frame |

Pinned, interleaved microbenchmark on 1920x1080 int16 data. The full decoder
benchmark was unchanged (39.983 fps vs 39.998 fps baseline) and the stage
profile was unchanged (34.23% vs 34.49%), confirming the stage is
memory-bound in the real decode (reads the coefficient plane, writes the
uint16 output plane), so compute SIMD cannot help.

Root cause: Haswell executes shuffles and `cvtepi16_epi32` widenings on a
single port (p5); the extraction-heavy kernel needs ~1-2 p5 operations per
output, whereas the MSVC-generated scalar is already well scheduled and the
real decode is store-limited. The compact de-interleave alternative would
still be p5-limited and cannot beat the memory wall.

## Actions

- Reverted the final-H kernels, dispatch, and test additions (working tree
  restored to Round 5 commit `7a55308`; encoder `__restrict` VLC work in the
  working tree was untouched).
- Verified hashes still match after revert: DD9
  `45B4DA1EBC8559B47223DF2084433BFBAEC53A1C96E1D9377D8F9680BC9B5277`, DD13
  `C08EB0D68953FDD1AE4DB36362E538981B77695D73BE2DBBF834737CABEB3AC0`.

## Research follow-up

The negative result isolates the boundary between compute-bound and
memory-bound stages on this CPU. The DD final-H is memory-bound, so the next
transform target should be the inverse-vertical stage (~29% of DD9/7 decoder
time, already SSE4.2; an AVX2 eight-column variant would halve loop/state
overhead) or the VLC decode stage (24% for DD9/7, 61% for Haar0), not the
final-H arithmetic.

---

# Round 7 � DD9/7 and DD13/7 AVX2 inverse vertical

Date: 2026-08-13

## Coverage

Round 5 left the DD9/7 and DD13/7 finest-level inverse vertical transforms as
SSE4.2 four-column kernels (one SIMD lane per image column, int16 -> int32
widening). The inverse-vertical stage was still 29.25% (DD9/7) of decoder
worker time, and it was the last compute-bound transform stage (Round 6
showed the final horizontal stage is memory-bound, so SIMD cannot help
there).

## Retained change

- Added an AVX2 eight-column variant of the DD9/7 and DD13/7 finest-level
  inverse vertical kernels in a new `vc2invtransform_avx2` static library
  (compiled with `/arch:AVX2`, mirroring the encoder's `vc2transform_avx2`).
  The lifting structure is identical to the SSE4.2 kernels; only the width
  doubles (8 lanes per iteration, `_mm256_cvtepi16_epi32` loads and
  truncating `shuffle_epi8` + `unpacklo_epi64` stores).
- New `get_invvtransform_avx2` dispatch returns the AVX2 kernels for the DD
  finest level and otherwise falls back to `get_invvtransform_sse4_2`.
- The decoder selects the AVX2 dispatch when runtime detection reports AVX2
  (checked after the SSE4.2 block, so AVX2 wins only for the transforms that
  have AVX2 kernels).
- Added DD9/7 and DD13/7 vertical test rows (with an AVX2 check) to the
  inverse-transform test suite.

## Same-session A/B results

Pinned, interleaved microbenchmark on 1920x1080 int16 data (scalar vs SSE4.2
vs AVX2 full-plane finest-level V):

| Kernel | Scalar | SSE4.2 | AVX2 |
|---|---:|---:|---:|
| DD9/7 V | 4.25 ms/frame | 1.42 ms/frame | 0.76 ms/frame |
| DD13/7 V | 4.61 ms/frame | 1.62 ms/frame | 0.83 ms/frame |

Full decoder (seven pinned runs, median):

| Transform | Before (SSE4.2 V) | After (AVX2 V) | Improvement |
|---|---:|---:|---:|
| DD9/7 decoder | 39.998 fps | 41.377 fps | **+3.45%** |
| DD13/7 decoder | 37.521 fps | 39.130 fps | **+4.29%** |

The inverse-vertical stage profile dropped from 29.25% to 25.10% of decoder
worker time (DD9/7). The final-horizontal+output stage is now the largest
(36.55%) but is memory-bound (Round 6).

## Verification

- Six of six native CTest targets passed (including the new AVX2 DD V rows).
- Decoder output hashes unchanged:
  DD9 `45B4DA1EBC8559B47223DF2084433BFBAEC53A1C96E1D9377D8F9680BC9B5277`,
  DD13 `C08EB0D68953FDD1AE4DB36362E538981B77695D73BE2DBBF834737CABEB3AC0`.
- Fallback geometry (width not a multiple of 8) still routes to the scalar
  implementation.

## Research follow-up

The 8-column AVX2 kernel roughly halves the loop/state overhead of the
4-column SSE4.2 kernel and doubles the arithmetic width; on this Haswell CPU
the measured 1.9-2x kernel speedup confirms the vertical lifting recurrence
is compute-bound rather than port-bound at 8 lanes. The natural next targets
are the remaining compute-heavy stages: VLC decode (24% DD9/7, 61% Haar0) and
the coarser-level inverse vertical transforms (still scalar int16).

---

# Round 8 � Parallel DD9/7 and DD13/7 final horizontal + streaming stores

Date: 2026-08-14

## Coverage

Round 6 had rejected SIMD for the Deslauriers-Dubuc final horizontal stage
because the shuffle-heavy SSE4.2 attempt was slower than scalar, and the stage
appeared memory-bound. Re-investigating with the same benchmark/profile/optimize
cycle on the `bold-experiments` branch showed the stage was actually
compute-bound on the serial lifting recurrence (7.69 ms/frame cached, vs a
0.31 ms store floor), and that the earlier failure was the per-operand shuffle
extraction, not the idea of parallelizing.

## Retained changes

- **AVX2 two-pass DD9/7 and DD13/7 final horizontal kernels.** The scalar
  recurrence is a parallel column stencil. A two-pass-per-row scheme breaks
  the serial dependency without shuffle extraction:
  - pass 1 de-interleaves 8 int16 samples into even/odd, widens to int32,
    computes the compact even D vector, and stores it to a small stack
    scratch buffer (thread-safe, L1-resident);
  - pass 2 reads D contiguously, forms the four sliding update windows with
    two 128-bit loads and `alignr`, and writes 8 uint16 outputs with a
    non-temporal store.
  The 13/7 predict (5-tap) carries two odd samples and one look-ahead sample
  per pass-1 block; the update/output half is shared with the 9/7 version.
  Boundary mirrors (left X[-1]->X[1], X[-3]->X[3]; right D[W]->D[W-2],
  D[W+2]->D[W-4], X[W+1]->X[W-1]) are handled by padding the compact D
  buffer. The kernel supports the decoder's 32-pixel slice-overlap crop
  (computes the full input width, stores only the valid output range).
- **Non-temporal output stores in the Haar final horizontal kernel** (guarded
  by 16-byte alignment). The output plane is write-only, so streaming stores
  avoid the read-for-ownership traffic of a normal write-allocate store.
- Added `#pragma once` to the two scalar DD transform headers (they lacked
  include guards, which broke the new AVX2 headers that include them).

## Same-session A/B results

Pinned interleaved microbenchmark, 1920x1080 int16 (cached):

| Kernel | Scalar | AVX2 (new) |
|---|---:|---:|
| DD9/7 final-H | 7.69 ms/frame | 1.49 ms/frame |
| DD13/7 final-H | ~7.4 ms/frame | ~1.5 ms/frame |

Full decoder (eleven pinned runs, median):

| Workload | Before | After | Improvement |
|---|---:|---:|---:|
| Haar0 decoder | 73.353 fps | 76.566 fps | **+4.4%** |
| DD9/7 decoder | 41.866 fps | 52.167 fps | **+24.6%** |
| DD13/7 decoder | 39.135 fps | 48.781 fps | **+24.7%** |

The DD9/7 stage profile shows final-horizontal+output falling from 36.38% to
21.00% of decoder worker time; VLC decode (31.67%) and inverse vertical
(31.30%) are now the two largest stages.

## Verification

- Six of six native CTest targets passed.
- Decoder output hashes unchanged:
  DD9 `45B4DA1EBC8559B47223DF2084433BFBAEC53A1C96E1D9377D8F9680BC9B5277`,
  DD13 `C08EB0D68953FDD1AE4DB36362E538981B77695D73BE2DBBF834737CABEB3AC0`.
- Crop, short, and misaligned geometries fall back to the scalar
  implementation unchanged.

## Research follow-up

The two-pass "compute compact detail coefficients, then sweep with a sliding
window" pattern avoids both the serial recurrence and the shuffle extraction
that defeated earlier attempts; it generalizes to any separable lifting
stencil. The next largest targets are VLC decode (now the top DD9/7 stage at
~32%, and ~62% for Haar0) and the still-scalar coarser-level inverse vertical
transforms.

---

# Round 9 � AVX2 vectorized quantize + VLC lookup in the encoder

Date: 2026-08-14

## Coverage

The encoder was dominated by the quantiser-search stage (73.46% of worker
time), which for EIGHTHSEARCH is the full quantize+VLC of each slice
(`encode_slice_component<32,8,3,int16_t>` emits codewords during the search's
final size trial and reuses them, so the search is the encode). The per-sample
`encode_sample` does a reciprocal-multiply quantisation, a merged
codeword/length table lookup (`CLWLUT`), two scattered stores (codeword and
wordlength, in bitstream scan order), and a last-non-zero tracking branch.

## Retained change

- **AVX2 vectorization of the quantisation and CLWLUT lookup** in a new
  `encode_slice_component_32x8x3_avx2` path, selected at runtime when AVX2 is
  present, with the existing hand-unrolled scalar specialization as the
  fallback. Eight samples per iteration: `abs`, `mulhi` (16x16->32 high half,
  replacing the scalar reciprocal division), variable shifts (`srlv`), and an
  AVX2 gather over `CLWLUT` produce the codeword and length vectors. The
  codeword/wordlength stores stay scalar because the bitstream scan order
  scatters them; the last-non-zero tracking is done branchlessly with a
  horizontal max over the scan positions.
- Added a cached runtime AVX2 check (`encode_has_avx2`, CPUID leaf 7) local
  to the header.
- The scan-order permutation table was extracted programmatically from the
  hand-unrolled scalar specialization.

## Same-session A/B results

Pinned microbenchmark of the quantize+lookup kernel (8 samples/iter):

| Kernel | Scalar | AVX2 |
|---|---:|---:|
| quantize+lookup | 2.19 ns/sample | 0.61 ns/sample |

Full encoder (eleven pinned runs, median, 60 frames):

| Workload | Before | After | Improvement |
|---|---:|---:|---:|
| Encoder (Haar0) | 35.147 fps | 40.789 fps | **+16.1%** |

The quantiser-search stage dropped from 73.46% to 69.79% of encoder worker
time; serialise is now the second-largest stage (18.61%).

## Verification

- Six of six native CTest targets passed.
- Encoder stream byte-identical: the AVX2 and forced-scalar builds produced
  the same 1-frame stream SHA-256
  (`E27A23277771A6E294F91042C1011B24EFFA1EA044AA6CD62F0827CB04E46744`).
- Round-trip verified: the AVX2-encoded stream decodes successfully.

## Research follow-up

The remaining encoder cost is the scattered codeword/wordlength stores (scan
order) and the serialise bit-packer (18.61%). A future step could fuse the
bit-packing into the encode to eliminate the intermediate arrays and the
separate serialise pass, or reorder the coefficient storage so the scan order
is contiguous. The vectorized-gather pattern (mulhi + srlv + i32gather) is the
same one that made the decoder VLC lookup fast.

---

# Round 10 � AVX2 DD9/7 and DD13/7 forward vertical transforms (encoder)

Date: 2026-08-14

## Coverage

For Deslauriers-Dubuc content the encoder's vertical-wavelet stage was the
largest transform cost (32.37% of DD9/7 worker time) and was entirely scalar:
the encoder's AVX2 dispatch covered Haar and LeGall only. The DD forward
vertical lifting is the same per-column recurrence as the inverse (predict
`(a+b+2)>>2`, update `(-a+9b+9c-d+8)>>4`, plus the 13/7 five-tap predict),
just applied in the forward order, so the same 8-column SIMD strategy that
accelerated the decoder's inverse vertical applies.

## Retained change

- Added AVX2 eight-column forward vertical kernels for DD9/7 and DD13/7 in
  the encoder's `vc2transform_avx2` library (level 0 / skip 1, int16 -> int32
  widening, one SIMD lane per image column). The scalar lifting recurrence is
  preserved within each lane; the prologue/main/tail structure mirrors the
  scalar reference exactly. Short, unaligned, or coarser-level geometries
  fall back to the scalar template.
- Wired the two kernels into `get_vtransform_avx2` (runtime AVX2 dispatch,
  which the encoder already selects when AVX2 is present).

## Same-session A/B results

Pinned 60-frame medians, scalar dispatch vs AVX2 dispatch:

| Workload | Scalar | AVX2 | Improvement |
|---|---:|---:|---:|
| DD9/7 encoder | 25.63 fps | 30.69 fps | **+19.7%** |
| DD13/7 encoder | 23.23 fps | 28.14 fps | **+21.1%** |

The DD9/7 vertical-wavelet stage dropped from 32.37% to 14.31% of encoder
worker time. The Haar0 encoder is unaffected (41.24 fps).

## Verification

- Six of six native CTest targets passed.
- DD9/7 and DD13/7 encoder streams byte-identical between the AVX2 and
  forced-scalar builds:
  DD9 `DF99249F127BFA10C2407448165C92409F29811F5EB14DB0F129FE2418D3F08F`,
  DD13 `FF7EA76F32BD043F83A31A0E3A1EE5F1AA63D0519E033D37A5670EC481432F62`.

## Research follow-up

The encoder's DD forward horizontal transforms (input+horizontal-L0 at 18.22%
and horizontal-wavelet-L1+ at 5.31%) remain scalar and are the next
transform target, followed by the coarser-level forward vertical levels
(skip 2/4/8). The serialise bit-packer (13.15%) is the remaining
non-transform cost.

---

# Round 11 — encoder DD forward HORIZONTAL input transform (AVX2)

## Analysis

After the forward vertical kernels (Round 10), the DD9/7 encoder's remaining
transform cost was dominated by `input+horizontal-L0` (18.22%) — the initial
horizontal lifting applied to the raw 10P2 input row before the vertical
wavelet. That stage is the `Deslauriers_Dubuc_*_transform_H_inplace_10P2`
template, a serial lifting recurrence along each row.

The scalar recurrence is a parallel column stencil, so the same two-pass
approach used for the decoder's inverse final-H applies here:

1. de-interleave the uint16 row into even/odd int32 scratch, converting each
   sample with `(v-512)<<1`;
2. compute the odd outputs `O[j] = odd - update(even[j-1..j+2])` from compact
   even reads (sliding `alignr` windows);
3. compute the even outputs `E[j] = even + predict(O[j-1..O[j+1])`,
   interleave E/O and store as int16.

DD9/7 uses a 2-tap predict `(a+b+2)>>2`; DD13/7 uses the 5-tap
`(-a+9b+9c-d+16)>>5`. The mirror boundary pads (traced from the scalar
prologue/tail) are filled into the scratch arrays before passes 2 and 3, so
the inner loops have no branches.

## Retained change

- Added `Deslauriers_Dubuc_9_7_transform_H_inplace_10P2_avx2` and
  `Deslauriers_Dubuc_13_7_transform_H_inplace_10P2_avx2` in the encoder's
  `vc2transform_avx2` library, wired into `get_htransforminitial_avx2` for
  the 10P2/int16 dispatch. Short (<16), non-multiple-of-8, or short-height
  geometries fall back to the scalar template. The overlap mirror-fill and
  bottom-row copy stay scalar (identical to the reference).

## Same-session A/B results

Pinned 60-frame medians, before vs after (both builds already include the
Round 10 forward vertical kernels):

| Workload | Before (V only) | After (V+H) | Improvement |
|---|---:|---:|---:|
| DD9/7 encoder | 30.69 fps | 31.84 fps | **+3.7%** |
| DD13/7 encoder | 28.14 fps | 29.93 fps | **+6.4%** |

The DD9/7 input+horizontal-L0 stage dropped from 18.22% to 12.80% of encoder
worker time. Cumulative encoder gains (scalar baseline -> now): DD9/7
25.63 -> 31.84 fps (**+24.2%**), DD13/7 23.23 -> 29.93 fps (**+28.8%**).

## Verification

- Six of six native CTest targets passed.
- DD9/7 and DD13/7 encoder streams byte-identical between the AVX2 and
  scalar dispatches:
  DD9 `DF99249F127BFA10C2407448165C92409F29811F5EB14DB0F129FE2418D3F08F`,
  DD13 `FF7EA76F32BD043F83A31A0E3A1EE5F1AA63D0519E033D37A5670EC481432F62`.

## Research follow-up

Remaining encoder targets: the coarser forward vertical levels (skip 2/4/8),
the L1+ forward horizontal levels (5.01%), and the serialise bit-packer
(14.54%). The quantiser-search stage (54.20%) now dominates the DD encoder.

---

# Round 12 — encoder DD strided forward vertical (coarser levels, AVX2)

## Analysis

The DD forward vertical AVX2 kernels from Round 10 only handled level 0
(skip 1). Levels 1 and 2 (skip 2 and 4, the LL subbands) still fell through
to the scalar template. The wavelet levels store the LL subband interleaved
(even rows/columns), so the level-1/2 vertical transforms walk strided
memory rather than contiguous rows.

The scalar lifting recurrence is identical; only the column access changes.
For skip 2 the 8 columns sit at even int16 positions of a 16-sample span, so
the load is a two-lane de-interleave and the store is a read-modify-write
that re-interleaves the computed columns around the untouched odd columns.
For skip 4 the 4 columns sit at positions 0/4/8/12 of a 16-sample span, so a
128-bit four-lane kernel with `pshufb`+`pblendw` handles load/store.

## Retained change

- Added `*_V_inplace_avx2_s2` and `*_V_inplace_avx2_s4` for DD9/7 and DD13/7
  in the encoder's `vc2transform_avx2` library, wired into
  `get_vtransform_avx2` levels 1 and 2. Geometry falls back to the scalar
  template when the stride assumptions (width multiple of 16, sufficient
  height) do not hold.

## Same-session A/B results

Pinned 60-frame medians, before vs after (both builds include Rounds 9-11):

| Workload | Before | After | Improvement |
|---|---:|---:|---:|
| DD9/7 encoder | 31.84 fps | 33.67 fps | **+5.7%** |
| DD13/7 encoder | 29.93 fps | 31.61 fps | **+5.6%** |

The DD9/7 vertical-wavelet stage dropped from 13.44% to 9.27% of encoder
worker time. Cumulative encoder gains (scalar baseline -> now): DD9/7
25.63 -> 33.67 fps (**+31.4%**), DD13/7 23.23 -> 31.61 fps (**+36.1%**).

## Verification

- Six of six native CTest targets passed.
- DD9/7 and DD13/7 encoder streams byte-identical between the AVX2 and
  scalar dispatches:
  DD9 `DF99249F127BFA10C2407448165C92409F29811F5EB14DB0F129FE2418D3F08F`,
  DD13 `FF7EA76F32BD043F83A31A0E3A1EE5F1AA63D0519E033D37A5670EC481432F62`.

## Research follow-up

The quantiser-search stage (56.27%) now dominates the DD encoder — it is the
single full quantise+VLC trial per slice (already AVX2 from Round 9). The
serialise bit-packer (15.08%) and the L1+ forward horizontal levels (5.64%)
remain. Decoder-side work is unchanged this round.
---

# Round 13 — encoder serialise bit-packer (AVX2 prefix-sum + variable shift)

## Literature review (summary)

The serialise stage concatenates variable-length VLC codewords (1..18 bits)
into an MSB-first byte stream. The scalar loop has a serial dependency chain
on a 32-bit accumulator and bit counter. Candidate strategies surveyed:

- Fixed-width SIMD bit packing (BP128/BP256, `simdcomp`, `TurboPFor`,
  `FastPFor`) — the canonical primitive, but only for equal-width fields.
- Prefix-sum + variable-shift + OR-reduce (varint-G8IU; Stream VByte,
  arXiv:1709.08990) — the dominant general technique for variable widths:
  exclusive-prefix-sum the lengths (Kogge-Stone), shift each codeword by its
  cumulative offset (`vpsllvq`), OR-reduce the lanes.
- BMI2 `PDEP` (x86 BMI2, Haswell+; used in `facebook/openzl` entropy decode) —
  deposits a value's bits into a mask's set bits; fast on Intel, but scalar.
- SWAR/branch-free scalar tricks (64-bit accumulator, batched flush).
- AVX-512 primitives — unavailable on this AVX2-only machine.

## Retained change

`serialise.hpp` gained `serialise_pack_codewords`, which packs four codewords
per AVX2 group: a two-step Kogge-Stone prefix sum of the four lengths, four
64-bit variable shifts, an OR-reduce, and a merge into a 64-bit left-aligned
accumulator with a write-ahead flush (matching the scalar's byte-exact
MSB-first output). Groups whose total bit-length would overflow the remaining
accumulator headroom fall back to the scalar step. Both slice loop bodies now
delegate to this helper. A cached CPUID check selects the path at runtime.

## Results (pinned 60-frame medians, before vs after)

| Workload | Before | After | Improvement |
|---|---:|---:|---:|
| DD9/7 encoder | 33.67 fps | 34.55 fps | **+2.6%** |
| DD13/7 encoder | 31.61 fps | 32.10 fps | **+1.6%** |

The serialise stage dropped from ~26.9ms to ~25.8ms (5 frames).

## Bottleneck finding (measured)

A read-only experiment (loads without packing: ~10.2ms) versus the full pack
(~25.8ms) shows the stage is roughly 40% DRAM-bound on the intermediate
codeword/wordlength arrays (18.7MB/frame working set, which exceeds the 15MB
L3) and 60% packing compute. The SIMD trims only the compute half, which is
why the gain is modest. The codeword/wordlength arrays are written by the
quantiser-search and re-read by the serialiser; fusing the two stages (packing
directly during the final quantiser trial) would eliminate both the ~18.7MB
write and the ~18.7MB read and is the remaining large lever.

## Verification

- Six of six native CTest targets passed.
- DD9/7, DD13/7 and Haar0 encoder streams byte-identical between the AVX2 and
  scalar serialiser paths (forced-scalar A/B):
  Haar0 `E27A23277771A6E294F91042C1011B24EFFA1EA044AA6CD62F0827CB04E46744`,
  DD9 `DF99249F127BFA10C2407448165C92409F29811F5EB14DB0F129FE2418D3F08F`,
  DD13 `FF7EA76F32BD043F83A31A0E3A1EE5F1AA63D0519E033D37A5670EC481432F62`.

## Research follow-up

The quantiser-search (55%) remains the dominant encoder cost. The strongest
remaining lever is fusing quantise+encode with the serialiser so the final
quantiser trial packs bits directly into the output buffer, eliminating the
intermediate codeword/wordlength arrays (and their write+read traffic). The
L1+ forward horizontal levels (5.6%) are the last un-vectorised transform.

---

# Round 14 — fuse quantise+encode with the serialiser

## Change

The serialiser no longer re-reads the `codeword`/`wordlength` arrays. The
AVX2 quantise path scatters codewords into stack buffers and packs them
directly into a new per-slice `packed[c]` buffer (via the shared SIMD
bit-packer from Round 13, now in `bitpack.hpp`), setting a `packed_valid[c]`
flag. The serialiser memcpy's `packed[c]` (`length[c]` bytes) when the flag is
set, and falls back to packing the arrays otherwise (non-AVX2 or non-32x8x3
paths). The intermediate arrays are no longer written on the AVX2 path.

## Result

| Stage | Before | After |
|---|---:|---:|
| serialise | 25.8ms (16.3%) | 2.8ms (1.8%) |
| quantiser-search | 87.6ms (55.4%) | 108.6ms (70.5%) |
| total (5 frames) | 158ms | 154ms (~-2.5%) |

The packing compute moved from the serialiser into the quantiser-search
(where the final quantiser trial emits the bits), so the net single-thread
gain is modest (~2.5%). The intermediate memory traffic dropped ~3x (the
18.7MB/frame codeword+wordlength arrays are replaced by ~6.2MB/frame of
packed bits, which also fits in L3). The serialiser is now a trivial
memcpy+headers stage. This is also the necessary precondition for fusing the
packing directly into the quantise loop.

## Verification

- Six of six native CTest targets passed.
- Haar0, DD9/7 and DD13/7 encoder streams byte-identical:
  Haar0 `E27A23277771A6E294F91042C1011B24EFFA1EA044AA6CD62F0827CB04E46744`,
  DD9 `DF99249F127BFA10C2407448165C92409F29811F5EB14DB0F129FE2418D3F08F`,
  DD13 `FF7EA76F32BD043F83A31A0E3A1EE5F1AA63D0519E033D37A5670EC481432F62`.

## Research follow-up

The quantiser-search (70%) now bundles the quantise, the VLC lookup, and the
bit-packing; fusing the packing into the quantise loop (packing inline rather
than via the scan-order scatter + separate pack pass) is the next lever. The
L1+ forward horizontal levels (5.6%) remain the last un-vectorised transform.

---

# Round 15 — AVX2 quantise+pack for 16-wide chroma

## Change

The existing AVX2 quantise+VLC+bit-pack kernel was generalized from 32x8
luma slices to 16x8 chroma slices. A dedicated 16x8 scan-order table preserves
the scalar coefficient order; the non-AVX2 path continues to use the original
scalar fallback. Both chroma components now avoid the generic scalar
`encode_sample` loop and set `packed_valid` directly for the existing
serializer fast path.

## Result

Pinned 30-frame Haar0 encoder A/B medians, 21 interleaved pairs per run:

| Run | Clean baseline | AVX2 chroma | Improvement |
|---|---:|---:|---:|
| 1 | 39.343 fps | 48.680 fps | **+23.7%** |
| 2 | 39.948 fps | 49.123 fps | **+23.0%** |

The candidate passed the full tier: six native tests, three conformance
streams, and identical smoke stream/pixel hashes for Haar0, DD9/7, and
DD13/7. The change was promoted as commit `0c96986`.

## Research follow-up

The fastest-mode encoder now uses the AVX2 quantise+pack path for all three
components of the standard 32x8x3 slice. Remaining work should focus on the
DD L1+ forward horizontal transforms and any decoder-side bottlenecks.

---

# Round 16 — AVX2 DD9/DD13 forward horizontal levels

## Change

The DD9/7 and DD13/7 forward horizontal levels 1 and 2 (skip 2 and 4) now
use AVX2 eight-lane int32 lifting kernels with scratch buffers for the even,
odd, and intermediate prediction samples. The skip-2 path uses SIMD stride-4
deinterleaving; non-conforming widths and narrow rows retain the scalar
implementation. Dispatch is enabled for both transform families while the
existing scalar fallback remains available for other levels and platforms.

## Result

Fifteen interleaved profile pairs on DD13/7 showed the horizontal stage median
fall from 60.359ms to 44.677ms (**-26.0%**) and total worker time median fall
from 768.609ms to 754.808ms (**-1.8%**). Wall-clock encoder A/B runs were
noisy: the first pair improved 35.582 to 37.117 fps (+4.3%), while the repeat
was effectively flat at 37.366 to 37.096 fps. The stage-level and worker-time
profiles provide the stronger signal because the horizontal kernel is a
measured substage of the end-to-end run.

## Verification

- Post-promotion quick tier passed with all six native tests.
- Haar0, DD9/7, and DD13/7 pixel and stream hashes remained byte-identical.
- The candidate full tier passed six native tests and three validated streams.
- The change was promoted as commit `aa22c62`.

## Research follow-up

The next encoder targets are the remaining forward transform levels and
input/vertical work, followed by decoder-side profiling and any residual
quantise/pack costs.

---

# Round 17 — AVX2 DD9/DD13 10P2 input horizontal kernels

## Change

The DD9/7 and DD13/7 10P2 input horizontal transforms now process eight
even/odd sample pairs at a time with AVX2 int32 lifting arithmetic. The
existing four-lane implementation remains the fallback for widths that are
only multiples of eight, narrow rows, and short heights, preserving the prior
non-standard geometry behavior. The AVX2 path keeps the scalar boundary
mirrors and output overlap copy unchanged.

## Result

Fifteen interleaved profile pairs on the current head produced these medians:

| Workload | Input horizontal before | Input horizontal after | Total worker before | Total worker after |
|---|---:|---:|---:|---:|
| DD9/7 | 103.604ms | 91.916ms (**-11.3%**) | 720.759ms | 710.841ms (**-1.4%**) |
| DD13/7 | 130.915ms | 110.926ms (**-15.3%**) | 768.053ms | 746.627ms (**-2.8%**) |

The full-tier benchmark remained noisy, but the stage-level reductions were
consistent across the paired runs and the end-to-end worker medians improved
for both DD workloads.

## Verification

- Candidate full tier passed six native tests, three validated streams, and
  all pixel/stream hash gates.
- Post-promotion smoke and quick tiers passed; the quick tier remained clean
  with six native tests and identical hashes.
- The change was promoted as commit `b138c52`.

## Research follow-up

Remaining encoder work is concentrated in vertical wavelet/input memory
traffic and quantiser-search. Decoder-side profiling is the next separate
frontier once the remaining encoder transform opportunities are measured.

---

# Round 18 — AVX2 DD9/DD13 coarser inverse vertical levels

## Change

The DD9/7 and DD13/7 inverse vertical transforms now use eight-lane AVX2
lifting kernels at decomposition strides 2, 4, and 8. Gather loads preserve
the interleaved column geometry and scalar lane stores preserve the existing
sample narrowing behavior. Dispatch selects the matching kernel for levels
1 through 3; unsupported widths and short planes retain the scalar fallback.

## Result

The candidate was compared with a clean full-tier run of the immediately
preceding `ceaf5bb` head, then repeated after the focused profile. Nine-run
medians were:

| Workload | Clean baseline | AVX2 coarser V | Improvement |
|---|---:|---:|---:|
| DD9/7 decoder | 54.446 fps | 61.983 fps | **+13.8%** |
| DD13/7 decoder | 50.082 fps | 56.075 fps | **+12.0%** |
| Haar0 decoder | 98.041 fps | 97.720 fps | -0.3% |

The repeated candidate kept encoder medians within 0.14% of the clean
baseline. Focused alternating profiles also showed the inverse-vertical
stage falling by roughly 30% for both DD workloads.

## Verification

- The candidate full tier passed six native tests and three conformance
  validators.
- Haar0, DD9/7, and DD13/7 stream and decoded-pixel hashes remained exactly
  unchanged.
- A repeat full tier reproduced the decoder gains and kept the non-target
  encoder metrics effectively flat.
- The change was promoted as commit `eabed91`.

## Research follow-up

Decoder inverse-vertical is no longer the dominant DD stage. The next decoder
targets are inverse-horizontal and final output, while the encoder-side
quantiser-search and remaining transform memory traffic remain open.

---

# Round 19 — Compact decoder VLC lookup entries

## Change

The decoder VLC lookup entries now use their natural 16-byte stride while
retaining 16-byte alignment for the SSE lookup load. The previous 32-byte
alignment inflated the 1024-entry table from 16 KiB of payload to a 32 KiB
object, increasing cache pressure without adding fields or changing lookup
semantics.

## Result

Interleaved focused profiles used 8-frame samples on the same-head baseline.
The DD13/7 median showed the clearest end-to-end gain, with a consistent VLC
stage reduction across the DD workloads:

| Workload | Baseline total | Compact total | Improvement | Baseline VLC | Compact VLC |
|---|---:|---:|---:|---:|---:|
| DD13/7 | 141.913ms | 138.155ms | **-2.6%** | 34.384ms | 31.839ms (**-7.4%**) |
| DD9/7 | 127.650ms | 126.874ms | -0.6% | 33.512ms | 31.825ms (**-5.0%**) |
| Haar0 | 80.487ms | 80.218ms | -0.3% | 35.841ms | 35.931ms (+0.3%) |

The full tier measured 58.140 fps on DD13/7, 63.966 fps on DD9/7, and
100.335 fps on Haar0. Encoder measurements remained within the normal run
variation at 39.260, 41.142, and 48.544 fps respectively.

## Verification

- Smoke, quick, and full tiers passed all six native tests.
- The full tier passed all three conformance validators.
- Haar0, DD9/7, and DD13/7 stream and decoded-pixel hashes remained exactly
  unchanged.
- The change was promoted as commit `61925c8`.

## Research follow-up

The decoder VLC stage is now a smaller share of the DD workload. Remaining
decoder opportunities are inverse-horizontal and final output; encoder
quantiser-search remains the largest measured stage but earlier fused-path
experiments have not yet produced a safe improvement.

---

# Round 20 — Widen DD final inverse/output batches

## Change

The AVX2 DD9/7 and DD13/7 final inverse-horizontal kernels now process two
adjacent eight-pixel groups together with 256-bit arithmetic. The compact
intermediate D samples are contiguous across the pair, allowing one set of
sliding-window loads and two 128-bit non-temporal stores for each 16-pixel
batch. The existing eight-pixel loop remains as the tail path, and all prior
alignment/crop checks and scalar fallbacks are unchanged.

## Result

Twelve interleaved 8-frame profile pairs on the same head produced these
medians:

| Workload | Baseline total | Widened total | Improvement | Baseline final stage | Widened final stage |
|---|---:|---:|---:|---:|---:|
| DD9/7 | 122.556ms | 116.805ms | **-4.7%** | 33.520ms | 27.508ms (**-17.9%**) |
| DD13/7 | 139.314ms | 134.823ms | **-3.2%** | 43.099ms | 37.311ms (**-13.4%**) |
| Haar0 | 75.584ms | 75.958ms | +0.5% | 13.460ms | 13.559ms |

The full tier measured 62.893 fps on DD13/7, 68.493 fps on DD9/7, and
104.889 fps on Haar0. The Haar path does not use this kernel; its small
cross-binary variation stayed within the experiment tolerance.

## Verification

- Smoke, quick, and full tiers passed all six native tests.
- The full tier passed all three conformance validators.
- Haar0, DD9/7, and DD13/7 stream and decoded-pixel hashes remained exactly
  unchanged.
- The change was promoted as commit `61dad41`.

## Research follow-up

Final output is no longer the largest DD decoder stage after widening the
batch. Remaining work is focused on inverse-horizontal and any safe encoder
quantiser-search reductions.

---

# Round 21 — DD9 geometric quantiser-search bound

## Change

The DD9/7 encoder now doubles the QI trial step after each invalid size
trial, up to the maximum QI, before entering the existing binary refinement.
The search remains exact because coded size is monotonic with QI; DD13/7,
Haar0, eighth-search, and the refinement phase retain their prior paths.

## Result

The candidate was compared with a clean full-tier run of `e947e44` using
nine 30-frame runs. The full-tier medians were:

| Workload | Clean baseline | DD9 exponential bound | Improvement |
|---|---:|---:|---:|
| DD9/7 encoder | 40.841 fps | 41.347 fps | **+1.2%** |
| DD13/7 encoder | 39.460 fps | 38.437 fps | -2.6%* |
| Haar0 encoder | 49.313 fps | 48.943 fps | -0.8%* |

*The non-target paths are unchanged; their full-tier medians varied across
the matched runs. A second full candidate run measured DD13/7 at 39.355 fps
and Haar0 at 48.965 fps. The focused quick tier kept DD9/7 at 40.60–41.05
fps across all three runs.

## Verification

- Smoke, quick, and full tiers passed all six native tests.
- The full tier passed all three conformance validators.
- Haar0, DD9/7, and DD13/7 stream and decoded-pixel hashes remained exactly
  unchanged.
- The change was promoted as commit `cdc8a2b`.

## Research follow-up

DD9/7 quantiser selection now has a measured encoder gain. DD13/7 remains
the next encoder target, while decoder work is still focused on intermediate
inverse-horizontal lifting.
