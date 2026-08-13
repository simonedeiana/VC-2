# CUDA decoder profile

Date: 2026-08-13

## Goal and workload

Add a fused CUDA decoder to the `cuda-implementation` branch and compare it
against the branch's CPU decoder. Measurements use planar 10-bit 4:2:2,
Haar-0 depth 3, 32x8 slices, `fastest` quantiser, `vc2decode -d` (decode with
output disabled) at 1 thread on a GeForce GTX 1050 Ti (Pascal, SM 6.1) under
Windows/WDDM with driver 582.66 and CUDA 12.9.

## Final retained result

The tool's own steady-state measurement (which excludes process startup).
The `optimizations` branch CPU gains (vectorized Haar inverse transforms and
interleaved multi-stream VLC decode) were merged into this branch, so the CPU
numbers below are the optimized decoder.

| Path | 1080p | 720p |
|---|---:|---:|
| CPU decoder, 1 thread (optimized) | ~50 fps | ~145 fps |
| CUDA fused decoder | ~106 fps | ~209 fps |
| Speedup | ~2.1x | ~1.4x |

(On this machine with its background load the absolute numbers vary between
sessions; the interleaved A/B ratio is the reliable comparison. Before the
CPU optimizations were merged, the CPU decoder was 36-40 fps at 1080p and the
CUDA speedup measured ~3.4x.)

Output is byte-for-byte identical to the CPU decoder. A 30-picture 1080p
Haar-0 decode matches the CPU SHA-256 exactly
(`48a27492179d8c7181fd99508cc1b9fd902d6859b9f39bbae10e3a0346e85d46`);
single-picture Haar-1 and 720p Haar-0 decodes also match exactly.

## Final Nsight Systems evidence (1080p, per picture)

| CUDA work | Per picture |
|---|---:|
| Frame upload (whole compressed frame, frame-relative offsets) | ~0.22 ms |
| Coefficient plane zeroing (two memsets) | ~0.08 ms |
| VLC kernel (one lane per stream, 24,300 streams) | ~1.6-1.7 ms |
| Inverse transform kernel (one warp per slice) | ~1.0 ms |
| Output downloads (Y 4.1 MB + Cb/Cr 2.1 MB each, overlapped) | ~0.6 ms |

The output download went from ~1.3 ms to ~0.6 ms by pinning a persistent
staging buffer once and issuing the three plane downloads on separate
streams (all gated on one event) so they overlap instead of serializing on
the decode stream; a host memcpy then moves each plane into the caller's
buffer.

Nsight Compute counters remain unavailable (non-administrator counter access
is disabled, `ERR_NVGPUCTRPERM`), so the kernels were tuned with Nsight
Systems timing plus A/B experiments.

## Decode design

A VC-2 HQ slice is a complete wavelet tree, so every slice is independent and
the inverse transform is slice-local. The backend therefore:

- Uploads the compressed frame directly; per-slice stream offsets are
  frame-relative, avoiding a per-frame copy of the payload.
- Runs a VLC kernel with every lane decoding one of the 24,300 independent
  component streams and dequantizing inline into the coefficient plane. The
  decode LUT is staged in read-only global memory because 32 divergent
  per-warp lookups would serialize on the constant cache.
- Runs a transform kernel with one warp per slice, using the global plane and
  writing clipped pixels to the output planes (only pixels cross back).
- Zeroes the coefficient planes each picture so undecoded coefficients stay
  zero, matching the CPU's zero-filled VLC scratch.

## Optimization loop

- A single fused kernel (one warp per slice, lanes 0-2 decoding) was
  byte-exact but spent ~8.5 ms in the kernel because only three of 32 lanes
  were active during the serial VLC decode.
- Splitting into a VLC kernel (one lane per stream) plus a transform kernel
  reduced the VLC work from ~8.5 ms to ~1.7 ms; the two-kernel pipeline is
  byte-exact and about 1.6x faster end to end.
- Uploading the whole frame instead of a packed payload copy removed the
  per-frame `std::vector` payload build (24,300 inserts) and roughly halved
  the host-side gap between frames.
- Pinning the caller's output buffers directly was rejected: the benchmark
  tool rotates output buffers every frame, so re-registering cost ~1.5 ms per
  frame (60 `cudaHostRegister` calls over 20 frames). Instead the downloads
  land in persistent pinned staging buffers (registered once) that are then
  memcpy'd to the caller's planes; a plain 8 MB write to the caller's
  pageable buffers costs ~1.8 ms because the OS evicts the 250 MB of rotated
  output buffers between uses, so writing the caller's buffer directly was
  not the win it looked like.
- The frame upload was staged through a persistent pinned buffer
  (`cudaMallocHost` + memcpy once), cutting the H2D from ~0.58 ms to
  ~0.22 ms, and the transform kernel got `__launch_bounds__(128, 12)`
  (40 registers, neutral but kept as headroom).
- Grouping VLC streams by component (all Y first, then chroma) regressed
  (1.99 ms vs 1.66 ms): the natural interleaved layout keeps each warp's
  payload reads within ~11 consecutive slices, which the reordering lost.
- Interleaving two independent streams per VLC lane (the ILP trick that won
  on the CPU decoder) regressed 127 to 115 fps on the GPU: warp-level
  parallelism already hides the LUT-load latency, and the extra decode state
  raised register pressure.
- Staging the slice region in shared memory for the transform kernel was not
  faster than the global-plane version (1.34 ms vs 1.03 ms in Nsight), so the
  global version is retained.
- Disabling the scattered dequant stores changed the frame rate by only
  ~2.5%, confirming the VLC kernel is bound by the dependent LUT-load chain,
  not by the stores.

## Verification

- Full MSVC/CUDA Release build succeeds; all six CTest targets pass.
- Haar-0 1080p, Haar-1 1080p and Haar-0 720p CUDA decodes are byte-identical
  to the CPU decoder.
- Unsupported presets (e.g. LeGall) log "CUDA decoder preset unsupported;
  using CPU" and keep the untouched CPU path (output still correct).
