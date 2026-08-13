# CUDA single-picture latency profile

Date: 2026-08-12

## Goal and workload

The target is one picture in flight: less than 1 ms for each 720p encode/decode
and less than 2 ms for each 1080p encode/decode. Measurements use planar 10-bit
4:2:2, Haar0 depth 3, 32x8 slices, fastest quantizer selection, 2:1 compression,
and 12 CPU workers. Hardware is a GeForce GTX 1050 Ti (Pascal, SM 6.1) under
Windows/WDDM with driver 582.66 and CUDA 12.9.

## Final retained result

Each number is derived from repeated 120-picture, output-disabled runs. There
are never multiple pictures in flight.

| Path | Representative rate | Latency | Target |
|---|---:|---:|---:|
| AVX2 encoder, 720p | ~334 fps | ~2.99 ms | <1 ms |
| CUDA fused encoder, 720p | ~348 fps median | ~2.87 ms | <1 ms |
| AVX2 encoder, 1080p | 166-171 fps | 5.85-6.02 ms | <2 ms |
| CUDA fused encoder, 1080p | ~244 fps median | ~4.10 ms | <2 ms |
| AVX2 decoder, 720p | ~408 fps median | ~2.45 ms | <1 ms |
| AVX2 decoder, 1080p | ~235 fps median | ~4.25 ms | <2 ms |

(Measured with the latest round of encoder optimizations; the 1080p CUDA
encoder went from ~190 fps at the start of the round to ~244 fps median,
~1.56x the 12-thread AVX2 encoder.)

The retained encoder keeps coefficients resident and performs transform,
quantizer selection, VLC generation, and complete fixed-size slice
serialization on the GPU. It uses persistent pinned buffers, three transform
streams feeding one encoding stream, one host synchronization, shared-memory
warp-parallel bit packing, and selector-to-serializer metric reuse. Only the
compressed payload returns over PCIe.

## Final Nsight Systems evidence

The final 60-picture 1080p capture is `cuda-metric-reuse.nsys-rep`:

| CUDA work | Per picture |
|---|---:|
| Transform kernels (one warp per tile, 3 launches) | ~0.24 ms GPU time |
| Quantizer selector | ~0.80 ms GPU time |
| Warp-parallel VLC serializer | ~0.70 ms GPU time |
| Host-to-device copies (summed; three planes overlap) | ~0.60 ms GPU time |
| Compressed device-to-host copy | ~0.16 ms GPU time |
| One stream synchronization | ~3.5 ms API time |

Nsight Compute counters are unavailable because NVIDIA performance-counter
access is disabled for non-administrator users (`ERR_NVGPUCTRPERM`). Nsight
Systems timing is sufficient to show that the necessary device work and
transfers already exceed 2 ms on this GPU before application overhead.

## Optimization loop

- Persistent allocations removed context and `cudaMalloc` cost from picture
  latency.
- Pinned plane transfers and three streams reduced the transform-only path,
  but returning the full coefficient picture still made it slower than AVX2.
- GPU quantizer selection followed by CPU VLC emission duplicated the dominant
  coefficient pass and regressed 1080p latency to roughly 9.2 ms.
- Moving quantization, VLC, and serialization together made the output boundary
  the compressed picture rather than the full coefficients.
- A fused kernel with lane 0 serializing each slice was bit-exact but spent
  about 6.17 ms per picture in that kernel.
- One serializer thread per slice reduced that kernel to about 2.49 ms.
- Warp-parallel shared-memory bit packing reduced serialization to about
  0.77 ms, and reusing the selector's final bit counts reduced it to ~0.50 ms.
- Replacing per-thread selector band arrays and scan loops with shared band
  reductions and inverse scan tables reduced selector time from ~1.59 ms to
  ~0.69 ms.
- CUDA events and a dedicated encoding stream removed intermediate host
  synchronization. A tiled depth-3 Haar kernel reduced 18 transform launches
  to three, although transform compute time remained around 0.66 ms.
- The quantiser selector's candidate loop is invariant in its matrix for
  qi >= 32 (the loop steps by 8, so qindex = 28+(qi&3) is fixed): hoisting the
  loop-invariant `((v*m)>>16 + v) >> sh` out of the loop (and only recomputing
  the qshift) removed all plane/matrix re-reads from the search iterations
  (~+12-17% end to end).
- Replacing the shared-memory tiled depth-3 Haar with one warp per 8x8 tile
  using register shuffles removed all `__syncthreads` and shared memory: Y
  transform 431 -> 235 us, chroma 216 -> 112 us (~+12% end to end). The
  multi-resolution levels only touch the LL subband (even rows/cols), so the
  deeper passes update only those lanes.
- `__launch_bounds__(256, 4)` on the selector cut registers 72 -> 64 (3 -> 4
  resident blocks) and the kernel 968 -> 805 us.
- Packing the (multiplier, shift) matrix pair into one uint32 and reading one
  packed word instead of two arrays was neutral and was reverted.

## Boundary and next work

The end-to-end CUDA encoder is now bit-exact and faster than the local AVX2
encoder. There is no remaining micro-optimization with a credible path to the
requested 2 ms 1080p ceiling on the GTX 1050 Ti: the profiled GPU work and
required PCIe transfers alone exceed it. The next credible experiments require
different conditions—newer hardware, GPU-native input/output surfaces, or a
persistent kernel that avoids WDDM launch costs.

The decoder remains CPU-only. Its hotspot is entropy decoding/dequantization,
followed by output conversion and inverse vertical transform. A useful CUDA
decoder must fuse VLC decode, dequantization, inverse transform, and final pixel
output so only pixels cross back to the host. Inverse-transform-only offload is
not a useful boundary.

## Verification

- Full MSVC/CUDA Release build succeeded.
- All six native CTest targets passed.
- One-frame conformance validation reported no errors.
- The 30-frame 1080p CUDA output matched the CPU SHA-256 exactly:
  `d27e9c049ce776c3b14e2153286429e94ee312d440863fd20023e1f6b5d35a15`.
