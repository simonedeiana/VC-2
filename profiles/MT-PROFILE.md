# VC-2 HQ Multi-threaded performance analysis

Date: 2026-08-13

Machine: Intel Xeon E5-1650 v4 (Broadwell-EP, 6 physical cores / 12 logical),
MSVC 19.44 Release. Workload: 1920x1080 10-bit 4:2:2, Haar0, depth 3, 32x8
slices, 2:1, `fastest`, 120 (dec) / 60 (enc) frames, output disabled.

## Benchmarking methodology (important)

Logical-processor topology on this machine (verified via
`GetLogicalProcessorInformationEx`):

| Physical core | Logical CPUs (mask) |
|---|---|
| 0 | 0, 1 (0x3) |
| 1 | 2, 3 (0xC) |
| 2 | 4, 5 (0x30) |
| 3 | 6, 7 (0xC0) |
| 4 | 8, 9 (0x300) |
| 5 | 10, 11 (0xC00) |

Logical 0/1 are HT siblings of the SAME physical core. Multi-threaded
benchmarks MUST pin the process to distinct physical cores, e.g. 6 threads to
`0x555` (logical 0,2,4,6,8,10). Pinning 2 threads to `0x3` puts both on one
physical core and measures only the ~17% HT gain, not real parallelism.

## Baselines (pinned to N physical cores, medians)

| Threads | Decoder fps | Encoder fps |
|---|---:|---:|
| 1 | 72.6 | 31.5 |
| 2 | 132.0 | 60.3 |
| 4 | 239.9 | 107.7 |
| 6 | 287.8 | 138.1 |

Scaling at 6 cores: decoder 3.96x, encoder 4.38x. The machine has heavy
background load on the same cores (VS Code/Chrome), so quiet-session numbers
are higher; same-session interleaved A/B is the only reliable comparison.

## Decoder analysis

- Job grid: `n_jobs` = power of two >= 4 x threads (6 threads -> 32 jobs, 8x4
  grid). The 4x multiplier is OPTIMAL: 2x (-12% at 6t) and 8x (-19%) are both
  worse (load-balance vs overlap trade-off).
- Stage profile at 6 threads: vlc-decode 66.6%, dequant 14.1%, transforms
  ~19%. VLC+dequant calls are 29% above the no-overlap count -> the 8x4 grid
  decodes each boundary slice twice (~29% redundant VLC+dequant).
- **Overlap elimination is cache-hostile on this CPU**: a two-phase refactor
  (phase 1 decodes each slice once into shared full-frame planes; phase 2 jobs
  copy their region and transform) measured **-17% (4t) / -20% (6t)** in a
  same-session A/B, despite removing the 29% redundancy. The shared-buffer
  round-trip pushes all coefficients through L3 and adds a second barrier per
  frame; the per-job decode+transform design keeps data in local cache and
  wins. Reverted byte-exactly.
- Hyper-threading adds nothing at the top end: 12 threads (all logical) =
  287.7 fps, same as the 6-thread physical peak. The decoder is not
  latency-bound at scale.

Conclusion: the per-job grid design is the practical optimum for this CPU;
the ~29% boundary-slice redundancy is the price of cache-friendly
parallelism. Remaining inefficiency is small (imbalance, barriers).

## Encoder analysis

- Jobs are horizontal bands (full width), `n_jobs` = 4 x threads. Haar has
  zero transform overlap (`TRANSFORM_OVERLAP[HAAR] = 0`), so bands add no
  redundancy. 2x multiplier is not better.
- Two parallel phases per frame: `Transform` (32 bands, ~13% of time), then
  `EncodePartial` (540 chunks of 15 slices, ~87%: quantiser-search 72.8% +
  serialise 13.8%), separated by a barrier.
- NFACTOR (chunk count, 540) vs 32 made no difference within noise; the
  chunked encode phase is already well balanced.

The encoder is search-bound; its MT scaling is near the parallel limit for
this workload (~73% efficiency at 6 cores).

## Rejected MT experiments (all reverted, byte-identical)

- Decoder job multiplier 2x / 8x (worse than 4x).
- Decoder two-phase shared-buffer overlap elimination (-17..-20%).
- Decoder hyper-threading (neutral at 12 threads).
- Encoder NFACTOR 540 -> 32 (neutral).
- Encoder job multiplier 2x (neutral/worse).

## Take-aways for future MT work

- The decoder's only remaining lever is removing the boundary-slice
  redundancy without breaking cache locality (e.g., thread-affine two-phase
  where each thread decodes AND transforms the same jobs, copying only the
  overlap from neighbours). Expected gain ~+10-15% at 6 threads, but high
  complexity/risk; the simple shared-buffer version regressed.
- The encoder's lever is reducing the quantiser-search itself (per-thread
  work), which helps MT proportionally.
- Always benchmark MT with physical-core affinity and same-session interleaved
  A/B; HT-sibling pinning (0x3 for "2 threads") is a classic false baseline.
