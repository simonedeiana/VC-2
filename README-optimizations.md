# optimizations

The **mainline CPU-performance and experimental research** branch. It takes
the `main` baseline and applies a sequence of byte-identical SIMD (SSE4.2 /
AVX2) optimisations to the VC-2 HQ encoder and decoder, one reviewable commit
per round. The former `bold-experiments` branch has been folded into this one;
successful and rejected high-risk experiments now share a single history and
README.

## Why this branch exists

- It is the "safe" home for measured, byte-identical CPU optimisations — each
  round is verified against the reference before landing.
- It is the branch the `cuda-implementation` work periodically merges from, so
  the GPU work always builds on the latest CPU state.
- It is also the playground for aggressive ideas such as breaking serial
  lifting recurrences into two-pass stencils. Candidates still have to remain
  byte-identical and beat the baseline under repeated measurement.

## What was done (Rounds 1-14)

Decoder:

- interleave multi-stream VLC decode to hide the lookup-table latency chain
- vectorised Haar inverse transforms for 10-bit decode
- AVX2/SSE4.2 inverse vertical transforms for Deslauriers-Dubuc 9/7 and 13/7
- AVX2 two-pass final-horizontal transforms for DD9/7 and DD13/7

Encoder:

- AVX2 forward vertical and horizontal transforms for DD9/7 and DD13/7
  (including strided coarser-level vertical kernels)
- AVX2 quantise + merged codeword/length table lookup (8 samples/iteration)
- AVX2 SIMD bit-packing of the serialiser (prefix-sum + `vpsllvq`)
- fuse quantise+encode with the serialiser so bits are packed during the
  final quantiser trial (serialiser stage 16.3% -> 1.8%)

## Results (single-thread, 1920x1080, `--speed=fastest`, medians)

| workload | baseline | now | gain |
|---|---:|---:|---:|
| encoder DD9/7 | 25.6 fps | 34.3 fps | **+34%** |
| encoder DD13/7 | 23.2 fps | 32.1 fps | **+38%** |
| encoder Haar0 | 35.2 fps | 40.9 fps | +16% |
| decoder DD9/7 | 41.9 fps | 52.2 fps | +25% |
| decoder DD13/7 | 39.1 fps | 48.8 fps | +25% |

Every change is byte-identical (SHA-256 of stream and pixels) and the 6/6
native CTest targets stay green.

## Notable rejected experiments

- SSE4.2 DD final-horizontal transform: the stage was memory-bound and the
  SIMD implementation was slower than scalar.
- matched-length VLC stream grouping: the additional grouping work did not
  recover enough lookup-chain latency.

Rejected experiments are reverted and recorded in the optimization profile so
future research loops receive them as negative examples.

## Automated research loop

`autoresearch/` implements an AlphaEvolve-style evolutionary optimization
loop. It keeps a scored program database, samples across subsystem islands,
generates targeted SEARCH/REPLACE mutations, evaluates candidates in isolated
Git worktrees, and promotes only byte-identical latency improvements. See
`autoresearch/README.md` and start with:

```powershell
py -3 .\autoresearch\run.py `
  --config .\autoresearch\config.optimizations.json `
  baseline --tier full
```

## Documentation

- `profiles/OPTIMIZATION-PROFILE.md` — per-round analysis, measurements, and
  the rejected experiments (with the reasons they were rejected).
- `autoresearch/README.md` — evolutionary search setup, evaluator cascade,
  proposer interface, and promotion rules.
