# cuda-implementation

The **experimental NVIDIA CUDA GPU acceleration** branch. It layers GPU
offload of the codec's hottest kernels on top of the CPU optimisations from
`optimizations`.

## Why this branch exists

- To offload the Haar-0 transform (the `--speed=fastest` kernel) and parts of
  the quantiser search to the GPU, and to stream decoder output through
  pinned buffers.
- It is kept separate because it requires the CUDA toolkit and a CUDA-capable
  GPU (targets compute capability 6.1) and should not affect the pure-CPU
  builds.

## What was done

Encoder (CUDA):

- depth-3 Haar-0 transform as one warp per tile, using register shuffles
  instead of shared memory
- quantiser selector search: hoisted loop-invariant maths and raised occupancy
  to 4 blocks via launch bounds

Decoder (CUDA):

- CUDA inverse transform with pinned frame upload and capped transform
  registers
- decoder output staged in persistent pinned buffers with overlapped plane
  downloads

Integration:

- merged with the full CPU optimisation line (AVX2 transforms, quantise,
  SIMD serialise bit-packing, quantise+serialise fusion)

## Building and running

- Configure with `-DVC2_ENABLE_CUDA=ON` (falls back to CPU-only if no CUDA
  compiler is found).
- Enable at runtime with the `VC2HQ_CUDA=1` environment variable.

## Verification

- CPU-only and CUDA builds both pass the 6/6 native CTest targets.
- CUDA encoder and decoder run on a GeForce GTX 1050 Ti (sm_61); byte-identical
  output is preserved when the GPU path is disabled.

## Automated research loop

The shared `autoresearch/` framework runs an AlphaEvolve-style search over GPU
kernels, transfers, launch amortization, and CPU/GPU scheduling. The CUDA
configuration builds with `VC2_ENABLE_CUDA=ON`, benchmarks both
`VC2HQ_CUDA=1` and `VC2HQ_CUDA=0`, and requires exact stream/pixel hashes for
both variants before a candidate can enter the program database.

```powershell
py -3 .\autoresearch\run.py `
  --config .\autoresearch\config.cuda.json `
  baseline --tier full
```

See `autoresearch/README.md` for the proposer interface, evaluation cascade,
island selection, and safe promotion workflow.

## Documentation

- `CUDA.md`, `profiles/CUDA-DECODER.md`, `profiles/CUDA-LOW-LATENCY.md`.
- `autoresearch/README.md` and `autoresearch/BACKGROUND-CUDA.md`.
