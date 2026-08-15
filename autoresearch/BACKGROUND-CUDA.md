# CUDA optimization context

Read `CUDA.md`, `profiles/CUDA-DECODER.md`, and
`profiles/CUDA-LOW-LATENCY.md` before proposing an experiment.

Hard constraints:

- CUDA 12.x and MSVC 2022; the deployed baseline includes sm_61 hardware.
- Optimize end-to-end one-worker latency, including launch and transfer costs.
- `VC2HQ_CUDA=0` must retain the CPU behavior and performance contract.
- CUDA and CPU paths must produce byte-identical streams and decoded pixels.
- Do not weaken synchronization, error handling, tests, validator checks, or
  hash gates.
- Avoid ideas that only move time between measured stages without lowering
  end-to-end latency.

Useful search dimensions include kernel fusion, register pressure, occupancy,
pinned-buffer reuse, asynchronous transfers, launch amortization, CPU/GPU
overlap, and CPU fallbacks for workloads below the offload break-even point.
