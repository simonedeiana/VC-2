# LLM handoff: continue CUDA autoresearch

Point a future coding LLM at this file and tell it: **continue the VC-2 CUDA
autoresearch loop**. This document is the complete entry point. Read it before
editing code or launching experiments.

## Mission

Reduce end-to-end, single-worker encoding and decoding latency on the
`cuda-implementation` branch while preserving:

- byte-identical CUDA and CPU VC-2 encoded streams;
- byte-identical CUDA and CPU decoded 10-bit 4:2:2 pixels;
- the `VC2HQ_CUDA=0` CPU fallback and public API;
- all native tests and VC-2 conformance validation;
- correct operation on NVIDIA compute capability 6.1.

Optimize wall-clock latency, including allocation, launch, synchronization,
and transfer costs. A faster kernel that makes end-to-end fps worse is a
rejected candidate.

The repository implements an evolutionary coding loop inspired by
AlphaEvolve. The controller, evaluator, prompt construction, program database,
and selection logic are already implemented. Continue using them; do not
replace the framework with informal one-off timing.

## Read these files first

1. `autoresearch/README.md` — controller usage and proposal format.
2. `autoresearch/BACKGROUND-CUDA.md` — hard constraints and search frontier.
3. `CUDA.md` — build, runtime selection, and architecture.
4. `profiles/CUDA-DECODER.md` and `profiles/CUDA-LOW-LATENCY.md` — retained and
   rejected CUDA work.
5. `README-cuda-implementation.md` — current branch summary.
6. `autoresearch/config.cuda.json` — workloads, tiers, islands, and acceptance
   thresholds.
7. `autoresearch/core.py`, `evaluate.py`, and `run.py` — source of truth for
   selection and validation behavior.

Also inspect `profiles/OPTIMIZATION-PROFILE.md` when touching shared CPU code.
Do not repeat a rejected experiment until you can state what materially
changed in the new formulation.

## Establish the current state

Run these commands from the repository root:

```powershell
git branch --show-current
git status --short
git log -1 --oneline
nvcc --version
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
py -3 -m unittest discover -s .\autoresearch\tests -v
py -3 .\autoresearch\run.py `
  --config .\autoresearch\config.cuda.json status
```

The intended branch is `cuda-implementation`. Never discard a dirty worktree.
If the branch is checked out elsewhere, work there or create an isolated
worktree.

Generated state is intentionally untracked:

- `.autoresearch/cuda-implementation.sqlite3` — program database and elites;
- `.autoresearch/data/` — deterministic raw inputs;
- `.autoresearch/artifacts/` — CPU/CUDA streams and decoded frames;
- `.autoresearch/validator/` — validator output;
- `build-autoresearch/` — MSVC/CUDA Release build.

If the database is absent, or its newest full baseline commit differs from
`git rev-parse HEAD`, create a new full baseline:

```powershell
py -3 .\autoresearch\run.py `
  --config .\autoresearch\config.cuda.json `
  baseline --tier full
```

Do not compare candidates from different tiers. The database and elite cells
are partitioned by `smoke`, `quick`, and `full`.

## Last known full baseline

Recorded on 2026-08-15 at commit `36b08d3f169f` on a GeForce GTX 1050 Ti
(sm_61), CUDA 12.9, and driver 582.66. It used nine 30-frame runs, 1920x1080
10-bit 4:2:2 input, Haar0, depth 3, and one worker. Both CPU and CUDA streams
passed the installed VC-2 validator and all 6 native tests passed.

| Metric | Median |
|---|---:|
| CUDA encoder Haar0 | 240.724 fps |
| CPU-fallback encoder Haar0 | 39.901 fps |
| CUDA decoder Haar0 | 90.290 fps |
| CPU-fallback decoder Haar0 | 99.777 fps |

The decoder currently loses to the CPU fallback on this workload. Reducing
decoder launch, synchronization, transfer, or staging overhead is therefore a
high-value search direction. Preserve the much larger encoder gain.

Correctness hashes were identical for CUDA and CPU:

| Artifact | SHA-256 |
|---|---|
| Haar0 stream | `64E69879FF57B664A81A63AF9947956133FB1A47D928EE32712C8AD20A4250E6` |
| Haar0 decoded pixels | `8C391AAEA4257065028876E6A36EC21180F9B447324AEF23E1B2AB3B556FCEFC` |

These numbers are historical orientation, not permanent acceptance
thresholds. Re-baseline after code, toolkit, driver, clock, power-plan, or
hardware changes.

## Search islands and source map

The configured islands preserve diverse CUDA research paths:

- `encoder-transform`: `vc2hqencode/vc2transform_cuda/transform_cuda.cu` and
  `.hpp`;
- `encoder-selector`: CUDA selector/quantizer kernels and their integration in
  `vc2hqencode/vc2hqencode/VC2Encoder.cpp` and `quantise.cpp`;
- `encoder-transfer`: pinned buffers, upload/download lifetime, batching, and
  reuse around the encoder CUDA API;
- `decoder-transform`: `vc2hqdecode/vc2invtransform_cuda/decode_cuda.cu` and
  `.hpp`;
- `decoder-transfer`: persistent pinned output, plane copies, stream/event
  synchronization, and staging in `VC2Decoder.cpp`;
- `cpu-gpu-scheduling`: offload break-even logic, launch amortization, overlap,
  and safe fallback choices spanning `VC2Encoder.cpp` and `VC2Decoder.cpp`.

Shared AVX2 code remains under `vc2hqencode/vc2transform_avx2/` and
`vc2hqdecode/vc2inversetransform_avx2/`. General CPU improvements belong on
`optimizations` first and should be merged into this branch.

## Running an evolutionary generation

The controller is proposer-agnostic. Configure an executable that reads
`{prompt_file}` and writes `{output_file}`. CUDA evaluation defaults to one
parallel candidate to avoid GPU contention:

```powershell
py -3 .\autoresearch\run.py `
  --config .\autoresearch\config.cuda.json `
  run --generations 4 --candidates 2 --parallel 1 `
  --full-if-promising `
  --proposer-command 'agent-command --prompt {prompt_file} --output {output_file}'
```

The proposer must return the metadata and exact `FILE`/`SEARCH`/`REPLACE`
format documented in `autoresearch/README.md`. Do not grant it a shortcut
around the evaluator.

If the current LLM cannot invoke itself as a subprocess, it may act as the
proposer interactively, but it must preserve the same discipline:

1. create a detached candidate worktree from the selected full-tier parent;
2. make one coherent CUDA experiment;
3. run `evaluate.py --tier smoke` and reject any build, runtime, or hash
   failure;
4. run `evaluate.py --tier quick` to screen both CUDA and CPU-fallback latency;
5. run `evaluate.py --tier full` before retaining the experiment;
6. compare all four fps metrics and reject material regressions;
7. preserve the candidate with a commit or `refs/autoresearch/...` ref;
8. document both retained and rejected results in the CUDA profile.

Run an individual tier with:

```powershell
py -3 .\autoresearch\evaluate.py `
  --config .\autoresearch\config.cuda.json `
  --tier smoke
```

Replace `smoke` with `quick` or `full` as the candidate advances.

Never modify tests, inputs, evaluator workloads, runtime environment variants,
hashes, validator invocation, or tier duration to make a candidate pass.

## Acceptance and promotion

A candidate is eligible only when:

- CUDA Release build and all native tests pass;
- `VC2HQ_CUDA=1` and `VC2HQ_CUDA=0` stream/pixel hashes match the same-tier
  parent and one another;
- the full tier reports validator success for both variants;
- at least one target metric improves beyond the configured minimum;
- no other configured metric exceeds the regression tolerance;
- the result survives repeated measurement without GPU contention;
- CUDA error handling, synchronization, and fallback behavior remain sound.

Accepted candidates are kept under
`refs/autoresearch/cuda-implementation/<candidate-id>`. Inspect the complete
diff before promotion. `--promote` cherry-picks only into a clean worktree.
After promotion, rerun the full baseline, update the relevant CUDA profile,
commit, and push `cuda-implementation`.

## Known pitfalls

- CUDA smoke uses four benchmark frames because one frame can round decoder
  time to zero and print `inffps`. It still uses one-frame correctness hashes.
- Warm-up, power state, and concurrent GPU activity can dominate short runs.
  Smoke only proves viability; full-tier medians decide retention.
- Higher occupancy is not automatically faster. Track register pressure,
  spills, memory traffic, synchronization, and end-to-end wall time together.
- Asynchronous APIs are useful only when work actually overlaps; hidden
  synchronization can erase the gain.
- Keep sm_61 compatibility unless the deployment requirement is explicitly
  changed.
- The CPU fallback is a scored objective, not merely a correctness fallback.
- Do not copy shared CPU changes directly. Merge them from `optimizations`.

## Copy/paste instruction for the next LLM

```text
Continue the VC-2 CUDA autoresearch loop on the cuda-implementation branch.
Read autoresearch/LLM-HANDOFF.md completely, then read every file it marks as
required. Preserve existing work and exact CPU/CUDA output equivalence.
Verify the CUDA device/toolchain and the full-tier baseline, inspect the
program database and previous negative results, run focused candidates through
smoke, quick, and full evaluation, retain only repeatable end-to-end latency
improvements with no CPU-fallback regression, update the CUDA profile, and
leave the branch tested and documented. Do not stop at ideation when a safe
candidate can be implemented and measured.
```
