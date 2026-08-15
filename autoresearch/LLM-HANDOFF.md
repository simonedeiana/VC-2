# LLM handoff: continue CPU autoresearch

Point a future coding LLM at this file and tell it: **continue the VC-2 CPU
autoresearch loop**. This document is the complete entry point. Read it before
editing code or launching experiments.

## Mission

Reduce end-to-end, single-worker encoding and decoding latency on the
`optimizations` branch while preserving:

- byte-identical VC-2 encoded streams;
- byte-identical decoded 10-bit 4:2:2 pixels;
- the public API and supported input geometries;
- scalar and SSE fallback paths;
- all native tests and VC-2 conformance validation.

The repository implements an evolutionary coding loop inspired by
AlphaEvolve. The controller, evaluator, prompt construction, program database,
and selection logic are already implemented. Continue using them; do not
replace the framework with informal one-off timing.

## Read these files first

1. `autoresearch/README.md` — controller usage and proposal format.
2. `autoresearch/BACKGROUND-CPU.md` — hard constraints and search frontier.
3. `profiles/OPTIMIZATION-PROFILE.md` — every retained and rejected trial.
4. `README-optimizations.md` — branch history and current optimization set.
5. `autoresearch/config.optimizations.json` — workloads, tiers, islands, and
   acceptance thresholds.
6. `autoresearch/core.py`, `evaluate.py`, and `run.py` — source of truth for
   selection and validation behavior.

Do not repeat a rejected experiment until you can state what materially
changed in the new formulation.

## Establish the current state

Run these commands from the repository root:

```powershell
git branch --show-current
git status --short
git log -1 --oneline
py -3 -m unittest discover -s .\autoresearch\tests -v
py -3 .\autoresearch\run.py `
  --config .\autoresearch\config.optimizations.json status
```

The intended branch is `optimizations`. Never discard a dirty worktree. If
the branch is checked out in another worktree, either work there or create a
new isolated worktree.

Generated state is intentionally untracked:

- `.autoresearch/optimizations.sqlite3` — program database and elites;
- `.autoresearch/data/` — deterministic raw inputs;
- `.autoresearch/artifacts/` — streams and decoded frames;
- `.autoresearch/validator/` — validator output;
- `build-autoresearch/` — MSVC Release build.

If the database is absent, or its newest full baseline commit differs from
`git rev-parse HEAD`, create a new full baseline:

```powershell
py -3 .\autoresearch\run.py `
  --config .\autoresearch\config.optimizations.json `
  baseline --tier full
```

Do not compare candidates from different tiers. The database and elite cells
are partitioned by `smoke`, `quick`, and `full` for this reason.

## Last known full baseline

Recorded on 2026-08-15 at commit `ab05d42d635f` using nine 30-frame runs,
1920x1080 10-bit 4:2:2 input, `--speed=fastest`, depth 3, and one worker.
Three streams passed the installed VC-2 validator and all 6 native tests
passed.

| Metric | Median |
|---|---:|
| CPU encoder Haar0 | 40.372 fps |
| CPU encoder DD9/7 | 33.934 fps |
| CPU encoder DD13/7 | 31.702 fps |
| CPU decoder Haar0 | 100.610 fps |
| CPU decoder DD9/7 | 56.258 fps |
| CPU decoder DD13/7 | 50.212 fps |

Correctness hashes:

| Workload | Stream SHA-256 | Decoded-pixel SHA-256 |
|---|---|---|
| Haar0 | `64E69879FF57B664A81A63AF9947956133FB1A47D928EE32712C8AD20A4250E6` | `8C391AAEA4257065028876E6A36EC21180F9B447324AEF23E1B2AB3B556FCEFC` |
| DD9/7 | `17E3DEC7990A2D956C2E6DE3B7AD82471FB3B202E6312046B0FE81B2F93A3F42` | `E8827E19AC2F6CBD3519E0C6337AA8E71AF3B5D9669915E21CA73F88D84739AD` |
| DD13/7 | `D2156BE23BFF167F8D8008FC5A128EF6671EF02C5046811DED0F727737870EF2` | `1CD6D773CFD6D131D11F486B08D7C72CB0DB9B56C8946BAC1C7ED2DE2E0A0388` |

These numbers are historical orientation, not permanent acceptance
thresholds. Re-baseline after code, compiler, power-plan, or hardware changes.

## Search islands and source map

The configured islands deliberately preserve diverse lines of research:

- `encoder-transform`: `vc2hqencode/vc2transform_avx2/`, especially the DD
  kernels and transform dispatch;
- `encoder-entropy`: `vc2hqencode/vc2hqencode/quantise.cpp`,
  `quantiserselection.hpp`, `encode_slice_component_optimised.hpp`, and
  `serialise.hpp`;
- `decoder-transform`: `vc2hqdecode/vc2inversetransform_avx2/` and transform
  dispatch;
- `decoder-entropy`: `vc2hqdecode/vc2inversetransform_sse4_2/vlc_sse4_2.cpp`
  and dequantisation code;
- `memory-layout`: coefficient buffers, stride handling, temporary buffers,
  cache traffic, and output stores;
- `pipeline-scheduling`: `VC2Encoder.cpp`, `VC2Decoder.cpp`, job construction,
  and stage overlap that does not add worker threads.

The primary target is latency with `--threads=1`. Do not optimize only an
isolated kernel while regressing end-to-end fps.

## Running an evolutionary generation

The controller is proposer-agnostic. Configure an executable that reads
`{prompt_file}` and writes `{output_file}`. Then run, for example:

```powershell
py -3 .\autoresearch\run.py `
  --config .\autoresearch\config.optimizations.json `
  run --generations 4 --candidates 4 --parallel 2 `
  --full-if-promising `
  --proposer-command 'agent-command --prompt {prompt_file} --output {output_file}'
```

The proposer must return the metadata and exact `FILE`/`SEARCH`/`REPLACE`
format documented in `autoresearch/README.md`. Do not grant the proposer a
shortcut around the evaluator.

If the current LLM cannot invoke itself as a subprocess, it may act as the
proposer interactively, but it must preserve the same discipline:

1. create a detached candidate worktree from the selected full-tier parent;
2. make one coherent experiment;
3. run `evaluate.py --tier smoke` and reject any failure or hash change;
4. run `evaluate.py --tier quick` to screen latency;
5. run `evaluate.py --tier full` before retaining the experiment;
6. compare every fps metric and reject material regressions;
7. preserve the candidate with a commit or `refs/autoresearch/...` ref;
8. record the result, including rejected ideas, in the optimization profile.

Run an individual tier with:

```powershell
py -3 .\autoresearch\evaluate.py `
  --config .\autoresearch\config.optimizations.json `
  --tier smoke
```

Replace `smoke` with `quick` or `full` as the candidate advances.

Never modify tests, inputs, evaluator workloads, hashes, validator invocation,
or tier duration to make a candidate pass.

## Acceptance and promotion

A candidate is eligible only when:

- Release build and all native tests pass;
- every stream and decoded-pixel hash matches its same-tier parent;
- the full tier reports validator success;
- at least one target metric improves beyond the configured minimum;
- no other configured metric exceeds the regression tolerance;
- the result survives repeated measurement and is technically explainable.

Accepted controller candidates are kept under
`refs/autoresearch/optimizations/<candidate-id>`. Inspect the complete diff
before promotion. `--promote` cherry-picks only into a clean target worktree.
After promotion, rerun the full baseline, update
`profiles/OPTIMIZATION-PROFILE.md`, commit, and push `optimizations`.

Periodically merge `optimizations` into `cuda-implementation`; never copy CPU
changes independently into CUDA if a branch merge preserves the history.

## Known pitfalls

- Fidelity is exposed in some command-line help but is rejected by the current
  encoder dispatch; do not treat it as a valid benchmark workload.
- Single-sample measurements are noisy. Smoke only proves viability; it is not
  evidence for retention.
- The DD horizontal/output stages can be memory-bound. Earlier SIMD attempts
  lost to scalar because strided traffic outweighed arithmetic savings.
- Matched-length VLC grouping and several recurrence-breaking experiments
  were measured and rejected. Read the profile before revisiting them.
- Keep boundary mirroring, signed rounding, narrowing, and short/non-multiple
  geometry fallbacks exactly equivalent.
- Profilers may use software sampling. Do not assume hardware-counter data is
  available.

## Copy/paste instruction for the next LLM

```text
Continue the VC-2 CPU autoresearch loop on the optimizations branch. Read
autoresearch/LLM-HANDOFF.md completely, then read every file it marks as
required. Preserve existing work and exact-output correctness. Verify or
recreate the full-tier baseline, inspect the program database and previous
negative results, run focused candidates through smoke, quick, and full
evaluation, retain only repeatable end-to-end latency improvements, update the
optimization profile, and leave the branch tested and documented. Do not stop
at ideation when a safe candidate can be implemented and measured.
```
