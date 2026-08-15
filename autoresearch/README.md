# VC-2 autoresearch

This directory implements an evolutionary coding loop inspired by
[AlphaEvolve](https://arxiv.org/abs/2506.13131) for single-thread VC-2 encoder
and decoder latency.

The implementation mirrors the paper's core strategy:

- a SQLite program database stores candidates, parentage, scores, output
  hashes, measurements, and evaluator feedback;
- island/MAP-Elites-style sampling balances exploitation of the fastest
  candidates with exploration across codec subsystems;
- prompts contain repository context, the parent result, and diverse prior
  programs;
- proposers return targeted `SEARCH`/`REPLACE` mutations;
- every candidate runs in a detached Git worktree;
- an evaluation cascade prunes build, test, or correctness failures before
  spending time on repeated benchmarks;
- multiple encoder and decoder fps metrics steer the search, while exact
  stream/pixel hashes and the VC-2 validator are hard gates;
- independent candidates can be evaluated concurrently.

No model SDK is required. The controller accepts any proposer executable that
can read a prompt file and write a proposal file. This keeps credentials and
model choice outside the repository.

For continuation by another coding agent, start with
`autoresearch/LLM-HANDOFF.md`; it contains the current baseline, source map,
required reading, operational protocol, and a ready-to-use kickoff prompt.

## Baseline

From the desired branch worktree:

```powershell
py -3 .\autoresearch\run.py `
  --config .\autoresearch\config.optimizations.json `
  baseline --tier full
```

For CUDA, use `config.cuda.json`. The CUDA evaluator measures both
`VC2HQ_CUDA=1` and `VC2HQ_CUDA=0`, preventing GPU improvements from silently
regressing the CPU fallback.

## Evolution loop

Configure `proposer.command` as an argument array, or pass it on the command
line. Three placeholders are expanded: `{worktree}`, `{prompt_file}`, and
`{output_file}`.

```powershell
py -3 .\autoresearch\run.py `
  --config .\autoresearch\config.optimizations.json `
  run --generations 4 --candidates 4 --parallel 2 `
  --full-if-promising `
  --proposer-command 'my-agent --prompt {prompt_file} --output {output_file}'
```

Accepted candidates are preserved under `refs/autoresearch/<branch>/<id>`.
They remain isolated unless `--promote` is supplied. Promotion requires a
clean target worktree and uses `git cherry-pick`.

## Evaluation tiers

- `smoke`: Release build, all native CTest targets, one-frame exact-output
  checks, and a single short latency sample.
- `quick`: the same gates plus three alternating 8-frame measurements per
  configured encoder/decoder workload.
- `full`: nine 30-frame measurements and conformance validation of every
  smoke stream.

Generated inputs, builds, databases, prompts, streams, and decoded frames live
under `.autoresearch/` or `build-autoresearch/` and are ignored by Git.

## Proposal format

```text
TARGET: decoder
STRATEGY: reduce-vlc-dependency-chain
RISK: medium
RATIONALE: concise explanation

FILE: vc2hqdecode/path/to/file.cpp
<<<<<<< SEARCH
exact existing text
=======
replacement text
>>>>>>> REPLACE
```

Each `SEARCH` block must match exactly once. Path traversal and edits outside
the candidate worktree are rejected.
