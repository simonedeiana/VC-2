# CPU optimization context

The branch already contains measured SSE4.2 and AVX2 work. Consult
`profiles/OPTIMIZATION-PROFILE.md` before proposing an experiment so rejected
ideas are not repeated.

Hard constraints:

- MSVC 2022, C++14, x86-64.
- One worker thread is the primary latency target.
- Encoded streams and decoded YUV must remain byte-identical.
- Preserve scalar/SSE fallback behavior and non-multiple geometry handling.
- Do not weaken tests, validator checks, benchmark duration, or hash gates.
- Prefer changes in codec kernels and dispatch over benchmark-specific tuning.

The current search frontier includes transform memory traffic, entropy/VLC
dependency chains, quantizer search, serialisation, cache layout, and pipeline
scheduling. Treat measured regressions in the optimization profile as negative
examples.
