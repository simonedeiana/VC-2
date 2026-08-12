# Upstream source provenance

This repository vendors four BBC VC-2 projects into one Windows-oriented
source tree. The initial snapshot was made from these upstream revisions:

| Directory | Revision | Branch | Upstream |
|---|---|---|---|
| `vc2-reference` | `46639dcf32d3e510991b0ce599967134d1f05360` | `master` | <https://github.com/bbc/vc2-reference.git> |
| `vc2hqencode` | `600adbd7325ce572d4b04ce98dc21be005d660ac` | `master` | <https://github.com/bbc/vc2hqencode.git> |
| `vc2hqdecode` | `091661e2cf93a545161397df5e26de5e80304253` | `master` | <https://github.com/bbc/vc2hqdecode.git> |
| `vc2_conformance` | `2a280349d4b9fd870ca6c0d92d301bbc22612adc` | `master` | <https://github.com/bbc/vc2_conformance.git> |

The original `.git` directories are retained locally under `.upstream-git/`
and ignored by this repository. They are not needed to build the project, but
can be used to inspect the original histories or reconstruct upstream diffs.
