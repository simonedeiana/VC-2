# Building VC-2 on Windows with MSVC

The workspace contains the BBC VC-2 reference codec, the high-performance HQ
encoder and decoder, and the Python conformance validator. The top-level CMake
project builds all native libraries, command-line programs, and native tests
with 64-bit MSVC from Visual Studio 2022.

## Requirements

- Visual Studio 2022 with **Desktop development with C++**
- Python 3 (used to generate two SIMD headers and run the validator)

The build script uses the CMake and vcpkg copies bundled with Visual Studio, so
neither needs to be on `PATH`.

## Build and test

From PowerShell:

```powershell
.\build-windows.ps1
```

Use `-Configuration Debug` for a debug build or `-Clean` to remove only the
local `build-msvc` directory before rebuilding. Installed output is placed in
`artifacts\bin`, `artifacts\lib`, and `artifacts\include`.

## Validator

The validator is installed editable in `.venv-validator`. Run it with:

```powershell
.\.venv-validator\Scripts\vc2-bitstream-validator.exe <stream.vc2>
```

Other installed conformance tools can be listed with:

```powershell
Get-ChildItem .\.venv-validator\Scripts\vc2-*.exe
```

The validator is Python software and is therefore installed rather than
compiled by MSVC. Its command-line entry points run natively on Windows.

## Lightweight codec stage profiling

Set `VC2HQ_PROFILE` to enable aggregate worker-stage timings on standard error:

```powershell
$env:VC2HQ_PROFILE = "1"
.\build-msvc\bin\vc2encode.exe --speed=fastest --wavelet=haar0 `
  --depth=3 --threads=12 --num-frames=30 --disable-output
Remove-Item Env:VC2HQ_PROFILE
```

The hooks are inactive when the environment variable is absent. Initial
profiling results are recorded in `profiles\INITIAL-PROFILE.md`.
