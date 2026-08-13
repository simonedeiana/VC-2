# Experimental CUDA backend

The `cuda-implementation` branch contains an experimental end-to-end CUDA
encoder backend for the lowest-latency preset. For planar 10-bit input using
Haar-0 depth 3, 32x8 slices, `fastest`, and no fragments, input conversion,
wavelet transform, quantizer selection, VLC generation, and fixed-size slice
serialization all remain on the GPU. Only the compressed picture payload is
downloaded. Other configurations use the existing CPU encoder.

The backend is optional. Normal builds continue to use the existing
C/SSE/AVX dispatch and do not require CUDA.

## Build on Windows

Install Visual Studio 2022 with Desktop development with C++, CMake, and a CUDA
toolkit that supports the installed MSVC toolset. This branch was built with
CUDA 12.9 on a GeForce GTX 1050 Ti (compute capability 6.1).

From a Developer PowerShell:

```powershell
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DVC2_ENABLE_CUDA=ON
cmake --build build-cuda --config Release --target vc2encode
```

## Run

CUDA use is opt-in at runtime:

```powershell
$env:VC2HQ_CUDA = "1"
build-cuda\bin\vc2encode.exe -n=30 -t=12 --speed=fastest -w=haar0 -d=3 `
  --slicewidth=32 --sliceheight=8 -r=2 input.raw output.vc2
```

The log prints `CUDA [X]` when a device is available and the environment switch
is enabled. Unsupported layouts, wavelets, slice geometries, and speed presets
continue through the CPU backend.

## Current result

- Persistent allocations and host registrations keep setup outside the
  steady-state picture path.
- Three transform streams feed one encoding stream through CUDA events, with a
  single host synchronization per picture.
- Quantizer selection uses one warp per slice. The warp-parallel serializer
  packs VLC bits through shared memory and reuses final bit counts from the
  selector.
- The 30-picture 1080p stream is byte-for-byte identical to the CPU encoder
  (`d27e9c049ce776c3b14e2153286429e94ee312d440863fd20023e1f6b5d35a15`),
  and the VC-2 validator reports no errors.
- On the GTX 1050 Ti/WDDM test system, representative steady-state latency is
  about 2.87 ms at 720p and 4.44 ms at 1080p. This beats the local CPU encoder,
  but does not meet the 1 ms / 2 ms objectives.

The final 1080p Nsight trace measures about 0.69 ms for quantizer selection,
0.50 ms for warp-parallel serialization, about 0.66 ms for the three tiled
transforms, 0.20 ms for compressed-payload download, and roughly 0.71 ms summed
input upload time (the three uploads overlap). On this Pascal/WDDM platform the
device pipeline and transfers already exceed 2 ms before application overhead,
so further launch-level tuning cannot reach the 1080p target. A newer GPU, a
persistent-kernel design, or GPU-native input/output surfaces is required for a
credible attempt at that ceiling.

## CUDA decoder

A fused CUDA decoder backend is available for the same preset (Haar-0/Haar-1,
depth 3, 32x8 slices, planar 10-bit/12-bit 4:2:2). It is opt-in with the same
`VC2HQ_CUDA=1` environment switch and decodes the whole picture as one job:

- Kernel 1 decodes every component VLC stream (three per slice, 24,300
  streams for 1080p) with one lane per stream and dequantizes each coefficient
  inline straight into the coefficient plane. The decode LUT lives in
  read-only global memory so 32 divergent per-warp lookups do not serialize on
  the constant cache.
- Kernel 2 runs the slice-local inverse Haar transform and writes clipped
  pixels to the output planes; only pixels cross back over PCIe.
- The compressed frame is uploaded directly (slice offsets are
  frame-relative), and the coefficient planes are zeroed per picture so
  undecoded coefficients stay zero.

On the GTX 1050 Ti test system, the 1080p decoder reaches about
125-140 fps steady state (roughly 7-8 ms per picture) versus 36-40 fps for the
single-threaded CPU decoder, and 720p reaches about 209 fps versus 108 fps.
Output is byte-for-byte identical to the CPU decoder: a 30-picture 1080p
Haar-0 decode matches the CPU SHA-256 exactly
(`48a27492179d8c7181fd99508cc1b9fd902d6859b9f39bbae10e3a0346e85d46`), and
Haar-1 and 720p Haar-0 matches are also verified.

Nsight Systems per-frame 1080p measurements: about 1.6-1.7 ms for the VLC
kernel, 1.0 ms for the transform kernel, 0.2 ms frame upload, and about 1.1 ms
for the three output downloads. Unsupported layouts, wavelets, slice
geometries, interlaced streams, partial decode and colourise continue through
the CPU backend.

Detailed profiler evidence and rejected experiments are in
`profiles/CUDA-LOW-LATENCY.md` and `profiles/CUDA-DECODER.md`.
