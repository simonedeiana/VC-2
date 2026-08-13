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

The decoder remains CPU-only. A useful CUDA decoder must fuse VLC decoding,
dequantization, inverse transform, and final pixel output; inverse-transform
offload alone repeats the transfer-bound failure of the original encoder
prototype. Multiple pictures in flight remain explicitly out of scope.

Detailed profiler evidence and rejected experiments are in
`profiles/CUDA-LOW-LATENCY.md`.
