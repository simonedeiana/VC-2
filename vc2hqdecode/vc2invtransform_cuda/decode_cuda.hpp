/*****************************************************************************
 * decode_cuda.hpp : Experimental CUDA fused decoder interface
 *****************************************************************************
 * Copyright (C) 2026
 *
 * This experimental backend keeps VLC decoding, dequantization, the inverse
 * wavelet transform, and final pixel output on the GPU so that only decoded
 * pixels cross back over PCIe. It is opt-in at run time (VC2HQ_CUDA=1) and
 * only handles the supported preset; everything else keeps the CPU path.
 *****************************************************************************/

#ifndef VC2_DECODE_CUDA_HPP
#define VC2_DECODE_CUDA_HPP

#include <stdint.h>
#include <stddef.h>

// Per-slice stream description built by the host from the parsed CodedSlice
// array. `offset[c]` is a byte offset into the (uploaded) picture frame, and
// `length[c]` the number of bytes of VLC data for component c. Because the
// slice data pointers already live inside the frame buffer, uploading the
// frame directly avoids copying the compressed payload.
struct VC2CudaSlice {
  int32_t qindex;
  int32_t offset[3];
  int32_t length[3];
};

// Returns true when a CUDA device is present (mirrors the encoder backend).
bool vc2_cuda_decode_available();
const char *vc2_cuda_decode_last_error();

#ifdef __cplusplus
extern "C" {
#endif
// Returns the CPU decoder's 1024-entry VLC lookup table (byte-exact copy of
// lut.hpp VLCLUT). Implemented in VC2Decoder.cpp; the CUDA layer uploads it
// to constant memory.
const void *vc2_cpu_vlclut();
#ifdef __cplusplus
}
#endif

// Configure persistent GPU buffers and constant tables for a fixed picture
// geometry. `qfactor`/`qoffset` are flat int32 tables indexed by
// [qindex * (depth+1) * 4 + l*4 + s] copied from the CPU quantisation
// matrices so the GPU dequantizes bit-exactly like the host.
//
// Supported preset (this branch): Haar-0/Haar-1, depth 3, 32x8 slices,
// planar 4:2:2, 10-bit or 12-bit output. `wavelet_shift` is 0 for Haar-0
// and 1 for Haar-1 (the horizontal rounding shift).
bool vc2_cuda_decode_prepare(
    int slices_x, int slices_y, int slice_width, int slice_height,
    int depth, int wavelet_shift, int active_bits,
    const int out_width[3], const int out_height[3],
    const int plane_stride[3],
    const int32_t *qfactor, const int32_t *qoffset, int qtable_levels);

// Decode one picture into `odata[3]` (host uint16 planes with `ostride[c]`).
// `table` has n_slices VC2CudaSlice entries whose offsets are relative to
// `frame`; `frame_bytes` is the whole compressed picture buffer (header plus
// slice payloads).
bool vc2_cuda_decode_picture(
    const VC2CudaSlice *table, int n_slices, int slices_x, int slices_y,
    const uint8_t *frame, size_t frame_bytes,
    uint16_t *odata[3], const int ostride[3]);

#endif
