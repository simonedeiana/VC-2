/*****************************************************************************
 * decode_cuda.cu : Experimental CUDA fused decoder
 *****************************************************************************
 * Copyright (C) 2026
 *
 * Two kernels per picture:
 *
 *   kernel 1 (VLC + dequant): every lane decodes one independent component
 *     VLC stream (three per slice) and dequantizes it inline straight into
 *     the coefficient plane. The whole picture is processed with 32 lanes
 *     active instead of the three that a single slice needs, because the
 *     serial VLC chains are the dominant cost. The decode LUT lives in
 *     read-only global memory so 32 divergent lookups per warp do not
 *     serialize on the constant cache.
 *
 *   kernel 2 (transform + output): one warp per slice runs the hierarchical
 *     inverse transform on the slice region and writes clipped pixels to the
 *     output planes.
 *
 * The slice is a complete wavelet tree, so every slice is independent and
 * the transform is slice-local. Only pixels cross back to the host.
 *****************************************************************************/

#include "decode_cuda.hpp"

#include <cuda_runtime.h>
#include <cstdlib>
#include <mutex>
#include <stddef.h>
#include <string>
#include <vector>
#include <algorithm>

namespace {

// Mirror of the CPU LUTEntry in vc2hqdecode/lut.hpp (16 bytes, exact field
// order so the table upload is a plain byte copy).
struct __align__(16) LutEntry {
  uint8_t state;
  uint8_t preshift;
  int8_t  sgn;
  int8_t  term;
  uint8_t V;
  int8_t  N;
  uint8_t _pad0;
  uint8_t _pad1;
  int8_t  val[8];   // layout matches lut.hpp val0..val7
};

// Device-side copy of VC2CudaSlice.
struct SliceEntry {
  int32_t qindex;
  int32_t off[3];
  int32_t len[3];
};

// Process-wide persistent state. The decoder may call the CUDA path from any
// pool thread, so this is a mutex-protected singleton like the encoder.
uint8_t *device_payload = nullptr;
size_t payload_capacity = 0;
uint8_t *pinned_payload = nullptr;
size_t pinned_payload_capacity = 0;
uint16_t *pinned_out[3] = {};
size_t pinned_out_capacity[3] = {};
SliceEntry *device_table = nullptr;
size_t table_capacity = 0;
LutEntry *device_lut = nullptr;
int16_t *device_plane[2] = {};
size_t plane_capacity[2] = {};
uint16_t *device_out[2] = {};
size_t out_capacity[2] = {};
uint32_t *device_scan_y = nullptr;
uint32_t *device_scan_c = nullptr;
int32_t *device_qfactor = nullptr;
int32_t *device_qoffset = nullptr;
size_t qfactor_bytes = 0;
size_t qoffset_bytes = 0;
cudaStream_t decode_stream = nullptr;
cudaStream_t copy_stream[3] = {};
cudaEvent_t transform_done = nullptr;
bool lut_uploaded = false;
bool tables_uploaded = false;
std::mutex cuda_mutex;
thread_local std::string last_error;

int g_slices_x = 0;
int g_slices_y = 0;
int g_slice_width = 0;
int g_slice_height = 0;
int g_depth = 0;
int g_shift = 0;
int g_active_bits = 0;
int g_plane_stride[3] = {};
int g_out_width[3] = {};
int g_out_height[3] = {};

bool cuda_ok(cudaError_t result, const char *operation) {
  if (result == cudaSuccess)
    return true;
  last_error = operation;
  last_error += ": ";
  last_error += cudaGetErrorString(result);
  return false;
}

bool reserve_buffer(void **buffer, size_t *capacity, size_t required) {
  if (*capacity >= required)
    return true;
  if (*buffer)
    cudaFree(*buffer);
  *buffer = nullptr;
  *capacity = 0;
  if (!cuda_ok(cudaMalloc(buffer, required), "cudaMalloc"))
    return false;
  *capacity = required;
  return true;
}

// ---------------------------------------------------------------------------
// Dequant helper shared by the VLC lanes. `scan` packs the true plane offset
// in the low 16 bits and the (l*4+s) quantiser table index in the high 16.
// The result is saturated to int16 exactly like _mm_packs_epi32 in the SSE4.2
// dequantiser that this machine's CPU decoder dispatches to, and a zero
// coefficient stays zero (the CPU multiplies by sgn(D)).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void dequant_store_one(
    int32_t D, int idx, const uint32_t *scan,
    const int32_t *qfrow, const int32_t *qorow,
    int16_t *plane, int base) {
  const uint32_t ps = scan[idx];
  if (D == 0) {
    plane[base + (ps & 0xFFFF)] = 0;
    return;
  }
  const int32_t qf = qfrow[ps >> 16];
  const int32_t qo = qorow[ps >> 16];
  const int32_t a = D < 0 ? -D : D;
  int32_t X = (a * qf + qo) >> 2;
  if (D < 0)
    X = -X;
  if (X > 32767)
    X = 32767;
  else if (X < -32768)
    X = -32768;
  plane[base + (ps & 0xFFFF)] = static_cast<int16_t>(X);
}

// Per-stream decode state for the ILP VLC loop.
struct VlcState {
  const uint8_t *idata;
  const uint32_t *scan;
  const int32_t *qfrow;
  const int32_t *qorow;
  int16_t *plane;
  int base;
  int ilength;
  int olength;
  int icounter;
  int ocounter;
  int32_t V;
  const LutEntry *lut;
  const LutEntry *next;
  bool active;
};

__device__ __forceinline__ void vlc_emit(const LutEntry &E, VlcState &st) {
  const int n = E.N;
  const int32_t base_val = (st.V - 1) * E.sgn;
  int idx = st.ocounter;
  if (idx < st.olength) {
    dequant_store_one(base_val, idx, st.scan, st.qfrow, st.qorow, st.plane, st.base);
    ++idx;
    for (int k = 1; k < n && idx < st.olength; ++k, ++idx)
      dequant_store_one(E.val[k], idx, st.scan, st.qfrow, st.qorow, st.plane, st.base);
  }
  if (E.term)
    st.V = E.V;
  st.ocounter += n;
}

// One main-loop iteration. Returns false when the stream is exhausted.
__device__ __forceinline__ bool vlc_step(VlcState &st) {
  const LutEntry E = *st.next;
  st.next = &st.lut[(static_cast<int>(E.state) << 8) + st.idata[st.icounter]];
  st.icounter++;
  st.V = (st.V << E.preshift) | E.val[0];
  vlc_emit(E, st);
  return (st.icounter < st.ilength) && (st.ocounter < st.olength);
}

// Tail handling after the main loop: the pending LUT entry and the terminal
// state, exactly as decode_c() does once the input bytes are exhausted.
__device__ void vlc_tail(VlcState &st) {
  if (st.ocounter >= st.olength)
    return;
  const LutEntry E = *st.next;
  const int state = static_cast<int>(E.state);
  st.V = (st.V << E.preshift) | E.val[0];
  vlc_emit(E, st);
  if (st.ocounter < st.olength) {
    switch (state) {
    case 2: // STATE_DATA
      st.V = (st.V << 1) + 1;
    case 1: // STATE_FOLLOW
    case 3: // STATE_SIGN
      dequant_store_one(-(st.V - 1), st.ocounter, st.scan, st.qfrow, st.qorow,
                        st.plane, st.base);
      break;
    }
    st.ocounter = (st.ocounter + 4) & ~3;
  }
}

__device__ void vlc_init(VlcState &st, const uint8_t *idata, int ilength, int olength,
                         const uint32_t *scan, const int32_t *qfrow, const int32_t *qorow,
                         const LutEntry *lut, int16_t *plane, int base) {
  st.idata = idata;
  st.scan = scan;
  st.qfrow = qfrow;
  st.qorow = qorow;
  st.plane = plane;
  st.base = base;
  st.ilength = ilength;
  st.olength = olength;
  st.icounter = 1;
  st.ocounter = 0;
  st.V = 0;
  st.lut = lut;
  st.next = (ilength > 0) ? &lut[idata[0]] : &lut[0xFF];
  st.active = ilength > 0;
}

// Serial VLC decode of one component stream with inline dequantization.
// Faithful port of decode_c() in vc2inversetransform_c/vlc_c.cpp; the plane
// was zeroed beforehand so undecoded coefficients stay zero.
__device__ void vlc_decode_dequant(VlcState &st) {
  if (st.active) {
    while (vlc_step(st)) {
    }
  }
  vlc_tail(st);
}

// Fill the decode state for one component stream from the slice table.
__device__ void vlc_stream_fill(
    VlcState &st, int stream_id,
    const uint8_t *__restrict__ payload, const SliceEntry *__restrict__ slices,
    const uint32_t *__restrict__ scan_y, const uint32_t *__restrict__ scan_c,
    const int32_t *__restrict__ qfactor, const int32_t *__restrict__ qoffset,
    const LutEntry *lut,
    int16_t *__restrict__ plane_y, int stride_y,
    int16_t *__restrict__ plane_c, int stride_c,
    int slices_x, int slices_y, int slice_width, int slice_height, int depth) {
  const int sid = stream_id / 3;
  const int comp = stream_id % 3;
  const int sx = sid % slices_x;
  const int sy = sid / slices_x;
  const SliceEntry s = slices[sid];
  const int32_t *qfrow = qfactor + s.qindex * ((depth + 1) * 4);
  const int32_t *qorow = qoffset + s.qindex * ((depth + 1) * 4);
  const bool is_y = (comp == 0);
  const int cw = is_y ? slice_width : slice_width / 2;
  const int olength = cw * slice_height;
  const uint32_t *scan = is_y ? scan_y : scan_c;
  int16_t *plane = plane_y;
  int stride = stride_y;
  if (!is_y) {
    plane = plane_c + (comp - 1) * static_cast<size_t>(stride_c) * (slice_height * slices_y);
    stride = stride_c;
  }
  const int base = sy * slice_height * stride + sx * cw;
  vlc_init(st, payload + s.off[comp], s.len[comp], olength, scan, qfrow, qorow,
           lut, plane, base);
}

// ---------------------------------------------------------------------------
// Kernel 1: VLC decode + dequant. Every lane decodes one component stream.
// ---------------------------------------------------------------------------
__global__ void vc2_decode_vlc_kernel(
    const uint8_t *__restrict__ payload,
    const SliceEntry *__restrict__ slices,
    const uint32_t *__restrict__ scan_y,
    const uint32_t *__restrict__ scan_c,
    const int32_t *__restrict__ qfactor,
    const int32_t *__restrict__ qoffset,
    const LutEntry *__restrict__ lut,
    int16_t *__restrict__ plane_y, int stride_y,
    int16_t *__restrict__ plane_c, int stride_c,
    int slices_x, int slices_y,
    int slice_width, int slice_height, int depth,
    int n_streams) {
  const int stream_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (stream_id >= n_streams)
    return;

  VlcState st;
  vlc_stream_fill(st, stream_id, payload, slices, scan_y, scan_c, qfactor, qoffset, lut,
                  plane_y, stride_y, plane_c, stride_c,
                  slices_x, slices_y, slice_width, slice_height, depth);
  vlc_decode_dequant(st);
}

// ---------------------------------------------------------------------------
// Inverse Haar passes. All operate in place on a slice region. The plane is
// int16 and every store truncates to int16 exactly like the CPU templates.
// ---------------------------------------------------------------------------
__device__ void haar_v_inplace(int16_t *plane, int stride, int base,
                               int sw, int sh, int skip, int lane) {
  const int xc = sw / skip;
  const int yc = sh / (2 * skip);
  for (int i = lane; i < yc * xc; i += 32) {
    const int y = (i / xc) * (2 * skip);
    const int x = (i % xc) * skip;
    const int32_t D = plane[base + y * stride + x];
    const int32_t Dp1 = plane[base + (y + skip) * stride + x];
    const int32_t X = D - ((Dp1 + 1) >> 1);
    const int32_t Xp1 = Dp1 + X;
    plane[base + y * stride + x] = static_cast<int16_t>(X);
    plane[base + (y + skip) * stride + x] = static_cast<int16_t>(Xp1);
  }
}

__device__ void haar_h_inplace(int16_t *plane, int stride, int base,
                               int sw, int sh, int skip, int shift, int lane) {
  const int xc = sw / (2 * skip);
  const int yc = sh / skip;
  for (int i = lane; i < yc * xc; i += 32) {
    const int y = (i / xc) * skip;
    const int x = (i % xc) * (2 * skip);
    const int32_t D = plane[base + y * stride + x];
    const int32_t Dp1 = plane[base + y * stride + x + skip];
    int32_t X = D - ((Dp1 + 1) >> 1);
    int32_t Xp1 = Dp1 + X;
    if (shift != 0) {
      X = (X + (1 << (shift - 1))) >> shift;
      Xp1 = (Xp1 + (1 << (shift - 1))) >> shift;
    }
    plane[base + y * stride + x] = static_cast<int16_t>(X);
    plane[base + y * stride + x + skip] = static_cast<int16_t>(Xp1);
  }
}

__device__ void haar_h_final(int16_t *plane, int stride, int base,
                             int sw, int sh, int shift, int active_bits,
                             uint16_t *odata, int ostride, int outbase,
                             int ow, int oh, int lane) {
  const int xc = sw / 2;
  const uint16_t clip = static_cast<uint16_t>((1u << active_bits) - 1);
  const int off = 1 << (active_bits - 1);
  for (int i = lane; i < sh * xc; i += 32) {
    const int y = i / xc;
    const int x = (i % xc) * 2;
    if (y >= oh || x >= ow)
      continue;
    const int32_t D = plane[base + y * stride + x];
    const int32_t Dp1 = plane[base + y * stride + x + 1];
    int32_t X = D - ((Dp1 + 1) >> 1);
    int32_t Xp1 = Dp1 + X;
    if (shift != 0) {
      X = (X + (1 << (shift - 1))) >> shift;
      Xp1 = (Xp1 + (1 << (shift - 1))) >> shift;
    }
    int32_t o0 = X + off;
    int32_t o1 = Xp1 + off;
    if (o0 < 0) o0 = 0;
    if (o0 > clip) o0 = clip;
    if (o1 < 0) o1 = 0;
    if (o1 > clip) o1 = clip;
    odata[outbase + y * ostride + x] = static_cast<uint16_t>(o0);
    odata[outbase + y * ostride + x + 1] = static_cast<uint16_t>(o1);
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: inverse transform + output, one warp per slice. The transform is
// slice-local so each warp reads/writes only its own region of the planes.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128, 12) vc2_decode_transform_kernel(
    int16_t *__restrict__ plane_y, int stride_y,
    int16_t *__restrict__ plane_c, int stride_c,
    uint16_t *__restrict__ out_y, int ostride_y,
    uint16_t *__restrict__ out_c, int ostride_c,
    int slices_x, int slices_y,
    int slice_width, int slice_height,
    int depth, int shift, int active_bits,
    int frame_width, int frame_height) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp_id >= slices_x * slices_y)
    return;

  const int sid = warp_id;
  const int sx = sid % slices_x;
  const int sy = sid / slices_x;

  for (int c = 0; c < 3; ++c) {
    const int cw = (c == 0) ? slice_width : slice_width / 2;
    int16_t *plane = plane_y;
    int stride = stride_y;
    if (c != 0) {
      plane = plane_c + (c - 1) * static_cast<size_t>(stride_c) * (slice_height * slices_y);
      stride = stride_c;
    }
    const int base = (sy * slice_height) * stride + sx * cw;

    for (int l = 0; l < depth - 1; ++l) {
      const int skip = 1 << (depth - 1 - l);
      haar_v_inplace(plane, stride, base, cw, slice_height, skip, lane);
      __syncwarp();
      haar_h_inplace(plane, stride, base, cw, slice_height, skip, shift, lane);
      __syncwarp();
    }
    haar_v_inplace(plane, stride, base, cw, slice_height, 1, lane);
    __syncwarp();

    uint16_t *out = out_y;
    int ostride = ostride_y;
    if (c != 0) {
      out = out_c + (c - 1) * static_cast<size_t>(ostride_c) * frame_height;
      ostride = ostride_c;
    }
    const int fw = (c == 0) ? frame_width : frame_width / 2;
    const int ow = (cw < fw - sx * cw) ? cw : (fw - sx * cw);
    const int oh = (slice_height < frame_height - sy * slice_height)
                     ? slice_height : (frame_height - sy * slice_height);
    const int outbase = (sy * slice_height) * ostride + sx * cw;
    haar_h_final(plane, stride, base, cw, slice_height, shift, active_bits,
                 out, ostride, outbase, ow, oh, lane);
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// Host-side scan table construction: builds the packed scan tables (plane
// offset | (l*4+s)<<16) in the exact order the dequantisers read the VLC
// scratch (dequantise_c loop order). The plane offset uses the real picture
// stride, so a coefficient for slice-local row y / column x lands at
// y*stride + x inside the picture plane.
// ---------------------------------------------------------------------------
void build_scan_table(std::vector<uint32_t> &scan, int width, int height, int depth,
                      int stride) {
  scan.resize(static_cast<size_t>(width) * height);
  int n = 0;
  int skip = 1 << depth;
  // LL band: level 0, subband 0
  for (int y = 0; y < height; y += skip)
    for (int x = 0; x < width; x += skip)
      scan[n++] = static_cast<uint32_t>((0 << 16) | (y * stride + x));
  // Detail bands per level: HL(s=1), LH(s=2), HH(s=3)
  for (int l = 1; l <= depth; ++l) {
    const uint32_t lvl = static_cast<uint32_t>(l * 4);
    for (int y = 0; y < height; y += skip)
      for (int x = skip / 2; x < width; x += skip)
        scan[n++] = static_cast<uint32_t>((lvl + 1) << 16) | static_cast<uint32_t>(y * stride + x);
    for (int y = skip / 2; y < height; y += skip)
      for (int x = 0; x < width; x += skip)
        scan[n++] = static_cast<uint32_t>((lvl + 2) << 16) | static_cast<uint32_t>(y * stride + x);
    for (int y = skip / 2; y < height; y += skip)
      for (int x = skip / 2; x < width; x += skip)
        scan[n++] = static_cast<uint32_t>((lvl + 3) << 16) | static_cast<uint32_t>(y * stride + x);
    skip /= 2;
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------
bool vc2_cuda_decode_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

const char *vc2_cuda_decode_last_error() {
  return last_error.c_str();
}

bool vc2_cuda_decode_prepare(
    int slices_x, int slices_y, int slice_width, int slice_height,
    int depth, int wavelet_shift, int active_bits,
    const int out_width[3], const int out_height[3],
    const int plane_stride[3],
    const int32_t *qfactor, const int32_t *qoffset, int qtable_levels) {
  std::lock_guard<std::mutex> lock(cuda_mutex);
  last_error.clear();

  if (slices_x <= 0 || slices_y <= 0 || slice_width <= 0 || slice_height <= 0 ||
      depth < 1 || active_bits < 8 || active_bits > 16)
    return false;
  if (slice_width % (1 << depth) != 0 || slice_height % (1 << depth) != 0)
    return false;

  g_slices_x = slices_x;
  g_slices_y = slices_y;
  g_slice_width = slice_width;
  g_slice_height = slice_height;
  g_depth = depth;
  g_shift = wavelet_shift;
  g_active_bits = active_bits;
  for (int c = 0; c < 3; ++c) {
    g_plane_stride[c] = plane_stride[c];
    g_out_width[c] = out_width[c];
    g_out_height[c] = out_height[c];
  }

  if (!cuda_ok(cudaFree(nullptr), "CUDA context initialization"))
    return false;
  if (!decode_stream &&
      !cuda_ok(cudaStreamCreateWithFlags(&decode_stream, cudaStreamNonBlocking),
               "CUDA decoder stream creation"))
    return false;
  for (int c = 0; c < 3; ++c)
    if (!copy_stream[c] &&
        !cuda_ok(cudaStreamCreateWithFlags(&copy_stream[c], cudaStreamNonBlocking),
                 "CUDA decoder copy stream creation"))
      return false;
  if (!transform_done &&
      !cuda_ok(cudaEventCreateWithFlags(&transform_done, cudaEventDisableTiming),
               "CUDA decoder transform event creation"))
    return false;

  // Decode LUT in read-only global memory (divergent per-lane lookups would
  // serialize on the constant cache).
  const size_t lut_bytes = 1024 * sizeof(LutEntry);
  if (!lut_uploaded) {
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_lut), lut_bytes),
                 "VLC LUT allocation") ||
        !cuda_ok(cudaMemcpy(device_lut, vc2_cpu_vlclut(), lut_bytes,
                            cudaMemcpyHostToDevice), "VLC LUT upload"))
      return false;
    lut_uploaded = true;
  }

  // Coefficient planes: full picture stride x height for Y, and the two
  // chroma planes stacked in one allocation for Cb/Cr.
  const size_t y_bytes = static_cast<size_t>(plane_stride[0]) * out_height[0] * sizeof(int16_t);
  const size_t c_bytes = static_cast<size_t>(plane_stride[1]) * out_height[1] * sizeof(int16_t);
  if (!reserve_buffer(reinterpret_cast<void **>(&device_plane[0]), &plane_capacity[0], y_bytes) ||
      !reserve_buffer(reinterpret_cast<void **>(&device_plane[1]), &plane_capacity[1], 2 * c_bytes))
    return false;

  // Scan tables.
  if (!tables_uploaded || device_scan_y == nullptr) {
    std::vector<uint32_t> scan_y;
    std::vector<uint32_t> scan_c;
    build_scan_table(scan_y, slice_width, slice_height, depth, plane_stride[0]);
    build_scan_table(scan_c, slice_width / 2, slice_height, depth, plane_stride[1]);
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_scan_y),
                            scan_y.size() * sizeof(uint32_t)), "scan table Y allocation") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_scan_c),
                            scan_c.size() * sizeof(uint32_t)), "scan table C allocation") ||
        !cuda_ok(cudaMemcpy(device_scan_y, scan_y.data(), scan_y.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice), "scan table Y upload") ||
        !cuda_ok(cudaMemcpy(device_scan_c, scan_c.data(), scan_c.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice), "scan table C upload"))
      return false;
  }

  // Quantiser tables.
  const size_t qbytes = static_cast<size_t>(256) * (qtable_levels * 4) * sizeof(int32_t);
  if (!reserve_buffer(reinterpret_cast<void **>(&device_qfactor), &qfactor_bytes, qbytes) ||
      !reserve_buffer(reinterpret_cast<void **>(&device_qoffset), &qoffset_bytes, qbytes))
    return false;
  if (!cuda_ok(cudaMemcpy(device_qfactor, qfactor, qbytes, cudaMemcpyHostToDevice),
               "qfactor table upload") ||
      !cuda_ok(cudaMemcpy(device_qoffset, qoffset, qbytes, cudaMemcpyHostToDevice),
               "qoffset table upload"))
    return false;

  tables_uploaded = true;
  return true;
}

bool vc2_cuda_decode_picture(
    const VC2CudaSlice *table, int n_slices, int slices_x, int slices_y,
    const uint8_t *frame, size_t frame_bytes,
    uint16_t *odata[3], const int ostride[3]) {
  std::lock_guard<std::mutex> lock(cuda_mutex);
  last_error.clear();

  if (!table || !frame || !odata[0] || !odata[1] || !odata[2] || n_slices <= 0)
    return false;
  if (!device_plane[0] || !device_plane[1] || !device_scan_y || !device_qfactor)
    return false;

  const size_t table_bytes = static_cast<size_t>(n_slices) * sizeof(SliceEntry);
  if (!reserve_buffer(reinterpret_cast<void **>(&device_payload), &payload_capacity, frame_bytes) ||
      !reserve_buffer(reinterpret_cast<void **>(&device_table), &table_capacity, table_bytes))
    return false;

  // Stage the frame in a persistent pinned buffer so the upload is not
  // staged through pageable memory on every picture.
  if (pinned_payload_capacity < frame_bytes) {
    if (pinned_payload) {
      cudaFreeHost(pinned_payload);
      pinned_payload = nullptr;
      pinned_payload_capacity = 0;
    }
    if (!cuda_ok(cudaMallocHost(reinterpret_cast<void **>(&pinned_payload), frame_bytes),
                 "frame staging allocation"))
      return false;
    pinned_payload_capacity = frame_bytes;
  }
  memcpy(pinned_payload, frame, frame_bytes);

  // Output buffers sized to the host strides so the download is contiguous;
  // grow here because the actual output stride is only known per frame.
  const size_t out_y_bytes = static_cast<size_t>(ostride[0]) * g_out_height[0] * sizeof(uint16_t);
  const size_t out_c_bytes = static_cast<size_t>(ostride[1]) * g_out_height[1] * sizeof(uint16_t);
  if (!reserve_buffer(reinterpret_cast<void **>(&device_out[0]), &out_capacity[0], out_y_bytes) ||
      !reserve_buffer(reinterpret_cast<void **>(&device_out[1]), &out_capacity[1], 2 * out_c_bytes))
    return false;
  // Downloads land in persistent pinned staging buffers (registered once) and
  // are then copied to the caller's planes; the caller may change output
  // pointers every picture, which would otherwise force a re-registration per
  // picture (much more expensive than the copy itself).
  const size_t plane_bytes[3] = {out_y_bytes, out_c_bytes, out_c_bytes};
  for (int c = 0; c < 3; ++c)
    if (pinned_out_capacity[c] < plane_bytes[c]) {
      if (pinned_out[c])
        cudaFreeHost(pinned_out[c]);
      pinned_out[c] = nullptr;
      pinned_out_capacity[c] = 0;
      if (!cuda_ok(cudaMallocHost(reinterpret_cast<void **>(&pinned_out[c]), plane_bytes[c]),
                   "output staging allocation"))
        return false;
      pinned_out_capacity[c] = plane_bytes[c];
    }

  if (!cuda_ok(cudaMemcpyAsync(device_payload, pinned_payload, frame_bytes,
                               cudaMemcpyHostToDevice, decode_stream), "frame upload") ||
      !cuda_ok(cudaMemcpyAsync(device_table, table, table_bytes,
                               cudaMemcpyHostToDevice, decode_stream), "slice table upload"))
    return false;

  const int frame_width = g_out_width[0];
  const int frame_height = g_out_height[0];
  const size_t plane_c_stride = static_cast<size_t>(g_plane_stride[1]);
  const size_t y_plane_bytes = static_cast<size_t>(g_plane_stride[0]) * g_out_height[0] * sizeof(int16_t);
  const size_t c_plane_bytes = 2 * plane_c_stride * g_out_height[1] * sizeof(int16_t);

  // Zero the coefficient planes so undecoded coefficients stay zero.
  if (!cuda_ok(cudaMemsetAsync(device_plane[0], 0, y_plane_bytes, decode_stream),
               "coefficient plane zeroing (Y)") ||
      !cuda_ok(cudaMemsetAsync(device_plane[1], 0, c_plane_bytes, decode_stream),
               "coefficient plane zeroing (C)"))
    return false;

  // Kernel 1: VLC decode + dequant, one stream per lane.
  const int n_streams = n_slices * 3;
  const int block1 = 128;
  const int grid1 = (n_streams + block1 - 1) / block1;
  vc2_decode_vlc_kernel<<<grid1, block1, 0, decode_stream>>>(
      device_payload, device_table,
      device_scan_y, device_scan_c,
      device_qfactor, device_qoffset,
      device_lut,
      device_plane[0], g_plane_stride[0],
      device_plane[1], g_plane_stride[1],
      slices_x, slices_y,
      g_slice_width, g_slice_height, g_depth,
      n_streams);

  // Kernel 2: inverse transform + output, one warp per slice.
  const int warps_per_block = 4;
  const int block2 = warps_per_block * 32;
  const int grid2 = (n_slices + warps_per_block - 1) / warps_per_block;
  vc2_decode_transform_kernel<<<grid2, block2, 0, decode_stream>>>(
      device_plane[0], g_plane_stride[0],
      device_plane[1], g_plane_stride[1],
      device_out[0], ostride[0],
      device_out[1], ostride[1],
      slices_x, slices_y,
      g_slice_width, g_slice_height,
      g_depth, g_shift, g_active_bits,
      frame_width, frame_height);
  if (!cuda_ok(cudaGetLastError(), "CUDA decoder kernel launch"))
    return false;
  if (!cuda_ok(cudaEventRecord(transform_done, decode_stream),
               "CUDA decoder transform event"))
    return false;

  // Download the three output planes into the persistent pinned staging
  // buffers. Each plane copies on its own stream so the transfers overlap
  // instead of serializing on the decode stream.
  const size_t y_bytes = static_cast<size_t>(ostride[0]) * g_out_height[0] * sizeof(uint16_t);
  const size_t c_bytes = static_cast<size_t>(ostride[1]) * g_out_height[1] * sizeof(uint16_t);
  const uint16_t *out_src[3] = {
      device_out[0],
      device_out[1],
      device_out[1] + plane_c_stride * g_out_height[1]};
  const size_t out_bytes[3] = {y_bytes, c_bytes, c_bytes};
  for (int c = 0; c < 3; ++c) {
    if (!cuda_ok(cudaStreamWaitEvent(copy_stream[c], transform_done, 0),
                 "CUDA decoder copy dependency") ||
        !cuda_ok(cudaMemcpyAsync(pinned_out[c], out_src[c], out_bytes[c],
                                 cudaMemcpyDeviceToHost, copy_stream[c]),
                 "CUDA decoder output download"))
      return false;
  }
  for (int c = 0; c < 3; ++c)
    if (!cuda_ok(cudaStreamSynchronize(copy_stream[c]), "CUDA decoder synchronization"))
      return false;
  // Copy the staged planes to the caller's buffers (pageable is fine here).
  for (int c = 0; c < 3; ++c)
    memcpy(odata[c], pinned_out[c], out_bytes[c]);
  return true;
}
