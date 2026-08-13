#include "transform_cuda.hpp"

#include <cuda_runtime.h>
#include <mutex>
#include <stddef.h>
#include <string>

namespace {

// The encoder may schedule the one-picture CUDA job on a different pool
// thread for each call.  Process-wide storage therefore provides genuinely
// persistent allocations; thread_local storage caused one allocation set per
// worker and large latency spikes.  The mutex also makes this experimental
// singleton safe if independent encoder objects are called concurrently.
uint16_t *device_input[3] = {};
int16_t *device_output[3] = {};
size_t input_capacity[3] = {};
size_t output_capacity[3] = {};
void *registered_input[3] = {};
size_t registered_input_bytes[3] = {};
int16_t *registered_output[3] = {};
size_t registered_output_bytes[3] = {};
cudaStream_t streams[3] = {};
cudaStream_t encode_stream = nullptr;
cudaEvent_t transform_done[3] = {};
int *device_max_sizes = nullptr;
int *device_output_offsets = nullptr;
uint8_t *device_qindices = nullptr;
int *device_slice_metrics = nullptr;
uint8_t *device_bitstream = nullptr;
size_t quantiser_capacity = 0;
size_t bitstream_capacity = 0;
uint16_t *device_matrix_m = nullptr;
uint8_t *device_matrix_sh = nullptr;
const uint16_t *cached_matrix_m = nullptr;
uint8_t *registered_bitstream = nullptr;
size_t registered_bitstream_bytes = 0;
int direct_n_slices = 0;
int direct_slices_per_line = 0;
bool scan_tables_uploaded = false;
std::mutex cuda_mutex;
thread_local std::string last_error;

__constant__ uint16_t scan_to_y_position[256];
__constant__ uint16_t scan_to_c_position[128];
__constant__ uint16_t y_position_to_scan[256];
__constant__ uint16_t c_position_to_scan[128];

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

bool register_host_buffer(void *pointer, size_t bytes, void **registered,
                          size_t *registered_bytes, const char *operation) {
  if (*registered == pointer && *registered_bytes >= bytes)
    return true;
  if (*registered) {
    cudaHostUnregister(*registered);
    *registered = nullptr;
    *registered_bytes = 0;
  }
  if (!cuda_ok(cudaHostRegister(pointer, bytes, cudaHostRegisterPortable), operation))
    return false;
  *registered = pointer;
  *registered_bytes = bytes;
  return true;
}

__global__ void haar_initial_10p2(const uint16_t *input,
                                  int input_stride,
                                  int16_t *output,
                                  int output_stride,
                                  int input_width,
                                  int input_height,
                                  int output_width,
                                  int output_height) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = pair * 2;
  if (x >= output_width || y >= output_height)
    return;

  const int source_y = y < input_height ? y : 2 * input_height - y - 1;
  const int source_x = x < input_width ? x : 2 * input_width - x - 2;
  const int d0 = static_cast<int>(input[source_y * input_stride + source_x]) - 512;
  const int d1 = static_cast<int>(input[source_y * input_stride + source_x + 1]) - 512;
  const int high = d1 - d0;
  output[y * output_stride + x] = static_cast<int16_t>(d0 + ((high + 1) >> 1));
  output[y * output_stride + x + 1] = static_cast<int16_t>(high);
}

__global__ void haar_horizontal(int16_t *data, int stride, int width, int height, int skip) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = pair * 2 * skip;
  const int y = row * skip;
  if (x >= width || y >= height)
    return;

  const int d0 = data[y * stride + x];
  const int d1 = data[y * stride + x + skip];
  const int high = d1 - d0;
  data[y * stride + x] = static_cast<int16_t>(d0 + ((high + 1) >> 1));
  data[y * stride + x + skip] = static_cast<int16_t>(high);
}

__global__ void haar_vertical(int16_t *data, int stride, int width, int height, int skip) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = column * skip;
  const int y = pair * 2 * skip;
  if (x >= width || y >= height)
    return;

  const int d0 = data[y * stride + x];
  const int d1 = data[(y + skip) * stride + x];
  const int high = d1 - d0;
  data[y * stride + x] = static_cast<int16_t>(d0 + ((high + 1) >> 1));
  data[(y + skip) * stride + x] = static_cast<int16_t>(high);
}

__global__ void haar0_depth3_tiled_10p2(const uint16_t *input,
                                         int input_stride,
                                         int16_t *output,
                                         int output_stride,
                                         int input_width,
                                         int input_height,
                                         int output_width,
                                         int output_height) {
  __shared__ int16_t tile[8][8];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * 8 + tx;
  const int gy = blockIdx.y * 8 + ty;
  if (gx >= output_width || gy >= output_height)
    return;
  const int pair_x = gx & ~1;
  const int source_pair_x = pair_x < input_width ? pair_x : 2 * input_width - pair_x - 2;
  const int source_x = source_pair_x + (gx & 1);
  const int source_y = gy < input_height ? gy : 2 * input_height - gy - 1;
  tile[ty][tx] = static_cast<int16_t>(
      static_cast<int>(input[source_y * input_stride + source_x]) - 512);
  __syncthreads();

#pragma unroll
  for (int level = 0; level < 3; ++level) {
    const int skip = 1 << level;
    if ((ty & (skip - 1)) == 0 && (tx & (2 * skip - 1)) == 0) {
      const int d0 = tile[ty][tx];
      const int d1 = tile[ty][tx + skip];
      const int high = d1 - d0;
      tile[ty][tx] = static_cast<int16_t>(d0 + ((high + 1) >> 1));
      tile[ty][tx + skip] = static_cast<int16_t>(high);
    }
    __syncthreads();
    if ((tx & (skip - 1)) == 0 && (ty & (2 * skip - 1)) == 0) {
      const int d0 = tile[ty][tx];
      const int d1 = tile[ty + skip][tx];
      const int high = d1 - d0;
      tile[ty][tx] = static_cast<int16_t>(d0 + ((high + 1) >> 1));
      tile[ty + skip][tx] = static_cast<int16_t>(high);
    }
    __syncthreads();
  }
  output[gy * output_stride + gx] = tile[ty][tx];
}

__device__ __forceinline__ int coefficient_weight_32x8(int x, int y) {
  if ((x & 7) == 0 && y == 0)
    return 16;
  const int skips[3] = {8, 4, 2};
  const int hl[3] = {12, 8, 4};
  const int hh[3] = {8, 4, 0};
  for (int level = 0; level < 3; ++level) {
    const int skip = skips[level];
    const int xm = x & (skip - 1);
    const int ym = y & (skip - 1);
    if (xm == skip / 2 && ym == 0)
      return hl[level];
    if (xm == 0 && ym == skip / 2)
      return hl[level];
    if (xm == skip / 2 && ym == skip / 2)
      return hh[level];
  }
  return 0;
}

__device__ __forceinline__ int coefficient_band_32x8(int x, int y) {
  if ((x & 7) == 0 && y == 0)
    return 0;
  const int skips[3] = {8, 4, 2};
  for (int level = 0; level < 3; ++level) {
    const int skip = skips[level];
    const int xm = x & (skip - 1);
    const int ym = y & (skip - 1);
    if (xm == skip / 2 && ym == 0) return 1 + 3 * level;
    if (xm == 0 && ym == skip / 2) return 2 + 3 * level;
    if (xm == skip / 2 && ym == skip / 2) return 3 + 3 * level;
  }
  return 0;
}

__device__ __forceinline__ int coefficient_scan_index(int x, int y, int width) {
  int n = 0;
  int skip = 8;
  for (int yy = 0; yy < 8; yy += skip)
    for (int xx = 0; xx < width; xx += skip, ++n)
      if (x == xx && y == yy) return n;
  for (int level = 0; level < 3; ++level) {
    for (int yy = 0; yy < 8; yy += skip)
      for (int xx = 0; xx < width; xx += skip, ++n)
        if (x == xx + skip/2 && y == yy) return n;
    for (int yy = 0; yy < 8; yy += skip)
      for (int xx = 0; xx < width; xx += skip, ++n)
        if (x == xx && y == yy + skip/2) return n;
    for (int yy = 0; yy < 8; yy += skip)
      for (int xx = 0; xx < width; xx += skip, ++n)
        if (x == xx + skip/2 && y == yy + skip/2) return n;
    skip >>= 1;
  }
  return 0;
}

__device__ __forceinline__ int vlc_length(unsigned int value) {
  return value ? 2 * (31 - __clz(value + 1)) + 2 : 1;
}

__device__ __forceinline__ unsigned int quantise_value(unsigned int value,
                                                        uint16_t multiplier,
                                                        uint8_t shift,
                                                        int qshift) {
  const unsigned int t = (value * static_cast<unsigned int>(multiplier)) >> 16;
  return ((t + value) >> shift) >> qshift;
}

__device__ __forceinline__ unsigned int warp_or(unsigned int value) {
  for (int offset = 16; offset; offset >>= 1)
    value |= __shfl_down_sync(0xffffffff, value, offset);
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ int warp_sum(int value) {
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ int warp_max(int value) {
  for (int offset = 16; offset; offset >>= 1)
    value = max(value, __shfl_down_sync(0xffffffff, value, offset));
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ unsigned int sample_codeword(unsigned int value,
                                                         bool negative,
                                                         int *length) {
  if (!value) {
    *length = 1;
    return 1;
  }
  unsigned int y = value + 1;
  const int unsigned_length = 2 * (31 - __clz(y)) + 1;
  y = (y | (y << 8)) & 0x00ff00ff;
  y = (y | (y << 4)) & 0x0f0f0f0f;
  y = (y | (y << 2)) & 0x33333333;
  y = (y | (y << 1)) & 0x55555555;
  y = ((y << 1) | 1) & ((1u << unsigned_length) - 1);
  *length = unsigned_length + 1;
  return (y << 1) | static_cast<unsigned int>(negative);
}

__device__ __forceinline__ void write_component(
    const int16_t *plane, int base, int stride, int width, int component,
    int samples, int qindex, int qshift, const uint16_t *matrix_m,
    const uint8_t *matrix_sh, uint8_t *output, int *offset) {
  const uint16_t *scan = width == 32 ? scan_to_y_position : scan_to_c_position;
  const int matrix_base = qindex * 512 + component * 128 + (component ? 128 : 0);
  unsigned int accumulator = 0;
  int bits = 0;
  for (int n = 0; n < samples; ++n) {
    const int position = scan[n];
    const int x = position & 31;
    const int y = position >> 5;
    const int value = plane[base + y * stride + x];
    const unsigned int magnitude = value < 0 ? -value : value;
    const unsigned int quantised = quantise_value(
        magnitude, matrix_m[matrix_base + y * width + x],
        matrix_sh[matrix_base + y * width + x], qshift);
    int length = 0;
    const unsigned int codeword = sample_codeword(quantised, value < 0, &length);
    accumulator |= codeword << (32 - length - bits);
    bits += length;
    while (bits >= 8) {
      output[(*offset)++] = static_cast<uint8_t>(accumulator >> 24);
      accumulator <<= 8;
      bits -= 8;
    }
  }
  if (bits)
    output[(*offset)++] = static_cast<uint8_t>((accumulator >> 24) | (0xff >> bits));
}

__device__ __forceinline__ void component_metrics(
    const int16_t *plane, int base, int stride, int width, int component,
    int qindex, int qshift, const uint16_t *matrix_m, const uint8_t *matrix_sh,
    int *raw_bytes, int *samples) {
  const uint16_t *scan = width == 32 ? scan_to_y_position : scan_to_c_position;
  const int count = width * 8;
  const int matrix_base = qindex * 512 + component * 128 + (component ? 128 : 0);
  int total_bits = 0;
  int last = -1;
  for (int n = 0; n < count; ++n) {
    const int position = scan[n];
    const int x = position & 31;
    const int y = position >> 5;
    const int value = plane[base + y * stride + x];
    if (value)
      last = n;
    const unsigned int magnitude = value < 0 ? -value : value;
    total_bits += vlc_length(quantise_value(
        magnitude, matrix_m[matrix_base + y * width + x],
        matrix_sh[matrix_base + y * width + x], qshift));
  }
  *samples = last + 1;
  *raw_bytes = (total_bits - (count - 1 - last) + 7) / 8;
}

__global__ void serialise_slices_32x8(
    const int16_t *y_plane, const int16_t *cb_plane, const int16_t *cr_plane,
    int y_stride, int c_stride, const int *max_sizes, const int *output_offsets,
    int n_slices, int slices_per_line, int slice_size_scalar,
    const uint16_t *matrix_m, const uint8_t *matrix_sh, const uint8_t *qindices,
    uint8_t *output) {
  const int slice = blockIdx.x * blockDim.x + threadIdx.x;
  if (slice >= n_slices)
    return;
  const int slice_x = slice % slices_per_line;
  const int slice_y = slice / slices_per_line;
  const int y_base = slice_y * 8 * y_stride + slice_x * 32;
  const int c_base = slice_y * 8 * c_stride + slice_x * 16;
  const int qi = qindices[slice];
  const int qindex = qi <= 31 ? qi : 28 + (qi & 3);
  const int qshift = qi <= 31 ? 0 : qi / 4 - 7;
  int raw[3];
  int samples[3];
  component_metrics(y_plane, y_base, y_stride, 32, 0, qindex, qshift,
                    matrix_m, matrix_sh, &raw[0], &samples[0]);
  component_metrics(cb_plane, c_base, c_stride, 16, 1, qindex, qshift,
                    matrix_m, matrix_sh, &raw[1], &samples[1]);
  component_metrics(cr_plane, c_base, c_stride, 16, 2, qindex, qshift,
                    matrix_m, matrix_sh, &raw[2], &samples[2]);

  int offset = output_offsets[slice];
  const int slice_end = offset + max_sizes[slice];
  output[offset++] = static_cast<uint8_t>(qi);
  int padding = max_sizes[slice] - 4 -
      ((raw[0] + slice_size_scalar - 1) / slice_size_scalar +
       (raw[1] + slice_size_scalar - 1) / slice_size_scalar +
       (raw[2] + slice_size_scalar - 1) / slice_size_scalar) * slice_size_scalar;
  const int16_t *planes[3] = {y_plane, cb_plane, cr_plane};
  const int bases[3] = {y_base, c_base, c_base};
  const int strides[3] = {y_stride, c_stride, c_stride};
  const int widths[3] = {32, 16, 16};
  for (int component = 0; component < 3; ++component) {
    const int units = (raw[component] + slice_size_scalar - 1) / slice_size_scalar;
    int component_padding = min((255 - units) * slice_size_scalar, padding);
    padding -= component_padding;
    component_padding += units * slice_size_scalar - raw[component];
    output[offset++] = static_cast<uint8_t>((raw[component] + component_padding) /
                                             slice_size_scalar);
    write_component(planes[component], bases[component], strides[component],
                    widths[component], component, samples[component], qindex,
                    qshift, matrix_m, matrix_sh, output, &offset);
    for (int i = 0; i < component_padding; ++i)
      output[offset++] = 0xff;
  }
  while (offset < slice_end)
    output[offset++] = 0xff;
}

__device__ __forceinline__ void shared_write_codeword(
    unsigned int *words, int bit_position, unsigned int codeword, int length) {
  int remaining = length;
  while (remaining) {
    const int byte = bit_position >> 3;
    const int intra = bit_position & 7;
    const int take = min(remaining, 8 - intra);
    const unsigned int mask = (1u << take) - 1;
    const unsigned int chunk = (codeword >> (remaining - take)) & mask;
    const unsigned int byte_bits = chunk << (8 - intra - take);
    atomicOr(&words[byte >> 2], byte_bits << ((byte & 3) * 8));
    bit_position += take;
    remaining -= take;
  }
}

__global__ void serialise_slices_32x8_warp(
    const int16_t *y_plane, const int16_t *cb_plane, const int16_t *cr_plane,
    int y_stride, int c_stride, const int *max_sizes, const int *output_offsets,
    int n_slices, int slices_per_line, int slice_size_scalar,
    const uint16_t *matrix_m, const uint8_t *matrix_sh, const uint8_t *qindices,
    const int *slice_metrics, uint8_t *output, int shared_stride) {
  extern __shared__ unsigned int shared_words[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slice = blockIdx.x * 8 + warp;
  if (slice >= n_slices)
    return;
  uint8_t *slice_buffer = reinterpret_cast<uint8_t *>(shared_words) + warp * shared_stride;
  unsigned int *slice_words = reinterpret_cast<unsigned int *>(slice_buffer);
  for (int i = lane; i < shared_stride / 4; i += 32)
    slice_words[i] = 0;
  __syncwarp();

  const int slice_x = slice % slices_per_line;
  const int slice_y = slice / slices_per_line;
  const int bases[3] = {slice_y * 8 * y_stride + slice_x * 32,
                        slice_y * 8 * c_stride + slice_x * 16,
                        slice_y * 8 * c_stride + slice_x * 16};
  const int strides[3] = {y_stride, c_stride, c_stride};
  const int widths[3] = {32, 16, 16};
  const int16_t *planes[3] = {y_plane, cb_plane, cr_plane};
  const int qi = qindices[slice];
  const int qindex = qi <= 31 ? qi : 28 + (qi & 3);
  const int qshift = qi <= 31 ? 0 : qi / 4 - 7;
  int raw[3];
  int samples[3];
  int component_bits[3];

  for (int component = 0; component < 3; ++component) {
    component_bits[component] = slice_metrics[6 * slice + component];
    samples[component] = slice_metrics[6 * slice + 3 + component];
    raw[component] = (component_bits[component] + 7) / 8;
  }

  int padding = max_sizes[slice] - 4 -
      ((raw[0] + slice_size_scalar - 1) / slice_size_scalar +
       (raw[1] + slice_size_scalar - 1) / slice_size_scalar +
       (raw[2] + slice_size_scalar - 1) / slice_size_scalar) * slice_size_scalar;
  int component_start[3];
  int component_padding[3];
  int cursor = 1;
  for (int component = 0; component < 3; ++component) {
    const int units = (raw[component] + slice_size_scalar - 1) / slice_size_scalar;
    int p = min((255 - units) * slice_size_scalar, padding);
    padding -= p;
    p += units * slice_size_scalar - raw[component];
    component_padding[component] = p;
    component_start[component] = cursor + 1;
    cursor += 1 + raw[component] + p;
  }

  for (int component = 0; component < 3; ++component) {
    const uint16_t *scan = component == 0 ? scan_to_y_position : scan_to_c_position;
    const int matrix_base = qindex * 512 + component * 128 + (component ? 128 : 0);
    int round_base = 0;
    for (int round = 0; round * 32 < samples[component]; ++round) {
      const int n = round * 32 + lane;
      int length = 0;
      unsigned int codeword = 0;
      if (n < samples[component]) {
        const int position = scan[n];
        const int x = position & 31;
        const int y = position >> 5;
        const int value = planes[component][bases[component] + y * strides[component] + x];
        const unsigned int quantised = quantise_value(value < 0 ? -value : value,
            matrix_m[matrix_base + y * widths[component] + x],
            matrix_sh[matrix_base + y * widths[component] + x], qshift);
        codeword = sample_codeword(quantised, value < 0, &length);
      }
      int inclusive = length;
      for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffff, inclusive, offset);
        if (lane >= offset)
          inclusive += other;
      }
      if (length)
        shared_write_codeword(slice_words,
            component_start[component] * 8 + round_base + inclusive - length,
            codeword, length);
      round_base += __shfl_sync(0xffffffff, inclusive, 31);
    }
  }
  __syncwarp();
  if (lane == 0) {
    slice_buffer[0] = static_cast<uint8_t>(qi);
    for (int component = 0; component < 3; ++component) {
      const int length_position = component_start[component] - 1;
      slice_buffer[length_position] = static_cast<uint8_t>(
          (raw[component] + component_padding[component]) / slice_size_scalar);
      if (raw[component]) {
        if (component_bits[component] & 7)
          slice_buffer[component_start[component] + raw[component] - 1] |=
              static_cast<uint8_t>(0xff >> (component_bits[component] & 7));
      }
      for (int i = 0; i < component_padding[component]; ++i)
        slice_buffer[component_start[component] + raw[component] + i] = 0xff;
    }
  }
  __syncwarp();
  const int output_base = output_offsets[slice];
  for (int i = lane; i < max_sizes[slice]; i += 32)
    output[output_base + i] = slice_buffer[i];
}

__global__ void encode_slices_32x8(
    const int16_t *y_plane, const int16_t *cb_plane, const int16_t *cr_plane,
    int y_stride, int c_stride, const int *max_sizes, const int *output_offsets,
    int n_slices, int slices_per_line, int slice_size_scalar,
    const uint16_t *matrix_m, const uint8_t *matrix_sh, uint8_t *qindices,
    uint8_t *output) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slice = blockIdx.x * 8 + warp;
  if (slice >= n_slices)
    return;
  const int slice_x = slice % slices_per_line;
  const int slice_y = slice / slices_per_line;
  const int y_base = slice_y * 8 * y_stride + slice_x * 32;
  const int c_base = slice_y * 8 * c_stride + slice_x * 16;
  const int weights[10] = {16, 12, 12, 8, 8, 8, 4, 4, 4, 0};

  unsigned int band_y[10] = {};
  unsigned int band_cb[10] = {};
  unsigned int band_cr[10] = {};
  int last_y = -1, last_cb = -1, last_cr = -1;
  for (int row = 0; row < 8; ++row) {
    const int16_t value = y_plane[y_base + row * y_stride + lane];
    const unsigned int magnitude = value < 0 ? -static_cast<int>(value) : value;
    band_y[coefficient_band_32x8(lane, row)] |= magnitude;
    if (value) last_y = max(last_y, coefficient_scan_index(lane, row, 32));
  }
  if (lane < 16) {
    for (int row = 0; row < 8; ++row) {
      const int16_t cb = cb_plane[c_base + row * c_stride + lane];
      const int16_t cr = cr_plane[c_base + row * c_stride + lane];
      const int band = coefficient_band_32x8(lane, row);
      band_cb[band] |= cb < 0 ? -static_cast<int>(cb) : cb;
      band_cr[band] |= cr < 0 ? -static_cast<int>(cr) : cr;
      const int scan = coefficient_scan_index(lane, row, 16);
      if (cb) last_cb = max(last_cb, scan);
      if (cr) last_cr = max(last_cr, scan);
    }
  }
  int qbase = 0;
  for (int band = 0; band < 10; ++band) {
    qbase = max(qbase, 4 * (24 - __clz(warp_or(band_y[band]) + 1)) + weights[band]);
    qbase = max(qbase, 4 * (24 - __clz(warp_or(band_cb[band]) + 1)) + weights[band]);
    qbase = max(qbase, 4 * (24 - __clz(warp_or(band_cr[band]) + 1)) + weights[band]);
  }
  qbase = warp_max(qbase);
  last_y = warp_max(last_y);
  last_cb = warp_max(last_cb);
  last_cr = warp_max(last_cr);

  int qi = max(qbase, static_cast<int>(qindices[slice]));
  qi = max(0, qi - 8);
  int fits = 0;
  int raw_y = 0, raw_cb = 0, raw_cr = 0;
  while (!fits && qi < 64) {
    qi = min(64, qi + 8);
    const int qindex = qi <= 31 ? qi : 28 + (qi & 3);
    const int qshift = qi <= 31 ? 0 : qi / 4 - 7;
    const int matrix_base = qindex * 512;
    int bits_y = 0, bits_cb = 0, bits_cr = 0;
    for (int row = 0; row < 8; ++row) {
      const int pos = row * 32 + lane;
      const int value = y_plane[y_base + row * y_stride + lane];
      bits_y += vlc_length(quantise_value(value < 0 ? -value : value,
          matrix_m[matrix_base + pos], matrix_sh[matrix_base + pos], qshift));
    }
    if (lane < 16) {
      for (int row = 0; row < 8; ++row) {
        const int pos = row * 16 + lane;
        const int cb = cb_plane[c_base + row * c_stride + lane];
        const int cr = cr_plane[c_base + row * c_stride + lane];
        bits_cb += vlc_length(quantise_value(cb < 0 ? -cb : cb,
            matrix_m[matrix_base + 256 + pos], matrix_sh[matrix_base + 256 + pos], qshift));
        bits_cr += vlc_length(quantise_value(cr < 0 ? -cr : cr,
            matrix_m[matrix_base + 384 + pos], matrix_sh[matrix_base + 384 + pos], qshift));
      }
    }
    bits_y = warp_sum(bits_y);
    bits_cb = warp_sum(bits_cb);
    bits_cr = warp_sum(bits_cr);
    if (lane == 0) {
      raw_y = (bits_y - (255 - last_y) + 7) / 8;
      raw_cb = (bits_cb - (127 - last_cb) + 7) / 8;
      raw_cr = (bits_cr - (127 - last_cr) + 7) / 8;
      const int rounded_y = (raw_y + slice_size_scalar - 1) / slice_size_scalar;
      const int rounded_cb = (raw_cb + slice_size_scalar - 1) / slice_size_scalar;
      const int rounded_cr = (raw_cr + slice_size_scalar - 1) / slice_size_scalar;
      fits = rounded_y <= 255 && rounded_cb <= 255 && rounded_cr <= 255 &&
             4 + (rounded_y + rounded_cb + rounded_cr) * slice_size_scalar <= max_sizes[slice];
    }
    fits = __shfl_sync(0xffffffff, fits, 0);
  }
  qi = min(64, max(qbase, qi));
  if (lane != 0)
    return;

  qindices[slice] = static_cast<uint8_t>(qi);
  const int qindex = qi <= 31 ? qi : 28 + (qi & 3);
  const int qshift = qi <= 31 ? 0 : qi / 4 - 7;
  int offset = output_offsets[slice];
  const int slice_end = offset + max_sizes[slice];
  output[offset++] = static_cast<uint8_t>(qi);
  int padding = max_sizes[slice] - 4 -
      ((raw_y + slice_size_scalar - 1) / slice_size_scalar +
       (raw_cb + slice_size_scalar - 1) / slice_size_scalar +
       (raw_cr + slice_size_scalar - 1) / slice_size_scalar) * slice_size_scalar;
  const int raw[3] = {raw_y, raw_cb, raw_cr};
  const int samples[3] = {last_y + 1, last_cb + 1, last_cr + 1};
  const int16_t *planes[3] = {y_plane, cb_plane, cr_plane};
  const int bases[3] = {y_base, c_base, c_base};
  const int strides[3] = {y_stride, c_stride, c_stride};
  const int widths[3] = {32, 16, 16};
  for (int component = 0; component < 3; ++component) {
    const int units = (raw[component] + slice_size_scalar - 1) / slice_size_scalar;
    int component_padding = min((255 - units) * slice_size_scalar, padding);
    padding -= component_padding;
    component_padding += units * slice_size_scalar - raw[component];
    output[offset++] = static_cast<uint8_t>((raw[component] + component_padding) /
                                             slice_size_scalar);
    write_component(planes[component], bases[component], strides[component],
                    widths[component], component, samples[component], qindex,
                    qshift, matrix_m, matrix_sh, output, &offset);
    for (int i = 0; i < component_padding; ++i)
      output[offset++] = 0xff;
  }
  while (offset < slice_end)
    output[offset++] = 0xff;
}

__global__ void select_quantisers_32x8(
    const int16_t *y_plane, const int16_t *cb_plane, const int16_t *cr_plane,
    int y_stride, int c_stride, const int *max_sizes, int n_slices,
    int slices_per_line, int slice_size_scalar, const uint16_t *matrix_m,
    const uint8_t *matrix_sh, uint8_t *qindices, int *slice_metrics) {
  __shared__ unsigned int shared_bands[8][30];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slice = blockIdx.x * 8 + warp;
  if (slice >= n_slices)
    return;
  const int slice_x = slice % slices_per_line;
  const int slice_y = slice / slices_per_line;
  const int y_base = slice_y * 8 * y_stride + slice_x * 32;
  const int c_base = slice_y * 8 * c_stride + slice_x * 16;
  const int weights[10] = {16, 12, 12, 8, 8, 8, 4, 4, 4, 0};

  if (lane < 30)
    shared_bands[warp][lane] = 0;
  __syncwarp();
  int last_y = -1, last_cb = -1, last_cr = -1;
  for (int row = 0; row < 8; ++row) {
    const int16_t value = y_plane[y_base + row * y_stride + lane];
    const unsigned int magnitude = value < 0 ? -static_cast<int>(value) : value;
    atomicOr(&shared_bands[warp][coefficient_band_32x8(lane, row)], magnitude);
    if (value) last_y = max(last_y, static_cast<int>(y_position_to_scan[row * 32 + lane]));
  }
  if (lane < 16) {
    for (int row = 0; row < 8; ++row) {
      const int16_t cb = cb_plane[c_base + row * c_stride + lane];
      const int16_t cr = cr_plane[c_base + row * c_stride + lane];
      const unsigned int cba = cb < 0 ? -static_cast<int>(cb) : cb;
      const unsigned int cra = cr < 0 ? -static_cast<int>(cr) : cr;
      const int band = coefficient_band_32x8(lane, row);
      atomicOr(&shared_bands[warp][10 + band], cba);
      atomicOr(&shared_bands[warp][20 + band], cra);
      const int scan = c_position_to_scan[row * 16 + lane];
      if (cb) last_cb = max(last_cb, scan);
      if (cr) last_cr = max(last_cr, scan);
    }
  }
  __syncwarp();
  int qbase = 0;
  if (lane == 0) {
    for (int band = 0; band < 10; ++band) {
      qbase = max(qbase, 4 * (24 - __clz(shared_bands[warp][band] + 1)) + weights[band]);
      qbase = max(qbase, 4 * (24 - __clz(shared_bands[warp][10 + band] + 1)) + weights[band]);
      qbase = max(qbase, 4 * (24 - __clz(shared_bands[warp][20 + band] + 1)) + weights[band]);
    }
  }
  qbase = __shfl_sync(0xffffffff, qbase, 0);
  last_y = warp_max(last_y);
  last_cb = warp_max(last_cb);
  last_cr = warp_max(last_cr);

  const int qi0 = max(qbase, static_cast<int>(qindices[slice]));
  int fits = 0;
  int selected_bits_y = 0, selected_bits_cb = 0, selected_bits_cr = 0;
  int qi;
  if (qi0 >= 32) {
    // For qi >= 32 the loop steps by 8, so qi&3 (hence qindex) is constant for
    // the whole search. Precompute the loop-invariant ((v*m)>>16 + v) >> sh once
    // per coefficient, then each candidate only needs a shift plus a VLC length:
    // no plane or matrix re-reads inside the loop.
    const int qindex = 28 + (qi0 & 3);
    const int matrix_base = qindex * 512;
    int qv_y[8], qv_cb[8], qv_cr[8];
    for (int row = 0; row < 8; ++row) {
      const int pos = row * 32 + lane;
      const int value = y_plane[y_base + row * y_stride + lane];
      const unsigned int magnitude = value < 0 ? -value : value;
      const unsigned int t = (magnitude * static_cast<unsigned int>(matrix_m[matrix_base + pos])) >> 16;
      qv_y[row] = static_cast<int>((t + magnitude) >> matrix_sh[matrix_base + pos]);
    }
    if (lane < 16) {
      for (int row = 0; row < 8; ++row) {
        const int pos = row * 16 + lane;
        int cb = cb_plane[c_base + row * c_stride + lane];
        int cr = cr_plane[c_base + row * c_stride + lane];
        const unsigned int cba = cb < 0 ? -cb : cb;
        const unsigned int cra = cr < 0 ? -cr : cr;
        const unsigned int tb = (cba * static_cast<unsigned int>(matrix_m[matrix_base + 256 + pos])) >> 16;
        const unsigned int tr = (cra * static_cast<unsigned int>(matrix_m[matrix_base + 384 + pos])) >> 16;
        qv_cb[row] = static_cast<int>((tb + cba) >> matrix_sh[matrix_base + 256 + pos]);
        qv_cr[row] = static_cast<int>((tr + cra) >> matrix_sh[matrix_base + 384 + pos]);
      }
    }
    qi = qi0;
    while (!fits && qi < 64) {
      qi = min(64, qi + 8);
      const int qshift = qi / 4 - 7;
      int bits_y = 0, bits_cb = 0, bits_cr = 0;
      for (int row = 0; row < 8; ++row)
        bits_y += vlc_length(static_cast<unsigned int>(qv_y[row] >> qshift));
      if (lane < 16) {
        for (int row = 0; row < 8; ++row) {
          bits_cb += vlc_length(static_cast<unsigned int>(qv_cb[row] >> qshift));
          bits_cr += vlc_length(static_cast<unsigned int>(qv_cr[row] >> qshift));
        }
      }
      bits_y = warp_sum(bits_y);
      bits_cb = warp_sum(bits_cb);
      bits_cr = warp_sum(bits_cr);
      selected_bits_y = bits_y;
      selected_bits_cb = bits_cb;
      selected_bits_cr = bits_cr;
      if (lane == 0) {
        int bytes_y = (bits_y - (255 - last_y) + 7) / 8;
        int bytes_cb = (bits_cb - (127 - last_cb) + 7) / 8;
        int bytes_cr = (bits_cr - (127 - last_cr) + 7) / 8;
        bytes_y = (bytes_y + slice_size_scalar - 1) / slice_size_scalar * slice_size_scalar;
        bytes_cb = (bytes_cb + slice_size_scalar - 1) / slice_size_scalar * slice_size_scalar;
        bytes_cr = (bytes_cr + slice_size_scalar - 1) / slice_size_scalar * slice_size_scalar;
        fits = bytes_y / slice_size_scalar <= 255 &&
               bytes_cb / slice_size_scalar <= 255 &&
               bytes_cr / slice_size_scalar <= 255 &&
               4 + bytes_y + bytes_cb + bytes_cr <= max_sizes[slice];
      }
      fits = __shfl_sync(0xffffffff, fits, 0);
    }
  } else {
    qi = max(0, qi0 - 8);
    while (!fits && qi < 64) {
      qi = min(64, qi + 8);
      const int qindex = qi <= 31 ? qi : 28 + (qi & 3);
      const int qshift = qi <= 31 ? 0 : qi / 4 - 7;
      const int matrix_base = qindex * 512;
      int bits_y = 0, bits_cb = 0, bits_cr = 0;
      for (int row = 0; row < 8; ++row) {
        const int pos = row * 32 + lane;
        const int value = y_plane[y_base + row * y_stride + lane];
        const unsigned int magnitude = value < 0 ? -value : value;
        bits_y += vlc_length(quantise_value(magnitude, matrix_m[matrix_base + pos],
                                            matrix_sh[matrix_base + pos], qshift));
      }
      if (lane < 16) {
        for (int row = 0; row < 8; ++row) {
          const int pos = row * 16 + lane;
          int cb = cb_plane[c_base + row * c_stride + lane];
          int cr = cr_plane[c_base + row * c_stride + lane];
          const unsigned int cba = cb < 0 ? -cb : cb;
          const unsigned int cra = cr < 0 ? -cr : cr;
          bits_cb += vlc_length(quantise_value(cba, matrix_m[matrix_base + 256 + pos],
                                               matrix_sh[matrix_base + 256 + pos], qshift));
          bits_cr += vlc_length(quantise_value(cra, matrix_m[matrix_base + 384 + pos],
                                               matrix_sh[matrix_base + 384 + pos], qshift));
        }
      }
      bits_y = warp_sum(bits_y);
      bits_cb = warp_sum(bits_cb);
      bits_cr = warp_sum(bits_cr);
      selected_bits_y = bits_y;
      selected_bits_cb = bits_cb;
      selected_bits_cr = bits_cr;
      if (lane == 0) {
        int bytes_y = (bits_y - (255 - last_y) + 7) / 8;
        int bytes_cb = (bits_cb - (127 - last_cb) + 7) / 8;
        int bytes_cr = (bits_cr - (127 - last_cr) + 7) / 8;
        bytes_y = (bytes_y + slice_size_scalar - 1) / slice_size_scalar * slice_size_scalar;
        bytes_cb = (bytes_cb + slice_size_scalar - 1) / slice_size_scalar * slice_size_scalar;
        bytes_cr = (bytes_cr + slice_size_scalar - 1) / slice_size_scalar * slice_size_scalar;
        fits = bytes_y / slice_size_scalar <= 255 &&
               bytes_cb / slice_size_scalar <= 255 &&
               bytes_cr / slice_size_scalar <= 255 &&
               4 + bytes_y + bytes_cb + bytes_cr <= max_sizes[slice];
      }
      fits = __shfl_sync(0xffffffff, fits, 0);
    }
  }
  if (lane == 0) {
    qindices[slice] = static_cast<uint8_t>(min(64, max(qbase, qi)));
    if (slice_metrics) {
      slice_metrics[6 * slice + 0] = selected_bits_y - (255 - last_y);
      slice_metrics[6 * slice + 1] = selected_bits_cb - (127 - last_cb);
      slice_metrics[6 * slice + 2] = selected_bits_cr - (127 - last_cr);
      slice_metrics[6 * slice + 3] = last_y + 1;
      slice_metrics[6 * slice + 4] = last_cb + 1;
      slice_metrics[6 * slice + 5] = last_cr + 1;
    }
  }
}

} // namespace

bool vc2_cuda_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

const char *vc2_cuda_last_error() {
  return last_error.c_str();
}

bool vc2_cuda_prepare_haar0_10p2_i16(const int input_width[3],
                                     const int input_height[3],
                                     int16_t *const output[3],
                                     const int output_stride[3],
                                     const int output_height[3]) {
  std::lock_guard<std::mutex> lock(cuda_mutex);
  last_error.clear();
  if (!cuda_ok(cudaFree(nullptr), "CUDA context initialization"))
    return false;
  for (int c = 0; c < 3; ++c) {
    const size_t input_bytes = static_cast<size_t>(input_width[c]) * input_height[c] * sizeof(uint16_t);
    const size_t output_bytes = static_cast<size_t>(output_stride[c]) * output_height[c] * sizeof(int16_t);
    if (!streams[c] &&
        !cuda_ok(cudaStreamCreateWithFlags(&streams[c], cudaStreamNonBlocking), "CUDA stream creation"))
      return false;
    if (!transform_done[c] &&
        !cuda_ok(cudaEventCreateWithFlags(&transform_done[c], cudaEventDisableTiming),
                 "CUDA transform event creation"))
      return false;
    if (!reserve_buffer(reinterpret_cast<void **>(&device_input[c]), &input_capacity[c], input_bytes) ||
        !reserve_buffer(reinterpret_cast<void **>(&device_output[c]), &output_capacity[c], output_bytes) ||
        !register_host_buffer(output[c], output_bytes,
                              reinterpret_cast<void **>(&registered_output[c]),
                              &registered_output_bytes[c], "coefficient buffer pinning"))
      return false;
  }
  // A new encoder configuration must not inherit the previous stream's
  // per-slice lower-bound quantizers from the process-wide CUDA cache.
  direct_n_slices = 0;
  direct_slices_per_line = 0;
  return true;
}

bool vc2_cuda_haar0_transform_10p2_i16_3plane(
    const uint16_t *const input[3], const int input_stride[3],
    int16_t *const output[3], const int output_stride[3],
    const int input_width[3], const int input_height[3],
    const int output_width[3], const int output_height[3], int depth,
    bool download_coefficients) {
  std::lock_guard<std::mutex> lock(cuda_mutex);
  last_error.clear();
  const dim3 threads(32, 8);
  for (int c = 0; c < 3; ++c) {
    if (!input[c] || !output[c] || input_width[c] <= 0 || input_height[c] <= 0 ||
        output_width[c] < input_width[c] || output_height[c] < input_height[c] || depth < 1)
      return false;
    const size_t input_bytes = static_cast<size_t>(input_stride[c]) * input_height[c] * sizeof(uint16_t);
    const size_t output_bytes = static_cast<size_t>(output_stride[c]) * output_height[c] * sizeof(int16_t);
    if (!reserve_buffer(reinterpret_cast<void **>(&device_input[c]), &input_capacity[c], input_bytes) ||
        !reserve_buffer(reinterpret_cast<void **>(&device_output[c]), &output_capacity[c], output_bytes) ||
        !register_host_buffer(const_cast<uint16_t *>(input[c]), input_bytes,
                              &registered_input[c],
                              &registered_input_bytes[c], "input buffer pinning") ||
        !register_host_buffer(output[c], output_bytes,
                              reinterpret_cast<void **>(&registered_output[c]),
                              &registered_output_bytes[c], "coefficient buffer pinning") ||
        !cuda_ok(cudaMemcpy2DAsync(device_input[c], input_stride[c] * sizeof(uint16_t),
                                   input[c], input_stride[c] * sizeof(uint16_t),
                                   input_width[c] * sizeof(uint16_t), input_height[c],
                                   cudaMemcpyHostToDevice, streams[c]), "input upload"))
      return false;

    if (depth == 3) {
      const dim3 tiled_threads(8, 8);
      const dim3 tiled_blocks((output_width[c] + 7) / 8, (output_height[c] + 7) / 8);
      haar0_depth3_tiled_10p2<<<tiled_blocks, tiled_threads, 0, streams[c]>>>(
          device_input[c], input_stride[c], device_output[c], output_stride[c],
          input_width[c], input_height[c], output_width[c], output_height[c]);
    } else {
      const dim3 initial_blocks((output_width[c] / 2 + threads.x - 1) / threads.x,
                                (output_height[c] + threads.y - 1) / threads.y);
      haar_initial_10p2<<<initial_blocks, threads, 0, streams[c]>>>(device_input[c], input_stride[c],
                                                     device_output[c], output_stride[c],
                                                     input_width[c], input_height[c],
                                                     output_width[c], output_height[c]);
      for (int level = 0; level < depth; ++level) {
        const int skip = 1 << level;
        if (level > 0) {
          const dim3 horizontal_blocks((output_width[c] / (2 * skip) + threads.x - 1) / threads.x,
                                       (output_height[c] / skip + threads.y - 1) / threads.y);
          haar_horizontal<<<horizontal_blocks, threads, 0, streams[c]>>>(device_output[c], output_stride[c],
                                                           output_width[c], output_height[c], skip);
        }
        const dim3 vertical_blocks((output_width[c] / skip + threads.x - 1) / threads.x,
                                   (output_height[c] / (2 * skip) + threads.y - 1) / threads.y);
        haar_vertical<<<vertical_blocks, threads, 0, streams[c]>>>(device_output[c], output_stride[c],
                                                    output_width[c], output_height[c], skip);
      }
    }
    if (!cuda_ok(cudaGetLastError(), "Haar kernel launch"))
      return false;
    if (download_coefficients &&
        !cuda_ok(cudaMemcpy2DAsync(output[c], output_stride[c] * sizeof(int16_t),
                                   device_output[c], output_stride[c] * sizeof(int16_t),
                                   output_width[c] * sizeof(int16_t), output_height[c],
                                   cudaMemcpyDeviceToHost, streams[c]), "coefficient download"))
      return false;
    if (!download_coefficients &&
        !cuda_ok(cudaEventRecord(transform_done[c], streams[c]),
                 "transform completion event"))
      return false;
  }
  if (!download_coefficients)
    return true;
  for (int c = 0; c < 3; ++c)
    if (!cuda_ok(cudaStreamSynchronize(streams[c]), "transform synchronization"))
      return false;
  return true;
}

bool vc2_cuda_select_quantisers_32x8_i16(const int *max_sizes,
                                         int n_slices,
                                         int slices_per_line,
                                         int slice_size_scalar,
                                         const int output_stride[3],
                                         const uint16_t *matrix_m,
                                         const uint8_t *matrix_sh,
                                         uint8_t *qindices) {
  std::lock_guard<std::mutex> lock(cuda_mutex);
  last_error.clear();
  if (!max_sizes || !qindices || !matrix_m || !matrix_sh || n_slices <= 0)
    return false;
  const size_t max_size_bytes = static_cast<size_t>(n_slices) * sizeof(int);
  const size_t qindex_bytes = static_cast<size_t>(n_slices);
  if (quantiser_capacity < static_cast<size_t>(n_slices)) {
    if (device_max_sizes) cudaFree(device_max_sizes);
    if (device_output_offsets) cudaFree(device_output_offsets);
    if (device_qindices) cudaFree(device_qindices);
    if (device_slice_metrics) cudaFree(device_slice_metrics);
    device_max_sizes = nullptr;
    device_output_offsets = nullptr;
    device_qindices = nullptr;
    device_slice_metrics = nullptr;
    quantiser_capacity = 0;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_max_sizes), max_size_bytes),
                 "quantizer size allocation") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_qindices), qindex_bytes),
                 "quantizer result allocation"))
      return false;
    quantiser_capacity = n_slices;
  }
  constexpr size_t matrix_entries = 32 * 512;
  if (!device_matrix_m &&
      (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_matrix_m),
                           matrix_entries * sizeof(uint16_t)), "quantizer matrix allocation") ||
       !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_matrix_sh),
                           matrix_entries * sizeof(uint8_t)), "quantizer shift allocation")))
    return false;
  if (cached_matrix_m != matrix_m) {
    if (!cuda_ok(cudaMemcpy(device_matrix_m, matrix_m,
                            matrix_entries * sizeof(uint16_t), cudaMemcpyHostToDevice),
                 "quantizer matrix upload") ||
        !cuda_ok(cudaMemcpy(device_matrix_sh, matrix_sh,
                            matrix_entries * sizeof(uint8_t), cudaMemcpyHostToDevice),
                 "quantizer shift upload"))
      return false;
    cached_matrix_m = matrix_m;
  }
  if (!cuda_ok(cudaMemcpy(device_max_sizes, max_sizes, max_size_bytes,
                          cudaMemcpyHostToDevice), "quantizer budgets upload") ||
      !cuda_ok(cudaMemcpy(device_qindices, qindices, qindex_bytes,
                          cudaMemcpyHostToDevice), "previous quantizers upload"))
    return false;
  select_quantisers_32x8<<<(n_slices + 7) / 8, 256>>>(
      device_output[0], device_output[1], device_output[2], output_stride[0],
      output_stride[1], device_max_sizes, n_slices, slices_per_line,
      slice_size_scalar, device_matrix_m, device_matrix_sh, device_qindices,
      nullptr);
  if (!cuda_ok(cudaGetLastError(), "quantizer kernel launch") ||
      !cuda_ok(cudaMemcpy(qindices, device_qindices, qindex_bytes,
                          cudaMemcpyDeviceToHost), "quantizer results download"))
    return false;
  return true;
}

bool vc2_cuda_encode_32x8_i16(const int *max_sizes,
                              const int *output_offsets,
                              int n_slices,
                              int slices_per_line,
                              int slice_size_scalar,
                              const int output_stride[3],
                              const uint16_t *matrix_m,
                              const uint8_t *matrix_sh,
                              uint8_t *output,
                              int output_bytes) {
  std::lock_guard<std::mutex> lock(cuda_mutex);
  last_error.clear();
  if (!max_sizes || !output_offsets || !matrix_m || !matrix_sh || !output ||
      n_slices <= 0 || slices_per_line <= 0 || slice_size_scalar <= 0 ||
      output_bytes <= 0 || !device_output[0] || !device_output[1] || !device_output[2])
    return false;
  if (!encode_stream &&
      !cuda_ok(cudaStreamCreateWithFlags(&encode_stream, cudaStreamNonBlocking),
               "encoder stream creation"))
    return false;

  const size_t integer_bytes = static_cast<size_t>(n_slices) * sizeof(int);
  const size_t qindex_bytes = static_cast<size_t>(n_slices);
  if (quantiser_capacity < static_cast<size_t>(n_slices)) {
    if (device_max_sizes) cudaFree(device_max_sizes);
    if (device_output_offsets) cudaFree(device_output_offsets);
    if (device_qindices) cudaFree(device_qindices);
    if (device_slice_metrics) cudaFree(device_slice_metrics);
    device_max_sizes = nullptr;
    device_output_offsets = nullptr;
    device_qindices = nullptr;
    device_slice_metrics = nullptr;
    quantiser_capacity = 0;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_max_sizes), integer_bytes),
                 "encoder size allocation") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_output_offsets), integer_bytes),
                 "encoder offset allocation") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_qindices), qindex_bytes),
                 "encoder quantizer allocation") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_slice_metrics),
                            static_cast<size_t>(n_slices) * 6 * sizeof(int)),
                 "encoder metric allocation") ||
        !cuda_ok(cudaMemset(device_qindices, 0, qindex_bytes),
                 "encoder quantizer initialization"))
      return false;
    quantiser_capacity = n_slices;
    direct_n_slices = n_slices;
    direct_slices_per_line = slices_per_line;
  } else if (!device_output_offsets) {
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_output_offsets),
                            quantiser_capacity * sizeof(int)),
                 "encoder offset allocation"))
      return false;
  }
  if (!device_slice_metrics &&
      !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_slice_metrics),
                          quantiser_capacity * 6 * sizeof(int)),
               "encoder metric allocation"))
    return false;
  if (direct_n_slices != n_slices || direct_slices_per_line != slices_per_line) {
    if (!cuda_ok(cudaMemset(device_qindices, 0, qindex_bytes),
                 "encoder quantizer reset"))
      return false;
    direct_n_slices = n_slices;
    direct_slices_per_line = slices_per_line;
  }
  if (!reserve_buffer(reinterpret_cast<void **>(&device_bitstream),
                      &bitstream_capacity, static_cast<size_t>(output_bytes)) ||
      !register_host_buffer(output, static_cast<size_t>(output_bytes),
                            reinterpret_cast<void **>(&registered_bitstream),
                            &registered_bitstream_bytes, "bitstream buffer pinning"))
    return false;

  constexpr size_t matrix_entries = 32 * 512;
  if (!device_matrix_m &&
      (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_matrix_m),
                           matrix_entries * sizeof(uint16_t)), "encoder matrix allocation") ||
       !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_matrix_sh),
                           matrix_entries * sizeof(uint8_t)), "encoder shift allocation")))
    return false;
  if (cached_matrix_m != matrix_m) {
    if (!cuda_ok(cudaMemcpy(device_matrix_m, matrix_m,
                            matrix_entries * sizeof(uint16_t), cudaMemcpyHostToDevice),
                 "encoder matrix upload") ||
        !cuda_ok(cudaMemcpy(device_matrix_sh, matrix_sh,
                            matrix_entries * sizeof(uint8_t), cudaMemcpyHostToDevice),
                 "encoder shift upload"))
      return false;
    cached_matrix_m = matrix_m;
  }
  if (!scan_tables_uploaded) {
    uint16_t y_scan[256] = {};
    uint16_t c_scan[128] = {};
    auto build_scan = [](uint16_t *scan, int width) {
      int n = 0;
      int skip = 8;
      for (int y = 0; y < 8; y += skip)
        for (int x = 0; x < width; x += skip)
          scan[n++] = static_cast<uint16_t>((y << 5) | x);
      for (int level = 0; level < 3; ++level) {
        for (int y = 0; y < 8; y += skip)
          for (int x = 0; x < width; x += skip)
            scan[n++] = static_cast<uint16_t>((y << 5) | (x + skip / 2));
        for (int y = 0; y < 8; y += skip)
          for (int x = 0; x < width; x += skip)
            scan[n++] = static_cast<uint16_t>(((y + skip / 2) << 5) | x);
        for (int y = 0; y < 8; y += skip)
          for (int x = 0; x < width; x += skip)
            scan[n++] = static_cast<uint16_t>(((y + skip / 2) << 5) | (x + skip / 2));
        skip >>= 1;
      }
    };
    build_scan(y_scan, 32);
    build_scan(c_scan, 16);
    uint16_t y_inverse[256] = {};
    uint16_t c_inverse[128] = {};
    for (int n = 0; n < 256; ++n) {
      const int position = y_scan[n];
      y_inverse[(position >> 5) * 32 + (position & 31)] = static_cast<uint16_t>(n);
    }
    for (int n = 0; n < 128; ++n) {
      const int position = c_scan[n];
      c_inverse[(position >> 5) * 16 + (position & 31)] = static_cast<uint16_t>(n);
    }
    if (!cuda_ok(cudaMemcpyToSymbol(scan_to_y_position, y_scan, sizeof(y_scan)),
                 "luma scan upload") ||
        !cuda_ok(cudaMemcpyToSymbol(scan_to_c_position, c_scan, sizeof(c_scan)),
                 "chroma scan upload") ||
        !cuda_ok(cudaMemcpyToSymbol(y_position_to_scan, y_inverse, sizeof(y_inverse)),
                 "luma inverse scan upload") ||
        !cuda_ok(cudaMemcpyToSymbol(c_position_to_scan, c_inverse, sizeof(c_inverse)),
                 "chroma inverse scan upload"))
      return false;
    scan_tables_uploaded = true;
  }
  if (!cuda_ok(cudaMemcpyAsync(device_max_sizes, max_sizes, integer_bytes,
                               cudaMemcpyHostToDevice, encode_stream),
               "encoder budgets upload") ||
      !cuda_ok(cudaMemcpyAsync(device_output_offsets, output_offsets, integer_bytes,
                               cudaMemcpyHostToDevice, encode_stream),
               "encoder offsets upload"))
    return false;
  for (int c = 0; c < 3; ++c)
    if (!cuda_ok(cudaStreamWaitEvent(encode_stream, transform_done[c]),
                 "encoder transform dependency"))
      return false;

  select_quantisers_32x8<<<(n_slices + 7) / 8, 256, 0, encode_stream>>>(
      device_output[0], device_output[1], device_output[2], output_stride[0],
      output_stride[1], device_max_sizes, n_slices, slices_per_line,
      slice_size_scalar, device_matrix_m, device_matrix_sh, device_qindices,
      device_slice_metrics);
  int max_slice_size = 0;
  for (int i = 0; i < n_slices; ++i)
    max_slice_size = max(max_slice_size, max_sizes[i]);
  const int shared_stride = (max_slice_size + 3) & ~3;
  if (shared_stride * 8 <= 48 * 1024) {
    serialise_slices_32x8_warp<<<(n_slices + 7) / 8, 256,
                                  shared_stride * 8, encode_stream>>>(
        device_output[0], device_output[1], device_output[2], output_stride[0],
        output_stride[1], device_max_sizes, device_output_offsets, n_slices,
        slices_per_line, slice_size_scalar, device_matrix_m, device_matrix_sh,
        device_qindices, device_slice_metrics, device_bitstream, shared_stride);
  } else {
    serialise_slices_32x8<<<(n_slices + 255) / 256, 256, 0, encode_stream>>>(
        device_output[0], device_output[1], device_output[2], output_stride[0],
        output_stride[1], device_max_sizes, device_output_offsets, n_slices,
        slices_per_line, slice_size_scalar, device_matrix_m, device_matrix_sh,
        device_qindices, device_bitstream);
  }
  if (!cuda_ok(cudaGetLastError(), "encoder kernel launch") ||
      !cuda_ok(cudaMemcpyAsync(output, device_bitstream, static_cast<size_t>(output_bytes),
                               cudaMemcpyDeviceToHost, encode_stream),
               "bitstream download") ||
      !cuda_ok(cudaStreamSynchronize(encode_stream), "encoder synchronization"))
    return false;
  return true;
}
