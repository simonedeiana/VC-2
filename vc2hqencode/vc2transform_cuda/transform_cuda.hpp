#ifndef VC2_TRANSFORM_CUDA_HPP
#define VC2_TRANSFORM_CUDA_HPP

#include <stdint.h>

// Returns true when the complete three-level Haar-0 transform was executed on
// CUDA. A false result leaves the caller free to run the existing CPU path.
bool vc2_cuda_haar0_transform_10p2_i16_3plane(
    const uint16_t *const input[3], const int input_stride[3],
    int16_t *const output[3], const int output_stride[3],
    const int input_width[3], const int input_height[3],
    const int output_width[3], const int output_height[3], int depth,
    bool download_coefficients = true);

// Create the CUDA context and reserve the fixed coefficient/input storage at
// encoder configuration time, keeping one-time driver/allocation cost out of
// the first picture's latency.
bool vc2_cuda_prepare_haar0_10p2_i16(const int input_width[3],
                                     const int input_height[3],
                                     int16_t *const output[3],
                                     const int output_stride[3],
                                     const int output_height[3]);

// Select the fastest-preset quantizer for every 32x8 slice while transformed
// coefficients are still resident. Returns qindices in raster slice order.
bool vc2_cuda_select_quantisers_32x8_i16(const int *max_sizes,
                                         int n_slices,
                                         int slices_per_line,
                                         int slice_size_scalar,
                                         const int output_stride[3],
                                         const uint16_t *matrix_m,
                                         const uint8_t *matrix_sh,
                                         uint8_t *qindices);

bool vc2_cuda_encode_32x8_i16(const int *max_sizes,
                              const int *output_offsets,
                              int n_slices,
                              int slices_per_line,
                              int slice_size_scalar,
                              const int output_stride[3],
                              const uint16_t *matrix_m,
                              const uint8_t *matrix_sh,
                              uint8_t *output,
                              int output_bytes);

bool vc2_cuda_available();
const char *vc2_cuda_last_error();

#endif
