/*****************************************************************************
 * invtransform_avx2.cpp : Inverse transform dispatch: AVX2 version
 *****************************************************************************
 * DD9/7 and DD13/7 inverse vertical kernels use AVX2 at all supported
 * decomposition strides; everything else falls back to the SSE4.2 dispatch.
 *****************************************************************************/

#include "invtransform_avx2.hpp"
#include "../vc2inversetransform_c/invtransform_c.hpp"
#include "../vc2inversetransform_sse4_2/invtransform_sse4_2.hpp"
#include "deslauriers_dubuc_9_7_invtransform.hpp"
#include "deslauriers_dubuc_13_7_invtransform.hpp"
#include <malloc.h>

InplaceTransform get_invvtransform_avx2(int wavelet_index, int level, int depth, int sample_size) {
  if (sample_size == 2) {
    switch (wavelet_index) {
    case VC2DECODER_WFT_DESLAURIERS_DUBUC_9_7:
      switch (depth - level - 1) {
      case 0:
        return Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_int16_t;
      case 1:
        return Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_s2;
      case 2:
        return Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_s4;
      case 3:
        return Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_s8;
      }
      break;
    case VC2DECODER_WFT_DESLAURIERS_DUBUC_13_7:
      switch (depth - level - 1) {
      case 0:
        return Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_int16_t;
      case 1:
        return Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_s2;
      case 2:
        return Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_s4;
      case 3:
        return Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_s8;
      }
      break;
    default:
      break;
    }
  }

  return get_invvtransform_sse4_2(wavelet_index, level, depth, sample_size);
}

InplaceTransformFinal get_invhtransformfinal_avx2(int wavelet_index, int active_bits, int sample_size) {
  if (sample_size == 2) {
    switch (wavelet_index) {
    case VC2DECODER_WFT_DESLAURIERS_DUBUC_9_7:
      switch (active_bits) {
      case 10: return Deslauriers_Dubuc_9_7_invtransform_H_final_1_avx2_int16_t<10>;
      case 12: return Deslauriers_Dubuc_9_7_invtransform_H_final_1_avx2_int16_t<12>;
      }
      break;
    case VC2DECODER_WFT_DESLAURIERS_DUBUC_13_7:
      switch (active_bits) {
      case 10: return Deslauriers_Dubuc_13_7_invtransform_H_final_1_avx2_int16_t<10>;
      case 12: return Deslauriers_Dubuc_13_7_invtransform_H_final_1_avx2_int16_t<12>;
      }
      break;
    default:
      break;
    }
  }

  return get_invhtransformfinal_sse4_2(wavelet_index, active_bits, sample_size);
}
