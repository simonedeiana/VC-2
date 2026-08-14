/*****************************************************************************
 * invtransform_avx2.cpp : Inverse transform dispatch: AVX2 version
 *****************************************************************************
 * Only the DD9/7 and DD13/7 finest-level inverse vertical kernels have AVX2
 * implementations; everything else falls back to the SSE4.2 dispatch (which
 * in turn falls back to scalar).
 *****************************************************************************/

#include "invtransform_avx2.hpp"
#include "../vc2inversetransform_c/invtransform_c.hpp"
#include "../vc2inversetransform_sse4_2/invtransform_sse4_2.hpp"
#include "deslauriers_dubuc_9_7_invtransform.hpp"
#include "deslauriers_dubuc_13_7_invtransform.hpp"

InplaceTransform get_invvtransform_avx2(int wavelet_index, int level, int depth, int sample_size) {
  if (sample_size == 2) {
    switch (wavelet_index) {
    case VC2DECODER_WFT_DESLAURIERS_DUBUC_9_7:
      if (depth - level - 1 == 0)
        return Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_int16_t;
      break;
    case VC2DECODER_WFT_DESLAURIERS_DUBUC_13_7:
      if (depth - level - 1 == 0)
        return Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_int16_t;
      break;
    default:
      break;
    }
  }

  return get_invvtransform_sse4_2(wavelet_index, level, depth, sample_size);
}
