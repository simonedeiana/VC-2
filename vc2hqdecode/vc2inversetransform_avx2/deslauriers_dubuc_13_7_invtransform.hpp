#ifndef VC2_DD13_INVTRANSFORM_AVX2_HPP
#define VC2_DD13_INVTRANSFORM_AVX2_HPP

/****************************************************************************
 * deslauriers_dubuc_13_7_invtransform.hpp : AVX2 DD 13/7 inverse vertical
 * kernel (8 columns per iteration, int16 samples).
 ****************************************************************************/

#include <stdint.h>
#include <immintrin.h>
#include "deslauriers_dubuc_9_7_invtransform.hpp"
#include "../vc2inversetransform_c/deslauriers_dubuc_13_7_invtransform.hpp"

static inline __m256i dd13_predict(const __m256i xm3, const __m256i xm1,
                                   const __m256i xp1, const __m256i xp3) {
  const __m256i nine_xm1 = _mm256_add_epi32(xm1, _mm256_slli_epi32(xm1, 3));
  const __m256i nine_xp1 = _mm256_add_epi32(xp1, _mm256_slli_epi32(xp1, 3));
  __m256i v = _mm256_sub_epi32(_mm256_setzero_si256(), xm3);
  v = _mm256_add_epi32(v, nine_xm1);
  v = _mm256_add_epi32(v, nine_xp1);
  v = _mm256_sub_epi32(v, xp3);
  return _mm256_srai_epi32(_mm256_add_epi32(v, _mm256_set1_epi32(16)), 5);
}

// 8 columns at once, widening int16 -> int32, scalar fallback otherwise.
// Mirrors the SSE4.2 4-column kernel structure exactly (one SIMD lane per
// image column; the lifting recurrence stays within each lane).
void Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_int16_t(
    void *_idata, const int istride, const int width, const int height);

#endif
