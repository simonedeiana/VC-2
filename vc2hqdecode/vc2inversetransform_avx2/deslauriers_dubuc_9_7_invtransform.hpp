#ifndef VC2_DD9_INVTRANSFORM_AVX2_HPP
#define VC2_DD9_INVTRANSFORM_AVX2_HPP

/****************************************************************************
 * deslauriers_dubuc_9_7_invtransform.hpp : AVX2 DD 9/7 inverse vertical
 * kernel (8 columns per iteration, int16 samples).
 ****************************************************************************/

#include <stdint.h>
#include <immintrin.h>

static inline __m256i dd9_load8_i16(const int16_t *p) {
  return _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)p));
}

static inline void dd9_store8_i16(int16_t *p, const __m256i v) {
  // Truncate each of the 8 int32 lanes to its low int16 (matches scalar cast).
  // shuffle_epi8 puts the 4 int16 of each 128-bit lane into that lane's low
  // 64 bits, so pull both halves together with extract + unpack.
  const __m256i low_words = _mm256_setr_epi8(
    0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
    0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m256i s = _mm256_shuffle_epi8(v, low_words);
  const __m128i lo = _mm256_castsi256_si128(s);
  const __m128i hi = _mm256_extracti128_si256(s, 1);
  _mm_storeu_si128((__m128i *)p, _mm_unpacklo_epi64(lo, hi));
}

static inline __m256i dd9_predict(const __m256i a, const __m256i b) {
  return _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(a, b), _mm256_set1_epi32(2)), 2);
}

static inline __m256i dd9_update(const __m256i dm6, const __m256i dm4,
                                 const __m256i dm2, const __m256i d) {
  const __m256i nine_dm4 = _mm256_add_epi32(dm4, _mm256_slli_epi32(dm4, 3));
  const __m256i nine_dm2 = _mm256_add_epi32(dm2, _mm256_slli_epi32(dm2, 3));
  __m256i v = _mm256_sub_epi32(_mm256_setzero_si256(), dm6);
  v = _mm256_add_epi32(v, nine_dm4);
  v = _mm256_add_epi32(v, nine_dm2);
  v = _mm256_sub_epi32(v, d);
  return _mm256_srai_epi32(_mm256_add_epi32(v, _mm256_set1_epi32(8)), 4);
}

// 8 columns at once, widening int16 -> int32, scalar fallback otherwise.
// Mirrors the SSE4.2 4-column kernel structure exactly (one SIMD lane per
// image column; the lifting recurrence stays within each lane).
void Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_int16_t(
    void *_idata, const int istride, const int width, const int height);

#endif
