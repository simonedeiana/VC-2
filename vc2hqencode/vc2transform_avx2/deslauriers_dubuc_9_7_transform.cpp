/*****************************************************************************
 * deslauriers_dubuc_9_7_transform.cpp : AVX2 DD 9/7 forward vertical kernel
 * (8 columns per iteration, int16 samples). Mirrors the inverse-vertical
 * AVX2 kernel structure: one SIMD lane per image column, the lifting
 * recurrence kept within each lane.
 *****************************************************************************/

#include <immintrin.h>
#include "../vc2transform_c/deslauriers_dubuc_9_7_transform.hpp"

static inline __m256i dd9f_load8(const int16_t *p) {
  return _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)p));
}

static inline void dd9f_store8(int16_t *p, const __m256i v) {
  // Truncate each of the 8 int32 lanes to its low int16 (matches scalar cast).
  const __m256i low_words = _mm256_setr_epi8(
    0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
    0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m256i s = _mm256_shuffle_epi8(v, low_words);
  const __m128i lo = _mm256_castsi256_si128(s);
  const __m128i hi = _mm256_extracti128_si256(s, 1);
  _mm_storeu_si128((__m128i *)p, _mm_unpacklo_epi64(lo, hi));
}

static inline __m256i dd9f_predict(const __m256i a, const __m256i b) {
  return _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(a, b), _mm256_set1_epi32(2)), 2);
}

static inline __m256i dd9f_update(const __m256i a, const __m256i b,
                                  const __m256i c, const __m256i d) {
  const __m256i nine_b = _mm256_add_epi32(b, _mm256_slli_epi32(b, 3));
  const __m256i nine_c = _mm256_add_epi32(c, _mm256_slli_epi32(c, 3));
  __m256i v = _mm256_sub_epi32(_mm256_setzero_si256(), a);
  v = _mm256_add_epi32(v, nine_b);
  v = _mm256_add_epi32(v, nine_c);
  v = _mm256_sub_epi32(v, d);
  return _mm256_srai_epi32(_mm256_add_epi32(v, _mm256_set1_epi32(8)), 4);
}

void Deslauriers_Dubuc_9_7_transform_V_inplace_avx2(void *_idata,
                                                    const int istride,
                                                    const int width,
                                                    const int height,
                                                    const int skip) {
  if (skip != 1 || (width & 7) != 0 || height < 8) {
    Deslauriers_Dubuc_9_7_transform_V_inplace<1, int16_t>(
      _idata, istride, width, height, skip);
    return;
  }

  int16_t *idata = (int16_t *)_idata;
  for (int x = 0; x < width; x += 8) {
    int y = 0;
    __m256i Dm2, D, Dp1, Dp2, Dp3, Dp4;
    __m256i Xm1, X, Xp1;

    D   = dd9f_load8(idata + (y + 0) * istride + x);
    Dp1 = dd9f_load8(idata + (y + 1) * istride + x);
    Dp2 = dd9f_load8(idata + (y + 2) * istride + x);
    Dp3 = dd9f_load8(idata + (y + 3) * istride + x);
    Dp4 = dd9f_load8(idata + (y + 4) * istride + x);

    Dm2 = D;
    Xp1 = _mm256_sub_epi32(Dp1, dd9f_update(Dm2, D, Dp2, Dp4));
    Xm1 = Xp1;
    X   = _mm256_add_epi32(D, dd9f_predict(Xm1, Xp1));
    dd9f_store8(idata + (y + 0) * istride + x, X);
    dd9f_store8(idata + (y + 1) * istride + x, Xp1);

    Dm2 = D;
    D   = Dp2;
    Dp1 = Dp3;
    Dp2 = Dp4;
    Dp3 = dd9f_load8(idata + (y + 5) * istride + x);
    Dp4 = dd9f_load8(idata + (y + 6) * istride + x);
    Xm1 = Xp1;
    y += 2;

    for (; y < height - 6; y += 2) {
      Xp1 = _mm256_sub_epi32(Dp1, dd9f_update(Dm2, D, Dp2, Dp4));
      X   = _mm256_add_epi32(D, dd9f_predict(Xm1, Xp1));
      dd9f_store8(idata + (y + 0) * istride + x, X);
      dd9f_store8(idata + (y + 1) * istride + x, Xp1);

      Dm2 = D;
      D   = Dp2;
      Dp1 = Dp3;
      Dp2 = Dp4;
      Dp3 = dd9f_load8(idata + (y + 5) * istride + x);
      Dp4 = dd9f_load8(idata + (y + 6) * istride + x);
      Xm1 = Xp1;
    }

    {
      Xp1 = _mm256_sub_epi32(Dp1, dd9f_update(Dm2, D, Dp2, Dp4));
      X   = _mm256_add_epi32(D, dd9f_predict(Xm1, Xp1));
      dd9f_store8(idata + (y + 0) * istride + x, X);
      dd9f_store8(idata + (y + 1) * istride + x, Xp1);

      Dm2 = D;
      D   = Dp2;
      Dp1 = Dp3;
      Dp2 = Dp4;
      Dp3 = dd9f_load8(idata + (y + 5) * istride + x);
      Dp4 = Dp4;
      Xm1 = Xp1;
    }
    y += 2;

    {
      Xp1 = _mm256_sub_epi32(Dp1, dd9f_update(Dm2, D, Dp2, Dp4));
      X   = _mm256_add_epi32(D, dd9f_predict(Xm1, Xp1));
      dd9f_store8(idata + (y + 0) * istride + x, X);
      dd9f_store8(idata + (y + 1) * istride + x, Xp1);

      Dm2 = D;
      D   = Dp2;
      Dp1 = Dp3;
      Dp2 = D;
      Dp3 = Dp3;
      Dp4 = Dm2;
      Xm1 = Xp1;
    }
    y += 2;

    {
      Xp1 = _mm256_sub_epi32(Dp1, dd9f_update(Dm2, D, Dp2, Dp4));
      X   = _mm256_add_epi32(D, dd9f_predict(Xm1, Xp1));
      dd9f_store8(idata + (y + 0) * istride + x, X);
      dd9f_store8(idata + (y + 1) * istride + x, Xp1);
    }
  }
}
