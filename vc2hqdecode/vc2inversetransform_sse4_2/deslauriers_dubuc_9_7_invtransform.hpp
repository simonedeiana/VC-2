#ifndef VC2_DD9_INVTRANSFORM_SSE4_2_HPP
#define VC2_DD9_INVTRANSFORM_SSE4_2_HPP

/****************************************************************************
 * deslauriers_dubuc_9_7_invtransform.hpp : SSE4.2 DD 9/7 inverse kernels
 ****************************************************************************/

#include <stdint.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include "../vc2inversetransform_c/deslauriers_dubuc_9_7_invtransform.hpp"

static inline __m128i dd9_load4_i16(const int16_t *p) {
  return _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)p));
}

static inline void dd9_store4_i16(int16_t *p, const __m128i v) {
  const __m128i low_words = _mm_setr_epi8(
    0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
  _mm_storel_epi64((__m128i *)p, _mm_shuffle_epi8(v, low_words));
}

static inline __m128i dd9_predict(const __m128i a, const __m128i b) {
  return _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(a, b), _mm_set1_epi32(2)), 2);
}

static inline __m128i dd9_update(const __m128i dm6, const __m128i dm4,
                                 const __m128i dm2, const __m128i d) {
  const __m128i nine_dm4 = _mm_add_epi32(dm4, _mm_slli_epi32(dm4, 3));
  const __m128i nine_dm2 = _mm_add_epi32(dm2, _mm_slli_epi32(dm2, 3));
  __m128i v = _mm_sub_epi32(_mm_setzero_si128(), dm6);
  v = _mm_add_epi32(v, nine_dm4);
  v = _mm_add_epi32(v, nine_dm2);
  v = _mm_sub_epi32(v, d);
  return _mm_srai_epi32(_mm_add_epi32(v, _mm_set1_epi32(8)), 4);
}

static inline void Deslauriers_Dubuc_9_7_invtransform_V_inplace_sse4_2_int16_t(
    void *_idata, const int istride, const int width, const int height) {
  if ((width & 3) != 0 || height < 8) {
    Deslauriers_Dubuc_9_7_invtransform_V_inplace<1, int16_t>(
      _idata, istride, width, height);
    return;
  }

  int16_t *idata = (int16_t *)_idata;
  for (int x = 0; x < width; x += 4) {
    int y = 0;
    __m128i Xm3, Xm1, X, Xp1;
    __m128i Dm6, Dm4, Dm2, D;

    X = dd9_load4_i16(idata + (y + 0) * istride + x);
    Xp1 = dd9_load4_i16(idata + (y + 1) * istride + x);
    Xm1 = Xp1;
    D = _mm_sub_epi32(X, dd9_predict(Xm1, Xp1));
    Dm4 = D;
    Dm2 = D;

    y += 2;
    X = dd9_load4_i16(idata + (y + 0) * istride + x);
    Xp1 = dd9_load4_i16(idata + (y + 1) * istride + x);
    D = _mm_sub_epi32(X, dd9_predict(Xm1, Xp1));
    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    Xm3 = Xm1;
    Xm1 = Xp1;
    X = dd9_load4_i16(idata + (y + 2) * istride + x);
    Xp1 = dd9_load4_i16(idata + (y + 3) * istride + x);
    y += 2;

    for (; y < height - 2; y += 2) {
      D = _mm_sub_epi32(X, dd9_predict(Xm1, Xp1));
      const __m128i Dm3 = _mm_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store4_i16(idata + (y - 4) * istride + x, Dm4);
      dd9_store4_i16(idata + (y - 3) * istride + x, Dm3);

      Dm6 = Dm4;
      Dm4 = Dm2;
      Dm2 = D;
      Xm3 = Xm1;
      Xm1 = Xp1;
      X = dd9_load4_i16(idata + (y + 2) * istride + x);
      Xp1 = dd9_load4_i16(idata + (y + 3) * istride + x);
    }

    D = _mm_sub_epi32(X, dd9_predict(Xm1, Xp1));
    {
      const __m128i Dm3 = _mm_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store4_i16(idata + (y - 4) * istride + x, Dm4);
      dd9_store4_i16(idata + (y - 3) * istride + x, Dm3);
    }

    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    Xm3 = Xm1;
    Xm1 = Xp1;
    y += 2;
    {
      const __m128i Dm3 = _mm_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store4_i16(idata + (y - 4) * istride + x, Dm4);
      dd9_store4_i16(idata + (y - 3) * istride + x, Dm3);
    }

    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    D = Dm6;
    Xm3 = Xm1;
    Xm1 = Xp1;
    y += 2;
    {
      const __m128i Dm3 = _mm_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store4_i16(idata + (y - 4) * istride + x, Dm4);
      dd9_store4_i16(idata + (y - 3) * istride + x, Dm3);
    }
  }
}

#endif
