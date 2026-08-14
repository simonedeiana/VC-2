#ifndef VC2_DD13_INVTRANSFORM_SSE4_2_HPP
#define VC2_DD13_INVTRANSFORM_SSE4_2_HPP

/****************************************************************************
 * deslauriers_dubuc_13_7_invtransform.hpp : SSE4.2 DD 13/7 inverse kernel
 ****************************************************************************/

#include <stdint.h>
#include "deslauriers_dubuc_9_7_invtransform.hpp"
#include "../vc2inversetransform_c/deslauriers_dubuc_13_7_invtransform.hpp"

static inline __m128i dd13_predict(const __m128i xm3, const __m128i xm1,
                                   const __m128i xp1, const __m128i xp3) {
  const __m128i nine_xm1 = _mm_add_epi32(xm1, _mm_slli_epi32(xm1, 3));
  const __m128i nine_xp1 = _mm_add_epi32(xp1, _mm_slli_epi32(xp1, 3));
  __m128i v = _mm_sub_epi32(_mm_setzero_si128(), xm3);
  v = _mm_add_epi32(v, nine_xm1);
  v = _mm_add_epi32(v, nine_xp1);
  v = _mm_sub_epi32(v, xp3);
  return _mm_srai_epi32(_mm_add_epi32(v, _mm_set1_epi32(16)), 5);
}

static inline void Deslauriers_Dubuc_13_7_invtransform_V_inplace_sse4_2_int16_t(
    void *_idata, const int istride, const int width, const int height) {
  if ((width & 3) != 0 || height < 8) {
    Deslauriers_Dubuc_13_7_invtransform_V_inplace<1, int16_t>(
      _idata, istride, width, height);
    return;
  }

  int16_t *idata = (int16_t *)_idata;
  for (int x = 0; x < width; x += 4) {
    int y = 0;
    __m128i Xm6, Xm5, Xm4, Xm3, Xm2, Xm1, X, Xp1, Xp2, Xp3;
    __m128i Dm6, Dm4, Dm3, Dm2, D;

    X = dd9_load4_i16(idata + (y + 0) * istride + x);
    Xp1 = dd9_load4_i16(idata + (y + 1) * istride + x);
    Xp2 = dd9_load4_i16(idata + (y + 2) * istride + x);
    Xp3 = dd9_load4_i16(idata + (y + 3) * istride + x);
    Xm1 = Xp1;
    Xm3 = Xp3;
    D = _mm_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm4 = D;
    Dm2 = D;

    Xm3 = Xm1;
    Xm1 = Xp1;
    X = Xp2;
    Xp1 = Xp3;
    Xp2 = dd9_load4_i16(idata + (y + 4) * istride + x);
    Xp3 = dd9_load4_i16(idata + (y + 5) * istride + x);
    y += 2;

    D = _mm_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;

    Xm3 = Xm1;
    Xm1 = Xp1;
    X = Xp2;
    Xp1 = Xp3;
    Xp2 = dd9_load4_i16(idata + (y + 4) * istride + x);
    Xp3 = dd9_load4_i16(idata + (y + 5) * istride + x);
    y += 2;

    for (; y < height - 4; y += 2) {
      D = _mm_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
      Dm3 = _mm_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store4_i16(idata + (y - 4) * istride + x, Dm4);
      dd9_store4_i16(idata + (y - 3) * istride + x, Dm3);

      Dm6 = Dm4;
      Dm4 = Dm2;
      Dm2 = D;
      Xm3 = Xm1;
      Xm2 = X;
      Xm1 = Xp1;
      X = Xp2;
      Xp1 = Xp3;
      Xp2 = dd9_load4_i16(idata + (y + 4) * istride + x);
      Xp3 = dd9_load4_i16(idata + (y + 5) * istride + x);
    }

    D = _mm_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm3 = _mm_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd9_store4_i16(idata + (y - 4) * istride + x, Dm4);
    dd9_store4_i16(idata + (y - 3) * istride + x, Dm3);

    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    Xm4 = Xm2;
    Xm3 = Xm1;
    Xm2 = X;
    Xm1 = Xp1;
    X = Xp2;
    Xp1 = Xp3;
    y += 2;

    D = _mm_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm3 = _mm_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd9_store4_i16(idata + (y - 4) * istride + x, Dm4);
    dd9_store4_i16(idata + (y - 3) * istride + x, Dm3);

    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    Xp2 = Xm2;
    Xp3 = Xm3;
    Xm6 = Xm4;
    Xm5 = Xm3;
    Xm4 = Xm2;
    Xm3 = Xm1;
    Xm2 = X;
    Xm1 = Xp1;
    y += 2;

    D = Dm2;
    Dm3 = _mm_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd9_store4_i16(idata + (y - 4) * istride + x, Dm4);
    dd9_store4_i16(idata + (y - 3) * istride + x, Dm3);

    D = Dm4;
    Dm6 = Dm4;
    Dm4 = Dm2;
    X = Xp2;
    Xp1 = Xp3;
    Xp2 = Xm6;
    Xp3 = Xm5;
    Xm4 = Xm2;
    Xm3 = Xm1;
    y += 2;

    Dm3 = _mm_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd9_store4_i16(idata + (y - 4) * istride + x, Dm4);
    dd9_store4_i16(idata + (y - 3) * istride + x, Dm3);
  }
}

#endif
