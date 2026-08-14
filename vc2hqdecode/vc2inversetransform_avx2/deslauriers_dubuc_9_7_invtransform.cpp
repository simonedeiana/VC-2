/****************************************************************************
 * deslauriers_dubuc_9_7_invtransform.cpp : AVX2 DD 9/7 inverse vertical
 ****************************************************************************/

#include "deslauriers_dubuc_9_7_invtransform.hpp"
#include "../vc2inversetransform_c/deslauriers_dubuc_9_7_invtransform.hpp"

void Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_int16_t(
    void *_idata, const int istride, const int width, const int height) {
  if ((width & 7) != 0 || height < 8) {
    Deslauriers_Dubuc_9_7_invtransform_V_inplace<1, int16_t>(
      _idata, istride, width, height);
    return;
  }

  int16_t *idata = (int16_t *)_idata;
  for (int x = 0; x < width; x += 8) {
    int y = 0;
    __m256i Xm3, Xm1, X, Xp1;
    __m256i Dm6, Dm4, Dm2, D;

    X = dd9_load8_i16(idata + (y + 0) * istride + x);
    Xp1 = dd9_load8_i16(idata + (y + 1) * istride + x);
    Xm1 = Xp1;
    D = _mm256_sub_epi32(X, dd9_predict(Xm1, Xp1));
    Dm4 = D;
    Dm2 = D;

    y += 2;
    X = dd9_load8_i16(idata + (y + 0) * istride + x);
    Xp1 = dd9_load8_i16(idata + (y + 1) * istride + x);
    D = _mm256_sub_epi32(X, dd9_predict(Xm1, Xp1));
    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    Xm3 = Xm1;
    Xm1 = Xp1;
    X = dd9_load8_i16(idata + (y + 2) * istride + x);
    Xp1 = dd9_load8_i16(idata + (y + 3) * istride + x);
    y += 2;

    for (; y < height - 2; y += 2) {
      D = _mm256_sub_epi32(X, dd9_predict(Xm1, Xp1));
      const __m256i Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store8_i16(idata + (y - 4) * istride + x, Dm4);
      dd9_store8_i16(idata + (y - 3) * istride + x, Dm3);

      Dm6 = Dm4;
      Dm4 = Dm2;
      Dm2 = D;
      Xm3 = Xm1;
      Xm1 = Xp1;
      X = dd9_load8_i16(idata + (y + 2) * istride + x);
      Xp1 = dd9_load8_i16(idata + (y + 3) * istride + x);
    }

    D = _mm256_sub_epi32(X, dd9_predict(Xm1, Xp1));
    {
      const __m256i Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store8_i16(idata + (y - 4) * istride + x, Dm4);
      dd9_store8_i16(idata + (y - 3) * istride + x, Dm3);
    }

    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    Xm3 = Xm1;
    Xm1 = Xp1;
    y += 2;
    {
      const __m256i Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store8_i16(idata + (y - 4) * istride + x, Dm4);
      dd9_store8_i16(idata + (y - 3) * istride + x, Dm3);
    }

    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    D = Dm6;
    Xm3 = Xm1;
    Xm1 = Xp1;
    y += 2;
    {
      const __m256i Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store8_i16(idata + (y - 4) * istride + x, Dm4);
      dd9_store8_i16(idata + (y - 3) * istride + x, Dm3);
    }
  }
}
