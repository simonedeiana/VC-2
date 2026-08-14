/****************************************************************************
 * deslauriers_dubuc_13_7_invtransform.cpp : AVX2 DD 13/7 inverse vertical
 ****************************************************************************/

#include "deslauriers_dubuc_13_7_invtransform.hpp"

void Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_int16_t(
    void *_idata, const int istride, const int width, const int height) {
  if ((width & 7) != 0 || height < 8) {
    Deslauriers_Dubuc_13_7_invtransform_V_inplace<1, int16_t>(
      _idata, istride, width, height);
    return;
  }

  int16_t *idata = (int16_t *)_idata;
  for (int x = 0; x < width; x += 8) {
    int y = 0;
    __m256i Xm6, Xm5, Xm4, Xm3, Xm2, Xm1, X, Xp1, Xp2, Xp3;
    __m256i Dm6, Dm4, Dm3, Dm2, D;

    X = dd9_load8_i16(idata + (y + 0) * istride + x);
    Xp1 = dd9_load8_i16(idata + (y + 1) * istride + x);
    Xp2 = dd9_load8_i16(idata + (y + 2) * istride + x);
    Xp3 = dd9_load8_i16(idata + (y + 3) * istride + x);
    Xm1 = Xp1;
    Xm3 = Xp3;
    D = _mm256_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm4 = D;
    Dm2 = D;

    Xm3 = Xm1;
    Xm1 = Xp1;
    X = Xp2;
    Xp1 = Xp3;
    Xp2 = dd9_load8_i16(idata + (y + 4) * istride + x);
    Xp3 = dd9_load8_i16(idata + (y + 5) * istride + x);
    y += 2;

    D = _mm256_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;

    Xm3 = Xm1;
    Xm1 = Xp1;
    X = Xp2;
    Xp1 = Xp3;
    Xp2 = dd9_load8_i16(idata + (y + 4) * istride + x);
    Xp3 = dd9_load8_i16(idata + (y + 5) * istride + x);
    y += 2;

    for (; y < height - 4; y += 2) {
      D = _mm256_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
      Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store8_i16(idata + (y - 4) * istride + x, Dm4);
      dd9_store8_i16(idata + (y - 3) * istride + x, Dm3);

      Dm6 = Dm4;
      Dm4 = Dm2;
      Dm2 = D;
      Xm3 = Xm1;
      Xm2 = X;
      Xm1 = Xp1;
      X = Xp2;
      Xp1 = Xp3;
      Xp2 = dd9_load8_i16(idata + (y + 4) * istride + x);
      Xp3 = dd9_load8_i16(idata + (y + 5) * istride + x);
    }

    D = _mm256_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd9_store8_i16(idata + (y - 4) * istride + x, Dm4);
    dd9_store8_i16(idata + (y - 3) * istride + x, Dm3);

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

    D = _mm256_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd9_store8_i16(idata + (y - 4) * istride + x, Dm4);
    dd9_store8_i16(idata + (y - 3) * istride + x, Dm3);

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
    Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd9_store8_i16(idata + (y - 4) * istride + x, Dm4);
    dd9_store8_i16(idata + (y - 3) * istride + x, Dm3);

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

    Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd9_store8_i16(idata + (y - 4) * istride + x, Dm4);
    dd9_store8_i16(idata + (y - 3) * istride + x, Dm3);
  }
}
