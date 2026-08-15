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

template<int skip> static inline __m256i dd13_load8_i16_stride(const int16_t *p) {
  const __m256i indexes = _mm256_setr_epi32(0, skip, 2 * skip, 3 * skip,
                                            4 * skip, 5 * skip, 6 * skip, 7 * skip);
  const __m256i gathered = _mm256_i32gather_epi32((const int *)p, indexes, 2);
  return _mm256_srai_epi32(_mm256_slli_epi32(gathered, 16), 16);
}

template<int skip> static inline void dd13_store8_i16_stride(int16_t *p, const __m256i v) {
  p[0 * skip] = (int16_t)_mm256_extract_epi32(v, 0);
  p[1 * skip] = (int16_t)_mm256_extract_epi32(v, 1);
  p[2 * skip] = (int16_t)_mm256_extract_epi32(v, 2);
  p[3 * skip] = (int16_t)_mm256_extract_epi32(v, 3);
  p[4 * skip] = (int16_t)_mm256_extract_epi32(v, 4);
  p[5 * skip] = (int16_t)_mm256_extract_epi32(v, 5);
  p[6 * skip] = (int16_t)_mm256_extract_epi32(v, 6);
  p[7 * skip] = (int16_t)_mm256_extract_epi32(v, 7);
}

template<int skip> static void dd13_invtransform_V_inplace_avx2_stride(
    void *_idata, const int istride, const int width, const int height) {
  if ((width % (8 * skip)) != 0 || height < 8 * skip) {
    Deslauriers_Dubuc_13_7_invtransform_V_inplace<skip, int16_t>(
      _idata, istride, width, height);
    return;
  }

  int16_t *idata = (int16_t *)_idata;
  for (int x = 0; x < width; x += 8 * skip) {
    int y = 0;
    __m256i Xm6, Xm5, Xm4, Xm3, Xm2, Xm1, X, Xp1, Xp2, Xp3;
    __m256i Dm6, Dm4, Dm3, Dm2, D;

    X = dd13_load8_i16_stride<skip>(idata + (y + 0 * skip) * istride + x);
    Xp1 = dd13_load8_i16_stride<skip>(idata + (y + 1 * skip) * istride + x);
    Xp2 = dd13_load8_i16_stride<skip>(idata + (y + 2 * skip) * istride + x);
    Xp3 = dd13_load8_i16_stride<skip>(idata + (y + 3 * skip) * istride + x);
    Xm1 = Xp1;
    Xm3 = Xp3;
    D = _mm256_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm4 = D;
    Dm2 = D;

    Xm3 = Xm1;
    Xm1 = Xp1;
    X = Xp2;
    Xp1 = Xp3;
    Xp2 = dd13_load8_i16_stride<skip>(idata + (y + 4 * skip) * istride + x);
    Xp3 = dd13_load8_i16_stride<skip>(idata + (y + 5 * skip) * istride + x);
    y += 2 * skip;

    D = _mm256_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;

    Xm3 = Xm1;
    Xm1 = Xp1;
    X = Xp2;
    Xp1 = Xp3;
    Xp2 = dd13_load8_i16_stride<skip>(idata + (y + 4 * skip) * istride + x);
    Xp3 = dd13_load8_i16_stride<skip>(idata + (y + 5 * skip) * istride + x);
    y += 2 * skip;

    for (; y < height - 4 * skip; y += 2 * skip) {
      D = _mm256_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
      Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd13_store8_i16_stride<skip>(idata + (y - 4 * skip) * istride + x, Dm4);
      dd13_store8_i16_stride<skip>(idata + (y - 3 * skip) * istride + x, Dm3);

      Dm6 = Dm4;
      Dm4 = Dm2;
      Dm2 = D;
      Xm3 = Xm1;
      Xm2 = X;
      Xm1 = Xp1;
      X = Xp2;
      Xp1 = Xp3;
      Xp2 = dd13_load8_i16_stride<skip>(idata + (y + 4 * skip) * istride + x);
      Xp3 = dd13_load8_i16_stride<skip>(idata + (y + 5 * skip) * istride + x);
    }

    D = _mm256_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd13_store8_i16_stride<skip>(idata + (y - 4 * skip) * istride + x, Dm4);
    dd13_store8_i16_stride<skip>(idata + (y - 3 * skip) * istride + x, Dm3);

    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    Xm4 = Xm2;
    Xm3 = Xm1;
    Xm2 = X;
    Xm1 = Xp1;
    X = Xp2;
    Xp1 = Xp3;
    y += 2 * skip;

    D = _mm256_sub_epi32(X, dd13_predict(Xm3, Xm1, Xp1, Xp3));
    Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd13_store8_i16_stride<skip>(idata + (y - 4 * skip) * istride + x, Dm4);
    dd13_store8_i16_stride<skip>(idata + (y - 3 * skip) * istride + x, Dm3);

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
    y += 2 * skip;

    D = Dm2;
    Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd13_store8_i16_stride<skip>(idata + (y - 4 * skip) * istride + x, Dm4);
    dd13_store8_i16_stride<skip>(idata + (y - 3 * skip) * istride + x, Dm3);

    D = Dm4;
    Dm6 = Dm4;
    Dm4 = Dm2;
    X = Xp2;
    Xp1 = Xp3;
    Xp2 = Xm6;
    Xp3 = Xm5;
    Xm4 = Xm2;
    Xm3 = Xm1;
    y += 2 * skip;

    Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
    dd13_store8_i16_stride<skip>(idata + (y - 4 * skip) * istride + x, Dm4);
    dd13_store8_i16_stride<skip>(idata + (y - 3 * skip) * istride + x, Dm3);
  }
}

void Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_s2(
    void *idata, const int istride, const int width, const int height) {
  dd13_invtransform_V_inplace_avx2_stride<2>(idata, istride, width, height);
}

void Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_s4(
    void *idata, const int istride, const int width, const int height) {
  dd13_invtransform_V_inplace_avx2_stride<4>(idata, istride, width, height);
}

void Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_s8(
    void *idata, const int istride, const int width, const int height) {
  dd13_invtransform_V_inplace_avx2_stride<8>(idata, istride, width, height);
}
