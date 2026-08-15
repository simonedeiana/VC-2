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

template<int skip> static inline __m256i dd9_load8_i16_stride(const int16_t *p) {
  const __m256i indexes = _mm256_setr_epi32(0, skip, 2 * skip, 3 * skip,
                                            4 * skip, 5 * skip, 6 * skip, 7 * skip);
  const __m256i gathered = _mm256_i32gather_epi32((const int *)p, indexes, 2);
  return _mm256_srai_epi32(_mm256_slli_epi32(gathered, 16), 16);
}

template<int skip> static inline void dd9_store8_i16_stride(int16_t *p, const __m256i v) {
  p[0 * skip] = (int16_t)_mm256_extract_epi32(v, 0);
  p[1 * skip] = (int16_t)_mm256_extract_epi32(v, 1);
  p[2 * skip] = (int16_t)_mm256_extract_epi32(v, 2);
  p[3 * skip] = (int16_t)_mm256_extract_epi32(v, 3);
  p[4 * skip] = (int16_t)_mm256_extract_epi32(v, 4);
  p[5 * skip] = (int16_t)_mm256_extract_epi32(v, 5);
  p[6 * skip] = (int16_t)_mm256_extract_epi32(v, 6);
  p[7 * skip] = (int16_t)_mm256_extract_epi32(v, 7);
}

template<int skip> static void dd9_invtransform_V_inplace_avx2_stride(
    void *_idata, const int istride, const int width, const int height) {
  if ((width % (8 * skip)) != 0 || height < 8 * skip) {
    Deslauriers_Dubuc_9_7_invtransform_V_inplace<skip, int16_t>(
      _idata, istride, width, height);
    return;
  }

  int16_t *idata = (int16_t *)_idata;
  for (int x = 0; x < width; x += 8 * skip) {
    int y = 0;
    __m256i Xm3, Xm1, X, Xp1;
    __m256i Dm6, Dm4, Dm2, D;

    X = dd9_load8_i16_stride<skip>(idata + (y + 0 * skip) * istride + x);
    Xp1 = dd9_load8_i16_stride<skip>(idata + (y + 1 * skip) * istride + x);
    Xm1 = Xp1;
    D = _mm256_sub_epi32(X, dd9_predict(Xm1, Xp1));
    Dm4 = D;
    Dm2 = D;

    y += 2 * skip;
    X = dd9_load8_i16_stride<skip>(idata + (y + 0 * skip) * istride + x);
    Xp1 = dd9_load8_i16_stride<skip>(idata + (y + 1 * skip) * istride + x);
    D = _mm256_sub_epi32(X, dd9_predict(Xm1, Xp1));
    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    Xm3 = Xm1;
    Xm1 = Xp1;
    X = dd9_load8_i16_stride<skip>(idata + (y + 2 * skip) * istride + x);
    Xp1 = dd9_load8_i16_stride<skip>(idata + (y + 3 * skip) * istride + x);
    y += 2 * skip;

    for (; y < height - 2 * skip; y += 2 * skip) {
      D = _mm256_sub_epi32(X, dd9_predict(Xm1, Xp1));
      const __m256i Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store8_i16_stride<skip>(idata + (y - 4 * skip) * istride + x, Dm4);
      dd9_store8_i16_stride<skip>(idata + (y - 3 * skip) * istride + x, Dm3);

      Dm6 = Dm4;
      Dm4 = Dm2;
      Dm2 = D;
      Xm3 = Xm1;
      Xm1 = Xp1;
      X = dd9_load8_i16_stride<skip>(idata + (y + 2 * skip) * istride + x);
      Xp1 = dd9_load8_i16_stride<skip>(idata + (y + 3 * skip) * istride + x);
    }

    D = _mm256_sub_epi32(X, dd9_predict(Xm1, Xp1));
    {
      const __m256i Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store8_i16_stride<skip>(idata + (y - 4 * skip) * istride + x, Dm4);
      dd9_store8_i16_stride<skip>(idata + (y - 3 * skip) * istride + x, Dm3);
    }

    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    Xm3 = Xm1;
    Xm1 = Xp1;
    y += 2 * skip;
    {
      const __m256i Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store8_i16_stride<skip>(idata + (y - 4 * skip) * istride + x, Dm4);
      dd9_store8_i16_stride<skip>(idata + (y - 3 * skip) * istride + x, Dm3);
    }

    Dm6 = Dm4;
    Dm4 = Dm2;
    Dm2 = D;
    D = Dm6;
    Xm3 = Xm1;
    Xm1 = Xp1;
    y += 2 * skip;
    {
      const __m256i Dm3 = _mm256_add_epi32(Xm3, dd9_update(Dm6, Dm4, Dm2, D));
      dd9_store8_i16_stride<skip>(idata + (y - 4 * skip) * istride + x, Dm4);
      dd9_store8_i16_stride<skip>(idata + (y - 3 * skip) * istride + x, Dm3);
    }
  }
}

void Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_s2(
    void *idata, const int istride, const int width, const int height) {
  dd9_invtransform_V_inplace_avx2_stride<2>(idata, istride, width, height);
}

void Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_s4(
    void *idata, const int istride, const int width, const int height) {
  dd9_invtransform_V_inplace_avx2_stride<4>(idata, istride, width, height);
}

void Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_s8(
    void *idata, const int istride, const int width, const int height) {
  dd9_invtransform_V_inplace_avx2_stride<8>(idata, istride, width, height);
}
