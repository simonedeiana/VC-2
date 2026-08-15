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

// ---------------------------------------------------------------------------
// DD 9/7 forward HORIZONTAL input transform (10P2 -> int16), AVX2.
// The scalar recurrence along x is a parallel column stencil, so a two-pass
// scheme (mirroring the decoder's inverse final-H) breaks the serial chain:
//   pass 1 de-interleaves the uint16 row into even/odd and converts each
//          sample to int32 (v-512)<<1;
//   pass 2 computes the odd outputs O[j] = odd - update(even[j-1..j+2]) from
//          compact even reads;
//   pass 3 computes the even outputs E[j] = even + predict(O[j-1],O[j]),
//          interleaves E/O and stores int16.
// Boundary mirrors: even[-1]=even[-2]=even[0]; even[nE]=even[nE-1],
// even[nE+1]=even[nE-2]; O[-1]=O[0]. The overlap mirror-fill and the
// bottom-row copy stay scalar (identical to the reference).

static inline __m128i dd9h_update4(__m128i a, __m128i b, __m128i c, __m128i d) {
  __m128i nb = _mm_add_epi32(b, _mm_slli_epi32(b, 3));
  __m128i nc = _mm_add_epi32(c, _mm_slli_epi32(c, 3));
  __m128i v = _mm_sub_epi32(_mm_setzero_si128(), a);
  v = _mm_add_epi32(v, nb);
  v = _mm_add_epi32(v, nc);
  v = _mm_sub_epi32(v, d);
  return _mm_srai_epi32(_mm_add_epi32(v, _mm_set1_epi32(8)), 4);
}

static inline __m128i dd9h_predict2(__m128i a, __m128i b) {
  return _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(a, b), _mm_set1_epi32(2)), 2);
}

void Deslauriers_Dubuc_9_7_transform_H_inplace_10P2_avx2(const char *_idata,
                                                         const int istride,
                                                         void **_odata,
                                                         const int ostride,
                                                         const int iwidth,
                                                         const int iheight,
                                                         const int owidth,
                                                         const int oheight) {
  if (iwidth < 16 || (iwidth & 7) != 0 || iheight < 8) {
    Deslauriers_Dubuc_9_7_transform_H_inplace_10P2<int16_t>(
      _idata, istride, _odata, ostride, iwidth, iheight, owidth, oheight);
    return;
  }

  const uint16_t *idata = (const uint16_t *)_idata;
  int16_t *odata = *(int16_t **)_odata;
  const int nE = iwidth / 2;

  int32_t *even32 = (int32_t *)_alloca((nE + 4) * sizeof(int32_t));
  int32_t *odd32  = (int32_t *)_alloca((nE + 4) * sizeof(int32_t));
  int32_t *Oscr   = (int32_t *)_alloca((nE + 2) * sizeof(int32_t));
  int32_t *e = even32 + 2;  // real even at e[0..nE-1]
  int32_t *o = odd32 + 2;   // real odd  at o[0..nE-1]
  int32_t *Op = Oscr + 1;   // real O    at Op[0..nE-1]

  const __m128i EVEN = _mm_setr_epi8(0,1, 4,5, 8,9, 12,13, -1,-1,-1,-1,-1,-1,-1,-1);
  const __m128i ODD  = _mm_setr_epi8(2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1);
  const __m128i LO16 = _mm_setr_epi8(0,1, 4,5, 8,9, 12,13, -1,-1,-1,-1,-1,-1,-1,-1);
  const __m128i M512 = _mm_set1_epi32(512);

  for (int y = 0; y < iheight; y++) {
    const uint16_t *row = idata + (size_t)y * istride;
    int16_t *orow = odata + (size_t)y * ostride;

    // pass 1: de-interleave + convert (v-512)<<1
    for (int j = 0; j + 4 <= nE; j += 4) {
      __m128i s = _mm_loadu_si128((const __m128i *)(row + 2 * j));
      __m128i e32 = _mm_cvtepu16_epi32(_mm_shuffle_epi8(s, EVEN));
      __m128i o32 = _mm_cvtepu16_epi32(_mm_shuffle_epi8(s, ODD));
      e32 = _mm_slli_epi32(_mm_sub_epi32(e32, M512), 1);
      o32 = _mm_slli_epi32(_mm_sub_epi32(o32, M512), 1);
      _mm_storeu_si128((__m128i *)(e + j), e32);
      _mm_storeu_si128((__m128i *)(o + j), o32);
    }
    // boundary pads (mirrors)
    e[-1] = e[0];
    e[-2] = e[0];
    e[nE] = e[nE - 1];
    e[nE + 1] = e[nE - 2];

    // pass 2: O[j] = odd[j] - update(even[j-1..j+2])
    for (int j0 = 0; j0 + 4 <= nE; j0 += 4) {
      __m128i d_lo = _mm_loadu_si128((const __m128i *)(e + j0 - 2));
      __m128i d_hi = _mm_loadu_si128((const __m128i *)(e + j0 + 2));
      __m128i S1 = _mm_alignr_epi8(d_hi, d_lo, 4);
      __m128i S2 = _mm_alignr_epi8(d_hi, d_lo, 8);
      __m128i S3 = _mm_alignr_epi8(d_hi, d_lo, 12);
      __m128i odd = _mm_loadu_si128((const __m128i *)(o + j0));
      __m128i O4 = _mm_sub_epi32(odd, dd9h_update4(S1, S2, S3, d_hi));
      _mm_storeu_si128((__m128i *)(Op + j0), O4);
    }
    Op[-1] = Op[0];

    // pass 3: E[j] = even[j] + predict(O[j-1],O[j]); interleave; store int16
    for (int j0 = 0; j0 + 4 <= nE; j0 += 4) {
      __m128i e4 = _mm_loadu_si128((const __m128i *)(e + j0));
      __m128i o_prev = _mm_loadu_si128((const __m128i *)(Op + j0 - 1));
      __m128i o_cur = _mm_loadu_si128((const __m128i *)(Op + j0));
      __m128i E4 = _mm_add_epi32(e4, dd9h_predict2(o_prev, o_cur));
      __m128i OUT_lo = _mm_unpacklo_epi32(E4, o_cur);
      __m128i OUT_hi = _mm_unpackhi_epi32(E4, o_cur);
      __m128i out16 = _mm_unpacklo_epi64(_mm_shuffle_epi8(OUT_lo, LO16),
                                         _mm_shuffle_epi8(OUT_hi, LO16));
      _mm_storeu_si128((__m128i *)(orow + 2 * j0), out16);
    }
  }

  // overlap mirror-fill (identical to the scalar reference)
  for (int y = 0; y < iheight; y++) {
    for (int x = iwidth; x < owidth; x += 2) {
      odata[y * ostride + x + 0] = odata[y * ostride + (2 * iwidth - x - 2) + 0];
      odata[y * ostride + x + 1] = odata[y * ostride + (2 * iwidth - x - 2) + 1];
    }
  }
  for (int y = iheight; y < oheight; y++) {
    memcpy(&odata[y * ostride], &odata[(2 * iheight - y - 1) * ostride], owidth);
  }
}

// AVX2 forward horizontal transform for the strided LL levels. The scalar
// kernel has a serial lifting recurrence along each row; keeping the even,
// odd, and predicted samples in int32 scratch lets eight independent pairs
// advance together while preserving the scalar boundary mirrors.
static inline __m256i dd9h_update8(__m256i a, __m256i b, __m256i c, __m256i d) {
  __m256i v = _mm256_sub_epi32(_mm256_setzero_si256(), a);
  v = _mm256_add_epi32(v, _mm256_add_epi32(b, _mm256_slli_epi32(b, 3)));
  v = _mm256_add_epi32(v, _mm256_add_epi32(c, _mm256_slli_epi32(c, 3)));
  v = _mm256_sub_epi32(v, d);
  return _mm256_srai_epi32(_mm256_add_epi32(v, _mm256_set1_epi32(8)), 4);
}

static inline __m256i dd9h_load8_stride4(const int16_t *p) {
  const __m128i M = _mm_setr_epi8(0, 1, 8, 9, -1, -1, -1, -1,
                                  -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i a = _mm_loadu_si128((const __m128i *)(p + 0));
  const __m128i b = _mm_loadu_si128((const __m128i *)(p + 8));
  const __m128i c = _mm_loadu_si128((const __m128i *)(p + 16));
  const __m128i d = _mm_loadu_si128((const __m128i *)(p + 24));
  const __m128i ab = _mm_unpacklo_epi32(_mm_shuffle_epi8(a, M),
                                        _mm_shuffle_epi8(b, M));
  const __m128i cd = _mm_unpacklo_epi32(_mm_shuffle_epi8(c, M),
                                        _mm_shuffle_epi8(d, M));
  return _mm256_cvtepi16_epi32(_mm_unpacklo_epi64(ab, cd));
}

template<int skip> static void dd9h_transform_inplace_avx2(void *_idata,
                                                            const int istride,
                                                            const int width,
                                                            const int height,
                                                            const int) {
  if (width < 16 * skip || (width % (16 * skip)) != 0) {
    Deslauriers_Dubuc_9_7_transform_H_inplace<skip, int16_t>(
      _idata, istride, width, height, 0);
    return;
  }

  int16_t *idata = (int16_t *)_idata;
  const int nE = width / (2 * skip);
  int32_t *even32 = (int32_t *)_alloca((nE + 4) * sizeof(int32_t));
  int32_t *odd32  = (int32_t *)_alloca((nE + 4) * sizeof(int32_t));
  int32_t *Oscr   = (int32_t *)_alloca((nE + 2) * sizeof(int32_t));
  int32_t *e = even32 + 2;
  int32_t *o = odd32 + 2;
  int32_t *Op = Oscr + 1;

  for (int y = 0; y < height; y += skip) {
    int16_t *row = idata + (size_t)y * istride;
    if (skip == 2) {
      for (int j0 = 0; j0 < nE; j0 += 8) {
        const __m256i ev = dd9h_load8_stride4(row + 4 * j0);
        const __m256i ov = dd9h_load8_stride4(row + 4 * j0 + 2);
        _mm256_storeu_si256((__m256i *)(e + j0), _mm256_slli_epi32(ev, 1));
        _mm256_storeu_si256((__m256i *)(o + j0), _mm256_slli_epi32(ov, 1));
      }
    } else {
      for (int j = 0; j < nE; j++) {
        e[j] = (int32_t)row[(2 * j) * skip] << 1;
        o[j] = (int32_t)row[(2 * j + 1) * skip] << 1;
      }
    }

    e[-1] = e[0];
    e[-2] = e[0];
    e[nE] = e[nE - 1];
    e[nE + 1] = e[nE - 2];

    for (int j0 = 0; j0 < nE; j0 += 8) {
      const __m256i a = _mm256_loadu_si256((const __m256i *)(e + j0 - 1));
      const __m256i b = _mm256_loadu_si256((const __m256i *)(e + j0));
      const __m256i c = _mm256_loadu_si256((const __m256i *)(e + j0 + 1));
      const __m256i d = _mm256_loadu_si256((const __m256i *)(e + j0 + 2));
      const __m256i odd = _mm256_loadu_si256((const __m256i *)(o + j0));
      _mm256_storeu_si256((__m256i *)(Op + j0),
                          _mm256_sub_epi32(odd, dd9h_update8(a, b, c, d)));
    }
    Op[-1] = Op[0];

    for (int j0 = 0; j0 < nE; j0 += 8) {
      const __m256i ev = _mm256_loadu_si256((const __m256i *)(e + j0));
      const __m256i prev = _mm256_loadu_si256((const __m256i *)(Op + j0 - 1));
      const __m256i cur = _mm256_loadu_si256((const __m256i *)(Op + j0));
      const __m256i pred = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_add_epi32(prev, cur), _mm256_set1_epi32(2)), 2);
      _mm256_storeu_si256((__m256i *)(e + j0), _mm256_add_epi32(ev, pred));
    }

    for (int j = 0; j < nE; j++) {
      row[(2 * j) * skip] = (int16_t)e[j];
      row[(2 * j + 1) * skip] = (int16_t)Op[j];
    }
  }
}

void Deslauriers_Dubuc_9_7_transform_H_inplace_avx2_s2(void *idata,
                                                        const int istride,
                                                        const int width,
                                                        const int height,
                                                        const int depth) {
  dd9h_transform_inplace_avx2<2>(idata, istride, width, height, depth);
}

void Deslauriers_Dubuc_9_7_transform_H_inplace_avx2_s4(void *idata,
                                                        const int istride,
                                                        const int width,
                                                        const int height,
                                                        const int depth) {
  dd9h_transform_inplace_avx2<4>(idata, istride, width, height, depth);
}

// ---------------------------------------------------------------------------
// Strided forward vertical kernels for the coarser levels (skip 2 and 4).
// The lifting recurrence is identical to level 0; only the column load/store
// changes because the LL subband samples are interleaved in memory.

// skip 2: 8 columns at memory positions 0,2,...,14 (16 int16 span).
static inline __m256i dd9f_load8_s2(const int16_t *p) {
  __m128i a = _mm_loadu_si128((const __m128i *)(p + 0));
  __m128i b = _mm_loadu_si128((const __m128i *)(p + 8));
  const __m128i EV = _mm_setr_epi8(0,1,4,5,8,9,12,13,-1,-1,-1,-1,-1,-1,-1,-1);
  return _mm256_cvtepi16_epi32(_mm_unpacklo_epi64(_mm_shuffle_epi8(a, EV),
                                                  _mm_shuffle_epi8(b, EV)));
}

static inline void dd9f_store8_s2(int16_t *p, const __m256i v) {
  const __m256i low_words = _mm256_setr_epi8(
    0,1,4,5,8,9,12,13,-1,-1,-1,-1,-1,-1,-1,-1,
    0,1,4,5,8,9,12,13,-1,-1,-1,-1,-1,-1,-1,-1);
  const __m256i s = _mm256_shuffle_epi8(v, low_words);
  const __m128i n = _mm_unpacklo_epi64(_mm256_castsi256_si128(s),
                                       _mm256_extracti128_si256(s, 1));
  __m128i a = _mm_loadu_si128((const __m128i *)(p + 0));
  __m128i b = _mm_loadu_si128((const __m128i *)(p + 8));
  const __m128i OD = _mm_setr_epi8(2,3,6,7,10,11,14,15,-1,-1,-1,-1,-1,-1,-1,-1);
  const __m128i oa = _mm_shuffle_epi8(a, OD);
  const __m128i ob = _mm_shuffle_epi8(b, OD);
  const __m128i nh = _mm_srli_si128(n, 8);
  _mm_storeu_si128((__m128i *)(p + 0), _mm_unpacklo_epi16(n, oa));
  _mm_storeu_si128((__m128i *)(p + 8), _mm_unpacklo_epi16(nh, ob));
}

// skip 4: 4 columns at memory positions 0,4,8,12 (16 int16 span), 128-bit.
static inline __m128i dd9f_load4_s4(const int16_t *p) {
  __m128i a = _mm_loadu_si128((const __m128i *)(p + 0));
  __m128i b = _mm_loadu_si128((const __m128i *)(p + 8));
  const __m128i M = _mm_setr_epi8(0,1,8,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
  __m128i c = _mm_unpacklo_epi64(_mm_shuffle_epi8(a, M), _mm_shuffle_epi8(b, M));
  return _mm_cvtepi16_epi32(_mm_shuffle_epi8(c, _mm_setr_epi8(0,1,2,3,8,9,10,11,
                                                               -1,-1,-1,-1,-1,-1,-1,-1)));
}

static inline void dd9f_store4_s4(int16_t *p, const __m128i v) {
  const __m128i n = _mm_shuffle_epi8(v, _mm_setr_epi8(0,1,4,5,8,9,12,13,
                                                       -1,-1,-1,-1,-1,-1,-1,-1));
  __m128i a = _mm_loadu_si128((const __m128i *)(p + 0));
  __m128i b = _mm_loadu_si128((const __m128i *)(p + 8));
  const __m128i sa = _mm_shuffle_epi8(n, _mm_setr_epi8(0,1,-1,-1,-1,-1,-1,-1,
                                                        2,3,-1,-1,-1,-1,-1,-1));
  const __m128i sb = _mm_shuffle_epi8(n, _mm_setr_epi8(4,5,-1,-1,-1,-1,-1,-1,
                                                        6,7,-1,-1,-1,-1,-1,-1));
  _mm_storeu_si128((__m128i *)(p + 0), _mm_blend_epi16(a, sa, 0x11));
  _mm_storeu_si128((__m128i *)(p + 8), _mm_blend_epi16(b, sb, 0x11));
}

static inline __m128i dd9f_update4(__m128i a, __m128i b, __m128i c, __m128i d) {
  __m128i nb = _mm_add_epi32(b, _mm_slli_epi32(b, 3));
  __m128i nc = _mm_add_epi32(c, _mm_slli_epi32(c, 3));
  __m128i v = _mm_sub_epi32(_mm_setzero_si128(), a);
  v = _mm_add_epi32(v, nb);
  v = _mm_add_epi32(v, nc);
  v = _mm_sub_epi32(v, d);
  return _mm_srai_epi32(_mm_add_epi32(v, _mm_set1_epi32(8)), 4);
}

static inline __m128i dd9f_predict4(__m128i a, __m128i b) {
  return _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(a, b), _mm_set1_epi32(2)), 2);
}

void Deslauriers_Dubuc_9_7_transform_V_inplace_avx2_s2(void *_idata,
                                                       const int istride,
                                                       const int width,
                                                       const int height,
                                                       const int skip) {
  if (skip != 2 || (width & 15) != 0 || height < 16) {
    Deslauriers_Dubuc_9_7_transform_V_inplace<2, int16_t>(
      _idata, istride, width, height, skip);
    return;
  }
  int16_t *idata = (int16_t *)_idata;
  for (int x = 0; x < width; x += 16) {
    int y = 0;
    __m256i Dm2, D, Dp1, Dp2, Dp3, Dp4;
    __m256i Xm1, X, Xp1;

    D   = dd9f_load8_s2(idata + (y + 0) * istride + x);
    Dp1 = dd9f_load8_s2(idata + (y + 2) * istride + x);
    Dp2 = dd9f_load8_s2(idata + (y + 4) * istride + x);
    Dp3 = dd9f_load8_s2(idata + (y + 6) * istride + x);
    Dp4 = dd9f_load8_s2(idata + (y + 8) * istride + x);

    Dm2 = D;
    Xp1 = _mm256_sub_epi32(Dp1, dd9f_update(Dm2, D, Dp2, Dp4));
    Xm1 = Xp1;
    X   = _mm256_add_epi32(D, dd9f_predict(Xm1, Xp1));
    dd9f_store8_s2(idata + (y + 0) * istride + x, X);
    dd9f_store8_s2(idata + (y + 2) * istride + x, Xp1);

    Dm2 = D;
    D   = Dp2;
    Dp1 = Dp3;
    Dp2 = Dp4;
    Dp3 = dd9f_load8_s2(idata + (y + 10) * istride + x);
    Dp4 = dd9f_load8_s2(idata + (y + 12) * istride + x);
    Xm1 = Xp1;
    y += 4;

    for (; y < height - 12; y += 4) {
      Xp1 = _mm256_sub_epi32(Dp1, dd9f_update(Dm2, D, Dp2, Dp4));
      X   = _mm256_add_epi32(D, dd9f_predict(Xm1, Xp1));
      dd9f_store8_s2(idata + (y + 0) * istride + x, X);
      dd9f_store8_s2(idata + (y + 2) * istride + x, Xp1);

      Dm2 = D;
      D   = Dp2;
      Dp1 = Dp3;
      Dp2 = Dp4;
      Dp3 = dd9f_load8_s2(idata + (y + 10) * istride + x);
      Dp4 = dd9f_load8_s2(idata + (y + 12) * istride + x);
      Xm1 = Xp1;
    }

    {
      Xp1 = _mm256_sub_epi32(Dp1, dd9f_update(Dm2, D, Dp2, Dp4));
      X   = _mm256_add_epi32(D, dd9f_predict(Xm1, Xp1));
      dd9f_store8_s2(idata + (y + 0) * istride + x, X);
      dd9f_store8_s2(idata + (y + 2) * istride + x, Xp1);

      Dm2 = D;
      D   = Dp2;
      Dp1 = Dp3;
      Dp2 = Dp4;
      Dp3 = dd9f_load8_s2(idata + (y + 10) * istride + x);
      Dp4 = Dp4;
      Xm1 = Xp1;
    }
    y += 4;

    {
      Xp1 = _mm256_sub_epi32(Dp1, dd9f_update(Dm2, D, Dp2, Dp4));
      X   = _mm256_add_epi32(D, dd9f_predict(Xm1, Xp1));
      dd9f_store8_s2(idata + (y + 0) * istride + x, X);
      dd9f_store8_s2(idata + (y + 2) * istride + x, Xp1);

      Dm2 = D;
      D   = Dp2;
      Dp1 = Dp3;
      Dp2 = D;
      Dp3 = Dp3;
      Dp4 = Dm2;
      Xm1 = Xp1;
    }
    y += 4;

    {
      Xp1 = _mm256_sub_epi32(Dp1, dd9f_update(Dm2, D, Dp2, Dp4));
      X   = _mm256_add_epi32(D, dd9f_predict(Xm1, Xp1));
      dd9f_store8_s2(idata + (y + 0) * istride + x, X);
      dd9f_store8_s2(idata + (y + 2) * istride + x, Xp1);
    }
  }
}

void Deslauriers_Dubuc_9_7_transform_V_inplace_avx2_s4(void *_idata,
                                                       const int istride,
                                                       const int width,
                                                       const int height,
                                                       const int skip) {
  if (skip != 4 || (width & 15) != 0 || height < 32) {
    Deslauriers_Dubuc_9_7_transform_V_inplace<4, int16_t>(
      _idata, istride, width, height, skip);
    return;
  }
  int16_t *idata = (int16_t *)_idata;
  for (int x = 0; x < width; x += 16) {
    int y = 0;
    __m128i Dm2, D, Dp1, Dp2, Dp3, Dp4;
    __m128i Xm1, X, Xp1;

    D   = dd9f_load4_s4(idata + (y + 0) * istride + x);
    Dp1 = dd9f_load4_s4(idata + (y + 4) * istride + x);
    Dp2 = dd9f_load4_s4(idata + (y + 8) * istride + x);
    Dp3 = dd9f_load4_s4(idata + (y + 12) * istride + x);
    Dp4 = dd9f_load4_s4(idata + (y + 16) * istride + x);

    Dm2 = D;
    Xp1 = _mm_sub_epi32(Dp1, dd9f_update4(Dm2, D, Dp2, Dp4));
    Xm1 = Xp1;
    X   = _mm_add_epi32(D, dd9f_predict4(Xm1, Xp1));
    dd9f_store4_s4(idata + (y + 0) * istride + x, X);
    dd9f_store4_s4(idata + (y + 4) * istride + x, Xp1);

    Dm2 = D;
    D   = Dp2;
    Dp1 = Dp3;
    Dp2 = Dp4;
    Dp3 = dd9f_load4_s4(idata + (y + 20) * istride + x);
    Dp4 = dd9f_load4_s4(idata + (y + 24) * istride + x);
    Xm1 = Xp1;
    y += 8;

    for (; y < height - 24; y += 8) {
      Xp1 = _mm_sub_epi32(Dp1, dd9f_update4(Dm2, D, Dp2, Dp4));
      X   = _mm_add_epi32(D, dd9f_predict4(Xm1, Xp1));
      dd9f_store4_s4(idata + (y + 0) * istride + x, X);
      dd9f_store4_s4(idata + (y + 4) * istride + x, Xp1);

      Dm2 = D;
      D   = Dp2;
      Dp1 = Dp3;
      Dp2 = Dp4;
      Dp3 = dd9f_load4_s4(idata + (y + 20) * istride + x);
      Dp4 = dd9f_load4_s4(idata + (y + 24) * istride + x);
      Xm1 = Xp1;
    }

    {
      Xp1 = _mm_sub_epi32(Dp1, dd9f_update4(Dm2, D, Dp2, Dp4));
      X   = _mm_add_epi32(D, dd9f_predict4(Xm1, Xp1));
      dd9f_store4_s4(idata + (y + 0) * istride + x, X);
      dd9f_store4_s4(idata + (y + 4) * istride + x, Xp1);

      Dm2 = D;
      D   = Dp2;
      Dp1 = Dp3;
      Dp2 = Dp4;
      Dp3 = dd9f_load4_s4(idata + (y + 20) * istride + x);
      Dp4 = Dp4;
      Xm1 = Xp1;
    }
    y += 8;

    {
      Xp1 = _mm_sub_epi32(Dp1, dd9f_update4(Dm2, D, Dp2, Dp4));
      X   = _mm_add_epi32(D, dd9f_predict4(Xm1, Xp1));
      dd9f_store4_s4(idata + (y + 0) * istride + x, X);
      dd9f_store4_s4(idata + (y + 4) * istride + x, Xp1);

      Dm2 = D;
      D   = Dp2;
      Dp1 = Dp3;
      Dp2 = D;
      Dp3 = Dp3;
      Dp4 = Dm2;
      Xm1 = Xp1;
    }
    y += 8;

    {
      Xp1 = _mm_sub_epi32(Dp1, dd9f_update4(Dm2, D, Dp2, Dp4));
      X   = _mm_add_epi32(D, dd9f_predict4(Xm1, Xp1));
      dd9f_store4_s4(idata + (y + 0) * istride + x, X);
      dd9f_store4_s4(idata + (y + 4) * istride + x, Xp1);
    }
  }
}
