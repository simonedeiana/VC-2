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

void Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_s2(
    void *idata, const int istride, const int width, const int height);
void Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_s4(
    void *idata, const int istride, const int width, const int height);
void Deslauriers_Dubuc_13_7_invtransform_V_inplace_avx2_s8(
    void *idata, const int istride, const int width, const int height);

// Deslauriers-Dubuc 13/7 inverse horizontal final transform, full frame.
// Same two-pass parallel scheme as the DD9/7 version: pass 1 computes the
// compact even D vector (13/7 predict) into a per-row stack scratch; pass 2
// forms the sliding update windows from contiguous D reads and emits 8 uint16
// outputs with a non-temporal store. The 13/7 predict needs X[k-3..k+3], so
// pass 1 carries two odd samples and reads one look-ahead sample per block.
template<int active_bits>
static inline void Deslauriers_Dubuc_13_7_invtransform_H_final_1_avx2_int16_t(
    void *_idata, const int istride, const char *odata, const int ostride,
    const int iwidth, const int iheight,
    const int ooffset_x, const int ooffset_y,
    const int owidth, const int oheight) {
  const int W = iwidth;

  if (W < 16 || (W & 7) != 0 || (ooffset_x & 7) != 0 || (owidth & 7) != 0 ||
      ooffset_y + oheight > iheight || (ostride & 7) != 0 ||
      (((uintptr_t)odata & 15) != 0)) {
    Deslauriers_Dubuc_13_7_invtransform_H_final_1<active_bits, int16_t>(
      _idata, istride, odata, ostride, iwidth, iheight,
      ooffset_x, ooffset_y, owidth, oheight);
    return;
  }

  int16_t *idata = (int16_t *)_idata;
  const int clip = (1 << active_bits) - 1;
  const int offset = 1 << (active_bits - 1);
  const int nD = W / 2;
  int32_t *Dbuf = (int32_t *)_alloca((nD + 4) * sizeof(int32_t));
  int32_t *D = Dbuf + 2; // real D at D[0..nD-1]

  const __m128i EVEN = _mm_setr_epi8(0,1, 4,5, 8,9, 12,13, -1,-1,-1,-1,-1,-1,-1,-1);
  const __m128i ODD  = _mm_setr_epi8(2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1);
  const __m128i SIXTEEN = _mm_set1_epi32(16);
  const __m128i EIGHT = _mm_set1_epi32(8);
  const __m128i OFFSET = _mm_set1_epi32(offset);
  const __m128i CLIP = _mm_set1_epi32(clip);
  const __m128i ZERO = _mm_setzero_si128();

  for (int y = ooffset_y; y < ooffset_y + oheight; ++y) {
    const int16_t *row = idata + (size_t)y * istride;
    uint16_t *orow = (uint16_t *)odata + (size_t)(y - ooffset_y) * ostride;

    // ---- pass 1: compact D (even outputs), 13/7 predict ----
    // X[-3] -> X[3], X[-1] -> X[1] left mirrors for D[0].
    int prev_odd2 = row[3]; // Xodd[-2] = X[-3]
    int prev_odd1 = row[1]; // Xodd[-1] = X[-1]
    int j = 0;
    for (; j + 4 <= nD; j += 4) {
      const __m128i s = _mm_loadu_si128((const __m128i *)(row + 2 * j));
      const __m128i Xeven16 = _mm_shuffle_epi8(s, EVEN);
      const __m128i Xodd16  = _mm_shuffle_epi8(s, ODD);
      const __m128i Xeven = _mm_cvtepi16_epi32(Xeven16);
      const __m128i C = _mm_cvtepi16_epi32(Xodd16); // X[k+1] window: Xodd[j..j+3]

      // look-ahead sample Xodd[4] = X[2j+9], mirrored at the right edge.
      const int x9 = row[(2 * j + 9 < W) ? (2 * j + 9) : (W - 1)];
      const __m128i X9 = _mm_set_epi32(0, 0, 0, x9);
      const __m128i Dv = _mm_alignr_epi8(X9, C, 4); // [x3,x5,x7,x9]

      // A = X[k-3] window [Xodd[j-2..j+1]], B = X[k-1] window [Xodd[j-1..j+2]].
      const __m128i AB = _mm_set_epi32(0, 0, prev_odd1, prev_odd2); // [o2,o1,0,0]
      const __m128i A = _mm_or_si128(AB, _mm_slli_si128(C, 8));     // [o2,o1,x1,x3]
      const __m128i B = _mm_insert_epi32(_mm_slli_si128(C, 4), prev_odd1, 0); // [o1,x1,x3,x5]

      const __m128i nine_B = _mm_add_epi32(B, _mm_slli_epi32(B, 3));
      const __m128i nine_C = _mm_add_epi32(C, _mm_slli_epi32(C, 3));
      __m128i d = _mm_sub_epi32(ZERO, A);
      d = _mm_add_epi32(d, nine_B);
      d = _mm_add_epi32(d, nine_C);
      d = _mm_sub_epi32(d, Dv);
      d = _mm_sub_epi32(Xeven, _mm_srai_epi32(_mm_add_epi32(d, SIXTEEN), 5));
      _mm_storeu_si128((__m128i *)(D + j), d);

      prev_odd2 = (int32_t)row[2 * j + 5]; // Xodd[2]
      prev_odd1 = (int32_t)row[2 * j + 7]; // Xodd[3]
    }
    for (; j < nD; j++) { // scalar tail (unreachable when W % 8 == 0)
      auto Xat = [&](int k) -> int32_t {
        if (k < 0) k = -k;
        if (k >= W) k = 2 * W - k;
        return (int32_t)row[k];
      };
      D[j] = Xat(2 * j) - ((-Xat(2 * j - 3) + 9 * Xat(2 * j - 1) + 9 * Xat(2 * j + 1) - Xat(2 * j + 3) + 16) >> 5);
    }

    // boundary mirrors, as compact-D padding
    D[-1] = D[0];
    D[-2] = D[0];
    D[nD] = D[nD - 1];
    D[nD + 1] = D[nD - 2];

    // ---- pass 2: outputs over the valid crop, with streaming stores ----
    for (int p0 = ooffset_x; p0 + 8 <= ooffset_x + owidth; p0 += 8) {
      const int j0 = p0 / 2;
      const __m128i d_lo = _mm_loadu_si128((const __m128i *)(D + j0 - 2));
      const __m128i d_hi = _mm_loadu_si128((const __m128i *)(D + j0 + 2));
      const __m128i S1 = _mm_alignr_epi8(d_hi, d_lo, 4);
      const __m128i S2 = _mm_alignr_epi8(d_hi, d_lo, 8);
      const __m128i S3 = _mm_alignr_epi8(d_hi, d_lo, 12);
      const __m128i E = S2;
      const __m128i nine_S2 = _mm_add_epi32(S2, _mm_slli_epi32(S2, 3));
      const __m128i nine_S3 = _mm_add_epi32(S3, _mm_slli_epi32(S3, 3));
      __m128i UPD = _mm_sub_epi32(ZERO, S1);
      UPD = _mm_add_epi32(UPD, nine_S2);
      UPD = _mm_add_epi32(UPD, nine_S3);
      UPD = _mm_sub_epi32(UPD, d_hi);
      UPD = _mm_srai_epi32(_mm_add_epi32(UPD, EIGHT), 4);

      const __m128i xs = _mm_loadu_si128((const __m128i *)(row + p0));
      const __m128i xodd16 = _mm_shuffle_epi8(xs, ODD);
      const __m128i Xodd = _mm_cvtepi16_epi32(xodd16);
      const __m128i O = _mm_add_epi32(Xodd, UPD);

      __m128i OUT_lo = _mm_unpacklo_epi32(E, O);
      __m128i OUT_hi = _mm_unpackhi_epi32(E, O);
      OUT_lo = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(_mm_srai_epi32(OUT_lo, 1), OFFSET), ZERO), CLIP);
      OUT_hi = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(_mm_srai_epi32(OUT_hi, 1), OFFSET), ZERO), CLIP);
      _mm_stream_si128((__m128i *)(orow + p0 - ooffset_x), _mm_packus_epi32(OUT_lo, OUT_hi));
    }
  }
}

#endif
