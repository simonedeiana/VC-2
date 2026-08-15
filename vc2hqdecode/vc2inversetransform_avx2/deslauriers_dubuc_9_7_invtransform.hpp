#ifndef VC2_DD9_INVTRANSFORM_AVX2_HPP
#define VC2_DD9_INVTRANSFORM_AVX2_HPP

/****************************************************************************
 * deslauriers_dubuc_9_7_invtransform.hpp : AVX2 DD 9/7 inverse vertical
 * kernel (8 columns per iteration, int16 samples).
 ****************************************************************************/

#include <stdint.h>
#include <immintrin.h>
#include <malloc.h>
#include "../vc2inversetransform_c/deslauriers_dubuc_9_7_invtransform.hpp"

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

void Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_s2(
    void *idata, const int istride, const int width, const int height);
void Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_s4(
    void *idata, const int istride, const int width, const int height);
void Deslauriers_Dubuc_9_7_invtransform_V_inplace_avx2_s8(
    void *idata, const int istride, const int width, const int height);

// Deslauriers-Dubuc 9/7 inverse horizontal final transform, full frame.
//
// The scalar recurrence along x is actually a parallel column stencil: the
// even outputs are D[k] = X[k] - ((X[k-1]+X[k+1]+2)>>2) and the odd outputs
// are X[k] + ((-D[k-3]+9*D[k-1]+9*D[k+1]-D[k+3]+8)>>4). A two-pass scheme
// breaks the apparent serial dependency without the per-operand shuffle
// extraction that made the SSE4.2 attempt slower than scalar:
//   pass 1 de-interleaves 8 samples into even/odd, computes the compact D
//          vector in int32, and stores it to a small per-row scratch buffer;
//   pass 2 reads D back contiguously, forms the four sliding update windows
//          with two 128-bit loads and alignr, and writes the 8 uint16 outputs
//          with a non-temporal store (the output plane is write-only).
// The scratch lives on the stack (thread-safe), and the boundary mirrors
// (X[-1]->X[1]; D[-2],D[-1]->D[0]; D[W]->D[W-2], D[W+2]->D[W-4]) are handled
// by padding the compact D buffer on both ends.
template<int active_bits>
static inline void Deslauriers_Dubuc_9_7_invtransform_H_final_1_avx2_int16_t(
    void *_idata, const int istride, const char *odata, const int ostride,
    const int iwidth, const int iheight,
    const int ooffset_x, const int ooffset_y,
    const int owidth, const int oheight) {
  const int W = iwidth;

  // Requires 8-pixel-aligned output crop and 16-byte-aligned rows so the
  // non-temporal stores can be used. The decoder's slice overlap geometry
  // (32-pixel margins) satisfies this; everything else falls back to scalar.
  if (W < 16 || (W & 7) != 0 || (ooffset_x & 7) != 0 || (owidth & 7) != 0 ||
      ooffset_y + oheight > iheight || (ostride & 7) != 0 ||
      (((uintptr_t)odata & 15) != 0)) {
    Deslauriers_Dubuc_9_7_invtransform_H_final_1<active_bits, int16_t>(
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
  const __m128i TWO = _mm_set1_epi32(2);
  const __m128i EIGHT = _mm_set1_epi32(8);
  const __m128i OFFSET = _mm_set1_epi32(offset);
  const __m128i CLIP = _mm_set1_epi32(clip);
  const __m128i ZERO = _mm_setzero_si128();

  for (int y = ooffset_y; y < ooffset_y + oheight; ++y) {
    const int16_t *row = idata + (size_t)y * istride;
    uint16_t *orow = (uint16_t *)odata + (size_t)(y - ooffset_y) * ostride;

    // ---- pass 1: compact D (even outputs) over the full input width ----
    int prev_odd = row[1]; // X[-1] mirror -> X[1] for D[0]
    int j = 0;
    for (; j + 4 <= nD; j += 4) {
      const __m128i s = _mm_loadu_si128((const __m128i *)(row + 2 * j));
      const __m128i Xeven16 = _mm_shuffle_epi8(s, EVEN);
      const __m128i Xodd16  = _mm_shuffle_epi8(s, ODD);
      const __m128i Xodd_s16 = _mm_insert_epi16(_mm_slli_si128(Xodd16, 2), prev_odd, 0);
      const __m128i Xeven = _mm_cvtepi16_epi32(Xeven16);
      const __m128i Xodd  = _mm_cvtepi16_epi32(Xodd16);
      const __m128i Xodd_s = _mm_cvtepi16_epi32(Xodd_s16);
      const __m128i d = _mm_sub_epi32(Xeven, _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(Xodd_s, Xodd), TWO), 2));
      _mm_storeu_si128((__m128i *)(D + j), d);
      prev_odd = row[2 * j + 7];
    }
    for (; j < nD; j++) { // scalar tail (unreachable when W % 8 == 0)
      const int32_t xm = (2 * j == 0) ? (int32_t)row[1] : (int32_t)row[2 * j - 1];
      D[j] = (int32_t)row[2 * j] - ((xm + (int32_t)row[2 * j + 1] + 2) >> 2);
    }

    // boundary mirrors, as compact-D padding
    D[-1] = D[0];
    D[-2] = D[0];
    D[nD] = D[nD - 1];
    D[nD + 1] = D[nD - 2];

    // ---- pass 2: outputs over the valid crop, with streaming stores ----
    // Process two adjacent 8-pixel groups together. The compact D samples
    // needed by both groups are contiguous, so the 256-bit path avoids the
    // per-group 128-bit window setup while preserving the same recurrence.
    const __m256i ODD256 = _mm256_setr_epi8(
      2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1,
      2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1);
    const __m256i EIGHT256 = _mm256_set1_epi32(8);
    const __m256i OFFSET256 = _mm256_set1_epi32(offset);
    const __m256i CLIP256 = _mm256_set1_epi32(clip);
    int p0 = ooffset_x;
    for (; p0 + 16 <= ooffset_x + owidth; p0 += 16) {
      const int j0 = p0 / 2;
      const __m256i S1 = _mm256_loadu_si256((const __m256i *)(D + j0 - 1));
      const __m256i E = _mm256_loadu_si256((const __m256i *)(D + j0));
      const __m256i S3 = _mm256_loadu_si256((const __m256i *)(D + j0 + 1));
      const __m256i Dhi = _mm256_loadu_si256((const __m256i *)(D + j0 + 2));
      const __m256i nine_E = _mm256_add_epi32(E, _mm256_slli_epi32(E, 3));
      const __m256i nine_S3 = _mm256_add_epi32(S3, _mm256_slli_epi32(S3, 3));
      __m256i UPD = _mm256_sub_epi32(_mm256_setzero_si256(), S1);
      UPD = _mm256_add_epi32(UPD, nine_E);
      UPD = _mm256_add_epi32(UPD, nine_S3);
      UPD = _mm256_sub_epi32(UPD, Dhi);
      UPD = _mm256_srai_epi32(_mm256_add_epi32(UPD, EIGHT256), 4);

      const __m256i xs = _mm256_loadu_si256((const __m256i *)(row + p0));
      const __m256i odd_shuffled = _mm256_shuffle_epi8(xs, ODD256);
      const __m128i odd16 = _mm_unpacklo_epi64(
          _mm256_castsi256_si128(odd_shuffled),
          _mm256_extracti128_si256(odd_shuffled, 1));
      const __m256i O = _mm256_add_epi32(_mm256_cvtepi16_epi32(odd16), UPD);

      __m256i OUT_lo = _mm256_unpacklo_epi32(E, O);
      __m256i OUT_hi = _mm256_unpackhi_epi32(E, O);
      OUT_lo = _mm256_min_epi32(_mm256_max_epi32(
          _mm256_add_epi32(_mm256_srai_epi32(OUT_lo, 1), OFFSET256),
          _mm256_setzero_si256()), CLIP256);
      OUT_hi = _mm256_min_epi32(_mm256_max_epi32(
          _mm256_add_epi32(_mm256_srai_epi32(OUT_hi, 1), OFFSET256),
          _mm256_setzero_si256()), CLIP256);
      const __m256i packed = _mm256_packus_epi32(OUT_lo, OUT_hi);
      _mm_stream_si128((__m128i *)(orow + p0 - ooffset_x),
                       _mm256_castsi256_si128(packed));
      _mm_stream_si128((__m128i *)(orow + p0 - ooffset_x + 8),
                       _mm256_extracti128_si256(packed, 1));
    }
    for (; p0 + 8 <= ooffset_x + owidth; p0 += 8) {
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
