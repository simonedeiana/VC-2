/*****************************************************************************
 * haar_invtransform.hpp : Haar filter inverse transform functions: 
 *                         SSE4.2 version
 *****************************************************************************
 * Copyright (C) 2014-2015 BBC
 *
 * Authors: James P. Weaver <james.barrett@bbc.co.uk>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at ipstudio@bbc.co.uk.
 *****************************************************************************/

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif // _WIN32

template<int skip>void Haar_invtransform_V_inplace_sse4_2(void *_idata,
                                                        const int istride,
                                                        const int width,
                                                        const int height);

template<>void Haar_invtransform_V_inplace_sse4_2<1>(void *_idata,
                                                   const int istride,
                                                   const int width,
                                                   const int height){
  int32_t *idata = (int32_t *)_idata;
  const __m128i ONE = _mm_set1_epi32(1);
  const int skip = 1;
  for (int y = 0; y < height; y += 2*skip) {
    for (int x = 0; x < width; x+=4) {
      __m128i D0 = _mm_load_si128((__m128i *)&idata[(y + 0*skip)*istride + x]);
      __m128i D1 = _mm_load_si128((__m128i *)&idata[(y + 1*skip)*istride + x]);

      __m128i X0 = _mm_sub_epi32(D0, _mm_srai_epi32(_mm_add_epi32(D1, ONE), 1));
      __m128i X1 = _mm_add_epi32(D1, X0);

      _mm_store_si128((__m128i *)&idata[(y + 0*skip)*istride + x], X0);
      _mm_store_si128((__m128i *)&idata[(y + 1*skip)*istride + x], X1);
    }
  }
}

// Haar inverse vertical transform, 16-bit samples. Only the columns belonging
// to this level's horizontal subband are updated; the rest are preserved via a
// lane blend (mirrors the scalar Haar_invtransform_V_inplace<skip, int16_t>).
template<int skip> void Haar_invtransform_V_inplace_sse4_2_int16_t(void *_idata,
                                                                  const int istride,
                                                                  const int width,
                                                                  const int height) {
  int16_t *idata = (int16_t *)_idata;
  const __m128i ONE = _mm_set1_epi16(1);
  // Valid 16-bit lanes within each 8-wide vector for this level's subband.
  const int MASK = (skip == 1) ? 0xFF : ((skip == 2) ? 0x55 : ((skip == 4) ? 0x11 : 0x01));
  for (int y = 0; y < height; y += 2*skip) {
    for (int x = 0; x < width; x += 8) {
      __m128i D0 = _mm_load_si128((__m128i *)&idata[(y + 0*skip)*istride + x]);
      __m128i D1 = _mm_load_si128((__m128i *)&idata[(y + 1*skip)*istride + x]);

      __m128i X0 = _mm_sub_epi16(D0, _mm_srai_epi16(_mm_add_epi16(D1, ONE), 1));
      __m128i X1 = _mm_add_epi16(D1, X0);

      _mm_store_si128((__m128i *)&idata[(y + 0*skip)*istride + x], _mm_blend_epi16(D0, X0, MASK));
      _mm_store_si128((__m128i *)&idata[(y + 1*skip)*istride + x], _mm_blend_epi16(D1, X1, MASK));
    }
  }
}

// Haar inverse horizontal transform at the finest level (skip == 1), 16-bit
// samples. De-interleaves even/odd samples, filters, and interleaves back
// (mirrors the scalar Haar_invtransform_H_inplace<1, shift, int16_t>).
template<int shift> void Haar_invtransform_H_inplace_1_sse4_2_int16_t(void *_idata,
                                                                     const int istride,
                                                                     const int width,
                                                                     const int height) {
  int16_t *idata = (int16_t *)_idata;
  const __m128i ONE = _mm_set1_epi16(1);
  const __m128i SHUF = _mm_set_epi8(15,14, 11,10, 7,6, 3,2,
                                    13,12,  9,8, 5,4, 1,0);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 16) {
      __m128i D0 = _mm_load_si128((__m128i *)&idata[y*istride + x + 0]);
      __m128i D8 = _mm_load_si128((__m128i *)&idata[y*istride + x + 8]);

      D0 = _mm_shuffle_epi8(D0, SHUF);
      D8 = _mm_shuffle_epi8(D8, SHUF);

      __m128i E = _mm_unpacklo_epi64(D0, D8);   // even samples
      __m128i O = _mm_unpackhi_epi64(D0, D8);   // odd samples

      __m128i X0 = _mm_sub_epi16(E, _mm_srai_epi16(_mm_add_epi16(O, ONE), 1));
      __m128i X1 = _mm_add_epi16(O, X0);

      if (shift != 0) {
        X0 = _mm_srai_epi16(_mm_add_epi16(X0, ONE), shift);
        X1 = _mm_srai_epi16(_mm_add_epi16(X1, ONE), shift);
      }

      __m128i R0 = _mm_unpacklo_epi16(X0, X1);
      __m128i R8 = _mm_unpackhi_epi16(X0, X1);

      _mm_store_si128((__m128i *)&idata[y*istride + x + 0], R0);
      _mm_store_si128((__m128i *)&idata[y*istride + x + 8], R8);
    }
  }
}

// Haar inverse horizontal transform for coarser levels (skip 2 and 4),
// 16-bit samples. The sample pairs are strided by `skip`; only those lanes are
// updated, the rest are preserved via a lane blend. Each 128-bit vector holds
// 8 samples, so skip == 8 pairs (spanning two vectors) are left to the C path.
template<int skip, int shift> void Haar_invtransform_H_inplace_sse4_2_int16_t(void *_idata,
                                                                             const int istride,
                                                                             const int width,
                                                                             const int height) {
  int16_t *idata = (int16_t *)_idata;
  const __m128i ONE = _mm_set1_epi16(1);
  // Extract the low (D) and high (P) subband lanes of each 8-wide vector.
  const __m128i SHUF_D = (skip == 2) ? _mm_setr_epi8(0,1, 8,9, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80)
                       : _mm_setr_epi8(0,1, 0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80);
  const __m128i SHUF_P = (skip == 2) ? _mm_setr_epi8(4,5, 12,13, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80)
                       : _mm_setr_epi8(8,9, 0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80);
  // Scatter the interleaved results back onto the valid lanes.
  const __m128i SPREAD = (skip == 2) ? _mm_setr_epi8(0,1, 0x80,0x80, 2,3, 0x80,0x80, 4,5, 0x80,0x80, 6,7, 0x80,0x80)
                       : _mm_setr_epi8(0,1, 0x80,0x80,0x80,0x80,0x80,0x80, 2,3, 0x80,0x80,0x80,0x80,0x80,0x80);
  const int BLEND = (skip == 2) ? 0x55 : 0x11;
  for (int y = 0; y < height; y += skip) {
    for (int x = 0; x < width; x += 8) {
      __m128i D16 = _mm_load_si128((__m128i *)&idata[y*istride + x]);

      __m128i D = _mm_shuffle_epi8(D16, SHUF_D);
      __m128i P = _mm_shuffle_epi8(D16, SHUF_P);

      __m128i X = _mm_sub_epi16(D, _mm_srai_epi16(_mm_add_epi16(P, ONE), 1));
      __m128i Y = _mm_add_epi16(P, X);

      if (shift != 0) {
        X = _mm_srai_epi16(_mm_add_epi16(X, ONE), shift);
        Y = _mm_srai_epi16(_mm_add_epi16(Y, ONE), shift);
      }

      __m128i I = _mm_unpacklo_epi16(X, Y);
      __m128i R = _mm_shuffle_epi8(I, SPREAD);

      _mm_store_si128((__m128i *)&idata[y*istride + x], _mm_blend_epi16(D16, R, BLEND));
    }
  }
}

template<int shift, int active_bits> void Haar_invtransform_H_final_1_sse4_2_int32_t(void *_idata,
																			   const int istride,
																			   const char *odata,
																			   const int ostride,
																			   const int iwidth,
																			   const int iheight,
																			   const int ooffset_x,
																			   const int ooffset_y,
																			   const int owidth,
																			   const int oheight) {
  int32_t *idata = (int32_t *)_idata;
  const int skip = 1;
  const __m128i ONE = _mm_set1_epi32(1);
  const __m128i OFFSET = _mm_set1_epi32(1 << (active_bits - 1));

  (void)iwidth;
  (void)iheight;

  for (int y = ooffset_y; y < ooffset_y + oheight; y+=skip) {
    for (int x = ooffset_x; x < ooffset_x + owidth; x += 8) {
      __m128i D0 = _mm_load_si128((__m128i *)&idata[y*istride + x + 0]);
      __m128i D4 = _mm_load_si128((__m128i *)&idata[y*istride + x + 4]);

      __m128i A0 = _mm_unpacklo_epi32(D0, D4);
      __m128i A2 = _mm_unpackhi_epi32(D0, D4);

      __m128i E0 = _mm_unpacklo_epi32(A0, A2);
      __m128i O1 = _mm_unpackhi_epi32(A0, A2);

      __m128i X0 = _mm_sub_epi32(E0, _mm_srai_epi32(_mm_add_epi32(O1, ONE), 1));
      __m128i X1 = _mm_add_epi32(O1, X0);

      __m128i Z0 = _mm_unpacklo_epi32(X0, X1);
      __m128i Z4 = _mm_unpackhi_epi32(X0, X1);

      if (shift != 0) {
        Z0 = _mm_add_epi32(Z0, ONE);
        Z4 = _mm_add_epi32(Z4, ONE);
        Z0 = _mm_srai_epi32(Z0, shift);
        Z4 = _mm_srai_epi32(Z4, shift);
      }

      Z0 = _mm_add_epi32(Z0, OFFSET);
      Z4 = _mm_add_epi32(Z4, OFFSET);

      Z0 = _mm_slli_epi32(Z0, (16 - active_bits));
      Z4 = _mm_slli_epi32(Z4, (16 - active_bits));

      __m128i R = _mm_packus_epi32(Z0, Z4);

      R = _mm_srli_epi16(R, (16 - active_bits));
      _mm_store_si128((__m128i *)&odata[2*((y - ooffset_y)*ostride + x - ooffset_x)], R);
    }
  }
}

template<int shift, int active_bits> void Haar_invtransform_H_final_1_sse4_2_int16_t(void *_idata,
                                                                       const int istride,
                                                                       const char *odata,
                                                                       const int ostride,
                                                                       const int iwidth,
                                                                       const int iheight,
                                                                       const int ooffset_x,
                                                                       const int ooffset_y,
                                                                       const int owidth,
                                                                       const int oheight) {
  int16_t *idata = (int16_t *)_idata;
  const int skip = 1;
  const __m128i ONE = _mm_set1_epi16(1);
  const __m128i OFFSET = _mm_set1_epi16(1 << (active_bits - 1));
  const __m128i SHUF = _mm_set_epi8(15,14, 11,10, 7,6, 3,2,
                                    13,12,   9,8, 5,4, 1,0);
  const __m128i CLIP = _mm_set1_epi16((1 << active_bits) - 1);
  const __m128i ZERO = _mm_set1_epi16(0);

  (void)iwidth;
  (void)iheight;

  for (int y = ooffset_y; y < ooffset_y + oheight; y+=skip) {
    for (int x = ooffset_x; x < ooffset_x + owidth; x += 16) {
      __m128i D0 = _mm_load_si128((__m128i *)&idata[y*istride + x + 0]);
      __m128i D8 = _mm_load_si128((__m128i *)&idata[y*istride + x + 8]);

      D0 = _mm_shuffle_epi8(D0, SHUF);
      D8 = _mm_shuffle_epi8(D8, SHUF);

      __m128i E0 = _mm_unpacklo_epi64(D0, D8);
      __m128i O1 = _mm_unpackhi_epi64(D0, D8);

      __m128i X0 = _mm_sub_epi16(E0, _mm_srai_epi16(_mm_add_epi16(O1, ONE), 1));
      __m128i X1 = _mm_add_epi16(O1, X0);

      __m128i Z0 = _mm_unpacklo_epi16(X0, X1);
      __m128i Z8 = _mm_unpackhi_epi16(X0, X1);

      if (shift != 0) {
        Z0 = _mm_add_epi16(Z0, ONE);
        Z8 = _mm_add_epi16(Z8, ONE);
        Z0 = _mm_srai_epi16(Z0, shift);
        Z8 = _mm_srai_epi16(Z8, shift);
      }

      Z0 = _mm_add_epi16(Z0, OFFSET);
      Z8 = _mm_add_epi16(Z8, OFFSET);

      Z0 = _mm_min_epi16(Z0, CLIP);
      Z8 = _mm_min_epi16(Z8, CLIP);

      Z0 = _mm_max_epi16(Z0, ZERO);
      Z8 = _mm_max_epi16(Z8, ZERO);

      _mm_store_si128((__m128i *)&odata[2*((y - ooffset_y)*ostride + x + 0 - ooffset_x)], Z0);
      _mm_store_si128((__m128i *)&odata[2*((y - ooffset_y)*ostride + x + 8 - ooffset_x)], Z8);
    }
  }
}
