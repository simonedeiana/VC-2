/*****************************************************************************
 * vlc_sse4_2.cpp : Variable Length Decoding functions: SSE4.2 version
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

#include "logger.hpp"
#include "internal.h"
#include "vlc.hpp"
#include "../vc2inversetransform_c/vlc_c.hpp"
#include "stage_profile.hpp"
#include <cstdio>
#include <immintrin.h>

#ifdef DEBUG
#include "debug.hpp"
#endif

#ifdef DEBUG_P_BLOCK
extern int DEBUG_P_JOB;
extern int DEBUG_P_SLICE_Y;
extern int DEBUG_P_SLICE_X;
extern int DEBUG_P_SLICE_W;
extern int DEBUG_P_SLICE_H;
#endif


/* These are used for decoding actual coded coefficients */
inline int decode_sse4_2(uint8_t *idata, int ilength, int32_t *odata, int olength) {
  int icounter = 1;
  int ocounter = 0;
  int32_t V = 0;
  int state;

  _mm_prefetch((char *)idata, _MM_HINT_T0);
  _mm_prefetch((char *)odata, _MM_HINT_T0);

  const __m128i ZERO = _mm_set1_epi8(0);

  const LUTEntry *next;
  if (ilength > 0)
    next = &VLCLUT[idata[0]];
  else
    next = &VLCLUT[0xFF];

  while (icounter < ilength && ocounter < olength) {
    const LUTEntry &entry = *next;
    __m128i E = _mm_load_si128((__m128i *)next);
    next  = &VLCLUT[(((int)entry.state) << 8) + idata[icounter++]];

    V <<= entry.preshift;
    V += entry.val0;

    __m128i A = _mm_unpackhi_epi8(ZERO, E);
    __m128i B = _mm_srai_epi32(_mm_unpacklo_epi16(ZERO, A), 24);
    __m128i C = _mm_srai_epi32(_mm_unpackhi_epi16(ZERO, A), 24);

    B = _mm_insert_epi32(B, (V - 1)*entry.sgn, 0);

    _mm_storeu_si128((__m128i *)&odata[ocounter],     B);
    _mm_storeu_si128((__m128i *)&odata[ocounter + 4], C);

    if (entry.term)
      V = entry.V;
    ocounter += entry.N;
  }

  if (icounter < ilength) {
    return ilength - icounter;
  }

  if (ocounter < olength) {
    const LUTEntry &entry = *next;
    __m128i E = _mm_load_si128((__m128i *)next);
    state = entry.state;

    V <<= entry.preshift;
    V += entry.val0;

    __m128i A = _mm_unpackhi_epi8(ZERO, E);
    __m128i B = _mm_srai_epi32(_mm_unpacklo_epi16(ZERO, A), 24);
    __m128i C = _mm_srai_epi32(_mm_unpackhi_epi16(ZERO, A), 24);

    B = _mm_insert_epi32(B, (V - 1)*entry.sgn, 0);

    _mm_storeu_si128((__m128i *)&odata[ocounter],     B);
    _mm_storeu_si128((__m128i *)&odata[ocounter + 4], C);

    if (entry.term)
      V = entry.V;
    ocounter += entry.N;
  }

  if (ocounter < olength) {
    __m128i TMP = ZERO;
    switch (state) {
    case STATE_DATA:
      V <<= 1;
      V += 1;
    case STATE_FOLLOW:
    case STATE_SIGN:
      TMP = _mm_insert_epi32(TMP, -(V - 1), 0);
    }
    _mm_storeu_si128((__m128i *)&odata[ocounter], TMP);
    ocounter = (ocounter + 4)&0xFFFFFFFC;
  }

  while (ocounter < olength) {
    _mm_store_si128((__m128i *)&odata[ocounter], ZERO);
    ocounter+=4;
  }

  return 0;
}

// One decode step for a single stream, state passed by reference so the
// compiler keeps it in registers. Two independent streams are decoded in a
// single interleaved loop so their serial LUT-latency chains overlap (the VLC
// stage is chain-latency bound: ~76% of the loop is the chain + control).
static inline void vlc_step_ex(uint8_t *__restrict idata, int ilength,
                               int32_t *__restrict odata, int olength,
                               int &icounter, int &ocounter, int32_t &V,
                               const LUTEntry *&next, const __m128i ZERO) {
  (void)ilength;
  (void)olength;
  const LUTEntry &entry = *next;
  __m128i E = _mm_load_si128((__m128i *)next);
  next  = &VLCLUT[(((int)entry.state) << 8) + idata[icounter++]];

  V <<= entry.preshift;
  V += entry.val0;

  __m128i A = _mm_unpackhi_epi8(ZERO, E);
  __m128i B = _mm_srai_epi32(_mm_unpacklo_epi16(ZERO, A), 24);
  __m128i C = _mm_srai_epi32(_mm_unpackhi_epi16(ZERO, A), 24);

  B = _mm_insert_epi32(B, (V - 1)*entry.sgn, 0);

  _mm_storeu_si128((__m128i *)&odata[ocounter],     B);
  _mm_storeu_si128((__m128i *)&odata[ocounter + 4], C);

  if (entry.term)
    V = entry.V;
  ocounter += entry.N;
}

// Tail handling for one stream: truncated-input padding, the final partial
// coefficient, and zero-fill of the remainder (mirrors decode_sse4_2).
static inline int vlc_tail_ex(uint8_t *idata, int ilength, int32_t *odata, int olength,
                              int icounter, int ocounter, int32_t V, const LUTEntry *next,
                              const __m128i ZERO) {
  int state;
  if (icounter < ilength) {
    return ilength - icounter;
  }

  if (ocounter < olength) {
    const LUTEntry &entry = *next;
    __m128i E = _mm_load_si128((__m128i *)next);
    state = entry.state;

    V <<= entry.preshift;
    V += entry.val0;

    __m128i A = _mm_unpackhi_epi8(ZERO, E);
    __m128i B = _mm_srai_epi32(_mm_unpacklo_epi16(ZERO, A), 24);
    __m128i C = _mm_srai_epi32(_mm_unpackhi_epi16(ZERO, A), 24);

    B = _mm_insert_epi32(B, (V - 1)*entry.sgn, 0);

    _mm_storeu_si128((__m128i *)&odata[ocounter],     B);
    _mm_storeu_si128((__m128i *)&odata[ocounter + 4], C);

    if (entry.term)
      V = entry.V;
    ocounter += entry.N;
  }

  if (ocounter < olength) {
    __m128i TMP = ZERO;
    switch (state) {
    case STATE_DATA:
      V <<= 1;
      V += 1;
    case STATE_FOLLOW:
    case STATE_SIGN:
      TMP = _mm_insert_epi32(TMP, -(V - 1), 0);
    }
    _mm_storeu_si128((__m128i *)&odata[ocounter], TMP);
    ocounter = (ocounter + 4)&0xFFFFFFFC;
  }

  while (ocounter < olength) {
    _mm_store_si128((__m128i *)&odata[ocounter], ZERO);
    ocounter+=4;
  }

  return 0;
}

// Decode two independent VLC streams in one interleaved loop so the two
// serial state chains run in parallel. Produces identical output to two calls
// of decode_sse4_2.
inline void decode_sse4_2_x2(uint8_t *idata0, int ilength0, int32_t *odata0, int olength0,
                             uint8_t *idata1, int ilength1, int32_t *odata1, int olength1,
                             int *padding0, int *padding1) {
  const __m128i ZERO = _mm_set1_epi8(0);
  int ic0 = 1, oc0 = 0;
  int32_t V0 = 0;
  const LUTEntry *next0 = (ilength0 > 0) ? &VLCLUT[idata0[0]] : &VLCLUT[0xFF];
  int ic1 = 1, oc1 = 0;
  int32_t V1 = 0;
  const LUTEntry *next1 = (ilength1 > 0) ? &VLCLUT[idata1[0]] : &VLCLUT[0xFF];

  _mm_prefetch((char *)idata0, _MM_HINT_T0);
  _mm_prefetch((char *)odata0, _MM_HINT_T0);
  _mm_prefetch((char *)idata1, _MM_HINT_T0);
  _mm_prefetch((char *)odata1, _MM_HINT_T0);

  while (ic0 < ilength0 && oc0 < olength0 && ic1 < ilength1 && oc1 < olength1) {
    vlc_step_ex(idata0, ilength0, odata0, olength0, ic0, oc0, V0, next0, ZERO);
    vlc_step_ex(idata1, ilength1, odata1, olength1, ic1, oc1, V1, next1, ZERO);
  }
  while (ic0 < ilength0 && oc0 < olength0)
    vlc_step_ex(idata0, ilength0, odata0, olength0, ic0, oc0, V0, next0, ZERO);
  while (ic1 < ilength1 && oc1 < olength1)
    vlc_step_ex(idata1, ilength1, odata1, olength1, ic1, oc1, V1, next1, ZERO);

  *padding0 = vlc_tail_ex(idata0, ilength0, odata0, olength0, ic0, oc0, V0, next0, ZERO);
  *padding1 = vlc_tail_ex(idata1, ilength1, odata1, olength1, ic1, oc1, V1, next1, ZERO);
}

// Decode three independent VLC streams (one slice's Y/C1/C2 components) in one
// interleaved loop for 3-way ILP on the serial state chains. Produces output
// identical to three calls of decode_sse4_2.
inline void decode_sse4_2_x3(uint8_t *idata0, int ilength0, int32_t *odata0, int olength0,
                             uint8_t *idata1, int ilength1, int32_t *odata1, int olength1,
                             uint8_t *idata2, int ilength2, int32_t *odata2, int olength2,
                             int *padding0, int *padding1, int *padding2) {
  const __m128i ZERO = _mm_set1_epi8(0);
  int ic0 = 1, oc0 = 0;
  int32_t V0 = 0;
  const LUTEntry *next0 = (ilength0 > 0) ? &VLCLUT[idata0[0]] : &VLCLUT[0xFF];
  int ic1 = 1, oc1 = 0;
  int32_t V1 = 0;
  const LUTEntry *next1 = (ilength1 > 0) ? &VLCLUT[idata1[0]] : &VLCLUT[0xFF];
  int ic2 = 1, oc2 = 0;
  int32_t V2 = 0;
  const LUTEntry *next2 = (ilength2 > 0) ? &VLCLUT[idata2[0]] : &VLCLUT[0xFF];

  _mm_prefetch((char *)idata0, _MM_HINT_T0);
  _mm_prefetch((char *)odata0, _MM_HINT_T0);
  _mm_prefetch((char *)idata1, _MM_HINT_T0);
  _mm_prefetch((char *)odata1, _MM_HINT_T0);
  _mm_prefetch((char *)idata2, _MM_HINT_T0);
  _mm_prefetch((char *)odata2, _MM_HINT_T0);

  while (ic0 < ilength0 && oc0 < olength0 && ic1 < ilength1 && oc1 < olength1 &&
         ic2 < ilength2 && oc2 < olength2) {
    vlc_step_ex(idata0, ilength0, odata0, olength0, ic0, oc0, V0, next0, ZERO);
    vlc_step_ex(idata1, ilength1, odata1, olength1, ic1, oc1, V1, next1, ZERO);
    vlc_step_ex(idata2, ilength2, odata2, olength2, ic2, oc2, V2, next2, ZERO);
  }
  while (ic0 < ilength0 && oc0 < olength0 && ic1 < ilength1 && oc1 < olength1) {
    vlc_step_ex(idata0, ilength0, odata0, olength0, ic0, oc0, V0, next0, ZERO);
    vlc_step_ex(idata1, ilength1, odata1, olength1, ic1, oc1, V1, next1, ZERO);
  }
  while (ic0 < ilength0 && oc0 < olength0)
    vlc_step_ex(idata0, ilength0, odata0, olength0, ic0, oc0, V0, next0, ZERO);
  while (ic1 < ilength1 && oc1 < olength1)
    vlc_step_ex(idata1, ilength1, odata1, olength1, ic1, oc1, V1, next1, ZERO);
  while (ic2 < ilength2 && oc2 < olength2)
    vlc_step_ex(idata2, ilength2, odata2, olength2, ic2, oc2, V2, next2, ZERO);

  *padding0 = vlc_tail_ex(idata0, ilength0, odata0, olength0, ic0, oc0, V0, next0, ZERO);
  *padding1 = vlc_tail_ex(idata1, ilength1, odata1, olength1, ic1, oc1, V1, next1, ZERO);
  *padding2 = vlc_tail_ex(idata2, ilength2, odata2, olength2, ic2, oc2, V2, next2, ZERO);
}



template<class T> void decode_slices_sse4_2(QuantisationMatrix *matrices,
                                       CodedSlice * const input,
                                       DecodedSlice ** scratch,
                                       int n_slices_x,
                                       int n_slices_y,
                                       VideoPlane **video_data,
                                       int slice_width,
                                       int slice_height,
                                       int depth,
                                       DequantiseFunction *dequant) {
  for (int Y = 0; Y < n_slices_y; Y++) {
    int X = 0;
    // Process two adjacent slices per iteration, interleaving their VLC streams
    // (same component type has similar length, so the overlap is maximised).
    // The three scratch buffers are rotated: slice n's components are decoded
    // and dequantised before slice m reuses the same buffers.
    for (; X + 1 < n_slices_x; X += 2) {
      const int n = Y*n_slices_x + X;
      const int m = n + 1;
      int p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0;
      // Each component must be dequantised before its scratch buffer is reused
      // by the next decode, so decode/dequant pairs stay adjacent.
      {
        vc2hq_stage_profile::Scope profile(vc2hq_stage_profile::VLC_DECODE);
        decode_sse4_2_x3((uint8_t *)input[n].data[0], input[n].length[0], scratch[0]->data, scratch[0]->size,
                         (uint8_t *)input[n].data[1], input[n].length[1], scratch[1]->data, scratch[1]->size,
                         (uint8_t *)input[n].data[2], input[n].length[2], scratch[2]->data, scratch[2]->size,
                         &p0, &p1, &p2);
      }
      {
        vc2hq_stage_profile::Scope profile(vc2hq_stage_profile::DEQUANTISE);
        _mm_prefetch((char *)&matrices[input[n].qindex], _MM_HINT_T0);
        _mm_prefetch((char *)&video_data[0]->as<T>()[Y*slice_height*video_data[0]->stride + X*slice_width], _MM_HINT_T0);
        dequant[0](&matrices[input[n].qindex], scratch[0]->data,
                   &video_data[0]->as<T>()[Y*slice_height*video_data[0]->stride + X*slice_width], video_data[0]->stride,
                   slice_width, slice_height, depth);
        _mm_prefetch((char *)&video_data[1]->as<T>()[Y*slice_height*video_data[1]->stride + X*slice_width], _MM_HINT_T0);
        dequant[1](&matrices[input[n].qindex], scratch[1]->data,
                   &video_data[1]->as<T>()[Y*slice_height*video_data[1]->stride + X*slice_width/2], video_data[1]->stride,
                   slice_width/2, slice_height, depth);
        _mm_prefetch((char *)&video_data[2]->as<T>()[Y*slice_height*video_data[2]->stride + X*slice_width], _MM_HINT_T0);
        dequant[2](&matrices[input[n].qindex], scratch[2]->data,
                   &video_data[2]->as<T>()[Y*slice_height*video_data[2]->stride + X*slice_width/2], video_data[2]->stride,
                   slice_width/2, slice_height, depth);
      }
      {
        vc2hq_stage_profile::Scope profile(vc2hq_stage_profile::VLC_DECODE);
        decode_sse4_2_x3((uint8_t *)input[m].data[0], input[m].length[0], scratch[0]->data, scratch[0]->size,
                         (uint8_t *)input[m].data[1], input[m].length[1], scratch[1]->data, scratch[1]->size,
                         (uint8_t *)input[m].data[2], input[m].length[2], scratch[2]->data, scratch[2]->size,
                         &p3, &p4, &p5);
      }
      {
        vc2hq_stage_profile::Scope profile(vc2hq_stage_profile::DEQUANTISE);
        _mm_prefetch((char *)&matrices[input[m].qindex], _MM_HINT_T0);
        _mm_prefetch((char *)&video_data[0]->as<T>()[Y*slice_height*video_data[0]->stride + (X+1)*slice_width], _MM_HINT_T0);
        dequant[0](&matrices[input[m].qindex], scratch[0]->data,
                   &video_data[0]->as<T>()[Y*slice_height*video_data[0]->stride + (X+1)*slice_width], video_data[0]->stride,
                   slice_width, slice_height, depth);
        _mm_prefetch((char *)&video_data[1]->as<T>()[Y*slice_height*video_data[1]->stride + (X+1)*slice_width], _MM_HINT_T0);
        dequant[1](&matrices[input[m].qindex], scratch[1]->data,
                   &video_data[1]->as<T>()[Y*slice_height*video_data[1]->stride + (X+1)*slice_width/2], video_data[1]->stride,
                   slice_width/2, slice_height, depth);
        _mm_prefetch((char *)&video_data[2]->as<T>()[Y*slice_height*video_data[2]->stride + (X+1)*slice_width], _MM_HINT_T0);
        dequant[2](&matrices[input[m].qindex], scratch[2]->data,
                   &video_data[2]->as<T>()[Y*slice_height*video_data[2]->stride + (X+1)*slice_width/2], video_data[2]->stride,
                   slice_width/2, slice_height, depth);
      }

      input[n].padding = p0 + p1 + p2;
      input[m].padding = p3 + p4 + p5;
    }

    // Odd leftover slice in this row (or odd row length): single-stream path.
    for (; X < n_slices_x; X++) {
      const int n = Y*n_slices_x + X;
      int padding = 0;
      {
        vc2hq_stage_profile::Scope profile(vc2hq_stage_profile::VLC_DECODE);
        padding += decode_sse4_2((uint8_t *)input[n].data[0], input[n].length[0], scratch[0]->data, scratch[0]->size);
        padding += decode_sse4_2((uint8_t *)input[n].data[1], input[n].length[1], scratch[1]->data, scratch[1]->size);
        padding += decode_sse4_2((uint8_t *)input[n].data[2], input[n].length[2], scratch[2]->data, scratch[2]->size);
      }
      {
        vc2hq_stage_profile::Scope profile(vc2hq_stage_profile::DEQUANTISE);
        _mm_prefetch((char *)&matrices[input[n].qindex], _MM_HINT_T0);
        _mm_prefetch((char *)&video_data[0]->as<T>()[Y*slice_height*video_data[0]->stride + X*slice_width], _MM_HINT_T0);
        dequant[0](&matrices[input[n].qindex], scratch[0]->data,
                   &video_data[0]->as<T>()[Y*slice_height*video_data[0]->stride + X*slice_width], video_data[0]->stride,
                   slice_width, slice_height, depth);
        _mm_prefetch((char *)&video_data[1]->as<T>()[Y*slice_height*video_data[1]->stride + X*slice_width], _MM_HINT_T0);
        dequant[1](&matrices[input[n].qindex], scratch[1]->data,
                   &video_data[1]->as<T>()[Y*slice_height*video_data[1]->stride + X*slice_width/2], video_data[1]->stride,
                   slice_width/2, slice_height, depth);
        _mm_prefetch((char *)&video_data[2]->as<T>()[Y*slice_height*video_data[2]->stride + X*slice_width], _MM_HINT_T0);
        dequant[2](&matrices[input[n].qindex], scratch[2]->data,
                   &video_data[2]->as<T>()[Y*slice_height*video_data[2]->stride + X*slice_width/2], video_data[2]->stride,
                   slice_width/2, slice_height, depth);
      }

      input[n].padding = padding;

#ifdef DEBUG_P_BLOCK_DEC
      {
        if (Y == 0 && X == 0) {
          printf("-----------------------------------------------------------------\n");
          printf("  Decoded\n");
          printf("-----------------------------------------------------------------\n");
          T *D = scratch[0]->data;
          for (int y = 0; y < slice_height; y++) {
            for (int x = 0; x < slice_width; x++)
              printf("  %+6d", D[y*slice_width + x]);
            printf("\n");
          }
          printf("-----------------------------------------------------------------\n");
        }

        if (Y == 0 && X == 0) {
          uint32_t D[slice_height*slice_width];
          int n = 0;
          int skip = 1 << depth;
          for (int y = 0; y < slice_height; y += skip) {
            for (int x = 0; x < slice_width; x += skip) {
              D[y*slice_width + x] = scratch[0]->data[n++];
            }
          }

          for (int l = 0; l < depth; l++) {
            for (int y = 0; y < slice_height; y += skip) {
              for (int x = 0; x < slice_width; x += skip) {
                D[(y + 0*skip/2)*slice_width + x + 1*skip/2] = scratch[0]->data[n++];
              }
            }

            for (int y = 0; y < slice_height; y += skip) {
              for (int x = 0; x < slice_width; x += skip) {
                D[(y + 1*skip/2)*slice_width + x + 0*skip/2] = scratch[0]->data[n++];
              }
            }

            for (int y = 0; y < slice_height; y += skip) {
              for (int x = 0; x < slice_width; x += skip) {
                D[(y + 1*skip/2)*slice_width + x + 1*skip/2] = scratch[0]->data[n++];
              }
            }

            skip /= 2;
          }

          printf("-----------------------------------------------------------------\n");
          printf("  Reordered\n");
          printf("-----------------------------------------------------------------\n");
          for (int y = 0; y < slice_height; y++) {
            for (int x = 0; x < slice_width; x++)
              printf("  %+6d", D[y*slice_width + x]);
            printf("\n");
          }
          printf("-----------------------------------------------------------------\n");
        }
      }
 #endif
    }
  }
}


SliceDecoderFunc get_slice_decoder_sse4_2(int sample_size) {
  if (sample_size == 4) {
    return decode_slices_sse4_2<int32_t>;
  } else if (sample_size == 2) {
    return decode_slices_sse4_2<int16_t>;
  }

  return get_slice_decoder_c(sample_size);
}
