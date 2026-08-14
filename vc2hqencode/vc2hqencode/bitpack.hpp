/*****************************************************************************
 * bitpack.hpp : SIMD bit-packing of VLC codewords (shared by encode+serialise)
 *****************************************************************************
 * Packs a run of (codeword, wordlength) pairs into an MSB-first byte stream.
 * The scalar serialiser is a serial dependency chain on an accumulator and a
 * bit counter; the AVX2 path breaks the chain by packing four codewords per
 * group: a two-step Kogge-Stone prefix sum of the lengths, four 64-bit
 * variable shifts, an OR-reduce, and a merge into a 64-bit left-aligned
 * accumulator with a write-ahead flush (byte-identical to the reference).
 *****************************************************************************/

#ifndef __BITPACK_HPP__
#define __BITPACK_HPP__

#include <x86intrin.h>
#include "platform_variant.hpp"

static inline bool bitpack_has_avx2() {
#ifdef _MSC_VER
  static const bool r = ([]() { int ci[4]; __cpuidex(ci, 7, 0); return (ci[1] & (1 << 5)) != 0; })();
  return r;
#else
  static const bool r = __builtin_cpu_supports("avx2");
  return r;
#endif
}

static inline void bitpack_flush(uint8_t *optr, int &ocounter, uint64_t &accum, int &bits) {
  // Write whole 4-byte chunks MSB-first, with write-ahead (like the scalar
  // reference), so a group costs 2-3 stores rather than one byte-store each.
  if (bits >= 32) {
    *((uint32_t *)&optr[ocounter]) = __builtin_bswap32((uint32_t)(accum >> 32));
    ocounter += 4;
    accum <<= 32;
    bits -= 32;
    if (bits >= 32) {
      *((uint32_t *)&optr[ocounter]) = __builtin_bswap32((uint32_t)(accum >> 32));
      ocounter += 4;
      accum <<= 32;
      bits -= 32;
    }
  }
  if (bits >= 8) {
    const int nbytes = bits >> 3;  // 1..3
    *((uint32_t *)&optr[ocounter]) = __builtin_bswap32((uint32_t)(accum >> 32));
    ocounter += nbytes;
    accum <<= nbytes * 8;
    bits -= nbytes * 8;
  }
}

// Pack `samples` VLC codewords (MSB-first) into dst. Returns bytes written.
static inline int bitpack_pack(uint8_t *dst, const uint16_t *codewords,
                               const uint8_t *wordlengths, int samples) {
  uint64_t accum = 0;
  int bits = 0;
  int n = 0;
  int ocounter = 0;

  if (bitpack_has_avx2()) {
    for (; n + 4 <= samples;) {
      const uint32_t l4 = *(const uint32_t *)(wordlengths + n);
      const int T = (int)(l4 & 0xFF) + (int)((l4 >> 8) & 0xFF) +
                    (int)((l4 >> 16) & 0xFF) + (int)((l4 >> 24) & 0xFF);
      if (bits + T > 64)
        break;  // rare (only near-maximum-length codes); finish the tail scalar

      __m128i l32 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)l4));
      __m128i p1  = _mm_add_epi32(l32, _mm_slli_si128(l32, 4));
      __m128i p2  = _mm_add_epi32(p1, _mm_slli_si128(p1, 8));  // inclusive prefix
      // Codeword 0 must end up at the TOP of the packed word (MSB-first
      // output), so its shift is T - l0; codeword i shifts by T - (l0..li).
      __m128i sh32 = _mm_sub_epi32(_mm_set1_epi32(T), p2);

      __m128i cw32 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(codewords + n)));
      __m256i cw64 = _mm256_cvtepu32_epi64(cw32);
      __m256i s64  = _mm256_cvtepu32_epi64(sh32);
      __m256i sh   = _mm256_sllv_epi64(cw64, s64);
      __m128i orm  = _mm_or_si128(_mm256_castsi256_si128(sh),
                                  _mm256_extracti128_si256(sh, 1));
      const uint64_t packed = (uint64_t)_mm_cvtsi128_si64(orm) |
                              (uint64_t)_mm_extract_epi64(orm, 1);

      accum |= packed << (64 - T - bits);
      bits += T;
      bitpack_flush(dst, ocounter, accum, bits);
      n += 4;
    }
  }

  for (; n < samples; n++) {
    const int l = wordlengths[n];
    accum |= (uint64_t)codewords[n] << (64 - l - bits);
    bits += l;
    bitpack_flush(dst, ocounter, accum, bits);
  }

  if (bits)
    dst[ocounter++] = (uint8_t)((accum >> 56) | (0xFF >> bits));
  return ocounter;
}

#endif /* __BITPACK_HPP__ */
