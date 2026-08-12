#ifndef VC2HQENCODE_PLATFORM_VARIANT_HPP
#define VC2HQENCODE_PLATFORM_VARIANT_HPP

#ifdef _MSC_VER
#include <intrin.h>
#include <cstdint>
#include <malloc.h>

static __forceinline int __builtin_clz(unsigned int value) {
  unsigned long bit = 0;
  return _BitScanReverse(&bit, value) ? 31 - static_cast<int>(bit) : 32;
}

#define __builtin_bswap32(value) _byteswap_ulong(value)
#define __builtin_expect(value, expected) (value)
#define VC2_ALIGNED_ALLOC(alignment, size) _aligned_malloc((size), (alignment))
#define VC2_ALIGNED_FREE(pointer) _aligned_free(pointer)
#else
#include <cstdlib>
#include <malloc.h>
#define VC2_ALIGNED_ALLOC(alignment, size) memalign((alignment), (size))
#define VC2_ALIGNED_FREE(pointer) free(pointer)
#endif

static inline void vc2_detect_cpu_features(bool &has_sse4_2, bool &has_avx, bool &has_avx2) {
  has_sse4_2 = false;
  has_avx = false;
  has_avx2 = false;
#ifdef _MSC_VER
  int cpuinfo[4] = {};
  __cpuid(cpuinfo, 0);
  const int maximum_leaf = cpuinfo[0];
  if (maximum_leaf >= 1) {
    __cpuidex(cpuinfo, 1, 0);
    has_sse4_2 = (cpuinfo[2] & (1 << 20)) != 0;
    const bool cpu_avx = (cpuinfo[2] & (1 << 28)) != 0;
    const bool osxsave = (cpuinfo[2] & (1 << 27)) != 0;
    has_avx = cpu_avx && osxsave && ((_xgetbv(0) & 0x6) == 0x6);
    if (has_avx && maximum_leaf >= 7) {
      __cpuidex(cpuinfo, 7, 0);
      has_avx2 = (cpuinfo[1] & (1 << 5)) != 0;
    }
  }
#else
  __builtin_cpu_init();
  has_sse4_2 = __builtin_cpu_supports("sse4.2");
  has_avx = __builtin_cpu_supports("avx");
  has_avx2 = __builtin_cpu_supports("avx2");
#endif
}

#endif
