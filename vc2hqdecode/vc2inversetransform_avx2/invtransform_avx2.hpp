#ifndef __INVTRANSFORM_AVX2_HPP__
#define __INVTRANSFORM_AVX2_HPP__

#include "common/attributes.h"
#include "invtransform.hpp"

VC2EXPORT InplaceTransform get_invvtransform_avx2(int wavelet_index, int level, int depth, int sample_size);

#endif /* __INVTRANSFORM_AVX2_HPP__ */
