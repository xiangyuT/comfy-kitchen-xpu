// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace comfy_kitchen::xpu {
void dequantize_convrot(sycl::queue& queue, const int8_t* q, const float* scale,
                       void* output, int64_t rows, int64_t width,
                       bool scalar_scale, int output_dtype);
}
