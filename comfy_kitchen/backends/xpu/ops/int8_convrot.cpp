// SPDX-License-Identifier: Apache-2.0
#include "../launchers.h"
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace comfy_kitchen::xpu {
template <typename Output> class DequantizeConvRot256;

template <typename Output>
static void launch(sycl::queue& queue, const int8_t* input, const float* scales,
                   Output* output, int64_t groups, int64_t groups_per_row, bool scalar) {
    // A subgroup owns one normalized H4 x H4 x H4 x H4 block. Keep all
    // dequantized values and Hadamard stages in registers, then narrow once.
    constexpr int subgroup_size = 16, values_per_lane = 16, workgroup_size = 256;
    if (!groups) return;
    const size_t global = (groups + 15) / 16 * workgroup_size;
    queue.parallel_for<DequantizeConvRot256<Output>>(
        sycl::nd_range<1>(global, workgroup_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            const auto subgroup = item.get_sub_group();
            const int lane = subgroup.get_local_linear_id();
            const int64_t group = item.get_global_linear_id() / subgroup_size;
            if (group >= groups) return;
            const float scale = scales[scalar ? 0 : group / groups_per_row];
            // Swapping the two H16 factors preserves H256 and lets adjacent
            // lanes access adjacent elements on both load and store.
            float values[values_per_lane];
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j)
                values[j] = static_cast<float>(input[group * 256 + j * 16 + lane]) * scale;
#pragma unroll
            for (int stride = 1; stride < values_per_lane; stride *= 4) {
#pragma unroll
                for (int base = 0; base < values_per_lane; base += 4 * stride) {
#pragma unroll
                    for (int offset = 0; offset < stride; ++offset) {
                        const float a = values[base + offset], b = values[base + stride + offset];
                        const float c = values[base + 2 * stride + offset], d = values[base + 3 * stride + offset];
                        values[base + offset] = a + b + c - d;
                        values[base + stride + offset] = a + b - c + d;
                        values[base + 2 * stride + offset] = a - b + c + d;
                        values[base + 3 * stride + offset] = -a + b + c + d;
                    }
                }
            }
#pragma unroll
            for (int stride = 1; stride < subgroup_size; stride *= 4) {
                const int base = lane / (4 * stride) * (4 * stride) + lane % stride;
                const int digit = lane / stride % 4;
#pragma unroll
                for (int j = 0; j < values_per_lane; ++j) {
                    const float a = sycl::select_from_group(subgroup, values[j], base);
                    const float b = sycl::select_from_group(subgroup, values[j], base + stride);
                    const float c = sycl::select_from_group(subgroup, values[j], base + 2 * stride);
                    const float d = sycl::select_from_group(subgroup, values[j], base + 3 * stride);
                    if (digit == 0) values[j] = a + b + c - d;
                    else if (digit == 1) values[j] = a + b - c + d;
                    else if (digit == 2) values[j] = a - b + c + d;
                    else values[j] = -a + b + c + d;
                }
            }
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j)
                output[group * 256 + j * 16 + lane] = static_cast<Output>(values[j] * 0.0625f);
        });
}

void dequantize_convrot(sycl::queue& queue, const int8_t* q, const float* scale,
                       void* output, int64_t rows, int64_t width, bool scalar, int dtype) {
    const int64_t groups = rows * (width / 256);
    if (dtype == 0) launch(queue, q, scale, static_cast<float*>(output), groups, width / 256, scalar);
    else if (dtype == 1) launch(queue, q, scale, static_cast<sycl::half*>(output), groups, width / 256, scalar);
    else launch(queue, q, scale, static_cast<sycl::ext::oneapi::bfloat16*>(output), groups, width / 256, scalar);
}
}
