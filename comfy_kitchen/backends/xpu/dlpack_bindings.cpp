// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>
#include <stdexcept>
#include "launchers.h"

namespace nb = nanobind;
using Weights = nb::ndarray<const int8_t, nb::ndim<2>, nb::c_contig, nb::device::oneapi>;
using Scale = nb::ndarray<const float, nb::c_contig, nb::device::oneapi>;
using Output = nb::ndarray<nb::ndim<2>, nb::c_contig, nb::device::oneapi>;

static sycl::queue& queue_from_address(std::uintptr_t address) {
    if (!address) throw std::invalid_argument("a current PyTorch SYCL queue is required");
    return *reinterpret_cast<sycl::queue*>(address);
}

static bool supports_device(std::uintptr_t address) {
    const auto device = queue_from_address(address).get_device();
    const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
    return device.has(sycl::aspect::usm_device_allocations) &&
           std::find(sizes.begin(), sizes.end(), 16) != sizes.end();
}

static void dequantize_convrot(Weights q, Scale scale, Output output, std::uintptr_t address) {
    if (q.shape(1) == 0 || q.shape(1) % 256 ||
        output.shape(0) != q.shape(0) || output.shape(1) != q.shape(1))
        throw std::invalid_argument("expected matching [N,K] weights/output, K divisible by 256");
    if (!(scale.size() == 1 || (scale.ndim() == 2 && scale.shape(0) == q.shape(0) && scale.shape(1) == 1)))
        throw std::invalid_argument("expected scalar or [N,1] FP32 scales");
    if (q.device_id() != scale.device_id() || q.device_id() != output.device_id())
        throw std::invalid_argument("all operands must share the XPU device");
    const auto dtype = output.dtype();
    int code = -1;
    if (dtype.lanes == 1) {
        if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float))
            code = dtype.bits == 32 ? 0 : dtype.bits == 16 ? 1 : -1;
        else if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat) && dtype.bits == 16)
            code = 2;
    }
    if (code < 0) throw std::invalid_argument("output must be FP32, FP16 or BF16");
    auto& queue = queue_from_address(address);
    for (const void* pointer : {static_cast<const void*>(q.data()),
                               static_cast<const void*>(scale.data()),
                               static_cast<const void*>(output.data())}) {
        if (q.size() && sycl::get_pointer_type(pointer, queue.get_context()) == sycl::usm::alloc::unknown)
            throw std::invalid_argument("operand is not USM storage in the current queue context");
    }
    // The caller allocates output with Torch and supplies that device's current
    // queue. DLPack only exchanges metadata; no tensor copies or queue waits.
    comfy_kitchen::xpu::dequantize_convrot(queue, q.data(), scale.data(), output.data(),
        q.shape(0), q.shape(1), scale.size() == 1, code);
}

NB_MODULE(_C, m) {
    m.def("supports_device", &supports_device);
    m.def("dequantize_convrot", &dequantize_convrot,
          nb::arg("q").noconvert(), nb::arg("scale").noconvert(),
          nb::arg("output").noconvert(), nb::arg("queue"));
}
