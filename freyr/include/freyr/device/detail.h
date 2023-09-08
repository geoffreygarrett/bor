// File: detail.h
#ifndef FREYR_DETAIL_H
#define FREYR_DETAIL_H

#include <type_traits>

#ifdef __APPLE__
#    include "metal/core.h"
#elif __CUDA_ARCH__
#    include <cuda.h>
#endif

namespace freyr::detail {

    enum class backend_types { METAL, CUDA, STL, TBB, DEFAULT };

    template<backend_types>
    struct device_traits {
        using type = void *;// Default type
    };


//    template<typename Device>
//    struct device_buffer_traits;
//
//    // Specialization for METAL_DEVICE
//    template<>
//    struct buffer_traits<metal_device> {
//        using type = metal_buffer;
//    };

    template<typename T>
    struct is_device : std::false_type {};

    template<typename T>
    inline constexpr bool is_device_v = is_device<T>::value;

    template<typename T>
    struct is_device_ptr : std::false_type {};

}// namespace freyr::detail

#endif// FREYR_DETAIL_H