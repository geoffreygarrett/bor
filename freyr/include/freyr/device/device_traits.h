// File: device_traits.h
#ifndef FREYR_DEVICE_TRAITS_H
#define FREYR_DEVICE_TRAITS_H

#include "detail.h"

#ifdef __APPLE__
namespace freyr::detail {
    template<>
    struct device_traits<backend_types::METAL> {
        using type = MTL::Device *;
    };

    template<>
    struct device_traits<backend_types::DEFAULT> {
        using type = MTL::Device *;
    };

}// namespace freyr::detail
#endif

#ifdef __CUDA_ARCH__
namespace freyr::detail {
    template<>
    struct device_traits<backend_type::CUDA> {
        using type = CUdevice;
    };

    template<>
    struct device_traits<backend_type::DEFAULT> {
        using type = CUdevice;
    };

}// namespace freyr::detail
#endif

#endif// FREYR_DEVICE_TRAITS_H
