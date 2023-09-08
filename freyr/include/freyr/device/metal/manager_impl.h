#ifndef FREYR_DEVICE_METAL_MANAGER_IMPL_H
#define FREYR_DEVICE_METAL_MANAGER_IMPL_H

#include "../_manager.h"
#include "core.h"

#include <iostream>
#include <vector>


namespace freyr::device {

    using namespace detail;
    using metal_device_type = device_traits<backend_types::METAL>::type;

    // File: metal/manager_impl.h
    // This would remain largely as you've written it, with the additional namespace of freyr::detail for consistency.


    template<>
    void manager<metal_device_type>::initialize() {
        // Nothing to do here
    }

    template<>
    bool manager<metal_device_type>::is_device_available() {
        return MTLCreateSystemDefaultDevice() != nullptr;
    }

    template<>
    metal_device_type manager<metal_device_type>::get_default_device() {
        // Return default Metal device
        metal_device_type device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device." << std::endl;
            return nullptr;
        }
        return MTLCreateSystemDefaultDevice();
    }

    template<>
    std::vector<metal_device_type> manager<metal_device_type>::get_all_devices() {
        // Return all available Metal devices
        // Convert NSArray to std::vector
        std::vector<metal_device_type> devices;
        NS::Array                *ns_array = MTLCopyAllDevices();
        for (unsigned long i = 0; i < ns_array->count(); i++) {
            devices.push_back((metal_device_type) ns_array->object(i));
        }
    }

}// namespace freyr::device

#endif// FREYR_DEVICE_METAL_MANAGER_IMPL_H