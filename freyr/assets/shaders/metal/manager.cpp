
#include "../device_manager.h"
#include <Metal/Metal.h>

namespace freyr::metal {

    template<>
    void device_manager<id<MTLDevice>>::initialize() {
        // Initialization code specific to Metal
    }

    template<>
    bool device_manager<id<MTLDevice>>::is_device_available() {
        // Check if Metal is available
        return true;// Simplified for the example
    }

    template<>
    id<MTLDevice> device_manager<id<MTLDevice>>::get_default_device() {
        // Return default Metal device
        return MTLCreateSystemDefaultDevice();
    }

    template<>
    std::vector<id<MTLDevice>> device_manager<id<MTLDevice>>::get_all_devices() {
        // Return all available Metal devices
        return MTLCopyAllDevices();
    }

}// namespace freyr::metal
