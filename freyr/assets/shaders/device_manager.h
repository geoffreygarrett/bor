#ifndef FREYR_DEVICE_MANAGER_H
#define FREYR_DEVICE_MANAGER_H


#include <string>
#include <vector>


namespace freyr {

    enum class DEVICE_TYPE {
        METAL,
        //        CUDA,
        //        OPENCL,
        //        VULKAN,
        //        CPU,
        //        GPU,
        OTHER
    };

    std::string to_string(DEVICE_TYPE device_type);

#ifdef __APPLE__
#def DEFAULT_DEVICE MTLDevice *
#elif __CUDA_ARCH__
#def DEFAULT_DEVICE CUdevice
#else
#def DEFAULT_DEVICE void *
#endif


    template<typename Device = DEFAULT_DEVICE>
    class device_manager {
    public:
        static void                initialize();
        static bool                is_device_available();
        static Device              get_default_device();
        static std::vector<Device> get_all_devices();
        // ... other utility methods ...
    };

#ifdef __APPLE__
#    include "./metal/device_manager_impl.h"
#elif __CUDA_ARCH__
#    include "./cuda/device_manager_impl.h"
#endif


}// namespace freyr


// Specializations will go in the implementation file

#endif// FREYR_DEVICE_MANAGER_H