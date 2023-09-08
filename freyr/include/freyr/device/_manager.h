// File: device_manager.h
#ifndef _FREYR_DEVICE_MANAGER_H
#define _FREYR_DEVICE_MANAGER_H

#include "device_traits.h"

#include <iostream>

namespace freyr::device {

    using namespace detail;

    template<typename T>
    class manager {
    public:
        using device_type = detail::device_traits<backend_types::METAL>::type;
        template<backend_types B>
        static typename device_traits<B>::type get_device();
        static void                            initialize();
        static bool                            is_device_available();
        static device_type                     get_default_device();
        static std::vector<device_type>        get_all_devices();
    };

}// namespace freyr::detail


#endif// _FREYR_DEVICE_MANAGER_H
