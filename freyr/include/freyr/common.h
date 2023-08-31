
#ifndef FREYR_COMMON_H
#define FREYR_COMMON_H

#ifdef __CUDA_ARCH__
#    define CROSS_PLATFORM __host__ __device__
#else
#    define CROSS_PLATFORM
#endif


#include <Eigen/Dense>


namespace freyr {

    template<typename T = float>
    using vector3_type = Eigen::Vector3<T>;

}// namespace freyr

#endif// FREYR_COMMON_H