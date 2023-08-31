#ifndef FREYR_ALGORITHMS_RAY_DIRECTION_STL_H
#define FREYR_ALGORITHMS_RAY_DIRECTION_STL_H

#include <execution>
#include <freyr/common.h>
#include <functional>
#include <vector>

namespace freyr {


    template<typename T>
    void generate_rays(std::vector<vector3_type> &rays, const RayGenerator &generator) {
        auto start_ptr = rays.data();     // Prefetch the start of the vector for quicker access
        std::for_each(std::execution::par,// Use parallel execution
                      std::begin(rays), std::end(rays),
                      [&generator, start_ptr](vector3_type &ray) {
                          std::ptrdiff_t idx = std::distance(start_ptr, &ray);
                          ray                = generator(idx);
                      });
    }

}// namespace freyr

#endif// FREYR_ALGORITHMS_RAY_DIRECTION_STL_H
