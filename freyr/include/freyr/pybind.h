#ifndef FREYR_PYBIND_H
#define FREYR_PYBIND_H

#include <Eigen/Core>

namespace freyr {

    template<typename Scalar, int Dim>
    using vector_type = Eigen::Vector<Scalar, Dim>;


}

#endif