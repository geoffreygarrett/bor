
#ifndef FREYR_DEVICE_ARRAY_H
#define FREYR_DEVICE_ARRAY_H
#include "manager_def.h"

#include <assert.h>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


std::ostream &operator<<(std::ostream &os, const std::vector<size_t> &vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i < vec.size() - 1) os << ", ";
    }
    return os << "]";
}

#include <initializer_list>
#include <iostream>
#include <vector>

class ndarray_shape {
public:
    using dimension_container = std::vector<size_t>;

    template<typename Container,
             std::enable_if_t<std::is_same_v<Container, dimension_container>
                                      || std::is_same_v<Container, std::initializer_list<size_t>>,
                              int>
             = 0>
    explicit ndarray_shape(Container &&dims) : dimensions(std::forward<Container>(dims)) {
        dimensions.reserve(8);// optional
    }

    size_t operator[](size_t i) const noexcept {
        return dimensions[i];// for performance, no boundary check
    }

    [[nodiscard]] size_t size() const noexcept { return dimensions.size(); }

    auto begin() noexcept { return dimensions.begin(); }
    auto end() noexcept { return dimensions.end(); }

    [[nodiscard]] auto begin() const noexcept { return dimensions.begin(); }
    [[nodiscard]] auto end() const noexcept { return dimensions.end(); }

    friend std::ostream &operator<<(std::ostream &os, const ndarray_shape &shape) {
        os << "[";
        for (size_t i = 0; i < shape.dimensions.size(); ++i) {
            os << shape.dimensions[i];
            if (i < shape.dimensions.size() - 1) os << ", ";
        }
        return os << "]";
    }

private:
    dimension_container dimensions;
};

// For enabling structured bindings
namespace std {
    template<>
    struct tuple_size<ndarray_shape> : std::integral_constant<std::size_t, 2> {};

    template<std::size_t N>
    struct tuple_element<N, ndarray_shape> {
        using type = size_t;
    };
}// namespace std

// Assuming shape is always of length 2 for this example
namespace std {
    template<std::size_t N>
    size_t get(const ndarray_shape &shape) {
        return shape[N];
    }
}// namespace std

namespace freyr::device {

    template<typename T>
    concept is_eigen_vector = requires(T a) {
        { a.size() } -> std::convertible_to<std::ptrdiff_t>;
    };

    namespace storage_layout {
        struct row_major {};
        struct column_major {};
    }// namespace storage_layout


    template<typename T,
             typename Backend       = backend_type::DEFAULT,
             typename StorageLayout = storage_layout::row_major>
    class ndarray {
    public:
        using device_type  = typename device_traits<device_type>::type;
        using buffer_type  = typename buffer_traits<device_type>::buffer_type;
        using value_type   = T;
        using size_type    = size_t;
        using backend_type = Backend;
        using storage_type = std::vector<T>;
        using index_type   = std::vector<size_t>;
        using layout_type  = StorageLayout;

        // Initialize from given storage
        explicit ndarray(const storage_type &data)
            : m_data(data), m_device(manager<device_type>::get_default_device()),
              m_buffer(manager<device_type>::get_default_device(), data.size() * sizeof(T), 0) {
            m_buffer.set_data(m_data);
        }

        // Move constructor
        ndarray(ndarray &&other) noexcept
            : m_device(std::move(other.m_device)), m_buffer(std::move(other.m_buffer)),
              m_data(std::move(other.m_data)), m_shape(std::move(other.m_shape)) {}

        // Move assignment operator
        ndarray &operator=(ndarray &&other) noexcept {
            if (this != &other) {
                m_device = std::move(other.m_device);
                m_buffer = std::move(other.m_buffer);
                m_data   = std::move(other.m_data);
                m_shape  = std::move(other.m_shape);
            }
            return *this;
        }
        // Overloaded version for simplicity
        template<typename... Indices>
        T &at(size_t first, Indices... rest) {
            return at({first, static_cast<size_t>(rest)...});
        }

        template<typename Derived>
        explicit ndarray(const Eigen::MatrixBase<Derived> &eigen_matrix) {
            if constexpr (std::is_same_v<layout_type, storage_layout::row_major>) {
                m_shape = {static_cast<size_t>(eigen_matrix.rows()),
                           static_cast<size_t>(eigen_matrix.cols())};
            } else {
                m_shape = {static_cast<size_t>(eigen_matrix.cols()),
                           static_cast<size_t>(eigen_matrix.rows())};
            }

            m_data.resize(eigen_matrix.size());

            size_t idx = 0;
            for (int i = 0; i < eigen_matrix.rows(); ++i) {
                for (int j = 0; j < eigen_matrix.cols(); ++j) {
                    m_data[idx++] = eigen_matrix(i, j);
                }
            }

            // Initialize buffer and device here, if needed
        }

        ndarray(size_t length, T value)
            : m_device(manager<m_shape>::get_default_device()),
              m_buffer(manager<m_shape>::get_default_device(), length * sizeof(T), 0) {
            m_data = storage_type(length, value);
            m_buffer.set_data(m_data);
        }

        T &operator[](size_t index) { return m_data[index]; }

        // Move array data to device
        void to_device() { m_buffer.copy_from_host(m_data); }

        // Move array data to host
        void to_host() { m_data = m_buffer.copy_to_host(m_data.size()); }

        // Get std::vector
        std::vector<T> &data() { return m_data; }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> to_eigen() {
            m_data = m_buffer.copy_to_host(m_data.size());
            return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(m_data.data(),
                                                                                m_data.size());
        }

        T &at(const index_type &indices) {
            size_t index = calculate_index(indices);
            return m_data[index];
        }

        [[nodiscard]] index_type from_linear_index(size_t linear_index) const {
            index_type indices(m_shape.size(), 0);
            if (std::is_same_v<layout_type, storage_layout::row_major>) {
                size_t factor = 1;
                for (const auto &dim: m_shape) { factor *= dim; }

                for (size_t i = 0; i < m_shape.size(); ++i) {
                    factor /= m_shape[i];
                    indices[i] = linear_index / factor;
                    linear_index %= factor;
                }
            } else {
                size_t factor = 1;
                for (size_t i = m_shape.size(); i-- > 0;) {
                    indices[i] = linear_index / factor;
                    linear_index %= factor;
                    factor *= m_shape[i];
                }
            }
            return indices;
        }

        // Function for iterating through each index in the ndarray
        void for_each_index(std::function<void(const index_type &)> func) const {
            index_type indices(m_shape.size(), 0);
            for_each_index_impl(func, indices, m_shape.size() - 1);
        }

        std::vector<size_t> shape() const { return m_shape; }

        size_t size() const { return m_data.size(); }

        void set_dimensions(const std::vector<size_t> &dims) { m_shape = dims; }

    private:
        device_type         m_device;
        buffer_type         m_buffer;
        storage_type        m_data;
        std::vector<size_t> m_shape;


        size_t calculate_index(const index_type &indices) const {
            if (indices.size() != m_shape.size()) {
                throw std::invalid_argument("Dimension mismatch.");
            }

            size_t index  = 0;
            size_t factor = 1;
            if (std::is_same_v<layout_type, storage_layout::row_major>) {
                for (size_t i = 0; i < indices.size(); ++i) {
                    index += factor * indices[i];
                    factor *= m_shape[i];
                }
            } else {
                for (size_t i = indices.size(); i-- > 0;) {
                    index += factor * indices[i];
                    factor *= m_shape[i];
                }
            }
            return index;
        }

        void for_each_index_impl(std::function<void(const index_type &)> func,
                                 index_type                             &indices,
                                 size_t                                  dim) const {
            if (dim == 0) {
                for (size_t i = 0; i < m_shape[dim]; ++i) {
                    indices[dim] = i;
                    func(indices);
                }
            } else {
                for (size_t i = 0; i < m_shape[dim]; ++i) {
                    indices[dim] = i;
                    for_each_index_impl(func, indices, dim - 1);
                }
            }
        }
    };

    template<typename T, typename Derived>
    ndarray<T> make_ndarray(const Eigen::MatrixBase<Derived> &data) {
        return ndarray<T>(data);
    }

    template<typename T>
    struct mgrid_args {
        T                  start;
        T                  end;
        std::optional<int> steps;
    };

    template<typename T>
    using matrix_type = std::vector<std::vector<T>>;


    using float32  = float;
    using float64  = double;
    using int32    = int;
    using int64    = long;
    using int128   = long long;
    using float16  = short;
    using float80  = long double;
    using float128 = long double;// TODO: must check architectures and compilers.

    using scalar_types
            = std::variant<float32, float64, int32, int64, int128, float16, float80, float128>;


}// namespace freyr::device

#endif//FREYR_DEVICE_ARRAY_H