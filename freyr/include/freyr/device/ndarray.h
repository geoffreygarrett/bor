// Header file (ndarray.h)
#ifndef FREYR_DEVICE_ARRAY_H
#define FREYR_DEVICE_ARRAY_H

#include "buffer.h"
#include "detail.h"
#include "device_traits.h"

#include <functional>
#include <vector>

#include <Eigen/Dense>

namespace freyr::device {

    namespace storage_policy {
        struct row_major {};
        struct column_major {};
        struct DEFAULT {};
    }// namespace storage_policy

    namespace backend_policy {
        struct metal {};
        struct cuda {};
        struct stl {};
        struct tbb {};
        struct DEFAULT {};
    }// namespace backend_policy

    template<typename T,
             typename Backend = backend_policy::DEFAULT,
             typename Storage = storage_policy::DEFAULT>
    class ndarray {
    public:
        using backend_type = Backend;
        using device_type  = typename detail::device_traits<backend_type>::type;
        using storage_type = std::vector<T>;
        using buffer_type  = typename detail::buffer_traits<device_type>::type;
        using index_type   = std::vector<size_t>;
        using layout_type  = Storage;
        using shape_type   = std::vector<size_t>;

        static default_backend() {
#ifdef __APPLE__
            return BackendType::Metal;
#elif defined(__CUDACC__)
            return BackendType::Cuda;
#else
            return BackendType::STL;
#endif
        }

        explicit ndarray(const storage_type &data);
        ndarray(ndarray &&other) noexcept;
        ndarray &operator=(ndarray &&other) noexcept;

        template<typename... Indices>
        T &at(size_t first, Indices... rest);

        template<typename Derived>
        explicit ndarray(const Eigen::MatrixBase<Derived> &eigen_matrix);

        void linspace(T start, T end, size_t num_elements, bool endpoint = true);
        ndarray(size_t length, T value);
        T &operator[](size_t index);

        void                                             to_device();
        void                                             to_host();
        std::vector<T>                                  &data();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> to_eigen();

        T                       &at(const index_type &indices);
        [[nodiscard]] index_type from_linear_index(size_t linear_index) const;
        void for_each_index(std::function<void(const index_type &)> func) const;
        [[nodiscard]] std::vector<size_t> shape() const;
        [[nodiscard]] size_t              size() const;

    private:
        device_type  m_device;
        buffer_type  m_buffer;
        storage_type m_data;
        shape_type   m_shape;

        size_t calculate_index(const index_type &indices) const;
        void   for_each_index_impl(std::function<void(const index_type &)> func,
                                   index_type                             &indices,
                                   size_t                                  dim) const;
    };
}// namespace freyr::device

#endif// FREYR_DEVICE_ARRAY_H
