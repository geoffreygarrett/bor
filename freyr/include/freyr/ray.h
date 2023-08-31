#ifndef FREYR_RAY_H
#define FREYR_RAY_H

// https://chat.openai.com/c/e48577cd-6a58-48cb-984f-bb52731d561b
// https://chat.openai.com/c/1f9db774-684a-4b05-9842-a469bb802e14
// https://chat.openai.com/c/7c911409-cc66-42ec-885e-7eb10b81f482

namespace freyr {

#include "Eigen/Core"

#include <algorithm>// For std::transform
#include <array>
#include <vector>


    // Transformation trait to be specialized
    template<typename Policy>
    struct transform_trait;

    //    // Specialization of transformation trait for default ray_policy
    //    template<typename VectorType, typename ScalarType>
    //    struct transform_trait<ray_policy<VectorType, ScalarType>> {
    //        static VectorType rotate_vector(const VectorType &vec, const RotationMatrix &rot_mat) {
    //            VectorType result;
    //            for (size_t i = 0; i < 3; ++i) {
    //                result[i] = std::inner_product(rot_mat[i].begin(), rot_mat[i].end(), vec.begin(), 0.0);
    //            }
    //            return result;
    //        }
    //
    //
    //        // Implement transformation logic for STL vectors
    //        static void apply(std::vector<VectorType> &origins,
    //                          std::vector<VectorType> &directions,
    //                          const void *) {
    //
    //            // For simplicity, let's do a translation
    //            std::transform(origins.begin(), origins.end(), origins.begin(),
    //                           [](const VectorType &vec) {
    //                               VectorType result;
    //                               std::transform(vec.begin(), vec.end(), global_translation.begin(), result.begin(), std::plus<>());
    //                               return result;
    //                           });
    //
    //            // Rotate directions
    //            std::transform(directions.begin(), directions.end(), directions.begin(),
    //                           [&](const VectorType &dir) { return rotate_vector(dir, rot_mat); });
    //        }
    //
    //        // Implement transformation logic for STL arrays
    //        static void apply(std::array<VectorType, 4> &origins,
    //                          std::array<VectorType, 4> &directions,
    //                          const void *) {
    //
    //            // For simplicity, let's do a translation
    //            std::transform(origins.begin(), origins.end(), origins.begin(),
    //                           [](const VectorType &vec) {
    //                               VectorType result;
    //                               std::transform(vec.begin(), vec.end(), global_translation.begin(), result.begin(), std::plus<>());
    //                               return result;
    //                           });
    //
    //            // Rotate directions
    //            std::transform(directions.begin(), directions.end(), directions.begin(),
    //                           [&](const VectorType &dir) { return rotate_vector(dir, rot_mat); });
    //        }
    //    };


    // Default ray_policy
    template<typename VectorType, typename ScalarType>
    struct ray_policy {
        using vector_type           = VectorType;
        using scalar_type           = ScalarType;
        using transform_matrix_type = void;// Default to void

        template<int N>
        struct soa_types {
            static_assert(N == -1 || N >= 1, "Compile size must be -1 (dynamic) or >= 1 (fixed).");

            // Define data_type based on N
            using data_type = std::
                    conditional_t<N == -1, std::vector<vector_type>, std::array<vector_type, N>>;
        };
    };


    // Specialization for Eigen
    template<typename U>
    struct ray_policy<Eigen::Matrix<U, 3, 1>, U> {
        using vector_type           = Eigen::Matrix<U, 3, 1>;
        using scalar_type           = U;
        using transform_matrix_type = Eigen::Matrix<U, 3, 4>;

        template<int N>
        struct soa_types {
            static_assert(N == -1 || N >= 1, "Compile size must be -1 (dynamic) or >= 1 (fixed).");

            // Specialized data_type for Eigen
            using data_type = std::conditional_t<N == -1,
                                                 Eigen::Matrix<U, 3, Eigen::Dynamic>,
                                                 Eigen::Matrix<U, 3, N>>;
        };
    };


    // Specialization of transformation trait for Eigen
    template<typename U>
    struct transform_trait<ray_policy<Eigen::Matrix<U, 3, 1>, U>> {
        static void apply(Eigen::Matrix<U, 3, Eigen::Dynamic> &origins,
                          Eigen::Matrix<U, 3, Eigen::Dynamic> &directions,
                          const Eigen::Matrix<U, 4, 4>        &matrix) {
            Eigen::Matrix<U, 3, 3> rotation_matrix    = matrix.block(0, 0, 3, 3);
            Eigen::Matrix<U, 3, 1> translation_vector = matrix.block(0, 3, 3, 1);
            origins += translation_vector.replicate(1, origins.cols());
            origins    = rotation_matrix * origins;
            directions = rotation_matrix * directions;
        }
    };

    // ray class template
    template<typename Policy = ray_policy<Eigen::Vector3f, float>>
    class ray {
    public:
        using vector_type = typename Policy::vector_type;
        using scalar_type = typename Policy::scalar_type;

        ray() = default;

        ray(const vector_type &origin, const vector_type &direction)
            : m_origin(origin), m_direction(direction) {}

        vector_type get_origin() const { return m_origin; }

        ray &set_origin(const vector_type &origin) {
            m_origin = origin;
            return *this;
        }

        vector_type get_direction() const { return m_direction; }

        ray &set_direction(const vector_type &direction) {
            m_direction = direction;
            return *this;
        }

        vector_type get_point(scalar_type t) const { return m_origin + t * m_direction; }

    private:
        vector_type m_origin;
        vector_type m_direction;
    };

    // raySoA class template
    template<typename Policy = ray_policy<Eigen::Vector3f, float>, int N = -1>
    class ray_soa {
    public:
        using vector_type           = typename Policy::vector_type;
        using scalar_type           = typename Policy::scalar_type;
        using transform_matrix_type = typename Policy::transform_matrix_type;
        using data_type             = typename Policy::template soa_types<N>::data_type;
        using ray_type              = ray<Policy>;

        ray_soa() = default;

        ray_soa(const data_type &origins, const data_type &directions)
            : origins(origins), directions(directions) {}

        void transform(const transform_matrix_type &matrix) {
            transform_trait<Policy>::apply(origins, directions, matrix);
        }

        data_type get_points(scalar_type t) const { return origins + t * directions; }

        ray_type get_ray(size_t i) const { return ray_type(origins[i], directions[i]); }
        [[nodiscard]] size_t size() const { return origins.size(); }
        data_type            get_origins() const { return origins; }
        data_type            get_directions() const { return directions; }

    private:
        data_type origins;
        data_type directions;
    };

}// namespace freyr


#endif//FREYR_RAY_H