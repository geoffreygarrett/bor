#include <algorithm>
#include <cmath>
#include <vector>
#ifdef USE_THRUST
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#endif

#ifdef USE_STL
#include <execution>
#endif

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __CUDA_ARCH__
#define CROSS_PLATFORM __host__ __device__
#else
#define CROSS_PLATFORM
#endif


//struct vector3_type {
//    float               x, y, z;
//    CROSS_PLATFORM void normalize() {
//        float norm = std::sqrt(x * x + y * y + z * z);
//        x /= norm;
//        y /= norm;
//        z /= norm;
//    }
//};

using vector3_type = Eigen::Vector3f;// Alias for 3D vector type


// CRTP base class for compile-time polymorphism
template<typename Derived>
struct ray_generator_base {
    CROSS_PLATFORM [[nodiscard]] vector3_type generate_ray(float u, float v) const {
        return static_cast<const Derived *>(this)->generate_ray_impl(u, v);
    }

    CROSS_PLATFORM [[nodiscard]] vector3_type generate_ray(int idx) const {
        float u = idx % int(static_cast<const Derived *>(this)->width);
        float v = idx / int(static_cast<const Derived *>(this)->width);
        return static_cast<const Derived *>(this)->generate_ray_impl(u, v);
    }
};


// PerspectiveRayGenerator
struct perspective_ray_generator : ray_generator_base<perspective_ray_generator> {
    float width, height, fov, aspect_ratio;

    CROSS_PLATFORM [[nodiscard]] vector3_type generate_ray_impl(float u, float v) const {
        return generate_ray_static(u, v, width, height, fov, aspect_ratio);
    }

    CROSS_PLATFORM [[nodiscard]] static vector3_type generate_ray_static(float u, float v, float width, float height, float fov, float aspect_ratio) {
        float        ndc_x         = (2.0f * u - width) / width;
        float        ndc_y         = (2.0f * v - height) / height;
        float        tan_fov_over2 = std::tan(fov / 2.0f);
        vector3_type ray_dir{ndc_x * tan_fov_over2 * aspect_ratio, ndc_y * tan_fov_over2, -1.0f};
        ray_dir.normalize();
        return ray_dir;
    }
};

// OrthographicRayGenerator
struct orthographic_ray_generator : ray_generator_base<orthographic_ray_generator> {
    float width, height, scale, aspect_ratio;

    CROSS_PLATFORM [[nodiscard]] vector3_type generate_ray_impl(float u, float v) const {
        return generate_ray_static(u, v, width, height, scale, aspect_ratio);
    }

    CROSS_PLATFORM static vector3_type generate_ray_static(float u, float v, float width, float height, float scale, float aspect_ratio) {
        float ndc_x = (2.0f * u - width) / width;
        float ndc_y = (2.0f * v - height) / height;
        return vector3_type{ndc_x * scale * aspect_ratio, ndc_y * scale, 0.0f};
    }
};

// FisheyeRayGenerator
struct fisheye_ray_generator : ray_generator_base<fisheye_ray_generator> {

    float width, height, angle_of_view, aspect_ratio;

    CROSS_PLATFORM [[nodiscard]] vector3_type generate_ray_impl(float u, float v) const {
        return generate_ray_static(u, v, width, height, angle_of_view, aspect_ratio);
    }

    CROSS_PLATFORM static vector3_type generate_ray_static(float u, float v, int width, int height, float angle_of_view, float aspect_ratio) {
        // Convert pixel coordinate to screen coordinate (-1 to 1)
        float s_u = (u / float(width) - 0.5f) * 2.0f;
        float s_v = (v / float(height) - 0.5f) * 2.0f;

        // Adjust coordinates for aspect ratio
        s_u *= aspect_ratio;

        // Calculate the radial distance from the center of the image
        float r = std::sqrt(s_u * s_u + s_v * s_v);

        // If the point is outside the circle described by the angle_of_view, it shouldn't produce a ray
        if (r > std::sin(angle_of_view / 2.0f)) {
            return {0, 0, 0};// Or any "null" representation of a ray
        }

        // Calculate the angle theta from the radial distance
        float theta = r * (angle_of_view / 2.0f);// Map r linearly to angle_of_view

        // Polar to Cartesian coordinates conversion
        float        phi = std::atan2(s_v, s_u);// Angle in polar coordinates
        vector3_type dir(
                std::sin(theta) * std::cos(phi),
                std::sin(theta) * std::sin(phi),
                -std::cos(theta));

        return dir.normalized();// Normalized direction vector for the ray
    }
};

#ifdef USE_THRUST
template<typename RayGenerator, typename DeviceVector>
void generate_rays(DeviceVector &rays, const RayGenerator &generator) {
    thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(generator.width * generator.height),
            rays.begin(),
            generator);
}
#endif// USE_THRUST

#ifdef USE_TBB
#include <tbb/parallel_for.h>

template<typename RayGenerator, typename DeviceVector>
void generate_rays(DeviceVector &rays, const RayGenerator &generator) {
    tbb::parallel_for(0, int(generator.width * generator.height), [&](int i) {
        rays[i] = generator(i);
    });
}

#endif// USE_TBB

#ifdef USE_OPENMP
#include <omp.h>

template<typename RayGenerator, typename DeviceVector>
void        generate_rays(DeviceVector &rays, const RayGenerator &generator) {
#pragma omp parallel for
    for (int i = 0; i < int(generator.width * generator.height); ++i) {
        rays[i] = generator(i);
    }
}

#endif// USE_OPENMP

#ifdef USE_STL
template<typename RayGenerator>
void generate_rays(std::vector<vector3_type> &rays, const RayGenerator &generator) {
    auto start_ptr = rays.data();     // Prefetch the start of the vector for quicker access
    std::for_each(std::execution::par,// Use parallel execution
                  std::begin(rays), std::end(rays),
                  [&generator, start_ptr](vector3_type &ray) {
                      std::ptrdiff_t idx = std::distance(start_ptr, &ray);
                      ray                = generator(idx);
                  });
}
#endif// USE_STL

#ifdef USE_CUDA
// CUDA kernel function
__global__ void generate_rays_kernel(vector3_type *rays, int width, int height, float fov, float aspect_ratio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) {
        return;// Skip excess threads
    }

    float u   = idx % width;
    float v   = idx / width;
    rays[idx] = PerspectiveRayGenerator::generate_ray_static(u, v, width, height, fov, aspect_ratio);
}

template<typename RayGenerator, typename DeviceVector>
void generate_rays(DeviceVector &rays, const RayGenerator &generator) {
    int block_size = 256;
    int num_blocks = (generator.width * generator.height + block_size - 1) / block_size;
    generate_rays_kernel<<<num_blocks, block_size>>>(rays.data().get(), generator.width, generator.height, generator.fov, generator.aspect_ratio);
    cudaDeviceSynchronize();
}
#endif// USE_CUDA


// Camera classes here
