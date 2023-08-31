
#ifndef FREYR_SHADER_H
#define FREYR_SHADER_H

#include "freyr/include/freyr/common.h"
#include "include/tbb/blocked_range2d.h"
#include "include/tbb/parallel_for.h"

#include <utility>

#include <Eigen/Core>
#include <freyr/common.h>
#include <freyr/core.h>


// temp
#include <random>

#include <freyr/camera.h>
#include <freyr/light.h>
#include <freyr/scene.h>


#ifdef USE_THRUST
#    include <thrust/device_vector.h>
#    include <thrust/for_each.h>

class ThrustBackendAdapter {
public:
    template<typename ShaderType>
    auto apply(const Scene &scene, size_t width, size_t height) {
        vector3f_type result = utility::zero_initialize<vector3f_type>();

        const auto &entities       = scene.get_entities();
        const auto &lights         = scene.get_lights();
        auto        camera_variant = scene.get_camera();

        size_t                        num_pixels = width * height;
        thrust::device_vector<size_t> d_pixels(num_pixels);// Create an index list

        thrust::for_each(d_pixels.begin(), d_pixels.end(), [=] __device__(size_t idx) {
            size_t        i            = idx / width;
            size_t        j            = idx % width;
            ray_type      ray          = generate_ray_util(camera_variant, i, j);
            vector3f_type local_result = compute_result_util<ShaderType>(entities, ray, lights);
            // Atomic addition or other synchronization to add `local_result` to `result`
        });

        return result;// Make sure to copy this back to the host if required
    }
};
#endif

#include <freyr/texture.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace freyr::shader {


    using vector3f_type = Eigen::Vector3f;
    using scene_type    = freyr::scene::scene;
    using ray_type      = freyr::ray<>;

    using material_type = std::variant<vector3f_type, texture::texture_type>;

    struct shader_args {
        vector3f_type normal;
        vector3f_type light_dir;
        vector3f_type view_dir;
        vector3f_type emission;
        material_type ambient;
        material_type diffuse;
        material_type specular;
        vector3f_type intensity;
        float         u;
        float         v;
        float         t_hit;
        float         shininess;
        bool          in_shadow;
        float         fresnel;
    };

    // Utility Functions
    namespace utility {


        // Initialize static variables for random engine
        static std::random_device               rd;
        static std::mt19937                     gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 1.0);

        // Function to generate a random float between 0 and 1
        inline float random_float() { return static_cast<float>(dis(gen)); }

        using image_type = Eigen::Tensor<float, 3>;

        template<typename T>
        T cwise_product(const T &vec1, const T &vec2);

        template<>
        vector3f_type cwise_product(const vector3f_type &vec1, const vector3f_type &vec2) {
            return vec1.cwiseProduct(vec2);
        }

        template<typename T>
        T zero_initialize();

        template<>
        vector3f_type zero_initialize() {
            return vector3f_type::Zero();
        }

        template<typename T>
        T set_zero(T &val);

        template<>
        image_type set_zero(image_type &val) {
            return val.setZero();
        }

        template<typename T, typename U>
        T clamp(const T &val, const U &min, const U &max) {
            return std::max(min, std::min(max, val));
        }

        template<>
        vector3f_type clamp(const vector3f_type &val, const float &min, const float &max) {
            return val.cwiseMax(min).cwiseMin(max);
        }

        void normalize_vectors(vector3f_type &vec1, vector3f_type &vec2) {
            vec1.normalize();
            vec2.normalize();
        }

        auto get_hit(const entity::entity &entity, const ray_type &ray) -> std::pair<bool, float> {
            if (shape::intersects_bounds(ray, CRTP_VAR_CALL(entity.shape, object_bounds)())) {
                return CRTP_VAR_CALL(entity.shape, intersect_p)(ray);
            } else {
                return {false, std::numeric_limits<float>::max()};
            }
        }


        // Utility to retrieve resolution from camera
        auto retrieve_resolution(const camera::camera_type &camera_variant)
                -> std::pair<size_t, size_t> {
            return {CRTP_VAR_CALL(camera_variant, get_width)(),
                    CRTP_VAR_CALL(camera_variant, get_height)()};
        }


    }// namespace utility


    template<typename Entity, typename Lights>
    shader_args compute_shader_args(const Entity   &entity,
                                    const ray_type &ray,
                                    float           t_hit,
                                    const Lights   &lights,
                                    const auto     &entities) {
        shader_args args;
        args.t_hit = t_hit;
        auto point = ray.get_point(t_hit);

        std::tie(args.u, args.v) = CRTP_VAR_CALL(entity.shape, get_uv_at)(point);

        // Fetch various quantities
        auto shape_normal = CRTP_VAR_CALL(entity.shape, get_normal)(point);
        auto btn          = CRTP_VAR_CALL(entity.shape, get_tbn_matrix)(point);
        auto material_normal
                = CRTP_VAR_CALL(entity.material.normal, get_normal_at)(args.u, args.v);

        // Transform the normal from the normal map into object/world space
        auto normal_map_contribution = (btn * material_normal);

        // Compute the final normal
        args.normal = (shape_normal + normal_map_contribution);

        const auto &light = lights[0];
        args.intensity    = CRTP_VAR_CALL(light, intensity_at)(point);
        args.light_dir    = CRTP_VAR_CALL(light, direction_from)(point);
        args.view_dir     = -ray.get_direction().normalized();


        args.emission  = entity.material.emission;
        args.shininess = entity.material.shininess;
        args.fresnel   = entity.material.fresnel;

        args.ambient  = entity.material.ambient;
        args.diffuse  = entity.material.diffuse;
        args.specular = entity.material.specular;

        args.in_shadow = false;
        for (const auto &l: lights) {
            auto     shadow_ray_origin = point + 1e-5f * args.normal;
            auto     shadow_ray_dir    = CRTP_VAR_CALL(l, direction_from)(point);
            ray_type shadow_ray(shadow_ray_origin, shadow_ray_dir);

            for (const auto &obstacle: entities) {
                auto [obstructs, _] = utility::get_hit(obstacle, shadow_ray);
                if (obstructs) {
                    args.in_shadow = true;
                    break;
                }
            }
            if (args.in_shadow) break;
        }

        return args;
    }


    // Common Lighting Calculations
    struct common_lighting_calculations {
        static vector3f_type
        extract_color(const std::variant<vector3f_type, texture::texture_type> &variant,
                      float                                                     u,
                      float                                                     v) {
            if (std::holds_alternative<vector3f_type>(variant)) {
                return std::get<vector3f_type>(variant);
            }
            return CRTP_VAR_CALL(std::get<texture::texture_type>(variant), get_color_at)(u, v);
        }

        [[nodiscard]] static vector3f_type ambient_lighting(const shader_args &args) {
            return utility::cwise_product(extract_color(args.ambient, args.u, args.v),
                                          args.intensity);
        }

        [[nodiscard]] static vector3f_type diffuse_lighting(const shader_args &args) {
            vector3f_type normal    = args.normal;
            vector3f_type light_dir = args.light_dir;
            utility::normalize_vectors(normal, light_dir);

            float val = utility::clamp(normal.dot(light_dir), 0.0f, 1.0f);
            return utility::cwise_product(extract_color(args.diffuse, args.u, args.v),
                                          (args.intensity * val).eval());
        }
        [[nodiscard]] static vector3f_type tone_mapping(const vector3f_type &color) {
            return color.array() / (color.array() + 1.0f);
        }
    };

    // CRTP-based Shader class
    template<typename Derived>
    class shader {
    public:
        [[nodiscard]] auto apply(const shader_args &args) const {
            return const_cast<Derived *>(static_cast<const Derived *>(this))->apply_impl(args);
        }
    };

    // Lambertian Shader
    class lambertian : public shader<lambertian>, public common_lighting_calculations {
    public:
        lambertian() = default;

        [[nodiscard]] vector3f_type apply_impl(const shader_args &args) const {
            vector3f_type color = utility::zero_initialize<vector3f_type>();

            // Handle ambient light
            color += ambient_lighting(args);

            if (!args.in_shadow) {
                // Add diffuse component
                color += diffuse_lighting(args);
            }

            return color;
        }
    };

    // Depth Shader
    class depth : public shader<depth> {
    public:
        depth(float near = 0.0f, float far = 100.0f) : m_near(near), m_far(far) {}

        [[nodiscard]] vector3f_type apply_impl(const shader_args &args) const {
            // Assuming the depth (t_hit) is directly available in args.
            // Replace with actual depth computation if needed.
            float depth = args.t_hit;

            // Normalize depth into [0, 1] if you have the far and near clipping planes.
            float normalized_depth = (depth - m_near) / (m_far - m_near);

            // Clamp normalized depth between 0 and 1
            normalized_depth = utility::clamp(normalized_depth, 0.0f, 1.0f);

            // Convert it to a grayscale color
            vector3f_type grayscale_color(depth, depth, depth);

            return grayscale_color;
        }

    private:
        float m_near;
        float m_far;
    };

    // Phong Shader class, inheriting from shader and common_lighting_calculations
    class phong : public shader<phong>, public common_lighting_calculations {
    public:
        phong() = default;

        // https://en.wikipedia.org/wiki/Phong_reflection_model
        // https://computergraphics.stackexchange.com/questions/1513/how-physically-based-is-the-diffuse-and-specular-distinction

        // Implement the Phong reflection model to compute the color at a point on the surface.
        CROSS_PLATFORM [[nodiscard]] static vector3f_type apply_impl(const shader_args &args) {
            vector3f_type color = utility::zero_initialize<vector3f_type>();

            // Check if the point is in shadow
            if (args.in_shadow) {
                return utility::cwise_product(extract_color(args.ambient, args.u, args.v),
                                              args.intensity);
            }

            // Normalize the relevant vectors
            auto N = args.normal.normalized();   // Surface Normal
            auto L = args.light_dir.normalized();// Light Direction
            auto V = args.view_dir.normalized(); // View Direction

            // Compute the reflection of L about N
            auto R = (2 * N.dot(L) * N - L).normalized();

            // Compute the ambient term: I_a = k_a * i_a
            auto ambient = utility::cwise_product(extract_color(args.ambient, args.u, args.v),
                                                  args.intensity);

            // Compute the diffuse term only when the surface is not in shadow
            float diff = std::max(N.dot(L), 0.0f);// L . N

            if (diff > 0.001f) {
                // Compute the diffuse term: I_d = k_d * (L . N) * i_d
                auto diffuse = diff
                             * utility::cwise_product(extract_color(args.diffuse, args.u, args.v),
                                                      args.intensity);

                // Compute the specular term: I_s = k_s * (R . V)^shininess * i_s
                float spec = std::pow(std::max(R.dot(V), 0.0f), args.shininess);
                auto  specular
                        = spec
                        * utility::cwise_product(extract_color(args.specular, args.u, args.v),
                                                 args.intensity);

                // Phong reflection model: I_p = I_a + I_d + I_s
                color = ambient + diffuse + specular;
            } else {
                // If in shadow, only ambient light contributes to the color
                color = ambient;
            }

            // Clamp the color components to [0, 1]
            color = color.cwiseMin(1.0f);

            return color;
        }
    };

    template<typename Derived>
    class sampling {
    public:
        using sample_type = std::tuple<float, float>;

        [[nodiscard]] std::vector<sample_type>
        generate_samples(size_t i, size_t j, size_t height, size_t width) const {
            return static_cast<const Derived *>(this)->generate_samples_impl(i, j, height, width);
        }
    };


    class uniform_sampling : public sampling<uniform_sampling> {
    public:
        explicit uniform_sampling(size_t num_samples) : m_n_samples(num_samples) {}

        [[nodiscard]] std::vector<sample_type>
        generate_samples_impl(size_t i, size_t j, size_t height, size_t width) const {
            std::vector<sample_type> samples;
            samples.reserve(m_n_samples);

            for (size_t k = 0; k < m_n_samples; ++k) {
                float u = i + 0.5f
                        + (utility::random_float() - 0.5f);// Random float between -0.5 and 0.5
                float v = j + 0.5f
                        + (utility::random_float() - 0.5f);// Random float between -0.5 and 0.5
                samples.emplace_back(u, v);
            }

            return samples;
        }

    private:
        size_t m_n_samples;
    };


    using sampling_type = std::variant<uniform_sampling>;

    using shader_node = std::variant<lambertian, phong, depth>;

    class tbb_backend_adapter {
    public:
        static Eigen::Tensor<float, 3> apply(const scene_type               &scene,
                                             const std::vector<shader_node> &nodes) {
            // Cache common scene information
            const auto &lights          = scene.get_lights();
            const auto &camera_variant  = scene.get_camera();
            const auto &[width, height] = utility::retrieve_resolution(camera_variant);
            const auto sampling_method  = uniform_sampling(4);

            // Initialize the image tensor
            using image_type = Eigen::Tensor<float, 3>;
            image_type image(height, width, 3);
            utility::set_zero<image_type>(image);

            // Perform parallel computation
            tbb::parallel_for(tbb::blocked_range2d<size_t, size_t>(0, width, 0, height),
                              [&](const tbb::blocked_range2d<size_t, size_t> &r) {
                                  process_pixel_range(scene,
                                                      nodes,
                                                      r,
                                                      image,
                                                      lights,
                                                      camera_variant,
                                                      sampling_method);
                              });

            return image;
        }

    private:
        // Function to handle each block of pixels
        static void process_pixel_range(const scene_type                           &original_scene,
                                        const std::vector<shader_node>             &nodes,
                                        const tbb::blocked_range2d<size_t, size_t> &range,
                                        Eigen::Tensor<float, 3>                    &image,
                                        const auto                                 &lights,
                                        const auto                                 &camera_variant,
                                        const auto                                 &sampling_method

        ) {
            // Local copy of the scene for this thread
            thread_local scene_type scene_copy = original_scene;
            thread_local auto      &entities   = scene_copy.get_entities();

            for (size_t i = range.rows().begin(); i != range.rows().end(); ++i) {
                for (size_t j = range.cols().begin(); j != range.cols().end(); ++j) {
                    process_pixel(i,
                                  j,
                                  entities,
                                  nodes,
                                  image,
                                  lights,
                                  camera_variant,
                                  sampling_method);
                }
            }
        }


        static void process_pixel(size_t                          i,
                                  size_t                          j,
                                  const auto                     &entities,
                                  const std::vector<shader_node> &nodes,
                                  Eigen::Tensor<float, 3>        &image,
                                  const auto                     &lights,
                                  const auto                     &camera_variant,
                                  const sampling_type            &sampling_method) {

            // Retrieve the resolution of the camera
            auto [height, width] = utility::retrieve_resolution(camera_variant);

            // Create samples using a sampling method
            auto samples = CRTP_VAR_CALL(sampling_method, generate_samples)(i, j, height, width);

            // Initialize the pixel value
            auto total_pixel_value = utility::zero_initialize<vector3f_type>();

            // For each sample, compute the color
            for (const auto &[u_s, v_s]: samples) {
                // Generate ray based on sample
                ray_type ray = CRTP_VAR_CALL(camera_variant, generate_ray)(u_s, v_s);

                float       closest_distance = std::numeric_limits<float>::max();
                auto        closest_entity   = std::optional<scene::entity_type>{};
                shader_args closest_args;

                // Find the closest entity hit by the ray
                for (const auto &entity: entities) {
                    const auto [hit_status, hit_distance] = utility::get_hit(entity, ray);
                    if (hit_status && hit_distance < closest_distance) {
                        closest_distance = hit_distance;
                        closest_entity   = entity;
                        closest_args
                                = compute_shader_args(entity, ray, hit_distance, lights, entities);
                    }
                }

                // If an entity was hit, compute the color
                if (closest_entity.has_value()) {
                    // Compute the color using the shader nodes
                    for (const auto &node: nodes) {
                        total_pixel_value += CRTP_VAR_CALL(node, apply)(closest_args);
                    }
                }
            }

            // Compute the average pixel value
            auto pixel_value = total_pixel_value / samples.size();

            // Populate image tensor
            long i_long = static_cast<long>(i), j_long = static_cast<long>(j);
            image(j_long, i_long, 0) = pixel_value(0);
            image(j_long, i_long, 1) = pixel_value(1);
            image(j_long, i_long, 2) = pixel_value(2);
        }
    };

    // Shader graph for single-pass processing
    class graph {
    public:
        graph() = default;
        void add_node(shader_node node) { nodes.push_back(node); }

        template<typename Backend>
        Eigen::Tensor<float, 3> execute_single_pass(const scene_type &scene, Backend &&backend) {
            return backend.apply(scene, nodes);
        }

    private:
        std::vector<shader_node> nodes;
    };


}// namespace freyr::shader


#endif// FREYR_SHADER_H