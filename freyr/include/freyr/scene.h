#ifndef FREYR_SCENE_H
#define FREYR_SCENE_H

#include <utility>

#include <freyr/camera.h>
#include <freyr/common.h>
#include <freyr/light.h>
#include <freyr/shape.h>


namespace freyr::tiling {

    //    template<typename Derived>
    //    class tiling_utility {
    //    public:
    //        void handle_tiling(float &u, float &v) const {
    //            static_cast<const Derived *>(this)->handle_tiling_impl(u, v);
    //        }
    //    };

    template<typename Derived>
    class tiling_strategy {
    public:
        //        tiling_strategy(float u, float v) {
        //            handle_tiling(u, v);
        //        }

        std::tuple<float, float> handle_tiling(float u, float v) const {
            return static_cast<const Derived *>(this)->handle_tiling_impl(u, v);
        }
    };

    class repeat_tiling : public tiling_strategy<repeat_tiling> {
    public:
        static std::tuple<float, float> handle_tiling_impl(float &u, float &v) {
            float u_tiled = std::fmod(u, 1.0f);
            float v_tiled = std::fmod(v, 1.0f);

            if (u_tiled < 0) u_tiled += 1.0f;
            if (v_tiled < 0) v_tiled += 1.0f;

            return {u_tiled, v_tiled};
        }
    };

    class clamp_tiling : public tiling_strategy<clamp_tiling> {
    public:
        static std::tuple<float, float> handle_tiling_impl(float &u, float &v) {
            return {std::clamp(u, 0.0f, 1.0f), std::clamp(v, 0.0f, 1.0f)};
        }
    };

    using tiling_strategy_type = std::variant<repeat_tiling, clamp_tiling>;

};// namespace freyr::tiling


namespace freyr::normal {
    //
    //    class normal_perturbation_strategy {
    //    public:
    //        virtual normal_type perturb_normal(float u, float v, float amplitude) const = 0;
    //    };
    //
    using namespace tiling;
    using normal_type = Eigen::Vector3f;

    template<typename Derived>
    class normal_map {
    public:
        // http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
        explicit normal_map(float                u_scale = 0.3f,
                            float                v_scale = 0.3f,
                            tiling_strategy_type tiling  = repeat_tiling())
            : tiling(tiling), m_u_scale(u_scale), m_v_scale(v_scale) {}

        [[nodiscard]] normal_type get_normal_at(float u, float v) const {
            u *= m_u_scale;
            v *= m_v_scale;
            auto [u_, v_] = CRTP_VAR_CALL(tiling, handle_tiling)(u, v);
            return static_cast<const Derived *>(this)->get_normal_impl(u_, v_);
        }

    protected:
        tiling_strategy_type tiling;
        float                m_u_scale;
        float                m_v_scale;
    };

    class tensor_based : public normal_map<tensor_based> {
    public:
        using index_type = Eigen::Index;
        using normal_map<tensor_based>::normal_map;

        // add copy constructor that copies eigen tensor properly

        tensor_based(const tensor_based &other)
            : normal_map<tensor_based>(other),   // Calls the base class copy constructor
              m_normal_data(other.m_normal_data),// Eigen Tensor copy
              m_amplitude(other.m_amplitude) {}

        explicit tensor_based(Eigen::Tensor<float, 3> normal_data,
                              float                   amplitude = 1.0f,
                              float                   u_scale   = 1.0f,
                              float                   v_scale   = 1.0f,
                              tiling_strategy_type    tiling    = repeat_tiling())
            : normal_map<tensor_based>(u_scale, v_scale, tiling),
              m_normal_data(std::move(normal_data)), m_amplitude(amplitude) {}

        [[nodiscard]] normal_type get_normal_impl(float u, float v) const {

#ifdef FREYR_DEBUG
            // Bounds check
            if (u < 0 || u > 1 || v < 0 || v > 1) {
                // Handle out-of-bounds u, v (e.g., return a default normal)
                throw std::runtime_error("u, v out of bounds");
                //return normal_type{0, 0, 1};
            }
#endif

            // Convert to indices
            auto u_idx = static_cast<index_type>(u * ((float) m_normal_data.dimension(0)));
            auto v_idx = static_cast<index_type>(v * ((float) m_normal_data.dimension(1)));

#ifdef FREYR_DEBUG
            // Check bounds
            if (u_idx < 0 || u_idx >= m_normal_data.dimension(0) || v_idx < 0
                || v_idx >= m_normal_data.dimension(1)) {
                // Handle out-of-bounds u_idx, v_idx (e.g., return a default normal)
                throw std::runtime_error("u_idx, v_idx out of bounds");
                //return normal_type{0, 0, 1};
            }
#endif

            // Fetch and remap the normal
            normal_type fetched_normal = {m_amplitude * (2 * m_normal_data(u_idx, v_idx, 0) - 1),
                                          m_amplitude * (2 * m_normal_data(u_idx, v_idx, 1) - 1),
                                          m_amplitude * (2 * m_normal_data(u_idx, v_idx, 2) - 1)};

            //
            //            // Fetch and remap the normal
            //            normal_type fetched_normal = {m_amplitude * (2 * m_normal_data(u_idx, v_idx, 0) - 1),
            //                                          m_amplitude * (2 * m_normal_data(u_idx, v_idx, 1) - 1),
            //                                          m_amplitude * (2 * m_normal_data(u_idx, v_idx, 2) - 1)};

            // Assuming you have a normalize function for your normal_type
            return fetched_normal;
        }


    private:
        Eigen::Tensor<float, 3> m_normal_data;
        float                   m_amplitude;
    };

    class none : public normal_map<none> {
    public:
        [[nodiscard]] static normal_type get_normal_impl(float u, float v) { return {0, 0, 0}; }
    };

    using normal_map_type = std::variant<tensor_based, none>;

}// namespace freyr::normal
//
//    class sine_wave_perturbation : public normal_perturbation_strategy {
//    public:
//        normal_type perturb_normal(float u, float v, float amplitude) const override {
//            float perturb_x = amplitude * sin(u * 2.0f * 3.14159f);
//            float perturb_y = amplitude * cos(v * 2.0f * 3.14159f);
//            return normal_type{perturb_x, perturb_y, sqrt(1 - perturb_x * perturb_x - perturb_y * perturb_y)};
//        }
//    };
//
//    class noise_perturbation : public normal_perturbation_strategy {
//    public:
//        normal_type perturb_normal(float u, float v, float amplitude) const override {
//            // Implement noise-based perturbation. Placeholder code here.
//            // ...
//            return normal_type{0, 0, 1}; // Replace with actual implementation.
//        }
//    };
//
//    // Add more strategies as needed.
//    class bumps : public normal_map<bumps> {
//    public:
//        using normal_map::normal_map;
//
//        bumps(std::shared_ptr<normal_perturbation_strategy> perturb_strategy,
//              float amplitude = 0.5f,
//              float u_scale   = 0.02f,
//              float v_scale   = 0.02f,
//              tiling_strategy_type tiling  = repeat_tiling())
//            : normal_map(u_scale, v_scale, tiling), m_amplitude(amplitude), m_perturb_strategy(perturb_strategy) {}
//
//        normal_type get_normal_impl(float u, float v) const override {
//            return m_perturb_strategy->perturb_normal(u, v, m_amplitude);
//        }
//
//    private:
//        float m_amplitude;
//        std::shared_ptr<normal_perturbation_strategy> m_perturb_strategy;
//    };
//
//}


namespace freyr::texture {

    using color_type = Eigen::Vector3f;

    using namespace tiling;

    template<typename Derived>
    class texture {
    public:
        explicit texture(float                u_scale = 1.0f,
                         float                v_scale = 1.0f,
                         tiling_strategy_type tiling  = repeat_tiling())
            : tiling(tiling), m_u_scale(u_scale), m_v_scale(v_scale) {}

        [[nodiscard]] color_type get_color_at(float u, float v) const {
            u *= m_u_scale;
            v *= m_v_scale;
            CRTP_VAR_CALL(tiling, handle_tiling)(u, v);
            return static_cast<const Derived *>(this)->get_color_impl(u, v);
        }

    protected:
        tiling_strategy_type tiling;
        float                m_u_scale;
        float                m_v_scale;
    };

    class checkerboard : public texture<checkerboard> {
    public:
        using texture::texture;

        explicit checkerboard(color_type           color1  = color_type{0.0f, 0.0f, 0.0f},
                              color_type           color2  = color_type{1.0f, 1.0f, 1.0f},
                              float                u_scale = 0.1f,
                              float                v_scale = 0.1f,
                              tiling_strategy_type tiling  = repeat_tiling())
            : texture(u_scale, v_scale, tiling), m_color1(std::move(color1)),
              m_color2(std::move(color2)) {}

        [[nodiscard]] color_type get_color_impl(float u, float v) const {

            if (std::abs(static_cast<int>(std::floor(u / m_u_scale)) % 2)
                == std::abs(static_cast<int>(std::floor(v / m_v_scale)) % 2)) {
                return m_color1;
            } else {
                return m_color2;
            }
        }

        void                     set_color1(const color_type &color) { m_color1 = color; }
        void                     set_color2(const color_type &color) { m_color2 = color; }
        [[nodiscard]] color_type get_color1() const { return m_color1; }
        [[nodiscard]] color_type get_color2() const { return m_color2; }


    private:
        color_type m_color1;
        color_type m_color2;
    };

    class constant : public texture<constant> {
    public:
        color_type color;

        explicit constant(color_type           color   = {1.0f, 1.0f, 1.0f},
                          float                u_scale = 1.0f,
                          float                v_scale = 1.0f,
                          tiling_strategy_type tiling  = repeat_tiling())
            : texture(u_scale, v_scale, tiling), color(std::move(color)) {}

        [[nodiscard]] color_type get_color_impl(float u, float v) const { return color; }

    private:
        color_type m_color;
    };

    class image : public texture<image> {
    public:
        Eigen::Tensor<float, 3> m_data;

        explicit image(const Eigen::Tensor<float, 3> &data,
                       float                          u_scale = 1.0f,
                       float                          v_scale = 1.0f,
                       tiling_strategy_type           tiling  = repeat_tiling())
            : texture(u_scale, v_scale, tiling), m_data(data) {}

        [[nodiscard]] color_type get_color_impl(float u, float v) const {
            long width  = m_data.dimension(0);
            long height = m_data.dimension(1);

            float fu = u * (float) (width - 1);
            float fv = v * (float) (height - 1);

            int x = static_cast<int>(fu);
            int y = static_cast<int>(fv);

            float u_ratio    = fu - (float) x;
            float v_ratio    = fv - (float) y;
            float u_opposite = 1 - u_ratio;
            float v_opposite = 1 - v_ratio;

            color_type c1 = color_type(m_data(x, y, 0), m_data(x, y, 1), m_data(x, y, 2));
            color_type c2
                    = color_type(m_data(x + 1, y, 0), m_data(x + 1, y, 1), m_data(x + 1, y, 2));
            color_type c3
                    = color_type(m_data(x, y + 1, 0), m_data(x, y + 1, 1), m_data(x, y + 1, 2));
            color_type c4 = color_type(m_data(x + 1, y + 1, 0),
                                       m_data(x + 1, y + 1, 1),
                                       m_data(x + 1, y + 1, 2));

            return (c1 * u_opposite + c2 * u_ratio) * v_opposite
                 + (c3 * u_opposite + c4 * u_ratio) * v_ratio;
        }
    };

    using texture_type = std::variant<constant, checkerboard, image>;

}// namespace freyr::texture

namespace freyr::material {
    using shape_type    = freyr::shape::shape_type;
    using color_type    = Eigen::Vector3f;
    using vector3f_type = Eigen::Vector3f;
    using texture_type  = freyr::texture::texture_type;
    using material_type = std::variant<vector3f_type, texture_type>;
    using normal_type   = freyr::normal::normal_map_type;

    struct material {
        color_type emission{0.9f, 0.5f, 0.5f};
        float      shininess = 1.0f;
        float      fresnel;

        normal_type normal{normal::none()};

        material_type ambient{
                vector3f_type{0.1f, 0.1f, 0.1f}
        };
        material_type diffuse{
                vector3f_type{0.5f, 0.5f, 0.5f}
        };
        material_type specular{
                vector3f_type{0.7f, 0.7f, 0.7f}
        };
    };

}// namespace freyr::material

namespace freyr::entity {
    using shape_type    = freyr::shape::shape_type;
    using color_type    = Eigen::Vector3f;
    using vector3f_type = Eigen::Vector3f;
    using material_type = freyr::material::material;

    struct entity {
        entity(shape_type shape, material_type material = material_type{})
            : shape(std::move(shape)), material(std::move(material)) {}
        shape_type    shape;
        material_type material{};
    };

}// namespace freyr::entity

namespace freyr::scene {

    using camera_type = freyr::camera::camera_type;
    using light_type  = freyr::light::light_type;
    using entity_type = entity::entity;

    struct global_settings {
        float ambient_intensity;
        // other global settings
    };

    class scene {
    public:
        explicit scene(camera_type camera) : m_camera(std::move(camera)) {}

        void add_light(const light_type &light) { m_lights.push_back(light); }

        void set_camera(const camera_type &camera) { this->m_camera = camera; }

        void set_global_settings(const global_settings &settings) { this->m_settings = settings; }

        [[nodiscard]] const std::vector<light_type> &get_lights() const { return m_lights; }

        [[nodiscard]] const camera_type &get_camera() const { return m_camera; }

        [[nodiscard]] const global_settings &get_global_settings() const { return m_settings; }

        void add_entity(const entity_type &entity) { m_entities.push_back(entity); }

        [[nodiscard]] const std::vector<entity_type> &get_entities() const { return m_entities; }

    private:
        std::vector<light_type>  m_lights;
        camera_type              m_camera;
        global_settings          m_settings{};
        std::vector<entity_type> m_entities;
    };


}// namespace freyr::scene

#endif// FREYR_SCENE_H