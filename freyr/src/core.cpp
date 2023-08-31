#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include <freyr/camera.h>
#include <freyr/common.h>
#include <freyr/light.h>
#include <freyr/ray.h>
#include <freyr/shader.h>
#include <freyr/shape.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace freyr;

template<typename T>//freyr_light
[[maybe_unused]] py::class_<T> freyr_light(py::module &m, const std::string &name) {
    return py::class_<T>(m, name.c_str())
            .def("get_intensity", &T::get_intensity, "Get light intensity")
            .def("set_intensity", &T::set_intensity, "Set light intensity")
            .def("get_color", &T::get_color, "Get light color")
            .def("set_color", &T::set_color, "Set light color")
            .def("intensity_at", &T::intensity_at, "point"_a, "Get light intensity at point")
            .def("direction_from",
                 &T::direction_from,
                 "point"_a,
                 "Get light direction from point");
}

template<typename T, typename U>//freyr_light
[[maybe_unused]] py::class_<T, U> freyr_light(py::module &m, const std::string &name) {
    return py::class_<T, U>(m, name.c_str())
            .def("get_intensity", &T::get_intensity, "Get light intensity")
            .def("set_intensity", &T::set_intensity, "Set light intensity")
            .def("get_color", &T::get_color, "Get light color")
            .def("set_color", &T::set_color, "Set light color")
            .def("intensity_at", &T::intensity_at, "point"_a, "Get light intensity at point")
            .def("direction_from",
                 &T::direction_from,
                 "point"_a,
                 "Get light direction from point");
}

template<typename T>//freyr_shape
[[maybe_unused]] py::class_<T> freyr_shape(py::module &m, const std::string &name) {
    return py::class_<T>(m, name.c_str())
            .def("set_position", &T::set_position, "Set shape position")
            .def("get_position", &T::get_position, "Get shape position")
            .def("get_normal_at", &T::get_normal, "point"_a, "Get shape normal at point")
            .def("get_uv_at", &T::get_uv_at, "point"_a, "Get shape uv at point")
            .def("intersect", &T::intersect, "ray"_a, "Intersect ray with shape")
            .def("object_bounds", &T::object_bounds, "Get shape object bounds")
            .def("intersects_bounds",
                 &T::intersects_bounds,
                 "ray"_a,
                 "Intersect ray with shape bounds")
            .def("intersect_p",
                 &T::intersect_p,
                 "ray"_a,
                 "Intersect ray with shape and return hit point");
    //            .def("set_rotation", &T::set_rotation, "Set shape rotation");
    //            .def("set_scale", &T::set_scale, "Set shape scale")
    //            .def("get_position", &T::get_position)
    //            .def("get_rotation", &T::get_rotation)
    //            .def("get_scale", &T::get_scale)
    //            .def("get_transform", &T::get_transform)
    //            .def("set_transform", &T::set_transform)
    //            .def("get_inverse_transform", &T::get_inverse_transform)
    //            .def("set_inverse_transform", &T::set_inverse_transform)
    //            .def("get_material", &T::get_material)
    //            .def("set_material", &T::set_material)
    //            .def("get_id", &T::get_id)
    //            .def("set_id", &T::set_id)
    //            .def("get_name", &T::get_name)
    //            .def("set_name", &T::set_name)
    //            .def("get_type", &T::get_type)
    //            .def("set_type", &T::set_type)
    //            .def("get_area", &T::get_area)
    //            .def("get_centroid", &T::get_centroid)
    //            .def("get_normal", &T::get_normal);
}

template<typename T, typename U>//freyr_shape
[[maybe_unused]] py::class_<T, U> freyr_shape(py::module &m, const std::string &name) {
    return py::class_<T, U>(m, name.c_str())
            .def("set_position", &T::set_position, "Set shape position")
            .def("get_position", &T::get_position, "Get shape position")
            .def("get_normal_at", &T::get_normal, "point"_a, "Get shape normal at point")
            .def("get_uv_at", &T::get_uv_at, "point"_a, "Get shape uv at point")
            .def("intersect", &T::intersect, "ray"_a, "Intersect ray with shape")
            .def("object_bounds", &T::object_bounds, "Get shape object bounds")
            .def("intersects_bounds",
                 &T::intersects_bounds,
                 "ray"_a,
                 "Intersect ray with shape bounds")
            .def("intersect_p",
                 &T::intersect_p,
                 "ray"_a,
                 "Intersect ray with shape and return hit point");
    //            .def("set_rotation", &T::set_rotation, "Set shape rotation");
}


//#define FREYR_VIRTUAL_BASE_CLASSES
#ifdef FREYR_VIRTUAL_BASE_CLASSES
//
//class base_shape : public shape<base_shape> {
//public:
//    explicit base_shape(vector3f_type position) : shape(std::move(position)) {}
//    virtual ~base_shape() = default;
//    //        virtual bool                    intersects_bounds_impl() const = 0;
//    virtual std::tuple<bool, float> intersect_p_impl(const ray_type &ray) const = 0;
//    virtual vector3f_type           get_normal_impl(vector3f_type point) const  = 0;
//    virtual bool                    intersect_impl(const ray_type &ray) const   = 0;
//    virtual bounds3f_type           object_bounds_impl() const                  = 0;
//};


class py_base_shape : public shape::base_shape {
public:
    /* Inherit the constructors */
    using shape::base_shape::base_shape;

    /* Trampoline (need one for each virtual function) */
    [[nodiscard]] vector3f_type get_normal_impl(vector3f_type point) const override {
        PYBIND11_OVERRIDE_PURE(
                vector3f_type,     /* Return type */
                shape::base_shape, /* Parent class */
                get_normal_impl,   /* Name of function in C++ (must match Python name) */
                point              /* Argument(s) */
        );
    }

    /* Trampoline (need one for each virtual function) */
    [[nodiscard]] bounds3f_type object_bounds_impl() const override {
        PYBIND11_OVERRIDE_PURE(
                bounds3f_type,      /* Return type */
                shape::base_shape,  /* Parent class */
                object_bounds_impl, /* Name of function in C++ (must match Python name) */
        );
    }

    using intersect_p_impl_return_type = std::pair<bool, float>;

    /* Trampoline (need one for each virtual function) */
    [[nodiscard]] std::tuple<bool, float> intersect_p_impl(const ray_type &ray) const override {
        PYBIND11_OVERRIDE_PURE(
                intersect_p_impl_return_type, /* Return type */
                shape::base_shape,            /* Parent class */
                intersect_p_impl, /* Name of function in C++ (must match Python name) */
                ray               /* Argument(s) */
        );
    }

    /* Trampoline (need one for each virtual function) */
    [[nodiscard]] bool intersect_impl(const ray_type &ray) const override {
        PYBIND11_OVERRIDE_PURE(
                bool,              /* Return type */
                shape::base_shape, /* Parent class */
                intersect_impl,    /* Name of function in C++ (must match Python name) */
                ray                /* Argument(s) */
        );
    }

    /* Trampoline (need one for each virtual function) */
    //    vector3f get_normal_impl(const vector3f &point) const override {
    //        PYBIND11_OVERRIDE_PURE(
    //                vector3f, /* Return type */
    //                base_shape,      /* Parent class */
    //                get_normal_impl,          /* Name of function in C++ (must match Python name) */
    //                point      /* Argument(s) */
    //        );
    //    }
};

#endif

template<typename... Ts>
[[maybe_unused]] py::class_<Ts...> freyr_camera(py::module &m, const std::string &name) {
    using T = typename std::tuple_element<0, std::tuple<Ts...>>::type;

    return py::class_<Ts...>(m, name.c_str())
            .def("set_position", &T::set_position, R"pbdoc(
                Set the camera's position in 3D space.

                Parameters
                ----------
                position : vector3_type
                    New camera position.

            )pbdoc")
            .def("set_aspect_ratio", &T::set_aspect_ratio, R"pbdoc(
                Set the aspect ratio of the camera view.

                Parameters
                ----------
                aspect_ratio : float
                    New aspect ratio.

            )pbdoc")
            .def("set_width", &T::set_width, R"pbdoc(
                Set the image width for the camera.

                Parameters
                ----------
                width : int
                    New image width.

            )pbdoc")
            .def("set_height", &T::set_height, R"pbdoc(
                Set the image height for the camera.

                Parameters
                ----------
                height : int
                    New image height.

            )pbdoc")
            .def("get_position", &T::get_position, R"pbdoc(
                Get the camera's current position in 3D space.

                Returns
                -------
                vector3_type
                    Current camera position.

            )pbdoc")
            .def("get_aspect_ratio", &T::get_aspect_ratio, R"pbdoc(
                Get the current aspect ratio of the camera view.

                Returns
                -------
                float
                    Current aspect ratio.

            )pbdoc")
            .def("get_width", &T::get_width, R"pbdoc(
                Get the current image width for the camera.

                Returns
                -------
                int
                    Current image width.

            )pbdoc")
            .def("get_height", &T::get_height, R"pbdoc(
                Get the current image height for the camera.

                Returns
                -------
                int
                    Current image height.

            )pbdoc")
            .def("generate_ray", &T::generate_ray, R"pbdoc(
                Generate a single ray from the camera.

                Returns
                -------
                Ray
                    Generated ray.

            )pbdoc")
            .def(
                    "generate_rays",
                    [](T &self, Eigen::ArrayXf u, Eigen::ArrayXf v) {
                        return self.generate_rays(u, v);
                    },
                    R"pbdoc(
                Generate a batch of rays using Eigen ArrayXf for u and v coordinates.

                Parameters
                ----------
                u : Eigen::ArrayXf
                    Array of u coordinates.
                v : Eigen::ArrayXf
                    Array of v coordinates.

                Returns
                -------
                Rays
                    A batch of generated rays.

            )pbdoc")
            .def(
                    "generate_rays",
                    [](T &self, Eigen::Tensor<float, 2> uu, Eigen::Tensor<float, 2> vv) {
                        return self.generate_rays(uu, vv);
                    },
                    R"pbdoc(
                Generate a batch of rays using Eigen Tensor for uu and vv coordinates.

                Parameters
                ----------
                uu : Eigen::Tensor<float, 2>
                    2D tensor of u coordinates.
                vv : Eigen::Tensor<float, 2>
                    2D tensor of v coordinates.

                Returns
                -------
                Rays
                    A batch of generated rays.

            )pbdoc");
}

template<typename T>
py::class_<T> freyr_texture(py::module &m, const std::string &name) {
    return py::class_<T>(m, name.c_str())
            .def("get_color_at", &T::get_color_at, "u"_a, "v"_a, "Get texture color");
}


PYBIND11_MODULE(core, m) {
    using vector3_type = freyr::vector3_type<float>;

    auto m_types      = m.def_submodule("types", "Types submodule");
    using camera_type = camera::camera_type;
    py::class_<camera_type>(m_types, "camera_type");

    auto m_camera = m.def_submodule("camera",
                                    R"pbdoc(
        Camera submodule
        ----------------

        This submodule provides different types of cameras used in 3D rendering.

        )pbdoc");

    freyr_camera<camera::perspective>(m_camera, "perspective")
            .def(py::init<vector3_type, float, float, int, int>(),
                 R"pbdoc(
        Create a perspective camera.

        Parameters
        ----------
        position : vector3_type
            The position of the camera in 3D space.
        fov : float
            The field of view angle, in degrees.
        aspect_ratio : float
            The aspect ratio of the view.
        width : int
            The width of the rendered image.
        height : int
            The height of the rendered image.

        Returns
        -------
        perspective
            An instance of a perspective camera.

        )pbdoc",
                 "position"_a,
                 "fov"_a,
                 "aspect_ratio"_a,
                 "width"_a,
                 "height"_a);

    freyr_camera<camera::orthographic>(m_camera, "orthographic")
            .def(py::init<vector3_type, float, float, int, int>(),
                 R"pbdoc(
        Create an orthographic camera.

        Parameters
        ----------
        position : vector3_type
            The position of the camera in 3D space.
        sov : float
            Size of view.
        aspect_ratio : float
            The aspect ratio of the view.
        width : int
            The width of the rendered image.
        height : int
            The height of the rendered image.

        Returns
        -------
        orthographic
            An instance of an orthographic camera.

        )pbdoc",
                 "position"_a,
                 "sov"_a,
                 "aspect_ratio"_a,
                 "width"_a,
                 "height"_a);

    freyr_camera<camera::fisheye>(m_camera, "fisheye")
            .def(py::init<vector3_type, float, float, int, int>(),
                 R"pbdoc(
        Create a fisheye camera.

        Parameters
        ----------
        position : vector3_type
            The position of the camera in 3D space.
        aov : float
            Angle of view, in degrees.
        aspect_ratio : float
            The aspect ratio of the view.
        width : int
            The width of the rendered image.
        height : int
            The height of the rendered image.

        Returns
        -------
        fisheye
            An instance of a fisheye camera.

        )pbdoc",
                 "position"_a,
                 "aov"_a,
                 "aspect_ratio"_a,
                 "width"_a,
                 "height"_a);

    // Define the ray class
    using ray_type = ray<ray_policy<Eigen::Vector3f, float>>;
    py::class_<ray_type>(m, "ray")
            .def(py::init<const Eigen::Vector3f &, const Eigen::Vector3f &>(),
                 R"pbdoc(
        Initialize a ray with an origin and a direction.

        Parameters
        ----------
        origin : Eigen::Vector3f
            The origin point of the ray.
        direction : Eigen::Vector3f
            The direction vector of the ray.
        )pbdoc",
                 "origin"_a,
                 "direction"_a)
            .def("get_origin", &ray_type::get_origin, "Retrieve the origin of the ray.")
            .def("set_origin", &ray_type::set_origin, "Set a new origin for the ray.")
            .def("get_direction", &ray_type::get_direction, "Retrieve the direction of the ray.")
            .def("get_point", &ray_type::get_point, "Get a point along the ray.")
            .def("set_direction", &ray_type::set_direction, "Set a new direction for the ray.");

    // Define shape types
    using shape_type = shape::shape_type;
    py::class_<shape_type>(m_types, "shape_type");

    // Define shape submodule
    auto m_shape = m.def_submodule("shape", "Submodule for various geometric shapes.");

    using box_type = Eigen::AlignedBox<float, 3>;
    py::class_<box_type>(m_shape, "aligned_box")
            .def(py::init<vector3_type, vector3_type>(),
                 R"pbdoc(
        Initialize an aligned box.

        Parameters
        ----------
        min : vector3_type
            The minimum corner of the box.
        max : vector3_type
            The maximum corner of the box.
        )pbdoc",
                 "min"_a,
                 "max"_a);

#ifdef FREYR_VIRTUAL_BASE_CLASSES
    freyr_shape<shape::base_shape, py_base_shape>(m_shape, "base_shape")
            .def(py::init<vector3_type>(),
                 R"pbdoc(
        Initialize a base shape with a position.

        Parameters
        ----------
        position : vector3_type
            The position of the shape in 3D space.
        )pbdoc",
                 "position"_a);
#endif

    freyr_shape<shape::sphere>(m_shape, "sphere")
            .def(py::init<vector3_type, float>(),
                 R"pbdoc(
        Initialize a sphere with a position and radius.

        Parameters
        ----------
        position : vector3_type
            The center position of the sphere.
        radius : float
            The radius of the sphere.
        )pbdoc",
                 "position"_a,
                 "radius"_a);

    freyr_shape<shape::plane>(m_shape, "plane")
            .def(py::init<vector3_type, vector3_type>(),
                 R"pbdoc(
        Initialize a plane with a position and normal vector.

        Parameters
        ----------
        position : vector3_type
            A point on the plane.
        normal : vector3_type
            The normal vector to the plane.
        )pbdoc",
                 "position"_a,
                 "normal"_a);

    using vertices_type = std::vector<vector3_type>;
    using faces_type    = std::vector<std::array<int, 3>>;

    freyr_shape<shape::mesh>(m_shape, "mesh")
            .def(py::init<vertices_type, faces_type, vector3_type>(),
                 R"pbdoc(
        Initialize a mesh shape with vertices, faces, and position.

        Parameters
        ----------
        vertices : list of vector3_type
            The vertices of the mesh.
        faces : list of arrays of int
            The faces of the mesh, each represented as an array of 3 vertex indices.
        position : vector3_type
            The position of the mesh in 3D space.
        )pbdoc",
                 "vertices"_a,
                 "faces"_a,
                 "position"_a);

    using light_type = light::light_type;
    py::class_<light_type>(m_types, "light_type");

    auto m_light = m.def_submodule("light",
                                   "Light submodule for handling various types of light sources.");

    using color_type = Eigen::Vector3f;

    freyr_light<light::point>(m_light, "point")
            .def(py::init<color_type, vector3_type, vector3_type, float, float, float>(),
                 R"pbdoc(
        Initialize a point light source.

        Parameters
        ----------
        position : vector3_type
            The 3D position of the point light source in the scene.
        color : vector3_type, optional
            The RGB color of the point light. Default is [1.0, 1.0, 1.0].
        intensity : vector3_type, optional
            The RGB intensity of the point light. Default is [1.0, 1.0, 1.0].
        c1 : float, optional
            The constant term in the attenuation formula. Default is 1.0.
        c2 : float, optional
            The linear term in the attenuation formula. Default is 1.0.
        c3 : float, optional
            The quadratic term in the attenuation formula. Default is 0.2.

        Notes
        -----
        The light attenuation is calculated as follows:
        `attenuation = 1 / (c1 + c2 * distance + c3 * distance^2)`
        )pbdoc",
                 "position"_a,
                 "color"_a     = vector3_type{1.0f, 1.0f, 1.0f},
                 "intensity"_a = vector3_type{1.0f, 1.0f, 1.0f},
                 "c1"_a        = 1.0f,
                 "c2"_a        = 1.0f,
                 "c3"_a        = 0.2f);

    freyr_light<light::directional>(m_light, "directional")
            .def(py::init<color_type, vector3_type, vector3_type>(),
                 R"pbdoc(
        Initialize a directional light source.

        Parameters
        ----------
        direction : vector3_type
            The direction vector of the light source.
        color : vector3_type, optional
            The RGB color of the directional light. Default is [1.0, 1.0, 1.0].
        intensity : vector3_type, optional
            The RGB intensity of the directional light. Default is [1.0, 1.0, 1.0].
        )pbdoc",
                 "direction"_a,
                 "color"_a     = vector3_type{1.0f, 1.0f, 1.0f},
                 "intensity"_a = vector3_type{1.0f, 1.0f, 1.0f});

    // Define Scene submodule
    auto m_scene
            = m.def_submodule("scene", "Submodule related to the scene settings and entities.");

    // Global Settings Class
    py::class_<scene::global_settings>(m_scene, "global_settings")
            .def(py::init<>(),
                 R"pbdoc(
        Initialize global settings for the scene.
        )pbdoc")
            .def_readwrite("ambient_intensity",
                           &scene::global_settings::ambient_intensity,
                           "The ambient intensity of the scene.");

    auto m_normal = m.def_submodule("normal", "Submodule related to the normal map.");

    // Normal Map Class
    //    explicit tensor_based(Eigen::Tensor<float, 3> normal_data,
    //                          float                   u_scale   = 1.0f,
    //                          float                   v_scale   = 1.0f,
    //                          tiling_strategy_type    tiling    = repeat_tiling(),
    //                          float                   amplitude = 1.0f)
    //        : normal_map<tensor_based>(u_scale, v_scale, tiling),
    //          m_normal_data(std::move(normal_data)), m_amplitude(amplitude) {}
    using image_type = Eigen::Tensor<float, 3>;
    py::class_<normal::tensor_based>(m_normal, "tensor_based")
            .def(py::init<image_type, float, float, float>(),
                 "data"_a,
                 "amplitude"_a = 1.0f,
                 "u_scale"_a   = 1.0f,
                 "v_scale"_a   = 1.0f);

    // Material Class
    py::class_<material::material>(m_scene, "material")
            .def(py::init<>(),
                 R"pbdoc(
        Initialize the material properties.
        )pbdoc")
            .def_readwrite("normal", &material::material::normal, "Normal map of the material.")
            .def_readwrite("emission",
                           &material::material::emission,
                           "Emission color of the material.")
            .def_readwrite("shininess",
                           &material::material::shininess,
                           "Shininess of the material.")
            .def_readwrite("fresnel",
                           &material::material::fresnel,
                           "Fresnel index of the material.")
            .def_readwrite("ambient",
                           &material::material::ambient,
                           "Ambient color of the material.")
            .def_readwrite("diffuse",
                           &material::material::diffuse,
                           "Diffuse color of the material.")
            .def_readwrite("specular",
                           &material::material::specular,
                           "Specular color of the material.");


    // Entity Class
    py::class_<entity::entity>(m_scene, "entity")
            .def(py::init<shape::shape_type, material::material>(),
                 R"pbdoc(
        Initialize an entity with a plane shape and material.

        Parameters
        ----------
        shape : shape::plane
            The shape of the entity.
        material : material::material
            The material properties of the entity.
        )pbdoc",
                 "shape"_a,
                 "material"_a)
            .def_readwrite("shape", &entity::entity::shape, "The shape of the entity.")
            .def_readwrite("material",
                           &entity::entity::material,
                           "The material properties of the entity.");

    // Define Texture submodule
    auto m_texture = m.def_submodule("texture", "Submodule related to textures.");

    // Texture Types
    py::class_<texture::texture_type>(m_types, "texture_type");

    // Constant Texture
    freyr_texture<texture::constant>(m_texture, "constant")
            .def(py::init<color_type>(),
                 R"pbdoc(
        Initialize a constant texture with a single color.

        Parameters
        ----------
        color : color_type
            The color of the texture.
        )pbdoc",
                 "color"_a);

    // Checkerboard Texture
    freyr_texture<texture::checkerboard>(m_texture, "checkerboard")
            .def(py::init<color_type, color_type, float, float>(),
                 R"pbdoc(
        Initialize a checkerboard texture with two colors and scales along the U and V axes.

        Parameters
        ----------
        color1 : color_type
            The first color in the checkerboard pattern.
        color2 : color_type
            The second color in the checkerboard pattern.
        u_scale : float
            The scale along the U axis.
        v_scale : float
            The scale along the V axis.
        )pbdoc",
                 "color1"_a  = color_type{1.0f, 1.0f, 1.0f},
                 "color2"_a  = color_type{0.0f, 0.0f, 0.0f},
                 "u_scale"_a = 1.0f,
                 "v_scale"_a = 1.0f)
            .def_property("color1",
                          &texture::checkerboard::get_color1,
                          &texture::checkerboard::set_color1,
                          "The first color in the checkerboard pattern.")
            .def_property("color2",
                          &texture::checkerboard::get_color2,
                          &texture::checkerboard::set_color2,
                          "The second color in the checkerboard pattern.")
            .def(
                    "__imul__",
                    [](texture::checkerboard &self, float scale) {
                        self.set_color1(self.get_color1() * scale);
                        self.set_color2(self.get_color2() * scale);
                        return self;
                    },
                    R"pbdoc(
        In-place scaling of both checkerboard colors.

        Parameters
        ----------
        f : float
            The scaling factor.
        )pbdoc",
                    "f"_a);

    py::class_<scene::scene>(m_scene, "scene")
            .def(py::init<camera::perspective>(), "camera"_a)
            .def(py::init<camera::fisheye>(), "camera"_a)
            .def(py::init<camera::orthographic>(), "camera"_a)
            .def("add_entity",
                 [](scene::scene &scene, shape::mesh &shape) {
                     scene.add_entity(entity::entity{shape});
                 })
            .def("add_entity",
                 [](scene::scene &scene, shape::sphere &shape) {
                     scene.add_entity(entity::entity{shape});
                 })
            .def("add_entity",
                 [](scene::scene &scene, shape::plane &shape) {
                     scene.add_entity(entity::entity{shape});
                 })
            .def("add_entity",
                 [](scene::scene &scene, entity::entity &entity) { scene.add_entity(entity); })
            .def("add_light",
                 [](scene::scene &scene, light::point &light) { scene.add_light(light); })
            .def("add_light",
                 [](scene::scene &scene, light::directional &light) { scene.add_light(light); });

    auto m_shader = m.def_submodule("shader", "Shader submodule");
    py::class_<shader::shader_node>(m_shader, "shader_node");

    py::class_<shader::shader_args>(m_shader, "shader_args");

    py::class_<shader::lambertian>(m_shader, "lambertian").def(py::init<>());
    py::class_<shader::phong>(m_shader, "phong").def(py::init<>());
    py::class_<shader::depth>(m_shader, "depth").def(py::init<>());

    //            .def(py::init<>())
    //            .def("apply", &shader::lambertian::apply, "shader_args"_a);
    //            .def("execute", &shader::shader::execute, "shader_args"_a);

    py::class_<shader::tbb_backend_adapter>(m_shader, "tbb_backend_adapter").def(py::init<>());

    using namespace freyr;
    py::class_<shader::graph>(m_shader, "graph")
            .def(py::init<>())
            .def("add_node", &shader::graph::add_node, "node"_a)
            .def(
                    "execute_single_pass",
                    [](shader::graph               &graph,
                       shader::scene_type          &scene,
                       shader::tbb_backend_adapter &adapter) {
                        return graph.execute_single_pass(scene, adapter);
                    },
                    "scene"_a,
                    "adapter"_a = shader::tbb_backend_adapter{});


    //            .def("add_camera", &scene::scene::add_camera)
    //            .def("get_shape", &scene::scene::get_shape)
    //            .def("get_light", &scene::scene::get_light)
    //            .def("get_camera", &scene::scene::get_camera)
    //            .def("get_shapes", &scene::scene::get_shapes)
    //            .def("get_lights", &scene::scene::get_lights);
    //            .def("get_cameras", &scene::scene::get_cameras)
    //            .def("get_shape_count", &scene::scene::get_shape_count)
    //            .def("get_light_count", &scene::scene::get_light_count)
    //            .def("get_camera_count", &scene::scene::get_camera_count);

    // Define the ray_soa class
    using ray_soa_type  = ray_soa<ray_policy<Eigen::Vector3f, float>>;
    using soa_data_type = typename ray_soa_type::data_type;
    py::class_<ray_soa_type>(m, "RaySOA")
            .def(py::init<>())// Default constructor
                              //            .def("transform", &ray_soa_type::transform)
                              //            .def("get_ray", &ray_soa_type::get_ray)
            .def("get_origins", &ray_soa_type::get_origins)
            .def("get_directions", &ray_soa_type::get_directions);
}
