#ifndef FREYR_SHAPE_H
#define FREYR_SHAPE_H

#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <freyr/ray.h>

namespace freyr::shape {


    using bounds3f_type = Eigen::AlignedBox3f;
    using vector3f_type = Eigen::Vector3f;
    using point3f_type  = Eigen::Vector3f;
    using ray_type      = ray<>;

    bool intersects_bounds(const ray_type &ray, const bounds3f_type &bounds) {
        float t_min = 0;// set to -INFINITY to get first hit on line
        float t_max = std::numeric_limits<float>::max();
        // set to max distance ray can travel (INFINITY)

        for (int i = 0; i < 3; ++i) {
            float inv_d = 1.0f / ray.get_direction()[i];
            float t0    = (bounds.min()[i] - ray.get_origin()[i]) * inv_d;
            float t1    = (bounds.max()[i] - ray.get_origin()[i]) * inv_d;
            if (inv_d < 0.0f) std::swap(t0, t1);
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min) return false;
        }
        return true;
    }


    template<typename Derived>
    class shape {
    public:
        using bounds3f_type = Eigen::AlignedBox3f;
        using ray_type      = ray<>;
        using ray_soa_type  = ray_soa<>;
        using vector3f_type = Eigen::Vector3f;
        using point3f_type  = Eigen::Vector3f;

        shape(vector3f_type position = vector3f_type::Zero()) : m_position(std::move(position)) {}

        shape(const shape &)            = default;
        shape(shape &&)                 = default;
        shape &operator=(const shape &) = default;
        shape &operator=(shape &&)      = default;

        // get u v coordinates for a point on the surface of the shape
        [[nodiscard]] std::tuple<float, float> get_uv_at(const vector3f_type &point) const {
            return static_cast<const Derived *>(this)->get_uv_coords_impl(point);
        }

        [[nodiscard]] bool intersect(const ray_type &ray) const {
            return static_cast<const Derived *>(this)->intersect_impl(ray);
        }

        [[nodiscard]] std::tuple<bool, float> intersect_p(const ray_type &ray) const {
            return static_cast<const Derived *>(this)->intersect_p_impl(ray);
        }

        [[nodiscard]] vector3f_type get_normal(const vector3f_type &point) const {
            return static_cast<const Derived *>(this)->get_normal_impl(point);
        }

        [[nodiscard]] vector3f_type get_tangent(const vector3f_type &point) const {
            return static_cast<const Derived *>(this)->get_tangent_impl(point);
        }

        [[nodiscard]] Eigen::Matrix3f get_tbn_matrix(const vector3f_type &point) const {
            // Get the normal at the point
            vector3f_type normal = this->get_normal(point);

            // Get the tangent at the point. This would be implemented in the derived class.
            vector3f_type tangent = static_cast<const Derived *>(this)->get_tangent_impl(point);

            // Compute the bitangent
            vector3f_type bitangent = normal.cross(tangent);

            // Create the TBN matrix
            Eigen::Matrix3f tbn;
            tbn.col(0) = tangent.normalized();
            tbn.col(1) = bitangent.normalized();
            tbn.col(2) = normal.normalized();

            return tbn;
        }


        [[nodiscard]] bounds3f_type object_bounds() const {
            return static_cast<const Derived *>(this)->object_bounds_impl();
        }

        [[nodiscard]] bool intersects_bounds(const ray_type &ray) const {
            return freyr::shape::intersects_bounds(ray, this->object_bounds());
        }

        void set_position(const vector3f_type &position) { this->m_position = position; }
        [[nodiscard]] vector3f_type get_position() const { return m_position; }


    protected:
        point3f_type m_position;
    };

    bool intersect_triangle(const ray_type      &r,
                            const vector3f_type &v0,
                            const vector3f_type &v1,
                            const vector3f_type &v2,
                            float               &t) {
        vector3f_type h, s, q;
        float         a, f, u, v;

        vector3f_type edge1 = v1 - v0;
        vector3f_type edge2 = v2 - v0;

        h = r.get_direction().cross(edge2);
        a = edge1.dot(h);

        if (a > -1e-6 && a < 1e-6) { return false; }

        f = 1.0 / a;
        s = r.get_origin() - v0;
        u = f * s.dot(h);

        if (u < 0.0 || u > 1.0) { return false; }

        q = s.cross(edge1);
        v = f * r.get_direction().dot(q);

        if (v < 0.0 || u + v > 1.0) { return false; }

        t = f * edge2.dot(q);

        if (t > 1e-6) { return true; }

        return false;
    }

    class mesh : public shape<mesh> {
    public:
        std::vector<vector3f_type>      vertices;
        std::vector<std::array<int, 3>> faces;       // Assuming triangular faces
        mutable int last_intersected_face_index = -1;// Store the last intersected face index

        mesh(std::vector<vector3f_type>      vertices,
             std::vector<std::array<int, 3>> faces,
             vector3f_type                   position = vector3f_type(0, 0, 0))
            : shape<mesh>(std::move(position)), vertices(std::move(vertices)),
              faces(std::move(faces)) {}

        mesh() = default;

        [[nodiscard]] std::tuple<float, float>
        get_uv_coords_impl(const vector3f_type &point) const {
            if (last_intersected_face_index == -1) { return {0, 0}; }

            const auto &face = faces[last_intersected_face_index];
            const auto &v0   = vertices[face[0]];
            const auto &v1   = vertices[face[1]];
            const auto &v2   = vertices[face[2]];

            // Compute vectors
            const auto v0v1 = v1 - v0;
            const auto v0v2 = v2 - v0;
            const auto p    = point - v0;

            // Compute dot products
            const auto dot00 = v0v2.dot(v0v2);
            const auto dot01 = v0v2.dot(v0v1);
            const auto dot02 = v0v2.dot(p);
            const auto dot11 = v0v1.dot(v0v1);
            const auto dot12 = v0v1.dot(p);

            // Compute barycentric coordinates
            const auto inv_denom = 1 / (dot00 * dot11 - dot01 * dot01);
            const auto u         = (dot11 * dot02 - dot01 * dot12) * inv_denom;
            const auto v         = (dot00 * dot12 - dot01 * dot02) * inv_denom;

            return {u, v};
        }

        [[nodiscard]] std::tuple<bool, float> intersect_p_impl(const ray_type &r) const {
            float closest_t             = std::numeric_limits<float>::max();
            bool  hit                   = false;
            last_intersected_face_index = -1;// Reset the last intersected face index

            for (size_t i = 0; i < faces.size(); ++i) {
                const auto &face = faces[i];
                float       t;
                if (intersect_triangle(r,
                                       m_position + vertices[face[0]],
                                       m_position + vertices[face[1]],
                                       m_position + vertices[face[2]],
                                       t)) {
                    if (t < closest_t) {
                        closest_t                   = t;
                        hit                         = true;
                        last_intersected_face_index = static_cast<int>(i);// Store the index
                    }
                }
            }

            return {hit, closest_t};
        }

        [[nodiscard]] bool intersect_impl(const ray_type &r) const {
            float closest_t             = std::numeric_limits<float>::max();
            bool  hit                   = false;
            last_intersected_face_index = -1;// Reset the last intersected face index

            for (size_t i = 0; i < faces.size(); ++i) {
                const auto &face = faces[i];
                float       t;
                if (intersect_triangle(r,
                                       m_position + vertices[face[0]],
                                       m_position + vertices[face[1]],
                                       m_position + vertices[face[2]],
                                       t)) {
                    if (t < closest_t) {
                        closest_t                   = t;
                        hit                         = true;
                        last_intersected_face_index = static_cast<int>(i);// Store the index
                    }
                }
            }

            return hit;
        }

        [[nodiscard]] bounds3f_type object_bounds_impl() const {
            bounds3f_type bounds;
            for (const auto &vertex: vertices) { bounds.extend(m_position + vertex); }
            return bounds;
        }

        [[nodiscard]] vector3f_type get_normal_impl(const vector3f_type &point) const {
            if (last_intersected_face_index == -1) {
                // No face was intersected
                return {0.0f, 0.0f, 0.0f};
            }

            const auto   &face   = faces[last_intersected_face_index];
            vector3f_type v1     = vertices[face[1]] - vertices[face[0]];
            vector3f_type v2     = vertices[face[2]] - vertices[face[0]];
            vector3f_type normal = v1.cross(v2);

            return normal;// Normalizing is often a good idea, depending on your application
        }

        [[nodiscard]] vector3f_type get_tangent_impl(const vector3f_type &point) const {
            if (last_intersected_face_index == -1) {
                // No face was intersected
                return {0.0f, 0.0f, 0.0f};
            }

            const auto   &face = faces[last_intersected_face_index];
            vector3f_type v1   = vertices[face[1]] - vertices[face[0]];
            return v1;// Normalizing is often a good idea, depending on your application
        }
    };

    class plane : public shape<plane> {
        vector3f_type normal;

    public:
        using shape<plane>::shape;

        plane(vector3f_type position, vector3f_type normal)
            : shape(std::move(position)), normal(std::move(normal)) {}

        [[nodiscard]] bool intersect_impl(const ray_type &ray) const {
            float denom = normal.dot(ray.get_direction());
            if (std::abs(denom) > 1e-6) {
                float t = (m_position - ray.get_origin()).dot(normal) / denom;
                if (t >= 0) { return true; }
            }
            return false;
        }

        [[nodiscard]] std::tuple<float, float>
        get_uv_coords_impl(const vector3f_type &point) const {
            vector3f_type u_basis, v_basis;

            // Explicitly orthonormalize the basis vectors
            if (std::abs(normal.x()) > std::abs(normal.y())) {
                u_basis = vector3f_type{-normal.z(), 0, normal.x()};
            } else {
                u_basis = vector3f_type{0, normal.z(), -normal.y()};
            }

            u_basis.normalize();
            v_basis = normal.cross(u_basis).normalized();

            vector3f_type point_vec = point - m_position;
            float         u         = point_vec.dot(u_basis);
            float         v         = point_vec.dot(v_basis);

            return std::make_tuple(u, v);
        }


        [[nodiscard]] vector3f_type get_normal_impl(const vector3f_type &point) const {
            return normal;
        }

        [[nodiscard]] vector3f_type get_tangent_impl(const vector3f_type & /*point*/) const {
            vector3f_type u_basis;

            // Create a u_basis vector that's orthogonal to the plane's normal
            if (std::abs(normal.x()) > std::abs(normal.y())) {
                u_basis = vector3f_type{-normal.z(), 0, normal.x()};
            } else {
                u_basis = vector3f_type{0, normal.z(), -normal.y()};
            }

            // Explicitly normalize the u_basis vector
            u_basis.normalize();

            return u_basis;
        }


        [[nodiscard]] std::tuple<bool, float> intersect_p_impl(const ray_type &ray) const {
            float denom = normal.dot(ray.get_direction());
            if (std::abs(denom) > 1e-6) {
                float t = (m_position - ray.get_origin()).dot(normal) / denom;
                if (t >= 0) { return {true, t}; }
            }
            return {false, std::numeric_limits<float>::max()};
        }

        [[nodiscard]] bounds3f_type object_bounds_impl() const {
            return {m_position
                            - vector3f_type(std::numeric_limits<float>::max(),
                                            std::numeric_limits<float>::max(),
                                            std::numeric_limits<float>::max()),
                    m_position
                            + vector3f_type(std::numeric_limits<float>::max(),
                                            std::numeric_limits<float>::max(),
                                            std::numeric_limits<float>::max())};
        }
    };


    // Sphere class
    class sphere : public shape<sphere> {
        float radius;

    public:
        using shape<sphere>::shape;

        sphere(vector3f_type position, float radius)
            : shape(std::move(position)), radius(radius) {}

        // make copyable
        sphere(const sphere &) = default;

        [[nodiscard]] std::tuple<float, float>
        get_uv_coords_impl(const vector3f_type &point) const {
            // Get the vector pointing from the center of the sphere to the point
            vector3f_type point_vec = point - m_position;

            // Normalize point_vec to get a point on the unit sphere
            point_vec.normalize();

            // Compute the spherical coordinates of the point
            float theta = std::atan2(point_vec.y(), point_vec.x());// Range [-pi, pi]
            float phi   = std::acos(point_vec.z());                // Range [0, pi]

            // Convert spherical coordinates to UV coordinates
            float u = (theta + M_PI) / (2 * M_PI);// Range [0, 1]
            float v = phi / M_PI;                 // Range [0, 1]

            //            std::cout<<u<<" "<<v<<std::endl;

            return std::make_tuple(u, v);
        }


        [[nodiscard]] std::tuple<bool, float> intersect_p_impl(const ray_type &ray) const {
            vector3f_type oc = ray.get_origin() - m_position;

            // Coefficients for quadratic equation ax^2 + bx + c = 0
            float a = ray.get_direction().dot(ray.get_direction());
            float b = 2.0f * oc.dot(ray.get_direction());
            float c = oc.dot(oc) - radius * radius;

            float discriminant = b * b - 4 * a * c;

            // No real roots means no intersection
            if (discriminant < 0) { return {false, std::numeric_limits<float>::max()}; }

            // Calculate the two possible intersections
            float t1 = (-b - std::sqrt(discriminant)) / (2 * a);
            float t2 = (-b + std::sqrt(discriminant)) / (2 * a);

            // If either t1 or t2 is in the range [0, infinity), the ray intersects the sphere
            float t = (t1 < 0) ? t2 : ((t2 < 0) ? t1 : std::min(t1, t2));

            if (t >= 0) { return {true, t}; }

            return {false, std::numeric_limits<float>::max()};
        }

        [[nodiscard]] bool intersect_impl(const ray_type &ray) const {
            vector3f_type oc = ray.get_origin() - m_position;

            // Coefficients for quadratic equation ax^2 + bx + c = 0
            float a = ray.get_direction().dot(ray.get_direction());
            float b = 2.0f * oc.dot(ray.get_direction());
            float c = oc.dot(oc) - radius * radius;

            float discriminant = b * b - 4 * a * c;

            // No real roots means no intersection
            if (discriminant < 0) { return false; }

            // Calculate the two possible intersections
            float t1 = (-b - std::sqrt(discriminant)) / (2 * a);
            float t2 = (-b + std::sqrt(discriminant)) / (2 * a);

            // If either t1 or t2 is in the range [0, infinity), the ray intersects the sphere
            if (t1 >= 0 || t2 >= 0) { return true; }

            return false;
        }

        [[nodiscard]] vector3f_type get_normal_impl(const vector3f_type &point) const {
            return (point - m_position).normalized();
        }

        [[nodiscard]] vector3f_type get_tangent_impl(const vector3f_type &point) const {
            // return tangent at point, on sphere
            vector3f_type point_vec = point - m_position;
            vector3f_type tangent   = point_vec.cross(vector3f_type(0, 0, 1));
            if (tangent.norm() < 1e-6) { tangent = point_vec.cross(vector3f_type(0, 1, 0)); }
            return tangent.normalized();
        }

        [[nodiscard]] bounds3f_type object_bounds_impl() const {
            return {m_position - vector3f_type(radius, radius, radius),
                    m_position + vector3f_type(radius, radius, radius)};
        }
    };

//#define FREYR_VIRTUAL_BASE_CLASSES
#ifdef FREYR_VIRTUAL_BASE_CLASSES
    // Virtual Base Class for Python polymorphism that derives from Shape
    class base_shape : public shape<base_shape> {
    public:
        explicit base_shape(vector3f_type position) : shape(std::move(position)) {}
        virtual ~base_shape() = default;
        //        virtual bool                    intersects_bounds_impl() const = 0;
        virtual std::tuple<bool, float> intersect_p_impl(const ray_type &ray) const = 0;
        virtual vector3f_type           get_normal_impl(vector3f_type point) const  = 0;
        virtual bool                    intersect_impl(const ray_type &ray) const   = 0;
        virtual bounds3f_type           object_bounds_impl() const                  = 0;
    };

    using shape_type = std::variant<plane, sphere, mesh, std::shared_ptr<base_shape>>;
#else
    using shape_type = std::variant<plane, sphere, mesh>;
#endif

}// namespace freyr::shape

#endif// FREYR_SHAPE_H