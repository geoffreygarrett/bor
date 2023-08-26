#ifndef FREYR_SHAPE_H
#define FREYR_SHAPE_H

namespace freyr {
#include <vector>

    // Base class for all 3D objects
    template<typename Derived>
    class Shape {
    public:
        bool intersect(const Ray &ray) const {
            return static_cast<const Derived *>(this)->intersect(ray);
        }

        Vector3 get_normal(const Vector3 &point) const {
            return static_cast<const Derived *>(this)->get_normal(point);
        }

        float get_albedo() const {
            return static_cast<const Derived *>(this)->get_albedo();
        }
    };

    // Sphere class
    class Sphere : public Shape<Sphere> {
        Vector3 center;
        float   radius;
        float   albedo;

    public:
        Sphere(const Vector3 &center, float radius, float albedo)
            : center(center), radius(radius), albedo(albedo) {}

        bool intersect(const Ray &ray) const {
            // Compute ray-sphere intersection
        }

        Vector3 get_normal(const Vector3 &point) const {
            // Compute surface normal at a point
        }

        float get_albedo() const {
            return albedo;
        }
    };

}// namespace freyr::camera

#endif// FREYR_SHAPE_H