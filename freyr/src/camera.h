#ifndef FREYR_CAMERA_H
#define FREYR_CAMERA_H

namespace freyr {

    // Base class for all cameras
    template<typename Derived>
    class Camera {
    protected:
        Vector3 position;
        float   fov;
        float   aspect_ratio;
        int     width;
        int     height;
        Vector3 light_position;

    public:
        Camera(const Vector3 &position, float fov, float aspect_ratio, int width, int height, const Vector3 &light_position)
            : position(position), fov(fov), aspect_ratio(aspect_ratio), width(width), height(height), light_position(light_position) {}

        Ray generate_ray(int x, int y) const {
            return static_cast<const Derived *>(this)->generate_ray(x, y);
        }

        void render_scene(const std::vector<Shape<Derived> *> &shapes) {
            static_cast<Derived *>(this)->render_scene(shapes);
        }
    };

    // Monochrome camera class
    class MonochromeCamera : public Camera<MonochromeCamera> {
    public:
        using Camera::Camera;

        Ray generate_ray(int x, int y) const {
            // Compute ray for a given pixel
        }

        void render_scene(const std::vector<Shape<Derived> *> &shapes) {
            // Render the scene as a monochrome image
        }
    };

}// namespace freyr

#endif// FREYR_CAMERA_H