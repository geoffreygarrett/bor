#ifndef FREYR_LIGHT_H
#define FREYR_LIGHT_H

#include <Eigen/Core>


namespace freyr::light {

    using vector3f_type = Eigen::Vector3f;
    using vector4f_type = Eigen::Vector4f;

    // Base Light Class using CRTP
    template<typename Derived>
    class base_light {
    public:
        using vector3f_type = Eigen::Vector3f;
        using color_type    = Eigen::Vector3f;// RGB color

        // Constructor
        explicit base_light(color_type    color     = vector3f_type(1.0f, 1.0f, 1.0f),
                            vector3f_type intensity = vector3f_type(1.0f, 1.0f, 1.0f))
            : color(std::move(color)), intensity(std::move(intensity)) {}

        // Getters and setters for color
        [[nodiscard]] color_type get_color() const { return color; }
        void                     set_color(const color_type &new_color) { color = new_color; }

        [[nodiscard]] vector3f_type intensity_at(const vector3f_type &point) const {
            return static_cast<const Derived *>(this)->intensity_at_impl(point).cwiseProduct(
                    color);
        }


        // Getters and setters for intensity
        [[nodiscard]] vector3f_type get_intensity() const { return intensity; }
        void set_intensity(const vector3f_type &new_intensity) { intensity = new_intensity; }

        [[nodiscard]] vector3f_type direction_from(const vector3f_type &point) const {
            return static_cast<const Derived *>(this)->direction_from_impl(point);
        }


    private:
        // The color of the light
        color_type    color;
        vector3f_type intensity;
    };

    // Point Light
    class point : public base_light<point> {
        vector3f_type position;
        float         c1, c2, c3;// drop-off coefficients

    public:
        // Constructor
        point(vector3f_type position,
              color_type    color     = vector3f_type(1.0f, 1.0f, 1.0f),
              vector3f_type intensity = vector3f_type(1.0f, 1.0f, 1.0f),
              float         c1        = 1.0f,
              float         c2        = 0.0f,
              float         c3        = 1.0f)
            : base_light<point>(color, intensity), position(std::move(position)), c1(c1), c2(c2),
              c3(c3) {}

        // Implementation of intensity at a point
        [[nodiscard]] vector3f_type intensity_at_impl(const vector3f_type &point) const {
            float dist = (point - position).norm();

            // TODO: find a better way to do this
            // cast to double before division to avoid floating point errors
            double attenuation = 1.0 / static_cast<double>(c1 + c2 * dist + c3 * dist * dist);
            return this->get_intensity() * static_cast<float>(attenuation);
        }

        // Implementation of light direction from a point
        [[nodiscard]] vector3f_type direction_from_impl(const vector3f_type &point) const {
            return (position - point).normalized();
        }

        // Getters and setters for position and intensity
        [[nodiscard]] vector3f_type get_position() const { return position; }
        void set_position(const vector3f_type &new_position) { position = new_position; }

        // Getters and setters for drop-off coefficients
        [[nodiscard]] std::tuple<float, float, float> get_dropoff_coefficients() const {
            return {c1, c2, c3};
        }
        void set_dropoff_coefficients(float new_c1, float new_c2, float new_c3) {
            c1 = new_c1;
            c2 = new_c2;
            c3 = new_c3;
        }
    };

    // Directional Light
    class directional : public base_light<directional> {
        vector3f_type direction;

    public:
        // Constructor
        directional(const vector3f_type &direction,
                    color_type           color     = vector3f_type(1.0f, 1.0f, 1.0f),
                    const vector3f_type &intensity = vector3f_type(1.0f, 1.0f, 1.0f))
            : base_light<directional>(std::move(color), intensity),
              direction(direction.normalized()) {}

        // Implementation of intensity at a point
        [[nodiscard]] vector3f_type intensity_at_impl(const vector3f_type & /* point */) const {
            return this->get_intensity();
        }

        // Implementation of light direction from a point
        [[nodiscard]] vector3f_type direction_from_impl(const vector3f_type & /* point */) const {
            return -direction;
        }

        // Getters and setters for direction and intensity
        [[nodiscard]] vector3f_type get_direction() const { return direction; }

        void set_direction(const vector3f_type &new_direction) {
            direction = new_direction.normalized();
        }
    };

    using light_type = std::variant<point, directional>;


}// namespace freyr::light

#endif// FREYR_LIGHT_H