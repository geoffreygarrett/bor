#include <iostream>

#include <freyr/core.h>

// The base class (non-virtual, to stick with CRTP)
template<typename Derived>
struct camera_base {
    auto get_width() const { return static_cast<const Derived *>(this)->get_width_impl(); }
    auto get_height() const { return static_cast<const Derived *>(this)->get_height_impl(); }
    auto test(int a) const { return static_cast<const Derived *>(this)->test_impl(a); }
};

// Derived classes
struct camera_a : camera_base<camera_a> {
    auto get_width_impl() const { return 1920; }
    auto get_height_impl() const { return 1080; }
    auto test_impl(int a) const { return a + 1; }
};

struct camera_b : camera_base<camera_b> {
    auto get_width_impl() const { return 1280; }
    auto get_height_impl() const { return 720; }
    auto test_impl(int a) const { return a + 2; }
};



int main() {
    using camera_type = std::variant<camera_a, camera_b>;

    camera_type my_camera = camera_b();

    auto width  = CRTP_VAR_CALL(my_camera, get_width)();
    auto height = CRTP_VAR_CALL(my_camera, get_height)();
    auto test   = CRTP_VAR_CALL(my_camera, test)(1);

    // The output would be based on the currently active variant.
    std::cout << "Width: " << width << "\nHeight: " << height << "\nTest: " << test << std::endl;

    return 0;
}