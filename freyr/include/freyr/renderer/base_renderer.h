#ifndef BASE_RENDERER_H
#define BASE_RENDERER_H

namespace freyr {

    template<typename Derived, typename Image, typename Scene, typename Camera>
    class base_renderer {

    public:
        using image_type  = Image;
        using scene_type  = Scene;
        using camera_type = Camera;

        // This function provides the generic rendering interface
        void render(image_type &image, scene_type &scene, camera_type &camera) {
            // static_cast ensures that we're calling the function of the correct derived type
            static_cast<Derived *>(this)->render_impl(image, scene, camera);
        }
    };

#endif// BASE_RENDERER_H
}