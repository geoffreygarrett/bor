#ifndef SIMPLE_RENDERER_H
#define SIMPLE_RENDERER_H

#include "base_renderer.h"

namespace freyr {

    template<typename Image, typename Scene, typename Camera>
    class simple_renderer : public base_renderer<simple_renderer<Image, Scene, Camera>, Image, Scene, Camera> {
    public:
        // Override this method to provide the actual rendering logic
        void render_impl(Image &image, Scene &scene, Camera &camera) {
            // Your rendering logic here
            // This function can freely modify the 'image' based on 'scene' and 'camera'
        }
    };

}// namespace freyr
#endif// SIMPLE_RENDERER_H
