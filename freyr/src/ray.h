#ifndef FREYR_RAY_H
#define FREYR_RAY_H

// https://chat.openai.com/c/e48577cd-6a58-48cb-984f-bb52731d561b
// https://chat.openai.com/c/1f9db774-684a-4b05-9842-a469bb802e14
// https://chat.openai.com/c/7c911409-cc66-42ec-885e-7eb10b81f482

namespace freyr {

    // Ray class
    class Ray {
        Vector3 origin;
        Vector3 direction;

    public:
        Ray(const Vector3 &origin, const Vector3 &direction)
            : origin(origin), direction(direction) {}
    };

    // Function for computing shading
    float compute_shading(const Vector3 &normal, const Vector3 &light_direction, float albedo) {
        // Compute shading based on the surface normal, light direction, and albedo
    }

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

    class RayTracer {
        // Your ray tracer implementation...
    };

    class TraceRays {
        RayTracer &rayTracer;
        Image     &image;

    public:
        TraceRays(RayTracer &rt, Image &im) : rayTracer(rt), image(im) {}

        void operator()(const tbb::blocked_range2d<size_t> &range) const {
            for (size_t y = range.rows().begin(); y != range.rows().end(); ++y) {
                for (size_t x = range.cols().begin(); x != range.cols().end(); ++x) {
                    Ray   ray   = rayTracer.generateRay(x, y);
                    Color color = rayTracer.traceRay(ray);
                    image.setPixel(x, y, color);
                }
            }
        }
    };

    int main() {
        RayTracer rayTracer;
        Image     image(width, height);

        // Trace rays in parallel with TBB
        tbb::parallel_for(tbb::blocked_range2d<size_t>(0, height, 0, width), TraceRays(rayTracer, image));

        // Now `image` contains the result of the ray tracing.
        // ...
    }

}// namespace freyr