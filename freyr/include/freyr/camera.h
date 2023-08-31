#ifndef FREYR_CAMERA_H
#define FREYR_CAMERA_H

#include <utility>

#include <Eigen/Core>// Eigen is used for vector types, replace with your own if needed
#include <Eigen/Geometry>
#include <freyr/algorithms/generate_rays.h>
#include <freyr/common.h>
#include <freyr/ray.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace freyr::camera {

    //        using vector3_type<float><float>    = Eigen::Vector3f;   // Alias for 3D vector type
    using quaternion_type = Eigen::Quaternionf;// Alias for quaternion type

    float compute_scale_from_fov(float fov, float focal_length) {
        return 2 * focal_length * std::tan(fov / 2);
    }

    float compute_fov_from_scale(float scale, float focal_length) {
        return 2 * std::atan(scale / (2 * focal_length));
    }

    template<typename Derived, typename RayPolicy = ray_policy<Eigen::Vector3f, float>>
    class camera {
    public:
        using ray_type     = ray<RayPolicy>;
        using rays_type    = ray_soa<RayPolicy>;
        using scalar_type  = typename RayPolicy::scalar_type;
        using vector3_type = vector3_type<float>;

        camera(vector3_type        pos,
               float               aspect,
               int                 w,
               int                 h,
               quaternion_type     q       = quaternion_type::Identity(),
               const vector3_type &up      = vector3_type::UnitY(),
               const vector3_type &forward = vector3_type::UnitZ(),
               const vector3_type &right   = vector3_type::UnitX())
            : position(std::move(pos)), orientation(std::move(q)), aspect_ratio(aspect), width(w),
              height(h), up(up), forward(forward), right(right) {}


        [[nodiscard]] ray_type generate_ray(float u, float v) const {
            return static_cast<const Derived *>(this)->generate_ray_impl(u, v);
        }

        [[nodiscard]] rays_type generate_rays(Eigen::ArrayXf uu, Eigen::ArrayXf vv) const {
            Eigen::Matrix<float, 3, Eigen::Dynamic> origins(3, uu.size());
            Eigen::Matrix<float, 3, Eigen::Dynamic> directions(3, uu.size());

            for (int i = 0; i < uu.size(); ++i) {
                const auto ray
                        = static_cast<const Derived *>(this)->generate_ray_impl(uu[i], vv[i]);
                directions.col(i) = ray.get_direction();
                origins.col(i)    = ray.get_origin();
            }

            return {origins, directions};
        }

        [[nodiscard]] rays_type generate_rays(const Eigen::Tensor<float, 2> &uu,
                                              const Eigen::Tensor<float, 2> &vv) const {

            assert(uu.dimension(0) == vv.dimension(0)), "uu and vv must have the same size";
            const auto num_rays = uu.size();

            Eigen::Matrix<float, 3, Eigen::Dynamic> origins(3, num_rays);
            Eigen::Matrix<float, 3, Eigen::Dynamic> directions(3, num_rays);

            int len_u = static_cast<int>(uu.dimension(0));
            int len_v = static_cast<int>(uu.dimension(1));

            tbb::parallel_for(tbb::blocked_range2d<int, int>(0, len_u, 0, len_v),
                              [=, this, &uu, &vv, &directions, &origins](
                                      const tbb::blocked_range2d<int, int> &r) {
                                  for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
                                      for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                                          const auto ray = static_cast<const Derived *>(this)
                                                                   ->generate_ray_impl(uu(i, j),
                                                                                       vv(i, j));
                                          directions.col(i * len_v + j) = ray.get_direction();
                                          origins.col(i * len_v + j)    = ray.get_origin();
                                      }
                                  }
                              });

            return {origins, directions};
        }

        // Common setter and getter methods for all derived classes
        void               set_position(vector3_type pos) { position = std::move(pos); }
        void               set_aspect_ratio(float a) { aspect_ratio = a; }
        void               set_width(int w) { width = w; }
        void               set_height(int h) { height = h; }
        [[nodiscard]] auto get_position() const { return position; }
        [[nodiscard]] auto get_aspect_ratio() const { return aspect_ratio; }
        [[nodiscard]] auto get_width() const { return width; }
        [[nodiscard]] auto get_height() const { return height; }
        void               set_orientation(quaternion_type q) { orientation = std::move(q); }
        [[nodiscard]] auto get_orientation() const { return orientation; }
        void               translate(const vector3_type &t) { position += t; }
        void               rotate(const quaternion_type &q) { orientation = q * orientation; }


    protected:
        vector3_type    position;
        vector3_type    up;
        vector3_type    forward;
        vector3_type    right;
        quaternion_type orientation;
        float           aspect_ratio;
        int             width;
        int             height;
    };

    // Derived class for perspective camera
    class perspective : public camera<perspective> {
    public:
        perspective(vector3_type           pos,
                    float                  fov,
                    float                  aspect,
                    int                    w,
                    int                    h,
                    const quaternion_type &q       = quaternion_type::Identity(),
                    const vector3_type    &up      = vector3_type::UnitY(),
                    const vector3_type    &forward = vector3_type::UnitZ(),
                    const vector3_type    &right   = vector3_type::UnitX())
            : camera<perspective>(std::move(pos), aspect, w, h, q, up, forward, right), fov(fov) {}

        [[nodiscard]] ray_type generate_ray_impl(float u, float v) const {
            auto direction = perspective_ray_generator::generate_ray_static(u,
                                                                            v,
                                                                            width,
                                                                            height,
                                                                            fov,
                                                                            aspect_ratio);
            return {position, direction};
        }

    private:
        float fov;
    };


    // Derived class for orthographic camera
    class orthographic : public camera<orthographic> {
    public:
        orthographic(vector3_type           pos,
                     float                  scale,
                     float                  aspect,
                     int                    w,
                     int                    h,
                     const quaternion_type &q       = quaternion_type::Identity(),
                     const vector3_type    &up      = vector3_type::UnitY(),
                     const vector3_type    &forward = vector3_type::UnitZ(),
                     const vector3_type    &right   = vector3_type::UnitX())
            : camera<orthographic>(std::move(pos), aspect, w, h, q, up, forward, right),
              scale(scale) {}

        [[nodiscard]] ray_type generate_ray_impl(float u, float v) const {
            auto origin = orthographic_ray_generator::generate_ray_static(u,
                                                                          v,
                                                                          width,
                                                                          height,
                                                                          scale,
                                                                          aspect_ratio);

            return {
                    position + origin,
                    {0, 0, -1}
            };
        }


    private:
        float scale;
    };


    class fisheye : public camera<fisheye> {
    public:
        fisheye(vector3_type           pos,
                float                  angle_of_view,
                float                  aspect,
                int                    w,
                int                    h,
                const quaternion_type &q       = quaternion_type::Identity(),
                const vector3_type    &up      = vector3_type::UnitY(),
                const vector3_type    &forward = vector3_type::UnitZ(),
                const vector3_type    &right   = vector3_type::UnitX())
            : camera<fisheye>(std::move(pos), aspect, w, h, q, up, forward, right),
              angle_of_view(angle_of_view) {}

        [[nodiscard]] ray_type generate_ray_impl(float u, float v) const {
            auto direction = fisheye_ray_generator::generate_ray_static(u,
                                                                        v,
                                                                        width,
                                                                        height,
                                                                        angle_of_view,
                                                                        aspect_ratio);

            return {position, direction};
        }

    private:
        float angle_of_view;
    };

    using camera_type = std::variant<perspective, orthographic, fisheye>;


}// namespace freyr::camera

#endif// FREYR_CAMERA_H
