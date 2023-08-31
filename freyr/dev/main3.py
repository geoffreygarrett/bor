from freyr.core import camera, shape, light, scene, shader, texture
from freyr.core import ray as Ray
import numpy as np
import time
from PIL import Image

camera_position = np.array([0, 0, 10], dtype=np.float32)
camera_width = w = 2888 * 2
camera_height = h = 1864 * 2
aspect_ratio = w / h

my_camera = camera.perspective(
    position=camera_position,
    fov=np.deg2rad(60),
    aspect_ratio=aspect_ratio,
    width=camera_width,
    height=camera_height,
)

# my_camera = camera.orthographic(
#     position=camera_position,
#     scale=10.0,
#     width=w,
#     height=h,
#     aspect_ratio=aspect_ratio,
# )

# #
# my_camera = camera.fisheye(
#     position=camera_position,
#     aov=np.deg2rad(180),
#     width=w,
#     height=h,
#     aspect_ratio=aspect_ratio,
# )

light_position = np.array([0, -1, 1], dtype=np.float32)
light_intensity = np.array([1, 1, 1], dtype=np.float32) * 0.5
light_color = np.array([1, 1, 1], dtype=np.float32)

# my_light = light.directional(
#     direction=-light_position,
#     intensity=light_intensity,
#     color=light_color,
# )

my_light = light.point(
    position=light_position * 26,
    intensity=light_intensity,
    color=light_color,
)


# Lambertian Shading
def lambertian_shading(normal, light_dir):
    dot_product = np.dot(normal, light_dir)
    dot_product = max(0, min(1, dot_product))
    return dot_product


# Phong Shading
def phong_shading(normal, light_dir, view_dir, specular_color, shininess):
    reflect_dir = 2.0 * np.dot(normal, light_dir) * normal - light_dir
    spec = np.maximum(np.dot(reflect_dir, view_dir), 0.0) ** shininess
    specular = spec * specular_color
    return specular


# Blinn-Phong Shading
def blinn_phong_shading(normal, light_dir, view_dir, specular_color, shininess):
    half_way = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
    spec = np.maximum(np.dot(normal, half_way), 0.0) ** shininess
    specular = spec * specular_color
    return specular


U, V = np.mgrid[0:h, 0:w].astype(np.float32)
output_image = np.zeros((h, w, 3), dtype=np.float32)

# Variables to adjust the shading model
pink = np.array([255, 182, 193], dtype=np.float32) / 255
ambient_color = np.ones(3, dtype=np.float32) * 0.3
diffuse_color = np.ones(3, dtype=np.float32) * 0.5
specular_color = np.ones(3, dtype=np.float32) * 0.9
shininess = 40

# Materials
my_material = scene.material()
my_material.ambient = ambient_color
my_material.diffuse = diffuse_color
my_material.specular = specular_color
my_material.shininess = shininess

my_material2 = scene.material()
my_material2.ambient = ambient_color
my_material2.diffuse = diffuse_color
my_material2.specular = specular_color
my_material2.shininess = shininess

my_material3 = scene.material()
my_material3.ambient = pink
my_material3.diffuse = pink
my_material3.specular = specular_color
my_material3.shininess = shininess

test_plane = scene.entity(
    shape=shape.plane(np.array([0, 0, -10], dtype=np.float32), np.array([0, 0, 2], dtype=np.float32)),
    material=my_material3)

test_plane2 = scene.entity(
    shape=shape.plane(np.array([0, -10, -10], dtype=np.float32), np.array([-0.5, 0, 1], dtype=np.float32)),
    material=my_material3)

test_sphere1 = scene.entity(
    shape=shape.sphere(np.array([0, 0, -2.0], dtype=np.float32), 2.0),
    material=my_material3)

test_sphere2 = scene.entity(
    shape=shape.sphere(np.array([0, 0, 1.], dtype=np.float32), 0.3),
    material=my_material3)

test_sphere3 = scene.entity(
    shape=shape.sphere(np.array([0, 0, -0.5], dtype=np.float32), 0.25),
    material=my_material)

my_scene = scene.scene(my_camera)
my_scene.add_light(my_light)
my_scene.add_entity(test_sphere2)
my_scene.add_entity(test_plane)
my_scene.add_entity(test_plane2)

# render_graph
my_shader_graph = shader.graph()
my_shader_graph.add_node(shader.phong())
# my_shader_graph.add_node(shader.lambertian())
time_render_cpp_start = time.time()
output_image_test = my_shader_graph.execute_single_pass(my_scene)
time_render_cpp_end = time.time()
print("Time taken for cpp render: ", time_render_cpp_end - time_render_cpp_start, " s")


image = output_image_test
image = (image * 255).clip(0, 255).astype('uint8')

# PIL
image = Image.fromarray(image, 'RGB')
image.show()
image.save('my_image.png')

# import time

# Choose shading model
# shading_model = "BlinnPhong"  # Options: "Lambertian", "Phong", "BlinnPhong"

# time_render_start = time.time()

#
# def find_closest_shape(ray, shapes):
#     closest_t = float("inf")
#     closest_shape = None
#
#     for shape in shapes:
#         if not shape.intersects_bounds(ray):
#             continue
#
#         it_hit, t_hit = shape.intersect_p(ray)
#         if it_hit and t_hit < closest_t:
#             closest_t = t_hit
#             closest_shape = shape
#
#     return closest_shape, closest_t
#
#
# def check_shadow(ray, shapes):
#     for shape in shapes:
#         if shape.intersects_bounds(ray):
#             it_hit, _ = shape.intersect_p(ray)
#             if it_hit:
#                 return True
#     return False

# Lambertian Shading
# def lambertian_shading(normal, light_dir):
#     dot_product = np.dot(normal, light_dir)
#     dot_product = max(0, min(1, dot_product))
#     return dot_product
#
#
# # Phong Shading
# def phong_shading(normal, light_dir, view_dir, specular_color, shininess):
#     reflect_dir = 2.0 * np.dot(normal, light_dir) * normal - light_dir
#     spec = np.maximum(np.dot(reflect_dir, view_dir), 0.0) ** shininess
#     specular = spec * specular_color
#     return specular
#
#
# # Blinn-Phong Shading
# def blinn_phong_shading(normal, light_dir, view_dir, specular_color, shininess):
#     half_way = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
#     spec = np.maximum(np.dot(normal, half_way), 0.0) ** shininess
#     specular = spec * specular_color
#     return specular

# def apply_shading(shading_model, closest_normal, light_direction, view_direction, diffuse_color, specular_color,
#                   shininess):
#     if shading_model == "Lambertian":
#         return lambertian_shading(closest_normal, light_direction) * diffuse_color
#     elif shading_model == "Phong":
#         return phong_shading(closest_normal, light_direction, view_direction, specular_color, shininess)
#     elif shading_model == "BlinnPhong":
#         return blinn_phong_shading(closest_normal, light_direction, view_direction, specular_color, shininess)
#     return np.array([0, 0, 0])

# def apply_advanced_shading(
#         shading_model, closest_normal, light_direction, view_direction,
#         ambient_color, diffuse_color, specular_color, shininess, fresnel_coefficient,
#         emissive_color, lights, transparent, reflectivity):
#     shading = np.zeros(3)
#
#     # Ambient lighting
#     shading += ambient_color
#
#     # Emissive materials
#     shading += emissive_color
#
#     # Standard shading models
#     shading += apply_shading(shading_model, closest_normal, light_direction, view_direction,
#                              diffuse_color, specular_color, shininess)
#
#     # Fresnel effect
#     fresnel_effect = fresnel_coefficient + (1 - fresnel_coefficient) * (
#             (1 - np.dot(view_direction, closest_normal)) ** 5)
#     shading *= fresnel_effect
#
#     # Multiple lights
#     for light in lights:
#         shading += apply_shading(shading_model, closest_normal, light.direction_from(closest_point), view_direction,
#                                  diffuse_color, specular_color, shininess)
#
#     if transparent:
#         # Apply transparency
#         shading *= 0.5  # Example: making it half transparent
#
#     if reflectivity > 0:
#         # Apply reflection effects (this is just a placeholder)
#         shading += reflectivity * np.array([0.8, 0.8, 0.8])
#
#     return shading

# # Sample shading variables
# ambient_color = np.array([0.2, 0.2, 0.2])
# diffuse_color = np.array([0.5, 0.5, 0.5])
# specular_color = np.array([1.0, 1.0, 1.0])
# shininess = 2.0
# fresnel_coefficient = 0.02
# emissive_color = np.array([0.1, 0.0, 0.0])
# lights = [my_light]  # Assuming my_light is your light object
# transparent = False
# reflectivity = 0.1

# for u, v in zip(U.flatten(), V.flatten()):
#     ray = my_camera.generate_ray(u, v)
#
#     closest_shape, closest_t = find_closest_shape(ray, shapes)
#
#     if closest_shape is not None:
#         closest_point = ray.get_point(closest_t)
#         closest_normal = closest_shape.get_normal_at(closest_point)
#         light_direction = my_light.direction_from(closest_point)
#         view_direction = -ray.get_direction()
#
#         shadow_ray = Ray(closest_point + 1e-4 * closest_normal, light_direction)
#         in_shadow = check_shadow(shadow_ray, shapes)
#
#         if in_shadow:
#             output_image[int(u), int(v)] = np.array([0, 0, 0])
#         else:
#             # shading = apply_shading(
#             #     shading_model, closest_normal, light_direction, view_direction,
#             #     diffuse_color, specular_color, shininess)
#             # In the main loop, replace apply_shading with apply_advanced_shading
#             shading = apply_advanced_shading(
#                 shading_model, closest_normal, light_direction, view_direction,
#                 ambient_color, diffuse_color, specular_color, shininess, fresnel_coefficient,
#                 emissive_color, lights, transparent, reflectivity)
#             output_image[int(u), int(v)] += shading

# time_render_end = time.time()
# print('time cost (rendering)', time_render_end - time_render_start, 's')
# import time
#
# # rays = camera.generate_rays(u, v)
# time_start = time.time()
# rays = my_camera.generate_rays(U, V)
# time_end = time.time()
# print('time cost', time_end - time_start, 's')
#
# # raise SystemExit
# # plot
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
#
# origins = rays.get_origins().T
# directions = rays.get_directions().T
#
# print(origins[:, 0])
# print(origins[:, 1])
# print(origins[:, 2])
# print(origins[:, 0].shape)

# points = np.vstack(points)
# points2 = np.vstack(points2)

# ax.quiver(origins[:, 0], origins[:, 1], origins[:, 2],
#           directions[:, 0], directions[:, 1], directions[:, 2])

# ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
# ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='b', marker='o')
#
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)
#
# plt.show()

# plot depth map

# def plot_image(image, name='output_image.png'):
#     fig = plt.figure()
#     ax = fig.add_subplot()
#
#     # Do not plot axes or white around
#     ax.set_axis_off()
#
#     # Plot aspect equal
#     ax.set_aspect('equal')
#
#     # Plot tight
#     # plt.axis('tight')
#
#     # Plot image
#     # ax.imshow(image)
#
#     # Show the plot
#     # plt.show()
#
#     # Save the figure, if needed
#     # fig.savefig(name, bbox_inches='tight', pad_inches=0, dpi=600)
#     # Create an image using Pillow\
#     print(image.min(), image.max())
#
#     # Normalize the array to be in the range [0, 255]
#     image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')
#
#     image = Image.fromarray(image, 'RGB')
#
#     # image_to_show = Image.fromarray(image, 'RGB')
#     image.show()
#
#     # Save the image
#     image.save('my_image.png')

#
# print(type(output_image))
# print(type(output_image_test))
# plot_image(output_image_test)

# ax.imshow(output_image)
# plt.show()

# # Suppose each ray is a 3D vector.
# rays = np.empty((512, 512, 3), dtype=np.float32)
#
# for i in range(512):
#     for j in range(512):
#         rays[i, j] = camera.generate_ray(i, j)
#
# print(rays.shape)
