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

my_camera = camera.orthographic(
    position=camera_position,
    scale=10.0,
    width=w,
    height=h,
    aspect_ratio=aspect_ratio,
)

# #
# my_camera = camera.fisheye(
#     position=camera_position,
#     aov=np.deg2rad(180),
#     width=w,
#     height=h,
#     aspect_ratio=aspect_ratio,
# )

light_position = np.array([0, -1, 1], dtype=np.float32)
light_intensity = np.array([1, 1, 1], dtype=np.float32)
light_color = np.array([1, 1, 1], dtype=np.float32)

my_light = light.directional(
    direction=-light_position,
    intensity=light_intensity,
    color=light_color,
)

my_light = light.point(
    position=light_position * 8,
    intensity=light_intensity,
    color=light_color,
    c1=0.0,
    c2=0.01,
    c3=0.005,
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
# ambient_color = pink * 0.5
diffuse_color = np.ones(3, dtype=np.float32) * 0.5
# diffuse_color = pink * 0.5
specular_color = np.ones(3, dtype=np.float32) * 0.9
# specular_color = pink * 0.5
shininess = 20

# Materials
my_texture1 = texture.checkerboard(
    color1=specular_color,
    color2=specular_color * 0.8,
    u_scale=0.01,
    v_scale=0.01)

my_material1 = scene.material()
my_material1.ambient = texture.checkerboard(
    color1=specular_color * 0.3,
    color2=specular_color * 0.8 * 0.3,
    u_scale=0.01,
    v_scale=0.01)
my_material1.diffuse = texture.checkerboard(
    color1=specular_color * 0.5,
    color2=specular_color * 0.8 * 0.5,
    u_scale=0.01,
    v_scale=0.01)
my_material1.specular = texture.checkerboard(
    color1=specular_color,
    color2=specular_color * 0.8,
    u_scale=0.01,
    v_scale=0.01)

my_material1.shininess = 0

my_material2 = scene.material()
# my_material2.ambient = pink * 0.2
my_material2.ambient = texture.checkerboard(
    color1=pink * 0.3,
    color2=np.array([0.9, 0.9, 0.9], dtype=np.float32) * 0.3,
    u_scale=0.01,
    v_scale=0.01
)
# my_material2.diffuse = pink * 0.5
my_material2.diffuse = texture.checkerboard(
    color1=pink * 0.5,
    color2=np.array([0.9, 0.9, 0.9], dtype=np.float32) * 0.5,
    u_scale=0.01,
    v_scale=0.01
)
# my_material2.specular = pink * 0.5
my_material2.specular = texture.checkerboard(
    color1=pink,
    color2=np.array([0.9, 0.9, 0.9], dtype=np.float32),
    u_scale=0.01,
    v_scale=0.01
)
my_material2.shininess = 0
my_material3 = scene.material()
my_material3.ambient = pink * 0.1
my_material3.diffuse = pink * 0.5
# my_material3.diffuse = texture.checkerboard(
#     color1=pink,
#     color2=np.array([0.9, 0.9, 0.9], dtype=np.float32),
#     u_scale=0.01,
#     v_scale=0.01
# )
my_material3.specular = pink * 0.5
# my_material3.specular = texture.checkerboard(
#     color1=pink,
#     color2=np.array([0.9, 0.9, 0.9], dtype=np.float32),
#     u_scale=0.01,
#     v_scale=0.01
# )
my_material3.shininess = shininess * 3

# my_material3.diffuse = texture.checkerboard(
#     color1=pink,
#     color2=np.array([0.9, 0.9, 0.9], dtype=np.float32),
#     u_scale=1.0,
#     v_scale=1.0
# )

test_plane1 = scene.entity(
    shape=shape.plane(np.array([0, 0, -10], dtype=np.float32), np.array([0, 0, 2], dtype=np.float32)),
    material=my_material2)

test_plane2 = scene.entity(
    shape=shape.plane(np.array([0, -10, -10], dtype=np.float32), np.array([-0.5, 0, 1], dtype=np.float32)),
    material=my_material1)

test_sphere1 = scene.entity(
    shape=shape.sphere(np.array([0, 0, -2.0], dtype=np.float32), 2.0),
    material=my_material3)

test_sphere2 = scene.entity(
    shape=shape.sphere(np.array([0, 0, 1.], dtype=np.float32), 0.3),
    material=my_material3)

test_sphere3 = scene.entity(
    shape=shape.sphere(np.array([0, 0, -0.5], dtype=np.float32), 0.25),
    material=my_material1)

my_scene = scene.scene(my_camera)
my_scene.add_light(my_light)
my_scene.add_entity(test_sphere1)
# my_scene.add_entity(test_sphere2)
my_scene.add_entity(test_plane1)
my_scene.add_entity(test_plane2)

# render_graph
my_shader_graph = shader.graph()
my_shader_graph.add_node(shader.phong())
# my_shader_graph.add_node(shader.lambertian())
# my_shader = shader.depth()
# my_shader_graph.add_node(my_shader)
time_render_cpp_start = time.time()
output_image_test = my_shader_graph.execute_single_pass(my_scene)
time_render_cpp_end = time.time()
print("Time taken for cpp render: ", time_render_cpp_end - time_render_cpp_start, " s")

print("output_image_test: ", output_image_test)
image = output_image_test
#
# image = (image - np.min(image)) / (np.max(image) - np.min(image))
# print(np.max(image))
# print(np.min(image))
image = (image * 255).clip(0, 255).astype('uint8')

# PIL
image = Image.fromarray(image, 'RGB')
image.show()
image.save('my_image.png')
