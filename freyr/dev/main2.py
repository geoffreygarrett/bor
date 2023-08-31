import numpy as np
from PIL import Image
from freyr.core import camera, shape, light, scene, texture, shader

# Initialize Camera Parameters
camera_position = np.array([0, 0, 10], dtype=np.float32)
camera_width, camera_height = 2888 * 2, 1864 * 2
aspect_ratio = camera_width / camera_height

# Initialize Light Parameters
light_position = np.array([0, -0.75, 1], dtype=np.float32)
light_intensity = light_color = np.array([1, 1, 1], dtype=np.float32) * 1.0

# Initialize Image Parameters
U, V = np.mgrid[0:camera_height, 0:camera_width].astype(np.float32)
output_image = np.zeros((camera_height, camera_width, 3), dtype=np.float32)

# Initialize Color and Material Parameters
pink = np.array([255, 182, 193], dtype=np.float32) / 255
blue = np.array([173, 216, 230], dtype=np.float32) / 255

# Create Cameras
my_perspective_camera = camera.perspective(
    position=camera_position, fov=np.deg2rad(60),
    aspect_ratio=aspect_ratio, width=camera_width, height=camera_height
)

my_orthographic_camera = camera.orthographic(
    position=camera_position, sov=10.0,
    width=camera_width, height=camera_height, aspect_ratio=aspect_ratio
)

my_fisheye_camera = camera.fisheye(
    position=camera_position, aov=np.deg2rad(180),
    width=camera_width, height=camera_height, aspect_ratio=aspect_ratio
)

# Create Lights
my_directional_light = light.directional(
    direction=-light_position, intensity=light_intensity, color=light_color
)

my_point_light = light.point(
    position=light_position * 8, intensity=light_intensity,
    color=light_color, c1=0.0, c2=0.00, c3=0.008
)

# Create Materials
my_material1, my_material2, my_material3 = scene.material(), scene.material(), scene.material()

checker_config = {'u_scale': 0.01, 'v_scale': 0.01}

# Configure Materials
my_material1.ambient = my_material1.diffuse = my_material1.specular = texture.checkerboard(
    color1=blue * 0.7, color2=np.ones(3), **checker_config
)

my_material2.ambient = my_material2.diffuse = my_material2.specular = texture.checkerboard(
    color1=pink * 0.7, color2=np.ones(3), **checker_config
)

c_ambient = 0.1
c_diffuse = 0.5
c_specular = 1.0

my_material1.ambient *= c_ambient
my_material1.diffuse *= c_diffuse
my_material1.specular *= c_specular
my_material1.shininess = 3.0

my_material2.ambient *= c_ambient
my_material2.diffuse *= c_diffuse
my_material2.specular *= c_specular
my_material2.shininess = 3.0

my_material3.ambient = pink * c_ambient
my_material3.diffuse = pink * c_diffuse
my_material3.specular = pink * c_specular
my_material3.shininess = 32.0

# Create Scene
my_scene = scene.scene(my_perspective_camera)
# my_scene = scene.scene(my_fisheye_camera)
my_scene.add_light(my_point_light)

# Add Entities to Scene
my_scene.add_entity(scene.entity(
    shape=shape.plane(np.array([0, 0, -10], dtype=np.float32), np.array([1.5, 0, 2], dtype=np.float32)),
    material=my_material1)
)

my_scene.add_entity(scene.entity(
    shape=shape.plane(np.array([0, 0, -10], dtype=np.float32), np.array([-1.5, 0, 2], dtype=np.float32)),
    material=my_material2)
)

my_scene.add_entity(scene.entity(
    shape=shape.sphere(np.array([0, -2.0, -2.0], dtype=np.float32), 2.0),
    material=my_material3)
)

# Render Scene
my_shader_graph = shader.graph()
my_shader_graph.add_node(shader.phong())
# my_shader_graph.add_node(shader.lambertian())
image = my_shader_graph.execute_single_pass(my_scene)
image = (image * 255).clip(0, 255).astype('uint8')

# Save Image
image = Image.fromarray(image, 'RGB')
image.show()
image.save('my_image.png')
