import numpy as np
import subprocess
from PIL import Image
from freyr.core import camera, shape, light, scene, texture, shader

# Get screen size
from screeninfo import get_monitors

screen_width, screen_height = 0, 0
for m in get_monitors():
    screen_width, screen_height = m.width, m.height

# Path to Eros mesh files
FACES_FILE = './freyr/examples/data/Eros.face'
NODES_FILE = './freyr/examples/data/Eros.node'

# Initialize empty lists for faces and vertices
faces = []
vertices = []

# Read faces from the faces.txt file
with open(FACES_FILE, "r") as f:
    for line in f:
        # Split the line into integers and append to the faces list
        face = list(map(int, line.strip().split()))
        faces.append(face[1:4])

# Read vertices from the vertices.txt file
with open(NODES_FILE, "r") as f:
    for line in f:
        # Split the line into a vertex ID and coordinates, then append to the vertices list
        vertex_id, x, y, z = map(float, line.strip().split())
        # vertex = {"id": int(vertex_id), "x": x, "y": y, "z": z}
        vertex = [x, y, z]
        vertices.append(np.array(vertex, dtype=np.float32))

# Convert the lists to numpy arrays
eros_shape = shape.mesh(faces=faces, vertices=vertices, position=np.array([0, 0, 0], dtype=np.float32))

# Initialize Camera Parameters
camera_position = np.array([0, 0, 1.5], dtype=np.float32)

# Get  current screen size
downscale_factor = 12
camera_width = screen_width // downscale_factor
camera_height = screen_height // downscale_factor
aspect_ratio = camera_width / camera_height

# Initialize Light Parameters
light_position = np.array([0, -0.75, 1], dtype=np.float32)
light_intensity = light_color = np.array([1, 1, 1.0], dtype=np.float32) * 1.0

# Initialize Image Parameters
U, V = np.mgrid[0:camera_height, 0:camera_width].astype(np.float32)
output_image = np.zeros((camera_height, camera_width, 3), dtype=np.float32)

# Initialize Color and Material Parameters
gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)

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
    color=light_color, c1=0.0, c2=0.00, c3=0.012
)

# Create Materials
my_material1 = scene.material()

c_ambient = 0.1
c_diffuse = 1.0
c_specular = 1.0

my_material1.ambient = gray * c_ambient
my_material1.diffuse = gray * c_diffuse
my_material1.specular = gray * c_specular
my_material1.shininess = 0.0

# Create Scene
my_scene = scene.scene(my_perspective_camera)
# my_scene = scene.scene(my_orthographic_camera)
# my_scene = scene.scene(my_fisheye_camera)
my_scene.add_light(my_point_light)

# Add Entities to Scene
my_scene.add_entity(scene.entity(
    shape=eros_shape,
    material=my_material1)
)

# Render Scene
my_shader_graph = shader.graph()
# my_shader_graph.add_node(shader.lambertian())
my_shader_graph.add_node(shader.phong())
image = my_shader_graph.execute_single_pass(my_scene)
image = (image * 255).clip(0, 255).astype('uint8')

# upscale to screen size
screen_width, screen_height = screen_width, screen_height
image = np.repeat(np.repeat(image, screen_width // camera_width, axis=1), screen_height // camera_height, axis=0)

# Save Image
image = Image.fromarray(image, 'RGB')
image.show()
image.save('my_image.png')
