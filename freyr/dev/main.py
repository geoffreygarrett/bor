from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)


def map_normal(u, v, normal_map):
    x, y = u, v
    height, width, _ = normal_map.shape

    # Convert uv coordinates to pixel coordinates
    x = int(x * (width - 1))
    y = int(y * (height - 1))

    # Fetch the color from the normal map
    normal_color = normal_map[y, x]

    # Convert it from [0, 255] to [-1, 1]
    normal_vector = normal_color / 127.5 - 1

    return normal_vector


class Material:
    def __init__(self, albedo, Ks, n, color=(1, 1, 1), normal_map=None):
        self.albedo = np.array(albedo)  # For diffuse reflection
        self.Ks = np.array(Ks)  # For specular reflection
        self.n = n  # Shininess factor
        self.color = np.array(color)
        self.normal_map = normal_map  # Store the normal map if any

    def get_effective_color(self, u, v):
        return self.albedo * self.color

    def get_mapped_normal(self, u, v):
        if self.normal_map is not None:
            # Implement your mapping function here
            return map_normal(u, v, self.normal_map)
        else:
            return None


class Shape(ABC):
    def __init__(self, normal_map=None):
        self.normal_map = normal_map

    @abstractmethod
    def intersect(self, ray):
        pass

    @abstractmethod
    def get_normal(self, point):
        pass

    @abstractmethod
    def get_uv_coords(self, point):
        pass

    def get_modified_normal(self, point):
        normal = self.get_normal(point)
        if self.normal_map is not None:
            u, v = self.get_uv_coords(point)

            # Get dimensions directly from the normal_map shape
            normal_map_height, normal_map_width = self.normal_map.shape[:2]

            # Scale UV coordinates
            u_scaled = u * normal_map_width
            v_scaled = v * normal_map_height

            # Wrap around (Tiling)
            u_index = int(u_scaled) % normal_map_width
            v_index = int(v_scaled) % normal_map_height

            # Alternatively, Clamping
            # u_index = min(int(u_scaled), normal_map_width - 1)
            # v_index = min(int(v_scaled), normal_map_height - 1)

            # Get normal modification from the normal map
            normal_modification = self.normal_map[v_index, u_index]

            # Normalizing the modification value
            normal_modification = normal_modification / np.linalg.norm(normal_modification)

            # Modify the original normal
            modified_normal = normal + normal_modification  # Use either addition or some other blending method

            # Normalize the modified normal
            modified_normal = modified_normal / np.linalg.norm(modified_normal)

            return modified_normal
        else:
            return normal


# New Object Class
class Entity:
    def __init__(self, shape, material):
        self.shape = shape
        self.material = material


# Abstract Base Camera Class
class Camera(ABC):
    def __init__(self, position, aspect_ratio, width, height, channel_type="rgb"):
        self.position = np.array(position)
        self.aspect_ratio = aspect_ratio
        self.width = width
        self.height = height
        self.channel_type = channel_type

    @property
    def direction(self):
        return np.array([0, 0, -1])

    @property
    def right(self):
        return np.array([1, 0, 0])

    @property
    def up(self):
        return np.array([0, 1, 0])

    @abstractmethod
    def generate_ray(self, u, v):
        pass


# Perspective Camera Class
class PerspectiveCamera(Camera):
    def __init__(self, position, fov, *args, **kwargs):
        super().__init__(position, *args, **kwargs)
        self.fov = fov

    def generate_ray(self, u, v):
        direction = self.direction + u * self.fov * self.right + v * self.fov * self.up
        direction = direction / np.linalg.norm(direction)  # Normalize
        return Ray(self.position, direction)


# Orthographic Camera Class
class OrthographicCamera(Camera):
    def __init__(self, position, scale, *args, **kwargs):
        super().__init__(position, *args, **kwargs)
        self.scale = scale

    def generate_ray(self, u, v):
        origin = self.position + u * self.scale * self.right + v * self.scale * self.up
        return Ray(origin, self.direction)


class Sphere(Shape):
    def __init__(self, center, radius, normal_map=None):
        self.center = center
        self.radius = radius
        super().__init__(normal_map)

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return np.inf, np.inf
        else:
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            return t1, t2

    def get_normal(self, point):
        return (point - self.center) / self.radius

    def get_uv_coords(self, point):
        normal = self.get_normal(point)
        phi = np.arctan2(normal[1], normal[0])
        theta = np.arccos(normal[2])

        u = 1 - (phi + np.pi) / (2 * np.pi)
        v = (theta + np.pi / 2) / np.pi

        u = int(u * self.normal_map.shape[1])
        v = int(v * self.normal_map.shape[0])

        return u, v


class Light:
    def __init__(self, position):
        self.position = position


class SamplingMethod(ABC):
    def __init__(self, samples_per_pixel):
        self.samples_per_pixel = samples_per_pixel

    @abstractmethod
    def sample(self, i, j, width, height):
        pass


class SimpleRandomSampling(SamplingMethod):
    def sample(self, i, j, width, height):
        samples = []
        for _ in range(self.samples_per_pixel):
            u = (i + np.random.rand()) / width - 0.5
            v = (j + np.random.rand()) / height - 0.5
            samples.append((u, v))
        return samples


class SimpleGridSampling(SamplingMethod):

    def sample(self, i, j, width, height):
        samples = []
        grid = np.meshgrid(np.linspace(0, 1, int(np.sqrt(self.samples_per_pixel)), endpoint=False),
                           np.linspace(0, 1, int(np.sqrt(self.samples_per_pixel)), endpoint=False))
        for u, v in zip(grid[0].flatten(), grid[1].flatten()):
            u = (i + u) / width - 0.5
            v = (j + v) / height - 0.5
            samples.append((u, v))
        return samples


class StratifiedJitteredSampling(SamplingMethod):
    def __init__(self, samples_per_pixel):
        super().__init__(samples_per_pixel)
        self.sub_pixels = int(np.sqrt(samples_per_pixel))

    def sample(self, i, j, width, height):
        samples = []
        sub_pixel_size = 1 / self.sub_pixels
        for sub_i in range(self.sub_pixels):
            for sub_j in range(self.sub_pixels):
                u = (i + (sub_i + np.random.rand() * sub_pixel_size)) / width - 0.5
                v = (j + (sub_j + np.random.rand() * sub_pixel_size)) / height - 0.5
                samples.append((u, v))
        return samples


class BaseRenderer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def render(self, scene, camera):
        pass


# class Renderer2(BaseRenderer):
#
#     def __init__(self, sampling_method, color_converter=None):
#         self.sampling_method = sampling_method
#         self.color_converter = color_converter if color_converter else self.default_converter
#
#     def default_converter(self, color):
#         return np.dot(color, [0.2989, 0.5870, 0.1140])
#
#     def cook_torrance_brdf(self, normal, light_direction, view_direction, material):
#         # In 1982, Robert Cook and Kenneth Torrance published a reflectance model that more accurately represented the physical reality of light reflectance than the Phong and Blinn-Phong models.
#         #
#         # The proposed method is general purpose - it has three “pluggable” components that can be replaced with equations of your choice. It is also effective at representing a variety of materials, whereas a BRDF like Blinn-Phong is really only good at representing plastics and some metals (though that point is debatable). As such, it is still in common usage today. Though that is perhaps a moot point, since the Phong reflection model is a weak approximation from 1975 that is still in common usage. Regardless…
#         roughness = 0.6
#         fresnel_reflectance = 0.3
#         epsilon = 1e-5
#
#         # Ensure vectors are normalized
#         normal = normalize(normal)
#         light_direction = normalize(light_direction)
#         view_direction = normalize(view_direction)
#
#         # Microfacet normal
#         h = normalize(light_direction + view_direction)
#
#         # Fresnel equation term (Schlick approximation)
#         fresnel = fresnel_reflectance + (1 - fresnel_reflectance) * np.power(1 - np.dot(view_direction, h), 5)
#
#         # Normal Distribution Function (Trowbridge-Reitz GGX)
#         alpha = roughness * roughness
#         dot_nh = np.clip(np.dot(normal, h), epsilon, 1 - epsilon)
#         ndf = (alpha ** 2) / (np.pi * ((dot_nh ** 2) * (alpha ** 2 - 1) + 1) ** 2)
#
#         # Geometry function (Smith's method with GGX)
#         k = roughness * roughness / 2
#         geometry_light = 2 / (1 + np.sqrt(1 + (1 - k) ** 2 / (k + np.dot(normal, light_direction)) ** 2))
#         geometry_view = 2 / (1 + np.sqrt(1 + (1 - k) ** 2 / (k + np.dot(normal, view_direction)) ** 2))
#         geometry = geometry_light * geometry_view
#
#         # Final BRDF computation
#         brdf = (fresnel * ndf * geometry) / (4 * np.dot(normal, light_direction) * np.dot(normal, view_direction))
#
#         return brdf
#
#     def compute_lighting(self, point, normal, material, scene, camera):
#         intensity = 50
#         if self.is_in_shadow(point, scene):
#             return np.zeros(3)  # No light if in shadow
#
#         light_direction = normalize(scene.light.position - point)
#         view_direction = normalize(camera.position - point)
#
#         brdf = self.cook_torrance_brdf(normal, light_direction, view_direction, material)
#         # color = brdf * material.get_effective_color() * scene.light.intensity
#         color = brdf * material.get_effective_color() * intensity
#         return color
#
#     def is_in_shadow(self, point, scene):
#         offset = 1e-6
#         direction_to_light = (scene.light.position - point).astype(float)
#         direction_to_light /= np.linalg.norm(direction_to_light)
#         shadow_ray_origin = point + offset * direction_to_light
#         shadow_ray = Ray(shadow_ray_origin, direction_to_light)
#         distance_to_light = np.linalg.norm(scene.light.position - point)
#         for entity in scene.entities:
#             t1, t2 = entity.shape.intersect(shadow_ray)
#             if 0 < t1 < distance_to_light or 0 < t2 < distance_to_light:
#                 return True
#         return False
#
#     def render(self, scene, camera):
#         if camera.channel_type == 'rgb':
#             image = np.zeros((camera.height, camera.width, 3))
#         else:
#             image = np.zeros((camera.height, camera.width))
#
#         for j in range(camera.height):
#             for i in range(camera.width):
#                 color = np.zeros(3)
#                 samples = self.sampling_method.sample(i, j, camera.width, camera.height)
#                 for u, v in samples:
#                     ray = camera.generate_ray(u, v)
#                     closest_t = np.inf
#                     closest_entity = None
#
#                     for entity in scene.entities:
#                         t1, t2 = entity.shape.intersect(ray)
#                         if 0 < t1 < closest_t:
#                             closest_t = t1
#                             closest_entity = entity
#
#                     if closest_entity is not None:
#                         intersection_point = ray.origin + closest_t * ray.direction
#                         normal = closest_entity.shape.get_normal(intersection_point)
#                         material = closest_entity.material
#                         lighting = self.compute_lighting(intersection_point, normal, material, scene, camera)
#                         color += lighting
#
#                 if camera.channel_type == 'monochrome':
#                     color = self.color_converter(color)
#                 image[j, i] = color / self.sampling_method.samples_per_pixel
#
#         return image
#     # Existing methods like render(), is_in_shadow(), etc. remain the same


def normalize(vector):
    return vector / np.linalg.norm(vector)


class Renderer(BaseRenderer):
    def __init__(self, sampling_method, color_converter=None):
        self.sampling_method = sampling_method
        self.color_converter = color_converter if color_converter else self.default_converter

    def default_converter(self, color):
        # Convert to grayscale by default
        return np.dot(color, [0.2989, 0.5870, 0.1140])

    def is_in_shadow(self, point, scene):
        offset = 1e-6
        direction_to_light = (scene.light.position - point).astype(float)
        direction_to_light /= np.linalg.norm(direction_to_light)
        shadow_ray_origin = point + offset * direction_to_light
        shadow_ray = Ray(shadow_ray_origin, direction_to_light)
        distance_to_light = np.linalg.norm(scene.light.position - point)
        for entity in scene.entities:
            t1, t2 = entity.shape.intersect(shadow_ray)
            if 0 < t1 < distance_to_light or 0 < t2 < distance_to_light:
                return True
        return False

    # def compute_lighting(self, point, normal, material, scene):
    #     if self.is_in_shadow(point, scene):
    #         return np.zeros(3)  # No light if in shadow
    #     light_direction = (scene.light.position - point).astype(float)
    #     light_direction /= np.linalg.norm(light_direction)
    #     color = max(0, np.dot(normal, light_direction)) * material.get_effective_color()
    #     return color

    def compute_lighting(self, point, normal, material, effective_color, scene, camera):
        if self.is_in_shadow(point, scene):
            return np.zeros(3)  # No light if in shadow

        light_direction = (scene.light.position - point).astype(float)
        light_direction /= np.linalg.norm(light_direction)

        # Diffuse term
        diffuse = max(0, np.dot(normal, light_direction)) * effective_color

        # Specular term
        view_direction = (camera.position - point).astype(float)
        view_direction /= np.linalg.norm(view_direction)
        reflection_direction = 2 * np.dot(light_direction, normal) * normal - light_direction
        reflection_direction /= np.linalg.norm(reflection_direction)
        specular = material.Ks * np.power(max(0, np.dot(reflection_direction, view_direction)), material.n)

        return diffuse + specular

    def render(self, scene, camera):
        if camera.channel_type == 'rgb':
            image = np.zeros((camera.height, camera.width, 3))
        else:
            image = np.zeros((camera.height, camera.width))

        for j in range(camera.height):
            for i in range(camera.width):
                color = np.zeros(3)
                samples = self.sampling_method.sample(i, j, camera.width, camera.height)
                for u, v in samples:
                    ray = camera.generate_ray(u, v)
                    closest_t = np.inf
                    closest_entity = None

                    for entity in scene.entities:
                        t1, t2 = entity.shape.intersect(ray)
                        if 0 < t1 < closest_t:
                            closest_t = t1
                            closest_entity = entity

                    # In your Renderer
                    if closest_entity is not None:
                        intersection_point = ray.origin + closest_t * ray.direction
                        normal = closest_entity.shape.get_modified_normal(intersection_point)
                        # u, v = closest_entity.shape.get_uv_coords(intersection_point)
                        material = closest_entity.material
                        effective_color = material.get_effective_color(0, 0)
                        lighting = self.compute_lighting(intersection_point, normal, material, effective_color, scene,
                                                         camera)
                        color += lighting

                if camera.channel_type == 'monochrome':
                    color = self.color_converter(color)
                image[j, i] = color / self.sampling_method.samples_per_pixel

        return image


# class MonochromeCamera(Camera):
#     def __init__(self, position, fov, aspect_ratio, width, height, advanced_shadow=False):
#         super().__init__(position, fov, aspect_ratio, width, height)
#         self.advanced_shadow = advanced_shadow
#
#     def generate_ray(self, u, v):
#         direction = self.direction + u * self.right + v * self.up
#         return Ray(self.position, direction)


class Scene:
    def __init__(self, entities, light):
        self.entities = entities
        self.light = light


def get_scene(n):
    if n == 1:

        # Define the light source
        light_position = np.array([-2, -2, 200])
        light = Light(light_position)
        # Define the shapes in the scene (spheres in this case)
        sphere1 = Sphere(np.array([-1, -1, -5]), 1, 1)
        sphere2 = Sphere(np.array([1, 1, -90]), 10, 0.5)

        objects = [Entity(sphere1, Material(albedo=0.5, color=np.array([0.5, 0.5, 0.5]), Ks=0.5, n=1)),
                   Entity(sphere2, Material(albedo=0.5, color=np.array([0, 1, 0]), Ks=0.5, n=1))]

        # Define the scene by combining shapes and light
        return Scene(objects, light)

    elif n == 2:
        # generate a grid of spheres
        # Define the light source
        light_position = np.array([0, 0, 200])
        # light_position = np.array([-2, -2, 10])
        light = Light(light_position)

        # Sphere positions
        sphere_pos_x = np.linspace(-2, 2, 2)
        sphere_pos_y = np.linspace(-2, 2, 2)
        sphere_pos_z = np.array([-5])
        sphere_positions = np.meshgrid(sphere_pos_x, sphere_pos_y, sphere_pos_z)
        sphere_positions = np.array(sphere_positions).reshape(3, -1).T

        # Sphere radii
        sphere_radii = np.array([1.] * len(sphere_positions))

        # sample normal map
        normal_map = np.random.rand(2, 2, 3)
        normal_map = np.repeat(np.repeat(normal_map, 2, axis=0), 2, axis=1)
        normal_map = normal_map.reshape(-1, 3)
        normal_map = normal_map / np.linalg.norm(normal_map, axis=1, keepdims=True)

        big_background_sphere = Sphere(np.array([0, 0, -190]), 100, normal_map)

        shapes = [Sphere(pos, rad) for pos, rad in zip(sphere_positions, sphere_radii)] + [big_background_sphere,
                                                                                           ]

        objects = [Entity(shape, Material(albedo=0.5, color=np.array([0.5, 0.5, 0.5]), Ks=0.5, n=1)) for shape in shapes]

        # Define the scene by combining shapes and light
        return Scene(objects, light)


def display(image):
    fig, ax = plt.subplots()

    # Show the image with an equal aspect ratio
    ax.imshow(image, cmap='gray', vmin=0, vmax=1, aspect='equal')

    # Remove the axes
    ax.axis('off')

    # Make the layout fit tightly around the image
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.show()


if __name__ == "__main__":
    sampling_method = StratifiedJitteredSampling(4)
    # sampling_method = SimpleGridSampling(8)

    # Define the camera
    # camera = MonochromeCamera(np.array([0, 0, 2]), np.pi / 3, 1, 60, 60, advanced_shadow=True)
    perspective_camera = PerspectiveCamera([0, 0, 4], np.pi / 3, 1, 120, 120)
    orthographic_camera = OrthographicCamera([0, 0, 4], 120, 1, 120, 120)
    camera = perspective_camera
    # camera = orthographic_camera

    # Get the scene
    scene = get_scene(2)

    # Create a Renderer
    renderer = Renderer(sampling_method)

    # Render a scene
    image = renderer.render(scene, camera)

    # Display the image
    display(image)
