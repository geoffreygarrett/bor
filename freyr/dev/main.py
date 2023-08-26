from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)


class Shape(ABC):
    @abstractmethod
    def intersect(self, ray):
        pass

    @abstractmethod
    def get_normal(self, point):
        pass

    @abstractmethod
    def get_monochrome_value(self, point):
        pass


class Camera(ABC):
    def __init__(self, position, fov, aspect_ratio, width, height, samples_per_pixel=1):
        self.position = position
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.width = width
        self.height = height
        self.samples_per_pixel = samples_per_pixel

    @abstractmethod
    def generate_ray(self, u, v):
        pass

    @abstractmethod
    def render_scene(self, scene):
        pass


class Sphere(Shape):
    def __init__(self, center, radius, monochrome_value):
        self.center = center
        self.radius = radius
        self.monochrome_value = monochrome_value

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

    def get_monochrome_value(self, point):
        return self.monochrome_value


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


class MonochromeCamera(Camera):
    def __init__(self, position, fov, aspect_ratio, width, height,
                 sampling_method, advanced_shadow=False):
        super().__init__(position, fov, aspect_ratio, width, height)
        self.advanced_shadow = advanced_shadow
        self.sampling_method = sampling_method

    def generate_ray(self, u, v):
        direction = self.direction + u * self.right + v * self.up
        return Ray(self.position, direction)

    # Corrected is_in_shadow method
    def is_in_shadow(self, point, scene):
        offset = 1e-6
        direction_to_light = (scene.light.position - point).astype(float)
        direction_to_light /= np.linalg.norm(direction_to_light)  # Normalize the direction
        shadow_ray_origin = point + offset * direction_to_light
        shadow_ray = Ray(shadow_ray_origin, direction_to_light)
        distance_to_light = np.linalg.norm(scene.light.position - point)
        for shape in scene.shapes:
            t1, t2 = shape.intersect(shadow_ray)
            # Debug prints
            if self.advanced_shadow:
                if 0 < t1 < distance_to_light or 0 < t2 < distance_to_light:
                    return True
            else:
                if 0 < t1 < distance_to_light:
                    return True
        return False

    def compute_lighting(self, point, normal, scene):  # Add 'point' as a parameter
        if self.is_in_shadow(point, scene):  # Change from 'self.position' to 'point'
            return 0.0
        light_direction = (scene.light.position - point).astype(float)  # Change from 'self.position' to 'point'
        light_direction /= np.linalg.norm(light_direction)
        return max(0, np.dot(normal, light_direction))

    def render_scene(self, scene):
        self.direction = np.array([0, 0, -1])
        self.right = np.array([1, 0, 0])
        self.up = np.array([0, 1, 0])

        image = np.zeros((self.height, self.width))
        for j in range(self.height):
            for i in range(self.width):
                brightness = 0.0
                samples = self.sampling_method.sample(i, j, self.width, self.height)
                for u, v in samples:
                    ray = self.generate_ray(u, v)
                    closest_t = np.inf
                    closest_shape = None
                    for shape in scene.shapes:
                        t1, t2 = shape.intersect(ray)
                        if 0 < t1 < closest_t:
                            closest_t = t1
                            closest_shape = shape
                    if closest_shape is not None:
                        intersection_point = ray.origin + closest_t * ray.direction
                        normal = closest_shape.get_normal(intersection_point)
                        monochrome_value = closest_shape.get_monochrome_value(intersection_point)
                        lighting = self.compute_lighting(intersection_point, normal, scene)
                        brightness += lighting * monochrome_value
                image[j, i] = brightness / self.sampling_method.samples_per_pixel
        return image


class Scene:
    def __init__(self, shapes, light):
        self.shapes = shapes
        self.light = light


def get_scene(n):
    if n == 1:

        # Define the light source
        light_position = np.array([-2, -2, 10])
        light = Light(light_position)
        # Define the shapes in the scene (spheres in this case)
        sphere1 = Sphere(np.array([-1, -1, -5]), 1, 1)
        sphere2 = Sphere(np.array([1, 1, -90]), 10, 0.5)

        # Define the scene by combining shapes and light
        return Scene([sphere1, sphere2], light)

    elif n == 2:
        # generate a grid of spheres
        # Define the light source
        light_position = np.array([0, 0, 10])
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

        big_background_sphere = Sphere(np.array([0, 0, -190]), 100, 0.5)
        center_sphere = Sphere(np.array([0, 0, -5]), 1, 0.8)

        # Define the scene by combining shapes and light
        return Scene(
            [Sphere(pos, rad, 1) for pos, rad in zip(sphere_positions, sphere_radii)] + [big_background_sphere,
                                                                                         center_sphere],
            light)


def render_and_display(camera, scene):
    image = camera.render_scene(scene)
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
    # sampling_method = StratifiedJitteredSampling(4)
    sampling_method = SimpleRandomSampling(26)

    # Define the camera
    camera = MonochromeCamera(np.array([0, 0, 2]), np.pi / 3, 1, 60, 60, sampling_method, advanced_shadow=True)

    # Get the scene
    scene = get_scene(2)

    # Render and display
    render_and_display(camera, scene)