import blenderproc as bproc
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


#

def initialize_scene():
    """
    Initializes the Blender scene with object and lighting setup.
    """
    bproc.init()
    light = bproc.types.Light('SUN')
    light.set_location([0, 50, 1000])
    light.set_rotation_euler([-np.pi / 4, np.pi / 3, 0])
    #    light.set_rotation_euler([-0.063, 0.6177, -0.1985])
    light.set_energy(64)


def setup_approach_camera_positions(positions=np.linspace(300, 5000, 10)):
    for pos in positions:
        bproc.camera.add_camera_pose(
            bproc.math.build_transformation_mat(np.array([pos, 0, 0]), np.array([0, np.pi / 2, 0])))


# def setup_camera(cam_positions):
#    """
#    Sets up the camera orientation based on multiple positions and orientations.
#
#    Args:
#        cam_positions (list): List of camera positions, each as a np.array.
#        cam_orientations (list): List of camera orientations, each as a np.array in Euler angles (degrees).
#    """
#
#    new_cam_pose = bproc.math.build_transformation_mat(new_cam_pos, new_cam_orient_rad)
#    bproc.camera.add_camera_pose(new_cam_pose)

#    if len(cam_positions) != len(cam_orientations):
#        raise ValueError("Length of cam_positions must match length of cam_orientations.")
#
#    for i in range(len(cam_positions)):
#        cam_pos = cam_positions[i]
#        cam_orient = cam_orientations[i]
#
#        cam_orient_rad = np.radians(cam_orient)
#        cam_pose = bproc.math.build_transformation_mat(cam_pos, cam_orient_rad)
#        bproc.camera.add_camera_pose(cam_pose)


# def setup_camera(initial_cam_pos, initial_cam_orient, axis, theta_deg):
#    """
#    Sets up the camera orientation based on initial position, initial orientation,
#    axis of rotation, and rotation angle in degrees.
#
#    Args:
#        initial_cam_pos (np.array): Initial camera position.
#        initial_cam_orient (np.array): Initial camera orientation in Euler angles (degrees).
#        axis (list): Axis of rotation.
#        theta_deg (float): Angle of rotation in degrees.
#    """
#    initial_cam_orient_rad = np.radians(initial_cam_orient)
#    initial_cam_pose = bproc.math.build_transformation_mat(initial_cam_pos, initial_cam_orient_rad)
#    bproc.camera.add_camera_pose(initial_cam_pose)
#
#    r = R.from_euler('xyz', initial_cam_orient, degrees=True)
#    r_new = R.from_rotvec(np.radians(theta_deg) * np.array(axis))
#    new_cam_pos = r_new.apply(initial_cam_pos)
#    r_combined = r * r_new
#    new_cam_orient = r_combined.as_euler('xyz', degrees=True)
#    new_cam_orient_rad = np.radians(new_cam_orient)
#    new_cam_pose = bproc.math.build_transformation_mat(new_cam_pos, new_cam_orient_rad)
#    bproc.camera.add_camera_pose(new_cam_pose)


def render_scene():
    """
    Renders the scene and returns the rendering data.

    Returns:
        dict: Dictionary containing rendered data.
    """
    bproc.renderer.set_world_background([0, 0, 0])
    bproc.renderer.enable_depth_output(activate_antialiasing=True)
    bproc.renderer.enable_normals_output()
    bproc.renderer.set_noise_threshold(0.01)
    return bproc.renderer.render()


def visualize_rendering(data):
    """
    Visualizes the rendering data including depth, normals, and colors.

    Args:
        data (dict): Dictionary containing rendered data.
    """
    #    # Visualize Depth
    #    depth_data = data["depth"][0]
    #    depth_data_normalized = (depth_data - np.min(depth_data)) / (np.max(depth_data) - np.min(depth_data))
    #    depth_colored = cm.plasma(depth_data_normalized)[:, :, :3]
    #    depth_img = Image.fromarray((depth_colored * 255).astype(np.uint8), 'RGB')
    #    depth_img.show()
    #
    #    # Visualize Normals
    #    normal_data = data["normals"][0]
    #    normal_data_normalized = (normal_data - np.min(normal_data)) / (np.max(normal_data) - np.min(normal_data))
    #    normal_img = Image.fromarray((normal_data_normalized * 255).astype(np.uint8), 'RGB')
    #    normal_img.show()

    # Visualize Colors
    color_data = data["colors"][0]

    # Step 1: Convert to grayscale
    gray_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)

    # Optional: Apply Gaussian blur to reduce noise and improve contour detection
    #    gray_data = cv2.GaussianBlur(gray_data, (5, 5), 0)
    gray_data = cv2.GaussianBlur(gray_data, (5, 5), 0)

    # Step 2: Apply Thresholding
    # Since we are in space and the background is black, a lower threshold might work well.
    _, thresh = cv2.threshold(gray_data, 30, 255, cv2.THRESH_BINARY)
    #    # Step 2: Thresholding (using Otsu's method here) alternative
    #    _, thresh = cv2.threshold(gray_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Find Contours
    #    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Filter Contours (Optional, e.g., based on area)
    # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # Draw contours on the original image
    result = cv2.drawContours(color_data, contours, -1, (0, 255, 0), 1)  # Drawing with green color

    # Convert to PIL Image and show
    result_img = Image.fromarray(result, 'RGB')
    result_img.show()


def generate_light_curve(image_arrays, time_points=None):
    """
    Generate a light curve data from a time series of image arrays.

    Args:
        image_arrays (list): List of image arrays.
        time_points (list, optional): List of time points corresponding to each image.

    Returns:
        dict: A dictionary containing 'time_points' and 'brightness_values'.
    """

    # Initialize lists to store brightness values and number of bright pixels
    brightness_values = []
    num_bright_pixels_list = []

    # Validate or generate time points
    if time_points is None:
        time_points = list(range(len(image_arrays)))
    elif len(time_points) != len(image_arrays):
        raise ValueError("Length of time_points must match length of image_arrays.")

    for image in image_arrays:
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Thresholding to isolate the object (assuming it's brighter than the background)
        _, thresh_image = cv2.threshold(blurred_image, 30, 255, cv2.THRESH_BINARY)

        # Find the number of non-zero (i.e., bright) pixels
        num_bright_pixels = cv2.countNonZero(thresh_image)
        num_bright_pixels_list.append(num_bright_pixels)

        # Calculate average brightness using the original grayscale image
        # for the pixels identified as "bright" in the thresholded image
        avg_brightness = np.sum(gray_image[thresh_image == 255]) / num_bright_pixels if num_bright_pixels > 0 else 0

        # Append to brightness list
        brightness_values.append(avg_brightness)

    return {
        'time_points': time_points,
        'brightness_values': brightness_values,
        'num_bright_pixels': num_bright_pixels_list  # Additional data that might be useful
    }


def plot_quiver(ax, flow, spacing, margin=0, **kwargs):
    """
    Plots a less dense quiver field for optical flow visualization.

    Args:
        ax (matplotlib.axis): Matplotlib axis to plot on.
        flow (np.array): Optical flow array.
        spacing (int): Spacing between arrows in pixels.
        margin (int, optional): Margin for the plot. Defaults to 0.
    """
    h, w, *_ = flow.shape
    nx = int((w - 2 * margin) / spacing)
    ny = int((h - 2 * margin) / spacing)
    x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
    y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)
    flow = flow[np.ix_(y, x)]
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}
    ax.quiver(x, y, u, v, **kwargs)
    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect("equal")


import bpy

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

if __name__ == "__main__":
    t_eros = 5.27025547 * 60
    t_end = t_eros * 3.0
    n_samples = 100
    f_sample = n_samples / t_end
    t_sample = 1 / f_sample

    initialize_scene()

    BASE_PATH = "freyr/examples/data"

    # obj = bproc.loader.load_obj("Eros Gaskell 50k poly.ply")
    obj = bproc.loader.load_obj(os.path.join(BASE_PATH, "eros_gaskell_50k_poly.ply"))

    for i in range(n_samples):
        th = 2 * np.pi / t_eros * t_sample * i
        obj[0].set_rotation_euler([0, 0, th], frame=i)

    # Create Lambertian material
    eros_texture = bpy.data.images.load(os.path.join(BASE_PATH, "eros_grayscale.jpg"))
    asteroid_material = obj[0].get_materials()[0]

    # Create a new image texture node
    asteroid_material.set_principled_shader_value("Base Color", eros_texture)
    asteroid_material.set_principled_shader_value("Roughness", 0.4)
    asteroid_material.set_principled_shader_value("Metallic", 0.0)
    asteroid_material.set_displacement_from_principled_shader_value("Base Color", multiply_factor=1.0)
    asteroid_material.set_principled_shader_value("Specular", 0.0)
    obj[0].set_material(0, asteroid_material)

    cam_0 = 5000
    cam_1 = 750
    initial_cam_pos = np.array([0, cam_0, 0])
    initial_cam_orient = np.array([0, np.pi / 2, 0])
    axis = [0, 1, 0]

    #    setup_camera(initial_cam_pos, initial_cam_orient, axis, theta_deg)
    setup_approach_camera_positions(np.linspace(cam_0, cam_1, n_samples))

    bproc.camera.set_intrinsics_from_blender_params(lens=np.deg2rad(5.5), lens_unit="FOV", clip_start=0.0,
                                                    clip_end=100000)
    bproc.camera.set_resolution(1020, 1020)
    #    bproc.camera.set_resolution(102, 102)
    import os
    # def load_
    # if "saved_images" not in os.listdir():
    data = render_scene()
    # else:
    #     data = load_rendered_images()

    # Create a directory to save images if it doesn't exist
    output_dir = "saved_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(n_samples):  # Assuming n_frames is an integer specifying the number of frames
        # Visualize Normals (or any other data you have)
        normal_data = data["colors"][i]  # Replace with the appropriate data source
        normal_data_normalized = (normal_data - np.min(normal_data)) / (np.max(normal_data) - np.min(normal_data))
        normal_img = Image.fromarray((normal_data_normalized * 255).astype(np.uint8), 'RGB')

        # Save the image
        image_path = os.path.join(output_dir, f"normal_image_frame_{i}.png")
        normal_img.save(image_path)

    print(data)

    time = np.linspace(0, t_end, n_samples)
    lc = generate_light_curve(data['colors'], time_points=time)

    # Generate a sample signal
    t = time
    # Subtract the mean to remove DC component
    y = lc["brightness_values"] - np.mean(lc["brightness_values"])

    # Perform FFT
    # Perform FFT
    Y = np.fft.fft(y)
    frequencies = np.fft.fftfreq(len(Y), 1 / f_sample)
    print(1 / frequencies / 60)

    # Find the peak frequency
    peak_freq = frequencies[np.argmax(abs(Y))]
    print(abs(Y))

    print("Fundamental frequency:", peak_freq, "Hz")
    print("Fundamental period:", 1 / peak_freq / 60, "Hr")
    # Initialize variable to store the lowest frequency with integer multiplicity
    lowest_multiple_freq = None

    # Loop through frequencies to find the lowest one with integer multiplicity of peak frequency
    for freq in frequencies:
        if freq > 0 and np.isclose(peak_freq % freq, 0, atol=1e-6):
            lowest_multiple_freq = freq
            break

    if lowest_multiple_freq is not None:
        print("Lowest frequency with integer multiplicity:", lowest_multiple_freq, "Hz")
        print("Corresponding period:", 1 / lowest_multiple_freq / 60, "Hr")
    else:
        print("No lower frequency with integer multiplicity found.")

    print(lc)

    plt.subplots()
    plt.plot(time, lc["brightness_values"])
    plt.show()

    plt.plot(frequencies, abs(Y))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.title('FFT')
    plt.show()

#    visualize_rendering(data)
#
#    fix, ax = plt.subplots()
#
#    # Convert the color images to grayscale
#    gray0 = cv2.cvtColor(data['colors'][0], cv2.COLOR_RGB2GRAY)
#    gray1 = cv2.cvtColor(data['colors'][1], cv2.COLOR_RGB2GRAY)
#
#    # Compute optical flow
#    flow = cv2.calcOpticalFlowFarneback(
#        gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0
#    )
#
#    plot_quiver(ax, flow, spacing=10, scale=1, color="#ff44ff")
#    plt.show()
