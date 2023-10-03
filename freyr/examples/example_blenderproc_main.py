import blenderproc as bproc
import cv2
import numpy as np
from PIL import Image


#


def initialize_scene(distance_au=1.5, illumination_angle=20):
    """
    Initializes the Blender scene with object and lighting setup.
    """
    bproc.init()
    light = bproc.types.Light("SUN")
    # light.set_location([0, 50, 1000])
    # # light.set_location([0, 50, 1000])
    # light.set_rotation_euler([-np.pi / 4, np.pi / 3, 0])
    # convert to W/km^2

    light.set_rotation_euler([0, np.pi / 2, -np.deg2rad(illumination_angle)])
    # light.set_rotation_euler([0, 0, 0])
    # Solar constant at Earth (W/m^2)
    original_energy = 1361 / 50
    # original_energy = 1361 / 50
    #
    # # Calculate new energy based on new distance in AU
    new_energy = original_energy * (1 / distance_au) ** 2

    light.set_energy(new_energy)


def setup_approach_camera_positions(
    position_mags=np.linspace(300, 5000, 10), approach_angle=0
):
    # Parametrize camera's position with declination and ascension
    def get_camera_position(ascension, declination, radius):
        x = radius * np.cos(declination) * np.cos(ascension)
        y = radius * np.cos(declination) * np.sin(ascension)
        z = radius * np.sin(declination)
        return np.array([x, y, z])

    look_at = np.array([0, 0, 0])
    for pos_mag in position_mags:
        position = get_camera_position(0, np.deg2rad(approach_angle), pos_mag)
        forward_vec = look_at - position
        forward_vec /= np.linalg.norm(forward_vec)
        pose_mat = bproc.camera.rotation_from_forward_vec(forward_vec, "Y")
        bproc.camera.add_camera_pose(
            bproc.math.build_transformation_mat(
                position,
                pose_mat,
            )
        )


def render_scene(directory=None):
    """
    Renders the scene and returns the rendering data.

    Returns:
        dict: Dictionary containing rendered data.
    """
    bproc.renderer.set_world_background([0, 0, 0])
    bproc.renderer.enable_depth_output(activate_antialiasing=True, output_dir=directory)
    bproc.renderer.enable_diffuse_color_output(output_dir=directory)
    bproc.renderer.enable_normals_output(output_dir=directory)
    # Render the optical flow (forward and backward) for all frames
    optical_flow = bproc.renderer.render_optical_flow(
        get_backward_flow=True,
        get_forward_flow=True,
        blender_image_coordinate_style=False,
    )
    bproc.renderer.set_noise_threshold(0.01)
    data = bproc.renderer.render()
    data["forward_flow"] = optical_flow["forward_flow"]
    data["backward_flow"] = optical_flow["backward_flow"]
    return data


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
    result = cv2.drawContours(
        color_data, contours, -1, (0, 255, 0), 1
    )  # Drawing with green color

    # Convert to PIL Image and show
    result_img = Image.fromarray(result, "RGB")
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
        avg_brightness = (
            np.sum(gray_image[thresh_image == 255]) / num_bright_pixels
            if num_bright_pixels > 0
            else 0
        )

        # Append to brightness list
        brightness_values.append(avg_brightness)

    return {
        "time_points": time_points,
        "brightness_values": brightness_values,
        "num_bright_pixels": num_bright_pixels_list,  # Additional data that might be useful
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


def rotation_matrix_x(angle):
    """Generate a rotation matrix for rotation about the X-axis by `angle` radians."""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def rotation_matrix_z(angle):
    """Generate a rotation matrix for rotation about the Z-axis by `angle` radians."""
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


import os

if __name__ == "__main__":
    # The best estimates for the J2000 right ascension and declination of the pole of Eros are α=11.3692±0.003° and δ=17.2273±0.006°. The rotation rate of Eros is 1639.38922±0.00015°/day, which gives a rotation period of 5.27025547 h. No wobble greater than 0.02° has been detected. Solar gravity gradient torques would introduce a wobble of at most 0.001°.

    # get command line args for
    # illumination_angle,
    # true_anomaly,
    # approach_angle,
    # camera_distance1,
    # camera_distance2,

    import sys

    illumination_angle = float(sys.argv[1])
    true_anomaly = float(sys.argv[2])
    approach_angle = float(sys.argv[3])
    camera_distance1 = float(sys.argv[4])
    camera_distance2 = float(sys.argv[5])
    n_samples = int(sys.argv[6])

    # example use:
    # blenderproc run example_blenderproc_main.py -- 0 0 0 500 500
    print("illumination_angle", illumination_angle)
    print("true_anomaly", true_anomaly)
    print("approach_angle", approach_angle)
    print("camera_distance1", camera_distance1)
    print("camera_distance2", camera_distance2)
    print("n_samples", n_samples)

    DIRECTORY_NAME = f"eros_{int(illumination_angle)}_{int(true_anomaly)}_{int(approach_angle)}_{int(camera_distance1)}_{int(camera_distance2)}_{n_samples}"

    cmd = """
         BLENDERPROC_BASE_PATH=./data blenderproc run example_blenderproc_main.py -- 15 0 0 500 500 720 && \
         BLENDERPROC_BASE_PATH=./data blenderproc run example_blenderproc_main.py -- 15 45 0 500 500 720 && \
         BLENDERPROC_BASE_PATH=./data blenderproc run example_blenderproc_main.py -- 15 90 0 500 500 720 && \
         BLENDERPROC_BASE_PATH=./data blenderproc run example_blenderproc_main.py -- 45 0 0 500 500 720 && \
         BLENDERPROC_BASE_PATH=./data blenderproc run example_blenderproc_main.py -- 45 45 0 500 500 720 && \
         BLENDERPROC_BASE_PATH=./data blenderproc run example_blenderproc_main.py -- 45 90 0 500 500 720 && \
        BLENDERPROC_BASE_PATH=./data blenderproc run example_blenderproc_main.py -- 75 0 0 500 500 720 && \
        BLENDERPROC_BASE_PATH=./data blenderproc run example_blenderproc_main.py -- 75 45 0 500 500 720 && \
        BLENDERPROC_BASE_PATH=./data blenderproc run example_blenderproc_main.py -- 75 90 0 500 500 720 
    """


    # if this exists, dont run
    if os.path.exists(DIRECTORY_NAME):
        print("directory exists, skipping")
        sys.exit(0)

    ra, dec = np.radians([11.3692, 17.2273])
    sma = 1.458104
    ecc = 0.222676
    inc = np.radians(10.830)
    argp = np.radians(178.796)
    node = np.radians(304.300)
    mass = 6.687e15

    t_eros = 5.27025547 * 60
    t_end = t_eros * 1.0
    # n_samples = 2
    f_sample = n_samples / t_end
    t_sample = 1 / f_sample
    distance_au = 1.5

    initialize_scene(distance_au, illumination_angle=illumination_angle)

    if "BLENDERPROC_BASE_PATH" in os.environ:
        BASE_PATH = os.environ["BLENDERPROC_BASE_PATH"]
    else:
        BASE_PATH = "freyr/examples/data"

    # obj = bproc.loader.load_obj("Eros Gaskell 50k poly.ply")
    obj = bproc.loader.load_obj(os.path.join(BASE_PATH, "eros_gaskell_200k_poly.ply"))[
        0
    ]

    th0 = np.pi / 2

    # α=11.3692±0.003° and δ=17.2273±0.006°
    # Note: This is in J2000 coordinates, however we will treat it as the asteroid's ecliptic coordinates

    # Given celestial longitude and latitude in degrees
    lambda_deg = 90 - 11.3692
    beta_deg = 90 - 17.2273

    # Convert to radians
    lambda_rad = np.radians(lambda_deg)
    beta_rad = np.radians(beta_deg)

    # Calculate the initial rotation matrix
    planetocentric_to_inertial = np.dot(
        rotation_matrix_z(lambda_rad), rotation_matrix_x(beta_rad)
    )

    planetocentric_to_inertial = np.linalg.inv(planetocentric_to_inertial)

    # Set the initial rotation of the object to align its pole
    obj.set_rotation_mat(planetocentric_to_inertial)

    asteroid_poses = []

    # Now proceed to rotate about the Z-axis
    for i in range(n_samples):
        th = 2 * np.pi / t_eros * t_sample * i + th0

        # Generate the rotation matrix for this sample
        rotation_matrix_sample = rotation_matrix_z(th)

        # position
        r = np.array([0, 0, 0])

        # Combine this with the initial rotation
        final_rotation_matrix = np.dot(
            planetocentric_to_inertial, rotation_matrix_sample
        )

        # phase rotation on z
        final_rotation_matrix = np.dot(
            rotation_matrix_z(np.radians(true_anomaly)), final_rotation_matrix
        )

        # Set the final rotation matrix for this frame
        obj.set_rotation_mat(final_rotation_matrix, frame=i)
        obj.set_location(r, frame=i)

        # pose
        pose = bproc.math.build_transformation_mat(r, final_rotation_matrix)

        # save the asteroid pose
        asteroid_poses.append(pose)

    texture_file = "eros_grayscale.jpg"  # Texture file name
    texture_path = os.path.join(BASE_PATH, texture_file)

    # Load texture
    eros_texture = bpy.data.images.load(texture_path)

    # Get object and set its shading mode
    # obj.set_shading_mode("SMOOTH")

    # Get material
    asteroid_material = obj.get_materials()[0]

    # Set principled shader values
    def set_asteroid_material(material):
        material.set_principled_shader_value("Subsurface", 0.0)
        material.set_principled_shader_value("Metallic", 0.1)
        material.set_principled_shader_value("Specular", 0.1)
        material.set_principled_shader_value("Roughness", 0.9)
        material.set_principled_shader_value("Anisotropic", 0.0)
        material.set_principled_shader_value("Sheen", 0.0)
        material.set_principled_shader_value("Clearcoat", 0.0)
        material.set_principled_shader_value("IOR", 1.45)

    set_asteroid_material(asteroid_material)

    # Set up nodes for mapping and displacement
    def setup_nodes(material, albedo_value=0.25):
        nodes = material.nodes
        links = material.links

        text_coord_node = nodes.new("ShaderNodeTexCoord")
        mapping_node = nodes.new("ShaderNodeMapping")
        environment_texture_node = nodes.new("ShaderNodeTexEnvironment")
        displacement_node = nodes.new("ShaderNodeDisplacement")
        multiply_for_displacement_node = nodes.new("ShaderNodeMath")
        multiply_for_albedo_node = nodes.new("ShaderNodeMath")
        material_output_node = nodes["Material Output"]

        # Configure nodes
        mapping_node.vector_type = "POINT"
        mapping_node.inputs[3].default_value[0] = -1
        mapping_node.inputs[2].default_value[2] = np.pi

        environment_texture_node.image = bpy.data.images.load(texture_path)

        displacement_node.inputs["Scale"].default_value = 1.0
        displacement_node.inputs["Midlevel"].default_value = 0.5

        multiply_for_displacement_node.operation = "MULTIPLY"
        multiply_for_displacement_node.inputs[1].default_value = 0.05

        multiply_for_albedo_node.operation = "MULTIPLY"
        multiply_for_albedo_node.inputs[
            1
        ].default_value = albedo_value  # Effective Albedo

        # Create links
        links.new(text_coord_node.outputs["Object"], mapping_node.inputs["Vector"])
        links.new(
            mapping_node.outputs["Vector"],
            environment_texture_node.inputs["Vector"],
        )
        links.new(
            environment_texture_node.outputs["Color"],
            multiply_for_albedo_node.inputs[0],
        )
        links.new(
            multiply_for_albedo_node.outputs[0],
            nodes["Principled BSDF"].inputs["Base Color"],
        )
        links.new(
            environment_texture_node.outputs["Color"],
            multiply_for_displacement_node.inputs[0],
        )
        links.new(
            multiply_for_displacement_node.outputs[0],
            displacement_node.inputs["Height"],
        )
        links.new(
            displacement_node.outputs["Displacement"],
            material_output_node.inputs["Displacement"],
        )

    setup_nodes(asteroid_material)

    # Set the material
    obj.set_material(0, asteroid_material)
    cam_0 = 357
    cam_1 = 354.240
    cam_0 = camera_distance1
    cam_1 = camera_distance2

    axis = [0, 1, 0]

    #    setup_camera(initial_cam_pos, initial_cam_orient, axis, theta_deg)
    setup_approach_camera_positions(
        np.linspace(cam_0, cam_1, n_samples), approach_angle
    )

    bproc.camera.set_intrinsics_from_blender_params(
        lens=np.deg2rad(5.5), lens_unit="FOV", clip_start=0.0, clip_end=100000
    )

    bproc.camera.set_resolution(1020, 1020)

    import os

    # Create a directory to save images if it doesn't exist
    if not os.path.exists(DIRECTORY_NAME):
        os.makedirs(DIRECTORY_NAME)

    if not os.path.exists(os.path.join(DIRECTORY_NAME, "images")):
        os.makedirs(os.path.join(DIRECTORY_NAME, "images"))

    if not os.path.exists(os.path.join(DIRECTORY_NAME, "camera_poses")):
        os.makedirs(os.path.join(DIRECTORY_NAME, "camera_poses"))

    if not os.path.exists(os.path.join(DIRECTORY_NAME, "asteroid_poses")):
        os.makedirs(os.path.join(DIRECTORY_NAME, "asteroid_poses"))

    if not os.path.exists(os.path.join(DIRECTORY_NAME, "blenderproc_output")):
        os.makedirs(os.path.join(DIRECTORY_NAME, "blenderproc_output"))

    # if "saved_images" not in os.listdir():
    data = render_scene(directory=None)

    # write the data to a .hdf5 container
    bproc.writer.write_hdf5(os.path.join(DIRECTORY_NAME, "blenderproc_output"), data)

    # save the camera matrix
    np.save(
        os.path.join(DIRECTORY_NAME, "camera_intrinsics.npy"),
        bproc.camera.get_intrinsics_as_K_matrix(),
    )

    for i in range(n_samples):
        normal_data = data["colors"][i]

        normal_img = Image.fromarray(normal_data.astype(np.uint8), "RGB")

        # Save the image
        image_path = os.path.join(DIRECTORY_NAME, "images", f"image_frame_{i}.png")
        normal_img.save(image_path)

        # save the camera pose
        np.save(
            os.path.join(DIRECTORY_NAME, "camera_poses", f"camera_pose_{i}.npy"),
            bproc.camera.get_camera_pose(i),
        )

        # save the asteroid poses
        np.save(
            os.path.join(DIRECTORY_NAME, "asteroid_poses", f"asteroid_pose_{i}.npy"),
            asteroid_poses[i],
        )


#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.fft import fft, fftfreq
#     from scipy.signal import find_peaks
#     from scipy.signal import detrend
#
#     # time = np.linspace(0, t_end, n_samples)
#     # Your other setup and rendering code here
#     # data = render_scene()
#
#     time = np.linspace(0, t_end, n_samples)
#     t = time
#     lc = generate_light_curve(data['colors'], time_points=time)
#     # Extract time points and brightness values
#     t_lc = lc["time_points"]
#     y_lc = lc["brightness_values"]
#     y_lc = y_lc - np.mean(y_lc)
#     y_lc = detrend(y_lc)
#
#     # Perform FFT on the detrended brightness values
#     yf_lc = fft(y_lc)
#     xf_lc = fftfreq(len(y_lc), time[1] - time[0])
#     power_spectrum_lc = np.abs(yf_lc)**2
#
#     # Identify the most significant peaks in the power spectrum (ignoring the zero-frequency term)
#     positive_indices = np.where(xf_lc > 0)[0]
#     peaks_lc_positive, _ = find_peaks(power_spectrum_lc[positive_indices])
#     peak_freqs_lc_positive = xf_lc[positive_indices][peaks_lc_positive]
#     peak_powers_lc_positive = np.sqrt(power_spectrum_lc[positive_indices][peaks_lc_positive])
#
#     # Sort peaks by power and select the top N (let's say 5 for demonstration)
#     N = 5
#     top_N_peaks_lc = sorted(zip(peak_powers_lc_positive, peak_freqs_lc_positive), reverse=True)[:N]
#
#     # Identify the primary frequency and its lower frequency harmonics
#     primary_freq = top_N_peaks_lc[0][1]
#     harmonics = [freq for power, freq in top_N_peaks_lc if np.isclose(primary_freq % freq, 0, atol=1e-5)]
#
#     # Reconstruct the brightness curve using only the harmonics of the primary frequency
#     reconstructed_curve_lc = np.zeros_like(y_lc)
#     for freq in harmonics:
#         amplitude = np.abs(yf_lc[int(freq * len(t))]) / len(t)
#         phase = np.angle(yf_lc[int(freq * len(t))])
#         reconstructed_curve_lc += amplitude * np.cos(2 * np.pi * freq * t + phase)
#
#     # Print the peaks and their frequencies in terms of minutes
#     for amp, freq in top_N_peaks_lc:
#         print(f"Frequency: {1/freq/60}, Amplitude: {amp}")
#     # Convert brightness to absolute magnitude using existing positions
#     # For demonstration, let's assume a constant distance in the `positions` array
#     positions = np.full_like(t_lc, 1e3)  # 1e3 meters for demonstration
#     C = 10  # Calibration constant for demonstration
#
#     # Convert to absolute magnitude
#     # apparent_magnitudes_lc = -2.5 * np.log10(y_lc) + C
#     # absolute_magnitudes_lc = apparent_magnitudes_lc - 5 * np.log10(positions) + 5
#
#     # Convert reconstructed curve to absolute magnitude
#     # apparent_magnitudes_reconstructed = -2.5 * np.log10(reconstructed_curve_lc) + C
#     # absolute_magnitudes_reconstructed = apparent_magnitudes_reconstructed - 5 * np.log10(positions) + 5
#
#     # Identify the primary frequency for rotation period
#     primary_freq_lc = top_N_peaks_lc[0][1]
#     primary_period_lc = 1 / primary_freq_lc
#
#     # Plot the original light curve, power spectrum, and reconstructed curve in terms of absolute magnitude
#     fig, axs = plt.subplots(3, 1, figsize=(12, 12))
#
#     # Plot the original absolute magnitudes
#     axs[0].plot(t_lc, y_lc, label='Original Light Curve')
#     axs[0].set_title('Original Light Curve in Absolute Magnitude')
#     axs[0].set_xlabel('Time')
#     axs[0].set_ylabel('Absolute Magnitude')
#
#     # Plot the power spectrum
#     axs[1].plot(xf_lc[1:], power_spectrum_lc[1:])
#     axs[1].set_title('Power Spectrum')
#     axs[1].set_xlabel('Frequency')
#     axs[1].set_ylabel('Power')
#
#     # Mark the selected frequencies in the power spectrum
#     for power, freq in top_N_peaks_lc:
#         axs[1].axvline(x=freq, color='r', linestyle='--', label=f'Frequency = {freq:.4f}')
#
#     # Plot the reconstructed absolute magnitudes
#     axs[2].plot(t_lc, reconstructed_curve_lc, label='Reconstructed Light Curve', linestyle='--')
#     axs[2].set_title('Reconstructed Light Curve in Absolute Magnitude')
#     axs[2].set_xlabel('Time')
#     axs[2].set_ylabel('Absolute Magnitude')
#
#     plt.tight_layout()
#     plt.show()
#
# # Generate a sample signal
#     # t = time
#     # # Subtract the mean to remove DC component
#     # y = lc["brightness_values"] - np.mean(lc["brightness_values"])
#     #
#     # # Perform FFT
#     # # Perform FFT
#     # Y = np.fft.fft(y)
#     # frequencies = np.fft.fftfreq(len(Y), 1 / f_sample)
#     # # print(1 / frequencies / 60)
#     #
#     # # Find the peak frequency
#     # peak_freq = frequencies[np.argmax(abs(Y))]
#     # # print(abs(Y))
#     #
#     # print("Fundamental frequency:", peak_freq, "Hz")
#     # print("Fundamental period:", 1 / peak_freq / 60, "Hr")
#     # # Initialize variable to store the lowest frequency with integer multiplicity
#     # lowest_multiple_freq = None
#     #
#     # # Loop through frequencies to find the lowest one with integer multiplicity of peak frequency
#     # for freq in frequencies:
#     #     if freq > 0 and np.isclose(peak_freq % freq, 0, atol=1e-6):
#     #         lowest_multiple_freq = freq
#     #         break
#     #
#     # if lowest_multiple_freq is not None:
#     #     print("Lowest frequency with integer multiplicity:", lowest_multiple_freq, "Hz")
#     #     print("Corresponding period:", 1 / lowest_multiple_freq / 60, "Hr")
#     # else:
#     #     print("No lower frequency with integer multiplicity found.")
#
#     # print(lc)
#
#     plt.subplots()
#     plt.plot(time, lc["brightness_values"])
#     plt.show()

# plt.plot(frequencies, abs(Y))
# plt.xlabel('Freq (Hz)')
# plt.ylabel('|Y(freq)|')
# plt.title('FFT')
# plt.show()

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
