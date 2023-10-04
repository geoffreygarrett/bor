import cv2


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


def find_largest_contour(image):
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def filter_points_within_contour(points, contour, min_distance=10):
    filter_mask = np.zeros(len(points), dtype=bool)

    for idx, point in enumerate(points):
        point_tuple = tuple(point.ravel())
        distance_to_contour = cv2.pointPolygonTest(contour, point_tuple, True)
        filter_mask[idx] = distance_to_contour >= min_distance

    return filter_mask


if __name__ == "__main__":
    # get all images in saved_images with glob

    import os

    # Merging data for frame 718 into eros_45_45_0_500_500_720/blenderproc_output/718.hdf5
    BASE_PATH = "freyr/examples/eros_45_45_0_500_500_720"

    IMAGES_PATH = os.path.join(BASE_PATH, "images")
    CAMERA_INTRINSICS_PATH = os.path.join(BASE_PATH, "camera_intrinsics.npy")
    CAMERA_EXTRINSICS_PATH = os.path.join(BASE_PATH, "camera_poses")
    ASTEROID_POSES_PATH = os.path.join(BASE_PATH, "asteroid_poses")

    image_paths = []
    for root, dirs, files in os.walk(IMAGES_PATH):
        for file in files:
            if file.startswith("image_") and file.endswith(".png"):
                image_paths.append(os.path.join(root, file))

    # sort images by name, but numbers are 0...9999, so sort by number
    image_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))

    image_paths = image_paths[::1]

    n_samples = len(image_paths)

    print(f"{n_samples=}")

    #    # Sort images by name
    # image_paths.sort()
    # image_paths = image_paths[:100]
    import matplotlib

    matplotlib.use("QtAgg")
    # frame0 = cv2.imread(image_paths[0])
    # frame1 = cv2.imread(image_paths[20])
    # #    # Convert to grayscale
    # gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    # gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # #    # Convert to float32
    # gray0 = np.float32(gray0)
    # gray1 = np.float32(gray1)

    # flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 40, 3, 5, 1.2, 0)

    import matplotlib.pyplot as plt

    #
    # print(flow.shape)

    # Initialize an empty list to store historical positions for each point
    point_history = []

    # fig, ax = plt.subplots()
    # # plot_quiver(ax, flow, spacing=10, scale=1, color="#ff44ff")
    # plt.show()

    import matplotlib.pyplot as plt
    import numpy as np

    def find_largest_contour(image, distance_threshold=10):
        ret, thresh = cv2.threshold(image, 3, 255, 0)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if distance_threshold != -1:
            contours = [
                contour
                for contour in contours
                if cv2.arcLength(contour, True) > distance_threshold
            ]
        return max(contours, key=cv2.contourArea)

    # def filter_points_within_contour(points, contour, min_distance=10):
    def filter_points_within_contour(points, contour, min_distance=5):
        mask = np.array(
            [
                cv2.pointPolygonTest(contour, tuple(point.ravel()), True)
                >= min_distance
                for point in points
            ]
        )
        print(type(mask))
        return mask

    def filter_image_within_contour(image, contour, threshold=10):
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)

        # Create the structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (threshold, threshold))

        # Erode the mask to remove a 'threshold'-pixel band from the boundary of the contour
        mask_eroded = cv2.erode(mask, kernel)

        # Bitwise AND to keep only the eroded area (or bitwise NOT to remove it)
        result = cv2.bitwise_and(image, image, mask=mask_eroded)

        return result

    def update_features(features, valid_points, old_points):
        updated_features = {}
        for idx, feature in features.items():
            old_point = feature["point"]
            # Find the new point corresponding to the old point
            for i, op in enumerate(old_points):
                if np.array_equal(old_point, op):
                    valid_point = valid_points[i]
                    updated_features[idx] = {
                        "point": valid_point,
                        "lifetime": feature["lifetime"] + 1,
                        "history": feature["history"] + [valid_point],
                        "active": True,  # Mark as active
                    }
                    break
            if idx not in updated_features:
                updated_features[idx] = {**feature, "active": False}  # Mark as inactive

        return updated_features

    from scipy.spatial import KDTree

    def add_new_features(features, new_points, min_distance=10):
        existing_points = np.array(
            [feature["point"] for feature in features.values()]
        ).reshape((-1, 2))
        if existing_points is not None:
            kdtree = KDTree(existing_points)
            idx = max(features.keys(), default=-1) + 1
            for point in new_points:
                # Query the KD-tree for the nearest point to the current point
                dist, _ = kdtree.query(point, k=1)
                if dist >= min_distance:
                    features[idx] = {
                        "point": point,
                        "lifetime": 0,
                        "history": [point],
                        "active": True,
                    }
                    idx += 1
        else:
            # If there are no existing points, add all new points
            idx = max(features.keys(), default=-1) + 1
            for point in new_points:
                features[idx] = {
                    "point": point,
                    "lifetime": 0,
                    "history": [point],
                    "active": True,
                }
                idx += 1
        return features

    # Initialize parameters
    lk_params = {
        "winSize": (10, 10),
        "maxLevel": 2,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }
    ft_params = {
        "maxCorners": 200,
        "qualityLevel": 0.7,
        "minDistance": 10,
        "blockSize": 10,
    }
    refresh_interval = 1
    frame_count = 0

    # Load initial frame and features
    old_image = cv2.imread(image_paths[0], 0)
    image_h, image_w = old_image.shape
    largest_contour = find_largest_contour(old_image)
    inside_contour = filter_image_within_contour(old_image, largest_contour)
    initial_features = cv2.goodFeaturesToTrack(
        old_image, mask=inside_contour, **ft_params
    )
    features = {
        idx: {"point": point, "lifetime": 0, "history": [point], "active": True}
        for idx, point in enumerate(initial_features)
    }

    # center of brigtness using otzu thresholding
    ret, thresh = cv2.threshold(old_image, 3, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    M = cv2.moments(thresh)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    cx_list = []
    cy_list = []

    # Setup plot
    fig, ax = plt.subplots()

    # Main loop
    for i, image_path in enumerate(image_paths[1:], start=1):
        print(f"Processing image {i}/{len(image_paths)}")
        cx_list.append(cx)
        cy_list.append(cy)
        frame_count += 1
        new_image = cv2.imread(image_path, 0)

        # Collect points for optical flow
        p0 = np.array([f["point"] for f in features.values() if f["active"]])
        largest_contour = find_largest_contour(new_image)

        # Perform optical flow
        if len(p0) == 0:
            # continue
            pass
        else:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_image, new_image, p0, None, **lk_params
            )

            # Filter points based on status and error
            st = st.ravel()
            err = err.ravel()
            valid_mask = (st == 1) & (err < 10)

            valid_points = p1[valid_mask]
            old_points = p0[valid_mask]

            # valid_points = p1
            filter_mask = filter_points_within_contour(
                valid_points, largest_contour, min_distance=10
            )

            if len(filter_mask) != 0:
                valid_points = valid_points[filter_mask]
                old_points = old_points[filter_mask]

                # Now update features
                features = update_features(features, valid_points, old_points)

        # Plotting
        color_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(color_image, [largest_contour], 0, (255, 0, 0), 1, cv2.LINE_AA)

        # for point in valid_points:
        #     a, b = map(int, point.ravel())
        #     cv2.circle(color_image, (a, b), 2, (0, 0, 255), -1)

        # Inside your main loop, in the plotting section:
        for feature in features.values():
            if feature["active"]:  # Only plot active features
                a, b = map(int, feature["point"].ravel())
                cv2.circle(color_image, (a, b), 2, (0, 0, 255), -1)
                history = np.array(
                    feature["history"], dtype=np.int32
                )  # Convert to int32
                history = history.reshape((-1, 1, 2))
                cv2.polylines(
                    color_image, [history], False, (0, 255, 0), 1, cv2.LINE_AA
                )

        ax.clear()
        ax.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        plt.pause(0.001)

        # if frame_count % refresh_interval == 0:
        #     new_points = cv2.goodFeaturesToTrack(
        #         new_image,
        #         mask=filter_image_within_contour(new_image, largest_contour),
        #         **ft_params,
        #     )
        #     if new_points is not None:
        #         features = add_new_features(features, new_points, ft_params["minDistance"])

        if frame_count % refresh_interval == 0:
            mask = filter_image_within_contour(new_image, largest_contour)
            new_points = cv2.goodFeaturesToTrack(new_image, mask=mask, **ft_params)
            if new_points is not None:
                features = add_new_features(
                    features, new_points, ft_params["minDistance"]
                )

        old_image = new_image.copy()
        # center of brigtness using otzu thresholding
        ret, thresh = cv2.threshold(
            old_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        M = cv2.moments(thresh)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    # print(features)

    plt.show()

    y_heur = 0.3
    y_sheaf = 0.8
    y_cn = 1e5

    # filter all features with a lifetime less than 5
    features = {
        k: v for k, v in features.items() if v["lifetime"] >= y_heur * n_samples
    }

    # find the 50 features with longest lifetimes
    longest_lifetimes = sorted(
        features.values(), key=lambda x: x["lifetime"], reverse=True
    )[:50]

    # Construct the accumulated lifetime
    # total_points = list(accumulate(x["lifetime"] for x in longest_lifetimes))
    # If you want to start from a different initial value, say 0:
    dim_m = sum(x["lifetime"] + 1 for x in longest_lifetimes)
    dim_n_d2 = len(longest_lifetimes) + 2
    dim_n_d1 = 3 * len(longest_lifetimes)
    print(dim_m)
    D2 = np.zeros((dim_m, dim_n_d2))
    D1 = np.zeros((dim_m, 3 * len(longest_lifetimes)))

    # Check conditioning, and whether we treat it as ellipsis or sheaf of lines
    S3_j = []
    D1_j = None
    point_count = 0
    D1_matrices = []

    def construct_ellipses_d_matrices(longest_lifetimes, image_h, image_w):
        dim_m = sum(x["lifetime"] + 1 for x in longest_lifetimes)
        dim_n_d2 = len(longest_lifetimes) + 2
        dim_n_d1 = 3 * len(longest_lifetimes)

        D1 = np.zeros((dim_m, dim_n_d1))
        D2 = np.zeros((dim_m, dim_n_d2))

        point_count = 0
        for j, feature in enumerate(longest_lifetimes):
            for i, point in enumerate(feature["history"]):
                point = point.ravel()
                x_norm, y_norm = point[0] / image_w, point[1] / image_h

                # Construct the row of the D2 matrix
                D2[point_count, :] = [
                    x_norm * y_norm,
                    y_norm**2,
                    *[
                        0 if k != j else x_norm**2 + y_norm**2
                        for k in range(len(longest_lifetimes))
                    ],
                ]

                # Construct the row of the D1 matrix
                D1[point_count, j * 3 : (j + 1) * 3] = [x_norm, y_norm, 1]

                point_count += 1

        S1 = np.matmul(D1.T, D1)
        S2 = np.matmul(D1.T, D2)
        S3 = np.matmul(D2.T, D2)

        # a1 is computed as the smallest positive eigenvalue of S1 - S2 * S3^-1 * S2^T
        T = -np.linalg.inv(S3) @ np.transpose(S2)
        M = S1 + S2 @ T
        eigvals, eigvecs = np.linalg.eig(M)
        a1 = eigvecs[:, np.where(eigvals == min(eigvals[eigvals > 0]))].flatten()
        a2 = T @ a1

        return a1, a2

    def calculate_conditioning(D1_matrix):
        return np.linalg.cond(np.matmul(D1_matrix.T, D1_matrix))

    def filter_features_by_lifetime(features, min_lifetime):
        return {k: v for k, v in features.items() if v["lifetime"] >= min_lifetime}

    def normalize_point(point, image_h, image_w):
        return point[0] / image_h, point[1] / image_w

    # 6.2. Detection of degenerate solutions
    for feature in longest_lifetimes:
        D1_matrix = np.zeros((feature["lifetime"] + 1, 3))
        for i, point in enumerate(feature["history"]):
            x_norm, y_norm = normalize_point(point.ravel(), image_h, image_w)
            D1_matrix[i, :] = [x_norm, y_norm, 1]
        D1_matrices.append(D1_matrix)

    j_cn = [calculate_conditioning(D1_matrix) for D1_matrix in D1_matrices]

    def construct_d_sheaf_of_lines(longest_lifetimes, image_h, image_w):
        dim_m = sum(x["lifetime"] + 1 for x in longest_lifetimes)
        dim_n_d1 = len(longest_lifetimes) + 2

        D_sheaf = np.zeros((dim_m, dim_n_d1))
        point_count = 0
        for j, feature in enumerate(longest_lifetimes):
            for i, point in enumerate(feature["history"]):
                point = point.ravel()
                x_norm, y_norm = point[0] / image_w, point[1] / image_h

                # Construct the row of the D2 matrix
                D_sheaf[point_count, :] = [
                    x_norm,
                    y_norm,
                    *[0 if k != j else 1 for k in range(len(longest_lifetimes))],
                ]

                point_count += 1

        # solve with svd
        U, S, V = np.linalg.svd(D_sheaf)

        # get the last column of V
        a = V[-1, :]

        return a

    # Correcting the function to fix the issue with a and b
    def quadratic_to_ellipse(A, B, C, D, E, F):
        M = np.array([[A, B / 2], [B / 2, C]])
        q = np.array([D, E])
        w = (
            -F
        )  # Taking the negative of F to ensure the term inside the sqrt is positive

        # Calculate the angle of rotation (theta) in radians
        theta = 0.5 * np.arctan2(B, (A - C))

        # Calculate the center (h, k) of the ellipse
        M_inv = np.linalg.inv(M)
        center = -0.5 * np.dot(M_inv, q)

        y0 = center[0]
        x0 = center[1]

        # Calculate the lengths of the semi-major and semi-minor axes
        eigenvalues, _ = np.linalg.eig(M)
        eigenvalues = sorted(eigenvalues, reverse=True)

        u = -0.5 * np.linalg.inv(M) @ q
        const = w + u.T @ M @ u

        a = np.sqrt(const / eigenvalues[0])
        b = np.sqrt(const / eigenvalues[1])

        return y0, x0, theta, a, b

    def plot_line(image, slope, y_intercept):
        h, w = image.shape[:2]
        y1 = int(h)
        x1 = int(y1 * slope + y_intercept)
        y2 = 0
        x2 = int(y2 * slope + y_intercept)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

    def plot_ellipse(img, h, k, theta, a, b):
        # Validate that it's an ellipse

        try:
            # Translate to the center of the image
            offset_x, offset_y = img.shape[1] // 2, img.shape[0] // 2
            print(f"{h=}")
            h *= img.shape[0]
            k *= img.shape[1]

            a *= img.shape[1]
            b *= img.shape[0]

            theta = np.rad2deg(theta)
            # Debug prints
            print(f"Center: ({h}, {k})")
            print(f"Semi-major axis: {a}")
            print(f"Semi-minor axis: {b}")
            print(f"Angle of rotation: {theta}")

            # Draw the ellipse
            cv2.ellipse(
                img,
                (int(h), int(k)),
                (int(a), int(b)),
                int(theta),
                0,
                360,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.circle(img, (int(h), int(k)), 1, (0, 0, 255), -1)
        except ValueError as e:
            print(e)
            return

    # Create a black image
    # img = np.zeros((500, 500, 3), dtype=np.uint8)

    # Create a black image
    img = np.zeros((old_image.shape[0], old_image.shape[1], 3), dtype=np.uint8)
    # copy old image into
    img = cv2.cvtColor(old_image, cv2.COLOR_GRAY2BGR)

    br_cx = np.mean(cx_list)
    br_cy = np.mean(cy_list)

    br_cx = int(br_cx)
    br_cy = int(br_cy)

    # cx and cy should be mean of all ellipse centers
    cx = int(0.5 * img.shape[1])
    cy = int(0.5 * img.shape[0])

    c = (cx, cy)

    yellow_bgr = (0, 255, 255)
    orange_bgr = (0, 165, 255)
    cv2.circle(img, (br_cx, br_cy), 3, yellow_bgr, -1)
    cv2.circle(img, (cx, cy), 3, orange_bgr, -1)

    # put text of true and estimated vectors, only put estimated that is closest to true
    # true_v = np.array([0, -np.cos(np.deg2rad(45)), np.cos(np.deg2rad(45))])
    # true_v = true_v / np.linalg.norm(true_v)
    # v1_mean = v1_mean / np.linalg.norm(v1_mean)
    # v2_mean = v2_mean / np.linalg.norm(v2_mean)
    # v3_mean = v3_mean / np.linalg.norm(v3_mean)
    # v4_mean = v4_mean / np.linalg.norm(v4_mean)

    # homogeneous = np.array(
    #     [
    #         [0.0, -0.70710677, 0.70710677, 353.55339059],
    #         [1.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.70710677, 0.70710677, 353.55339059],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )
    homogeneous = np.load(os.path.join(CAMERA_EXTRINSICS_PATH, "camera_pose_0.npy"))
    asteroid_pose = np.load(os.path.join(ASTEROID_POSES_PATH, "asteroid_pose_0.npy"))

    NC = homogeneous[:3, :3]
    pos = homogeneous[:3, 3]

    K = np.load(CAMERA_INTRINSICS_PATH)

    # rot_adjusted =
    def opencv_to_blender_matrix():
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # def opencv_to_blender_matrix():
    #     return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    rot_adjusted = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    blender_to_opencv = np.linalg.inv(opencv_to_blender_matrix())
    # opencv_to_blender = opencv_to_blender_matrix() @ np.linalg.inv(rot_adjusted)
    # opencv_to_blender = opencv_to_blender_matrix() @ np.linalg.inv(rot_adjusted)
    # blender_to_opencv = np.linalg.inv(opencv_to_blender_matrix())
    opencv_to_blender = np.linalg.inv(blender_to_opencv)
    # blender_to_opencv = np.linalg.inv(opencv_to_blender)

    fraction_exceeding_threshold = sum(1 for cond in j_cn if cond > y_cn) / len(j_cn)
    candidate_vs = []
    if fraction_exceeding_threshold > y_sheaf:
        print("sheaf of lines")
        a = construct_d_sheaf_of_lines(longest_lifetimes, image_h, image_w)

        # Plot the lines
        for i in range(len(longest_lifetimes)):
            E = a[0]
            D = a[1]
            F = a[2 + i]

            m = -D / E
            b = -F / E

            m *= image_h / image_w
            b *= image_h

            plot_line(img, m, b)
            parallel_vector = np.array([1, -m, 0])
            parallel_vector /= np.linalg.norm(parallel_vector)
            print(f"{homogeneous=}")
            r_camera = homogeneous[:3, 3]
            # r_camera = NC @ r_came@ra
            print(f"{r_camera=}")
            r_camera /= np.linalg.norm(r_camera)
            res = np.linalg.inv(K) @ parallel_vector
            res = res / np.linalg.norm(res)

            print(f"{res=}")
            normal_vector = parallel_vector #np.cross(r_camera, res)
            # normal
            normal_vector /= np.linalg.norm(normal_vector)
            candidate_vs.extend([normal_vector, -normal_vector])

    else:
        print("ellipsis")
        a1, a2 = construct_ellipses_d_matrices(longest_lifetimes, image_h, image_w)

        # # Offset to translate the ellipse to the center of the image
        # offset_x, offset_y = img.shape[1] // 2, img.shape[0] // 2

        print(f"{D1.shape=}")
        print(f"{D2.shape=}")
        print(f"{a1.shape=}")
        print(f"{a2.shape=}")
        print(f"{len(longest_lifetimes)=}")

        quadratic_params = []

        # Plot the ellipse with the translation
        for i in range(len(longest_lifetimes)):
            print(i)
            A = a2[2 + i]
            B = a2[0]
            k = a2[1]
            C = k + A
            D = a1[i * 3]
            E = a1[i * 3 + 1]
            F = a1[i * 3 + 2]

            A = A.real
            B = B.real
            C = C.real
            D = D.real
            E = E.real
            F = F.real

            print(A, B, C, D, E, F)

            # calculate eccentricity
            e = 4 * A * C - B**2
            h, k, theta, a, b = quadratic_to_ellipse(A, B, C, D, E, F)
            print("Eccentricity:", e)

            if e <= 0 or e >= 1:
                print("This conic is not an ellipse.")
                continue

            if a < 0 or b < 0:
                print("This is a degenerate ellipse.")
                continue

            if (
                np.isnan(h)
                or np.isnan(k)
                or np.isnan(theta)
                or np.isnan(a)
                or np.isnan(b)
            ):
                print("This is a degenerate ellipse.")
                continue

            print("Ellipse is valid")
            quadratic_params.append([A, B, C, D, E, F])
            plot_ellipse(img, h, k, theta, a, b)

        v1_list = []
        v2_list = []
        for i in range(len(quadratic_params)):
            A, B, C, D, E, F = quadratic_params[i]
            A *= img.shape[0] ** 2
            B *= img.shape[0] * img.shape[1]
            C *= img.shape[1] ** 2
            D *= img.shape[0]
            E *= img.shape[1]
            F *= 1
            MM = np.array([[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]])
            eigvals, eigvecs = np.linalg.eig(MM)
            sorted_idx = np.argsort(eigvals)
            eigvals = eigvals[sorted_idx]
            eigvecs = eigvecs[:, sorted_idx]

            # if np.linalg.det(MM) <= 0:
            print(f"{np.linalg.det(MM)=}")
            lam3, lam1, lam2 = eigvals
            # if lam1<0:
            #     lam1 = -lam1
            u3, u1, u2 = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]

        # else:
            #     # continue
            #     print(f"{np.linalg.det(MM)=}")
            #     lam1, lam2, lam3 = eigvals
            #     u1, u2, u3 = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]

            print(f"{lam1=}")
            print(f"{lam2=}")
            print(f"{lam3=}")
            # u3, u1, u2 = reordered_eigvecs[:, 0], reordered_eigvecs[:, 1], reordered_eigvecs[:, 2]

            v1 = (
                np.sqrt((lam2 - lam1) / (lam2 - lam3)) * u2
                + np.sqrt((lam1 - lam3) / (lam2 - lam3)) * u3
            )
            v2 = (
                np.sqrt((lam2 - lam1) / (lam2 - lam3)) * u2
                - np.sqrt((lam1 - lam3) / (lam2 - lam3)) * u3
            )
            # if no nan
            if not np.isnan(v1).any() and not np.isnan(v2).any():
                v1_list.append(v1)
                v2_list.append(v2)
                print(f"{v1=}")
                print(f"{v2=}")
                print(eigvals)
                print(eigvecs)

        # print the mean of v1 and v2
        print(f"{v1_list=}")
        v1_mean = np.mean(v1_list, axis=0)
        v2_mean = np.mean(v2_list, axis=0)

        # ensure normalized
        v1_mean = v1_mean / np.linalg.norm(v1_mean)
        v2_mean = v2_mean / np.linalg.norm(v2_mean)
        v3_mean = -v1_mean / np.linalg.norm(v1_mean)
        v4_mean = -v2_mean / np.linalg.norm(v2_mean)
        print(f"{v1_mean=}")
        print(f"{v2_mean=}")
        print(f"{v3_mean=}")
        print(f"{v4_mean=}")

        candidate_vs.extend([v1_mean, v2_mean, v3_mean, v4_mean])

    # PLOT these vectors on the image
    # Plot the ellipse with the translation
    c = np.array([cx, cy]).astype(int)

    NC = NC  # @ opencv_to_blender

    # True
    true_v_world = np.array([0, 0, 1])  # planeto-centric for mesh
    true_v_world = asteroid_pose[:3, :3] @ true_v_world
    true_v_camera = blender_to_opencv @ np.linalg.inv(NC) @ true_v_world
    true_v_camera = true_v_camera / np.linalg.norm(true_v_camera)
    cv2.arrowedLine(
        img,
        c,
        c + (200 * true_v_camera[:2]).astype(int),
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

    # # find closest
    print(f"{candidate_vs=}")
    for can in candidate_vs:
        print(can)
    closest_idx = np.argmin([np.linalg.norm(true_v_camera - v) for v in candidate_vs])

    # Candidates
    # for v in candidate_vs:
    cv2.arrowedLine(
        img,
        c,
        c + (200 * candidate_vs[closest_idx][:2]).astype(int),
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"True (C): {np.round(true_v_camera, 3)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"Est (C): {np.round(candidate_vs[closest_idx], 3)}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"True (I): {np.round(true_v_world, 3)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"Est (I): {np.round(NC @ opencv_to_blender @ candidate_vs[closest_idx], 3)}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )

    C_BR = np.array([br_cx, br_cy, 1])
    C_TR = np.array([cx, cy, 1])

    pos *= 1e3  # convert from km to m

    def project_image_point_to_camera_frame(point, intrinsic):
        return np.linalg.inv(intrinsic) @ point

    def project_image_point_to_world_frame(point, intrinsic, extrinsic):
        print(f"{point=}")
        print(f"{intrinsic=}")
        print(f"{extrinsic=}")
        camera_pos = extrinsic[:3, 3]
        camera_rot = extrinsic[:3, :3]
        camera_frame_point = project_image_point_to_camera_frame(point, intrinsic)
        return (
            camera_pos
            - np.linalg.norm(camera_pos)
            / np.linalg.norm(camera_frame_point)
            * camera_rot
            @ camera_frame_point
        )

    r_rc = project_image_point_to_world_frame(C_BR, K, homogeneous)
    r_tr = project_image_point_to_world_frame(C_TR, K, homogeneous)

    print(f"{r_rc=}")
    print(f"{r_tr=}")

    # Plot all feature histories as a dashed red line
    for feature in longest_lifetimes:
        history = np.array(feature["history"], dtype=np.int32)  # Convert to int32
        history = history.reshape((-1, 1, 2))
        cv2.polylines(img, [history], False, (255, 0, 0), 1, cv2.LINE_AA)

    # Show the image
    cv2.imshow("Translated Ellipse", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
