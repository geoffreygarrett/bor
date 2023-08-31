def generate_prism_vertices_faces():
    # Define vertices for the triangular prism
    vertices = [
        # Bottom triangle
        [0.0, 0.0, 0.0],  # vertex 0
        [1.0, 0.0, 0.0],  # vertex 1
        [0.5, 0.5 * (3 ** 0.5), 0.0],  # vertex 2

        # Top triangle
        [0.0, 0.0, 1.0],  # vertex 3
        [1.0, 0.0, 1.0],  # vertex 4
        [0.5, 0.5 * (3 ** 0.5), 1.0]  # vertex 5
    ]

    # Define faces for the triangular prism (each face is defined by 3 vertex indices)
    faces = [
        # Bottom triangle
        [0, 1, 2],
        # Top triangle
        [3, 4, 5],
        # Side rectangles (broken into two triangles each)
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4],
        [2, 0, 3],
        [2, 3, 5]
    ]
    return vertices, faces

def generate_pyramid_vertices_faces():
    vertices = [
        [0.0, 0.0, 0.0],  # base vertex 0
        [1.0, 0.0, 0.0],  # base vertex 1
        [1.0, 1.0, 0.0],  # base vertex 2
        [0.0, 1.0, 0.0],  # base vertex 3
        [0.5, 0.5, 1.0],  # apex vertex 4
    ]
    faces = [
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
        [0, 3, 2],
        [0, 2, 1],
    ]
    return vertices, faces

vertices, faces = generate_prism_vertices_faces()

# rotate vertices by 30 degrees around the x-axis
vertices = np.array(vertices, dtype=np.float32)
deg = 10
vertices = np.matmul(vertices, np.array(
    [[1, 0, 0],
     [0, np.cos(np.deg2rad(30)), -np.sin(np.deg2rad(deg))],
     [0, np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))]]))

position = [1.0, 1.0, 0.0]  # Position in world coordinates



# Create mesh shape
test_mesh = scene.entity(shape=shape.mesh(vertices, faces, position),
                         material=my_material,
                         # material=material.diffuse(color=[1.0, 0.0, 0.0]),
                         )
