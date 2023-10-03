load(
    "@pip_example//:requirements.bzl",
    "data_requirement",
    "entry_point",
    "requirement",
)

py_binary(
    name = "example_basic",
    srcs = ["example_basic.py"],
    data = [
        "@//freyr/examples/data:bricks.normal.jpeg",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@//freyr",
        requirement("matplotlib"),
        requirement("screeninfo"),
    ],
)

py_binary(
    name = "example_eros",
    srcs = ["example_eros.py"],
    data =
        [
            "@//freyr/examples/data:Eros.face",
            "@//freyr/examples/data:Eros.node",
        ],
    visibility = ["//visibility:public"],
    deps = [
        "@//freyr",
        requirement("matplotlib"),
        requirement("screeninfo"),
    ],
)

alias(
    name = "blenderproc",
    actual = entry_point(
        pkg = "blenderproc",
        script = "blenderproc",
    ),
)

#genrule(
#    name = "example_blenderproc",
#    srcs = [],
#    outs = [],
#    cmd = "$(location :blenderproc) --help",
#    tools = [
#        ":blenderproc",
#    ],
#)
#sh_binary(
#    name = "example_blenderproc",
#    srcs = [
#        "example_blenderproc.py",
#        "example_blenderproc_main.py",
#    ],
#    deps = [
#        ":blenderproc",
#    ],
#)
#

#sh_binary(
#    name = "example_blenderproc",
#    srcs = [
#        "example_blenderproc.sh",
#    ],
#    data =
#        [
#            "@//freyr/examples/data:Eros.node",
#            "@//freyr/examples/data:eros_gaskell_200k_poly.obj",
#            "@//freyr/examples/data:eros_gaskell_50k_poly.obj",
#            "@//freyr/examples/data:eros_grayscale.jpg",
#            ":blenderproc",
#            "example_blenderproc_main.py",
#
#            #        "@//freyr",
#            requirement("matplotlib"),
#            requirement("screeninfo"),
#            requirement("blenderproc"),
#        ],
#    env = {
#        "BLENDERPROC_BIN": "$(location :blenderproc)",
#    },
#    visibility = ["//visibility:public"],
#    #    deps = [
#    #        ":blenderproc",
#    #        #        "@//freyr",
#    #        requirement("matplotlib"),
#    #        requirement("screeninfo"),
#    #    ],
#)

py_binary(
    name = "example_blenderproc",
    srcs = [
        "example_blenderproc.py",
        "example_blenderproc_main.py",
    ],
    data =
        [
            "@//freyr/examples/data:Eros.node",
            "@//freyr/examples/data:eros_gaskell_200k_poly.ply",
            "@//freyr/examples/data:eros_gaskell_50k_poly.ply",
            "@//freyr/examples/data:eros_grayscale.jpg",
        ],
    env = {
        "BLENDERPROC_BIN": "$(location :blenderproc)",
        "BLENDERPROC_BASE_PATH": "freyr/examples/data",
    },
    visibility = ["//visibility:public"],
    deps = [
        ":blenderproc",
        #        "@//freyr",
        requirement("matplotlib"),
        requirement("screeninfo"),
        requirement("scipy"),
    ],
)

# fiklegroup for saved_images/*
filegroup(
    name = "saved_images",
    srcs = glob([
        "**/*",
        "saved_images/*",
        "saved_images2/*",
        "saved_images3/*",
        "saved_images4/*",
        "eros_500_500_45/*",
        "eros_15_0_0_500_500_720/**/*",
        "eros_15_45_0_500_500_720/**/*",
    ]),
)

py_binary(
    name = "example_optical_flow",
    srcs = [
        "example_optical_flow.py",
    ],
    data =
        [
            ":saved_images",
            "@//freyr/examples/data:Eros.node",
            "@//freyr/examples/data:eros_gaskell_200k_poly.ply",
            "@//freyr/examples/data:eros_gaskell_50k_poly.ply",
            "@//freyr/examples/data:eros_grayscale.jpg",
        ],
    env = {
        "BLENDERPROC_BIN": "$(location :blenderproc)",
        "BLENDERPROC_BASE_PATH": "$$(dir @//freyr/examples/data:eros_gaskell_50k_poly.obj)",
    },
    visibility = ["//visibility:public"],
    deps = [
        ":blenderproc",
        requirement("PySide6"),
        requirement("PyQt6"),
        requirement("matplotlib"),
        requirement("screeninfo"),
        requirement("scipy"),
        requirement("opencv-python"),
    ],
)

genrule(
    name = "default_metallib",
    srcs = ["@//freyr/assets/shaders/metal:default"],
    outs = ["default.metallib"],
    cmd = "cp $(location @//freyr/assets/shaders/metal:default) $@",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "example_metal_gpgpu",
    srcs = ["example_metal_gpgpu.cpp"],
    data = [
        #        "@//freyr/assets/shaders/metal:default",
        ":default_metallib",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@//freyr:freyr_cpp",
        "@//freyr/assets/shaders/metal:metal_backend",
        #        "@//freyr/examples:metal_gpgpu",
        #        "@//third_party:metal",
    ],
)

cc_binary(
    name = "example_gpgpu",
    srcs = ["example_gpgpu.cpp"],
    data = [
        #        "@//freyr/assets/shaders/metal:default",
        ":default_metallib",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@//freyr:freyr_cpp",
        "@//freyr/assets/shaders/metal:metal_backend",
        #        "@//freyr/examples:metal_gpgpu",
        #        "@//third_party:metal",
    ],
)

cc_library(
    name = "metal_extensions",
    hdrs = glob([
        "AppKit/**",
        "MetalKit/**",
    ]),
    deps = [],
)

cc_binary(
    name = "example_metal_window",
    srcs = ["example_metal_window.cpp"],
    data = [
        #        "@//freyr/assets/shaders/metal:default",
        ":default_metallib",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@//freyr:freyr_cpp",
        "@//freyr/assets/shaders/metal:metal_backend",
        ":metal_extensions",
        #        "@//freyr/examples:metal_gpgpu",
        #        "@//third_party:metal",
    ],
)
