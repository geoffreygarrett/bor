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
