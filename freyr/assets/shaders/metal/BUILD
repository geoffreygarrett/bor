load("@rules_metal//:metal.bzl", "metal_binary", "metal_library")

# Produces shaders.metallib
metal_binary(
    name = "default",
    srcs = [
        "op_addition.metal",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "metal_backend",
    srcs = [
        "metal_backend.cpp",
    ],
    hdrs = [
        "metal_backend.h",
        "op_addition.h",
        "op_base.h",
    ],
    data = [
        ":default",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@developer_apple_com_metal_cpp//:metal_cpp",
    ],
)
