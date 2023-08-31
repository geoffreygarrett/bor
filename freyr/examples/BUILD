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
