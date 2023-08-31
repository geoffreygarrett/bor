load(
    "@pip_example//:requirements.bzl",
    "data_requirement",
    "entry_point",
    "requirement",
)
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library", "pybind_stubgen")

cc_library(
    name = "freyr_cpp",
    hdrs = glob(["include/**/*.h"]),
    defines = [
        "USE_TBB",
    ],
    includes = ["include"],
    deps = [
        "@com_github_eigen_eigen//:header_lib",
        "@com_github_oneapi_onetbb//:tbb",
        #        "@com_github_opencv_opencv//:core",
    ],
)

pybind_library(
    name = "freyr_pybind",
    hdrs = glob(["include/pybind.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":freyr_cpp",
    ],
)

pybind_extension(
    name = "core",
    srcs = ["src/core.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":freyr_pybind",
    ],
)

py_library(
    name = "freyr",
    srcs = [
        #        ":py.files",
    ],
    data = [
        ":core",
    ],
    imports = [
        "..",
    ],
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
    ],
)

py_binary(
    name = "main2",
    srcs = ["dev/main2.py"],
    deps = [
        ":freyr",
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "dev_freyr",
    srcs = ["dev/main.py"],
    main = "dev/main.py",
    deps = [
        requirement("matplotlib"),
    ],
)

cc_binary(
    name = "test_crtp",
    srcs = ["dev/test_crtp.cpp"],
    deps = [":freyr_cpp"],
)
