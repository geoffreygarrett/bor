load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

config_setting(
    name = "linux",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

configure_make(
    name = "gsl",
    lib_source = "//:gsl_sources",
    configure_options = select({
        ":windows": ["--disable-static"],
        "//conditions:default": [],
    }),
    env = select({
        ":windows": {
            "CPPFLAGS": "-DGSL_DLL -DWIN32",
            "CXXFLAGS": "-DGSL_DLL -DWIN32",
            "CFLAGS": "-DGSL_DLL -DWIN32",
            "LDFLAGS": "-lcblas",
        },
        "//conditions:default": {
            "LIBS": "-lcblas -lm",
        },
    }),
    copts = select({
        ":windows": ["-DGSL_DLL", "-DWIN32"],
        "//conditions:default": [],
    }),
    linkopts = select({
        ":windows": ["-lcblas"],
        "//conditions:default": ["-lcblas", "-lm"],
    }),
    out_include_dir = "include",
    out_lib_dir = "lib",
    out_static_libs = select({
        ":linux": ["libgsl.a", "libgslcblas.a"],
        ":macos": ["libgsl.a", "libgslcblas.a"],
        ":windows": ["gsl.lib", "gslcblas.lib"],
        "//conditions:default": [],
    }),
    out_shared_libs = select({
        ":linux": ["libgsl.so", "libgslcblas.so"],
        ":macos": ["libgsl.dylib", "libgslcblas.dylib"],
        ":windows": ["gsl.dll", "gslcblas.dll"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "gsl_sources",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
