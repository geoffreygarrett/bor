load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

#
#def freyr_dependencies():
#    maybe(
#        http_archive,
#        name = "com_github_eigen_eigen",
#        build_file = "@//odin:external/eigen.BUILD",
#        sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
#        strip_prefix = "eigen-3.4.0",
#        urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],
#    )

#https://github.com/opencv/opencv/releases/tag/4.8.0

def freyr_dependencies():
    #    maybe(
    #        git_repository,
    #        name = "com_github_rookfighter_nvision",
    #        remote = "https://github.com/Rookfighter/nvision.git",
    #        commit = "5723269d336aa2d85a7772bddf6af29c60dfe72f",
    #        build_file_content = """
    #
    #load("@rules_cc//cc:defs.bzl", "cc_library")
    #
    ## header only
    #cc_library(
    #    name = "nvision",
    #    hdrs = glob(["**/*.h"]),
    #    includes = ["."],
    #    visibility = ["//visibility:public"],
    #    )
    #""",
    #    )

    maybe(
        http_archive,
        name = "com_github_opencv_opencv",
        build_file_content = """
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

# ----------------------------------
# OpenCV Configuration
# ----------------------------------
filegroup(
    name = "opencv_sources",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "core",
    copts = [
        "-GNinja",
        "-DBUILD_LIST=core,highgui,imgcodecs,imgproc",
    ],
    lib_source = ":opencv_sources",
#    hdrs = glob(["**/*.h", "**/*.hpp"]),
    out_include_dir = "include",
    visibility = ["//visibility:public"],
)
""",
        sha256 = "c20bb83dd790fc69df9f105477e24267706715a9d3c705ca1e7f613c7b3bad3d",
        urls = ["https://github.com/opencv/opencv/archive/4.5.4.tar.gz"],
        strip_prefix = "opencv-4.5.4",
    )
