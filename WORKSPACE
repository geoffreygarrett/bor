# ---------------------------------------------------------
# Workspace name
# ---------------------------------------------------------
workspace(name = "bor")

local_repository(
    name = "odin",
    path = "odin",
)

load("@odin//:deps.bzl", "odin_dependencies")

odin_dependencies()

local_repository(
    name = "pydin",
    path = "pydin",
)

load("@pydin//:deps.bzl", "pydin_dependencies")

pydin_dependencies()

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@rules_pkg//pkg:deps.bzl", "rules_pkg_dependencies")

rules_foreign_cc_dependencies()

rules_pkg_dependencies()

#register_execution_platforms(
#    ":x64_windows-clang-cl",
#)
#
#register_toolchains(
#    "@local_config_cc//:cc-toolchain-x64_windows-clang-cl",
#)

#
## ---------------------------------------------------------
## Initial setup and tools
## ---------------------------------------------------------
## Load HTTP functions for fetching archives
#load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
#
## Load and setup foreign_cc rules
##http_archive(
##    name = "rules_foreign_cc",
##    sha256 = "6041f1374ff32ba711564374ad8e007aef77f71561a7ce784123b9b4b88614fc",
##    strip_prefix = "rules_foreign_cc-0.8.0",
##    urls = ["https://github.com/bazelbuild/rules_foreign_cc/archive/0.8.0.tar.gz"],
##)
#
## ---------------------------------------------------------
## Package definitions
## ---------------------------------------------------------
##http_archive(
##    name = "rules_pkg",
##    sha256 = "360c23a88ceaf7f051abc99e2e6048cf7fe5d9af792690576554a88b2013612d",
##    strip_prefix = "rules_pkg-0.9.1",
##    urls = [
##        "https://github.com/bazelbuild/rules_pkg/archive/0.9.1.tar.gz",
##    ],
##)
#
## ---------------------------------------------------------
## Dependencies
## ---------------------------------------------------------
## Flags library
#
## PYBIND STUFF
##http_archive(
##    name = "com_github_pybind_pybind11",
##    build_file = "pybind11.BUILD",
##    sha256 = "5fbebeb7162d336ed06f119ba32f336f2a67c0a492c7802e48f16d4cfb8b0fbf",
##    strip_prefix = "pybind11-2.10.4",
##    urls = ["https://github.com/pybind/pybind11/archive/v2.10.4.zip"],
##)
#
##oad("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
#
#http_archive(
#    name = "rules_python",
#    sha256 = "8c8fe44ef0a9afc256d1e75ad5f448bb59b81aba149b8958f02f7b3a98f5d9b4",
#    strip_prefix = "rules_python-0.13.0",
#    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.13.0.tar.gz",
#)
#
#load("@rules_python//python:repositories.bzl", "python_register_toolchains")
#
#python_register_toolchains(
#    name = "python3_10",
#    python_version = "3.10",
#)
load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python3_10",
    python_version = "3.10",
)

load("@rules_python//python:pip.bzl", "pip_install", "pip_parse")
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
load("@python3_10//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_install", "pip_parse")

python_configure(
    name = "local_config_python",
    python_interpreter_target = interpreter,
)

#
#pip_parse(
#    python_interpreter_target = interpreter,
#)

pip_install(
    # (Optional) You can provide extra parameters to pip.
    # Here, make pip output verbose (this is usable with `quiet = False`).
    #extra_pip_args = ["-v"],

    # (Optional) You can exclude custom elements in the data section of the generated BUILD files for pip packages.
    # Exclude directories with spaces in their names in this example (avoids build errors if there are such directories).
    #pip_data_exclude = ["**/* */**"],

    # (Optional) You can provide a python_interpreter (path) or a python_interpreter_target (a Bazel target, that
    # acts as an executable). The latter can be anything that could be used as Python interpreter. E.g.:
    # 1. Python interpreter that you compile in the build file (as above in @python_interpreter).
    # 2. Pre-compiled python interpreter included with http_archive
    # 3. Wrapper script, like in the autodetecting python toolchain.
    #
    # Here, we use the interpreter constant that resolves to the host interpreter from the default Python toolchain.
    python_interpreter_target = interpreter,

    # (Optional) You can set quiet to False if you want to see pip output.
    #quiet = False,

    # (Optional) You can set an environment in the pip process to control its
    # behavior. Note that pip is run in "isolated" mode so no PIP_<VAR>_<NAME>
    # style env vars are read, but env vars that control requests and urllib3
    # can be passed.
    #environment = {"HTTP_PROXY": "http://my.proxy.fun/"},

    # Uses the default repository name "pip"
    requirements = "//:requirements-lock.txt",
)

load("@pip//:requirements.bzl", "install_deps")

# Initialize repositories for all packages in requirements.txt.
install_deps()

#http_archive(
#    name = "pybind11_bazel",
#    sha256 = "e1fe52ad3468629772c50c67a93449f235aed650a0fe8e89a22fbff285f677a1",
#    strip_prefix = "pybind11_bazel-9a24c33cbdc510fa60ab7f5ffb7d80ab89272799",
#    urls = ["https://github.com/pybind/pybind11_bazel/archive/b162c7c88a253e3f6b673df0c621aca27596ce6b.zip"],
#)
#
#http_archive(
#    name = "pybind11",
#    build_file = "@pybind11_bazel//:pybind11.BUILD",
#    sha256 = "115bc49b69133dd8a7a7f824b669826ff6a35bc70a28ce2cf987d57c50a6543a",
#    strip_prefix = "pybind11-2.10.4",
#    urls = ["https://github.com/pybind/pybind11/archive/v2.10.4.zip"],
#)
#
#load("@pybind11_bazel//:python_configure.bzl", "python_configure")
#
#python_configure(
#    name = "local_config_python",
#    python_interpreter_target = interpreter,
#)
#
## Pagmo2 library
##http_archive(
##    name = "com_github_esa_pagmo2",
##    build_file = "pagmo.BUILD",
##    sha256 = "9ffebf71e4b0f8591f8eb8c6a03ef33b5cd2c22a41bd16ed6e0e6ae0dd634e16",
##    strip_prefix = "pagmo2-2.19.0",
##    urls = ["https://github.com/esa/pagmo2/archive/v2.19.0.zip"],
##)
##
### NLOpt library
##http_archive(
##    name = "com_github_stevengj_nlopt",
##    build_file = "nlopt.BUILD",
##    sha256 = "a108541a18b3d230237a8a6a6c8783107dc8b5f10a98147f09abf863711960fc",
##    strip_prefix = "nlopt-2.7.1",
##    urls = ["https://github.com/stevengj/nlopt/archive/v2.7.1.zip"],
##)
##
### IPopt library
##http_archive(
##    name = "com_github_coin_or_ipopt",
##    build_file = "ipopt.BUILD",
##    sha256 = "4e52f279f0582d57895eb4516dd9e4607a0d5274c0d97978a91ee45aff0a66b5",
##    strip_prefix = "Ipopt-releases-3.14.12",
##    urls = ["https://github.com/coin-or/Ipopt/archive/refs/tags/releases/3.14.12.zip"],
##)
##
### LAPACK library
##http_archive(
##    name = "com_github_reference_lapack_lapack",
##    build_file = "lapack.BUILD",
##    sha256 = "4800d3c572ca58cbecfa73d24ce77b51a6896b570147ef96500e34f4036af8c1",
##    strip_prefix = "lapack-3.11.0",
##    urls = ["https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.11.0.zip"],
##)
##
##http_archive(
##    name = "com_github_coin_or_tools_thirdparty_mumps",
##    build_file = "mumps.BUILD",
##    sha256 = "cdbf3e54e4d7d82309469ac4158c756496201a94fff3ce3254b24eae41ddaaea",
##    strip_prefix = "ThirdParty-Mumps-releases-3.0.4",
##    urls = ["https://github.com/coin-or-tools/ThirdParty-Mumps/archive/refs/tags/releases/3.0.4.zip"],
##)
##http_archive(
##    name = "boost",
##    sha256 = "da3411ea45622579d419bfda66f45cd0f8c32a181d84adfa936f5688388995cf",
##    strip_prefix = "boost_1_68_0",
##    urls = ["https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz"],
##)
#
### https://github.com/spietras/rules_conda/releases/tag/0.2.0
##http_archive(
##    name = "rules_conda",
##    sha256 = "ee187c4902f5f8da85fcba9943064fd38b2a75f80ed0a2844b34cf9e27cb5990",
##    url = "https://github.com/spietras/rules_conda/releases/download/0.2.0/rules_conda-0.2.0.zip",
##)
#
##load("@rules_conda//:defs.bzl", "conda_create", "load_conda", "register_toolchain")
##load("@//:bazel/tools/conda_env.bzl", "conda_environment")
#
##conda_environment(
##    name = "conda",
##    build_file = "//:external/conda.BUILD",
##    #    env_name = "my_env",
##    environment_file = "//:environment.yml",
##    packages = [
##        "boost-cpp",
##        "pagmo-devel",
##    ],
##)

#load_conda(quiet = False)
#
#conda_create(
#    name = "env",
#    environment = "@//:environment.yml",
#    quiet = False,
#    #    use_mamba = True,
#    visibility = ["//visibility:public"],
#)
#
#register_toolchain(env = "env")
