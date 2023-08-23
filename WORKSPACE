# ---------------------------------------------------------
# Workspace name
# ---------------------------------------------------------
workspace(name = "bor")

local_repository(
    name = "pybind11_bazel",
    path = "pydin/external/pybind11_bazel",
)

load("@@bor//odin:repositories.bzl", "odin_dependencies")

odin_dependencies()

load("@@bor//pydin:repositories.bzl", "pydin_dependencies")

pydin_dependencies()

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@rules_pkg//pkg:deps.bzl", "rules_pkg_dependencies")

rules_foreign_cc_dependencies()

rules_pkg_dependencies()

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

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip",
    python_interpreter_target = interpreter,
    requirements_lock = "@@bor//pydin:requirements_lock.txt",
    requirements_windows = "@@bor//pydin:requirements_windows.txt",
    requirements_darwin = "@@bor//pydin:requirements_darwin.txt",
    requirements_linux = "@@bor//pydin:requirements_linux.txt",
)

pip_parse(
    name = "pip_build",
    python_interpreter_target = interpreter,
    requirements_lock = "@@bor//pydin/pip/requirements/build:requirements_lock.txt",
    requirements_windows = "@@bor//pydin/pip/requirements/build:requirements_windows.txt",
    requirements_darwin = "@@bor//pydin/pip/requirements/build:requirements_darwin.txt",
    requirements_linux = "@@bor//pydin/pip/requirements/build:requirements_linux.txt",
)

pip_parse(
    name = "pip_tests",
    python_interpreter_target = interpreter,
    requirements_lock = "@@bor//pydin/pip/requirements/tests:requirements_lock.txt",
    requirements_windows = "@@bor//pydin/pip/requirements/tests:requirements_windows.txt",
    requirements_darwin = "@@bor//pydin/pip/requirements/tests:requirements_darwin.txt",
    requirements_linux = "@@bor//pydin/pip/requirements/tests:requirements_linux.txt",
)

load("@pip//:requirements.bzl", pip_install_deps = "install_deps")
load("@pip_build//:requirements.bzl", pip_build_install_deps = "install_deps")
load("@pip_tests//:requirements.bzl", pip_tests_install_deps = "install_deps")

# Initialize repositories for all packages in requirements.txt.
pip_install_deps()
pip_build_install_deps()
pip_tests_install_deps()
