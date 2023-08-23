# ---------------------------------------------------------
# Workspace name
# ---------------------------------------------------------
workspace(name = "bor")

# ---------------------------------------------------------
# Local repository
# ---------------------------------------------------------
local_repository(
    name = "pybind11_bazel",
    path = "pydin/external/pybind11_bazel",
)

# ---------------------------------------------------------
# Odin dependencies
# ---------------------------------------------------------
load("@@bor//odin:repositories.bzl", "odin_dependencies")

odin_dependencies()

# ---------------------------------------------------------
# Pydin dependencies
# ---------------------------------------------------------
load("@@bor//pydin:repositories.bzl", "pydin_dependencies")

pydin_dependencies()

# ---------------------------------------------------------
# Rules foreign CC dependencies
# ---------------------------------------------------------
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@rules_pkg//pkg:deps.bzl", "rules_pkg_dependencies")

rules_foreign_cc_dependencies()

rules_pkg_dependencies()

# ---------------------------------------------------------
# Python toolchains
# ---------------------------------------------------------
load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python3_10",
    python_version = "3.10",
)

load("@python3_10//:defs.bzl", "interpreter")

# ---------------------------------------------------------
# Pybind11 interpreter configuration
# ---------------------------------------------------------
load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(
    name = "local_config_python",
    python_interpreter_target = interpreter,
)

# ---------------------------------------------------------
# Pip environment: build, test, example, benchmark
# ---------------------------------------------------------
load("@@bor//pydin/environment:dependency_set.bzl", "pip_dependency_set")

# Build dependencies
pip_dependency_set(
    name = "pip_build",
    interpreter = interpreter,
    requirements_path = "@@bor//pydin/environment/build",
)

load("@pip_build//:requirements.bzl", pip_install_build_deps = "install_deps")

pip_install_build_deps()

# Test dependencies #######################################
pip_dependency_set(
    name = "pip_test",
    interpreter = interpreter,
    requirements_path = "@@bor//pydin/environment/test",
)

load("@pip_test//:requirements.bzl", pip_install_test_deps = "install_deps")

pip_install_test_deps()

# Example dependencies ####################################
pip_dependency_set(
    name = "pip_example",
    interpreter = interpreter,
    requirements_path = "@@bor//pydin/environment/example",
)

load("@pip_example//:requirements.bzl", pip_install_example_deps = "install_deps")

pip_install_example_deps()

# Benchmark dependencies ##################################
pip_dependency_set(
    name = "pip_benchmark",
    interpreter = interpreter,
    requirements_path = "@@bor//pydin/environment/benchmark",
)

load("@pip_benchmark//:requirements.bzl", pip_install_benchmark_deps = "install_deps")

pip_install_benchmark_deps()
