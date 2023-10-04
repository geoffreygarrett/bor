import os
import subprocess

try:
    os.environ[
        "OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT"
    ] = "0"
    # subprocess.run(
    #     [
    #         os.environ.get("BLENDERPROC_BIN", "blenderproc"),
    #         "debug",
    #         "freyr/examples/example_blenderproc_main.py",
    #     ]
    # )
    subprocess.run(
        [
            os.environ.get("BLENDERPROC_BIN", "blenderproc"),
            "run",
            "freyr/examples/example_blenderproc_main.py",
        ]
    )

except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
