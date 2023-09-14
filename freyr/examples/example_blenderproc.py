import subprocess
import os

try:

    BASE_PATH = os.environ["BLENDERPROC_BASE_PATH"]

    print(BASE_PATH)
    # subprocess.run(["blenderproc", "example_blenderproc_main.py"], cwd='./')
    blender_proc_bin = os.environ["BLENDERPROC_BIN"]
    print(os.environ["BLENDERPROC_BIN"])
    os.environ["OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT"] = "0"
    subprocess.run([blender_proc_bin, "run", "freyr/examples/example_blenderproc_main.py"], cwd='./')
    # subprocess.run("blenderproc example_blenderproc_main.py", shell=True)

except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
