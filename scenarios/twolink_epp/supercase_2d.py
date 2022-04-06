import os, sys; sys.path.insert(0, os.path.abspath("."))
import subprocess
import numpy as np

path = os.path.join("scenarios", "twolink_epp")

# schedules the cases of 2dplots in
no_epp_cases = np.arange(6, 10)
epp_cases = np.arange(10, 14)


with open("environment_setup.txt", "r") as f:
    environment_setup_string = f.read()

for case in no_epp_cases:
    subprocess.run(["pipenv", "run", "python",
                    os.path.join(path, "orchestrate.py"),
                    "--time", "0-00:30:00", "--mailtype", "FAIL",
                    f"{case}"])


for case in epp_cases:
    subprocess.run(["pipenv", "run", "python",
                    os.path.join(path, "orchestrate.py"),
                    "--time", "0-00:45:00", "--mailtype", "FAIL",
                    f"{case}"])
