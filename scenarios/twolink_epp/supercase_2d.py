import os, sys; sys.path.insert(0, os.path.abspath("."))
import subprocess
import numpy as np
from time import sleep
import scenarios.twolink_epp.case_definition as case_definition

path = os.path.join("scenarios", "twolink_epp")
result_path = os.path.join("results", "twolink_epp")

no_epp_cases = np.arange(62, 283)
epp_cases = np.arange(283, 504)


with open("environment_setup.txt", "r") as f:
    environment_setup_string = f.read()

for case in no_epp_cases:
    case_path = os.path.join(result_path, case_definition.name(case), case_definition.subcase_name(case), "result.csv")
    if os.path.exists(case_path):
        print(f"skipping case {case} with subcase name {case_definition.subcase_name(case)}; result.csv already exists")
        continue
    subprocess.run(["pipenv", "run", "python",
                    os.path.join(path, "orchestrate.py"),
                    "--time", "1-00:00:00", "--mailtype", "FAIL,TIME_LIMIT",
                    "--bundle", "64", f"{case}"])
    print(f"Finished submitting case {case}.")
    sleep(3)


for case in epp_cases:
    case_path = os.path.join(result_path, case_definition.name(case), case_definition.subcase_name(case), "result.csv")
    if os.path.exists(case_path):
        print(f"skipping case {case} with subcase name {case_definition.subcase_name(case)}; result.csv already exists")
        continue
    subprocess.run(["pipenv", "run", "python",
                    os.path.join(path, "orchestrate.py"),
                    "--time", "1-00:00:00", "--mailtype", "FAIL,TIME_LIMIT",
                    "--bundle", "16", f"{case}"])
    print(f"Finished submitting case {case}.")
    sleep(3)
