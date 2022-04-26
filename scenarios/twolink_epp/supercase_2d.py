import os, sys; sys.path.insert(0, os.path.abspath("."))
import subprocess
import numpy as np
from time import sleep
import scenarios.twolink_epp.case_definition as case_definition

path = os.path.join("scenarios", "twolink_epp")
result_path = os.path.join("results", "twolink_epp")

# schedules the cases of 2dplots in
# no_epp_cases = np.arange(6, 227)
# epp_cases = np.arange(227, 448)
no_epp_cases = np.arange(890, 1111)
epp_cases = np.arange(1111, 1332)


with open("environment_setup.txt", "r") as f:
    environment_setup_string = f.read()

for case in no_epp_cases:
    case_path = os.path.join(result_path, case_definition.name(case), case_definition.subcase_name(case), "result.csv")
    if os.path.exists(case_path):
        print(f"skipping case {case} with subcase name {case_definition.subcase_name(case)}; result.csv already exists" )
        continue
    subprocess.run(["pipenv", "run", "python",
                    os.path.join(path, "orchestrate.py"),
                    "--time", "0-8:00:00", "--mailtype", "FAIL",
                    "--bundle", "32", f"{case}"])
    print(f"Finished submitting case {case}.")
    sleep(5)


for case in epp_cases:
    case_path = os.path.join(result_path, case_definition.name(case), case_definition.subcase_name(case), "result.csv")
    if os.path.exists(case_path):
        print(f"skipping case {case} with subcase name {case_definition.subcase_name(case)}; result.csv already exists" )
        continue
    subprocess.run(["pipenv", "run", "python",
                    os.path.join(path, "orchestrate.py"),
                    "--time", "0-09:00:00", "--mailtype", "FAIL",
                    "--bundle", "16", f"{case}"])
    print(f"Finished submitting case {case}.")
    sleep(5)
