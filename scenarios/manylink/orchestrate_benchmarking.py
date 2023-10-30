import os, sys;
sys.path.insert(0, os.path.abspath("."))
from multiprocessing import Pool
from scenarios.manylink.run_benchmarking import cases
import subprocess

run_file = os.path.join("scenarios", "manylink", "run_benchmarking.py")
run_command = ["pipenv", "run", "python", run_file]


def call_run(result_path, case, part):
    part_command = run_command + ["--runexisting", result_path, str(case), str(part)]
    submit = subprocess.run(part_command, capture_output=True)
    err = submit.stderr.decode("ascii")
    if err:
        print(err)


def call_collect(result_path, case):
    collect_command = run_command + ["--collect", result_path, str(case)]
    submit = subprocess.run(collect_command, capture_output=True)
    err = submit.stderr.decode("ascii")
    if err:
        print(err)


if __name__ == "__main__":
    for case, spec in cases.items():
        num_parts = spec["num_parts"]
        subcase_name = spec["subcase_name"]
        result_path = os.path.join("results", "manylink_benchmarking", subcase_name)
        star_args = [(result_path, case, part) for part in range(num_parts)]
        with Pool(6) as pool:
            pool.starmap(call_run, star_args, chunksize=1)
        call_collect(result_path, case)


