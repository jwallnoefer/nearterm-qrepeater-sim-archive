"""Orchestrate a case with slurm jobs.

This will setup all the necessary slurm jobs that will complete the run.
"""
import os, sys; sys.path.insert(0, os.path.abspath("."))
import subprocess
import argparse
import scenarios.twolink_epp.case_definition as case_definition
import re
from libs.aux_functions import assert_dir

email = "julius.wallnoefer@fu-berlin.de"
base_path = os.path.join("results", "twolink_epp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Orchestrate a case with slurm jobs.")
    # parser.add_argument("base_path", help="Where the results will be stored.")
    parser.add_argument("case", type=int, help="the case number")
    parser.add_argument("--time", required=True, help="specify time in DAYS-HH:MM:SS format")
    parser.add_argument("--parts", help="optionally, specify just some parts using sbatch --array syntax. Default: run all")
    parser.add_argument("--mem", default=2048, help="memory in MB per part run")
    parser.add_argument("--memcollect", default=1024, help="memory in MB for result collection step")
    args = parser.parse_args()
    case = args.case
    case_name = case_definition.name(case)
    case_path = os.path.join(base_path, case_name)
    subcase_name = case_definition.subcase_name(case)
    subcase_path = os.path.join(case_path, subcase_name)
    job_name = subcase_name + "_" + case_name
    if args.parts is None:
        nparts = case_definition.num_parts(case)
        parts = f"0-{nparts - 1}"
    else:
        parts = args.parts
    with open("environment_setup.txt", "r") as f:
        environment_setup_string = f.read()
    sbatch_text = f"""#!/bin/bash

#SBATCH --job-name={job_name}     # Job name, will show up in squeue output
#SBATCH --ntasks=1                     # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time={args.time}              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu={args.mem}              # Memory per cpu in MB (see also --mem)
#SBATCH --array={parts}
#SBATCH --output=out_files/%x_%a.out           # File to which standard out will be written
#SBATCH --error=out_files/%x_%a.err            # File to which standard err will be written
#SBATCH --mail-type=ALL                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user={email}   # Email to which notifications will be sent
#SBATCH --qos=standard

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID

{environment_setup_string}
pipenv run python scenarios/twolink_epp/run_two_link_epp.py {subcase_path} {case} $SLURM_ARRAY_TASK_ID
"""
    assert_dir("out_files")
    assert_dir(case_path)
    assert_dir(subcase_path)
    assert_dir(os.path.join(subcase_path, "parts"))
    sbatch_file = os.path.join(subcase_path, f"run_case_{case}.sh")
    with open(sbatch_file, "w") as f:
        f.write(sbatch_text)
    submit1 = subprocess.run(["sbatch", sbatch_file], capture_output=True)

    out1 = submit1.stdout.decode("ascii")
    err1 = submit1.stderr.decode("ascii")
    if err1:
        raise RuntimeError(err1)
    jid1 = re.search("([0-9]+)", out1).group(1)
    collect_text = f"""#!/bin/bash

#SBATCH --job-name=c_{job_name}     # Job name, will show up in squeue output
#SBATCH --ntasks=1                     # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=0-00:05:00            # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu={args.memcollect}              # Memory per cpu in MB (see also --mem)
#SBATCH --output=out_files/%x.out           # File to which standard out will be written
#SBATCH --error=out_files/%x.err            # File to which standard err will be written
#SBATCH --mail-type=ALL                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user={email}   # Email to which notifications will be sent
#SBATCH --qos=standard

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID

{environment_setup_string}
pipenv run python scenarios/twolink_epp/run_two_link_epp.py --collect {subcase_path} {case}
"""
    collect_file = os.path.join(subcase_path, f"collect_case_{case}.sh")
    with open(collect_file, "w") as f:
        f.write(collect_text)
    submit2 = subprocess.run(["sbatch", f"--dependency=afterok:{jid1}", "--deadline=now+14days", collect_file], capture_output=True)
    out2 = submit2.stdout.decode("ascii")
    err2 = submit2.stderr.decode("ascii")
    if err2:
        raise RuntimeError(err2)
