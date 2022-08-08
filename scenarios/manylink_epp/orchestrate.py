"""Orchestrate a case with slurm jobs.

This will setup all the necessary slurm jobs that will complete the run.
"""
import os, sys; sys.path.insert(0, os.path.abspath("."))
import subprocess
import argparse
import scenarios.manylink_epp.case_definition as case_definition
import re
from libs.aux_functions import assert_dir

email = "julius.wallnoefer@fu-berlin.de"
base_path = os.path.join("results", "manylink_epp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Orchestrate a case with slurm jobs.")
    # parser.add_argument("base_path", help="Where the results will be stored.")
    parser.add_argument("case", type=int, help="the case number")
    parser.add_argument("--time", required=True, help="specify time in DAYS-HH:MM:SS format")
    parser.add_argument("--parts", help="optionally, specify just some parts using sbatch --array syntax. Default: run all non-existing parts")
    parser.add_argument("--mem", default=2048, help="memory in MB per part run")
    parser.add_argument("--memcollect", default=1024, help="memory in MB for result collection step")
    parser.add_argument("--mailtype", default="ALL", help="mail-type option for sbatch. Default: ALL")
    parser.add_argument("--bundle", type=int, default=1, help="how many parts to bundle in one slurm job, works only if --parts is not specified and the number of all parts is divisible by bundle")
    parser.add_argument("--nocollect", default=False, action="store_const", const=True, help="Set this flag to skip the collection step. Useful if many orchestrates are launched at the same time and collection is handled via a supercase.")
    parser.add_argument("--runexisting", default=False, action="store_const", const=True, help="Set this flag to also run parts that already have results. If this is not specified and neither --bundle nor --parts are used, it avoids creating jobs for these parts in the first place.")
    args = parser.parse_args()
    case = args.case
    case_name = case_definition.name(case)
    case_path = os.path.join(base_path, case_name)
    subcase_name = case_definition.subcase_name(case)
    subcase_path = os.path.join(case_path, subcase_name)
    job_name = subcase_name + "_" + case_name
    if args.parts is None:
        if args.bundle == 1 and not args.runexisting:
            output_path = os.path.join(subcase_path, "parts")
            parts_to_run = []
            nparts = case_definition.num_parts(case)
            for part_index in range(nparts):
                if not os.path.exists(os.path.join(output_path, f"part{part_index}.csv")):
                    parts_to_run += [str(part_index)]
            if not parts_to_run:
                print(f"No parts for case {args.case} found that need to be run. Use --runexisting to run anyway.")
                sys.exit(0)
            if len(parts_to_run) == nparts:
                array_entry = f"0-{nparts-1}"
            else:
                array_entry = ",".join(parts_to_run)
        else:
            nparts = case_definition.num_parts(case)
            if nparts % args.bundle != 0:
                raise ValueError(f"The number of parts {nparts} must be divisible by --bundle {args.bundle}")
            num_array_jobs = nparts // args.bundle
            array_entry = f"0-{num_array_jobs - 1}"
    else:
        if args.bundle != 1:
            raise ValueError(f"--bundle only works if --parts is not specified. Was called with --bundle {args.bundle} and --parts {args.parts}")
        array_entry = args.parts
    with open("environment_setup.txt", "r") as f:
        environment_setup_string = f.read()
    if args.runexisting:
        run_string = f"pipenv run python scenarios/twolink_epp/run_two_link_epp.py --runexisting {subcase_path} {case}"
    else:
        run_string = f"pipenv run python scenarios/twolink_epp/run_two_link_epp.py {subcase_path} {case}"
    if args.bundle == 1:
        run_instructions = f"{run_string} $SLURM_ARRAY_TASK_ID"
    else:
        run_instructions = "\n".join([f"{run_string} $(({args.bundle} * $SLURM_ARRAY_TASK_ID + {i}))" for i in range(args.bundle)])

    sbatch_text = f"""#!/bin/bash

#SBATCH --job-name={job_name}     # Job name, will show up in squeue output
#SBATCH --ntasks=1                     # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time={args.time}              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu={args.mem}              # Memory per cpu in MB (see also --mem)
#SBATCH --array={array_entry}
#SBATCH --output=out_files/%x_%a.out           # File to which standard out will be written
#SBATCH --error=out_files/%x_%a.err            # File to which standard err will be written
#SBATCH --mail-type={args.mailtype}                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user={email}   # Email to which notifications will be sent
#SBATCH --qos=standard

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID

{environment_setup_string}
{run_instructions}
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
    if not args.nocollect:
        jid1 = re.search("([0-9]+)", out1).group(1)
        collect_text = f"""#!/bin/bash

#SBATCH --job-name=c_{job_name}     # Job name, will show up in squeue output
#SBATCH --ntasks=1                     # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=0-00:05:00            # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu={args.memcollect}              # Memory per cpu in MB (see also --mem)
#SBATCH --output=out_files/%x.out           # File to which standard out will be written
#SBATCH --error=out_files/%x.err            # File to which standard err will be written
#SBATCH --mail-type={args.mailtype}                # Type of email notification- BEGIN,END,FAIL,ALL
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
        submit2 = subprocess.run(["sbatch", f"--dependency=afterany:{jid1}", "--deadline=now+14days", collect_file], capture_output=True)
        out2 = submit2.stdout.decode("ascii")
        err2 = submit2.stderr.decode("ascii")
        if err2:
            raise RuntimeError(err2)
