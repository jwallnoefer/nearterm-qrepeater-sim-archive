import os, sys;
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import argparse
from scenarios.manylink.manylink_benchmarking import run, DefaultManylinkProtocol, CustomManylinkProtocol
from requsim.tools.evaluation import standard_bipartite_evaluation
import pandas as pd
from time import time

cases = {}

case_name = "simple_benchmark"
base_params = {"T_DP": 25, "F_INIT": 0.95}
num_parts = 128
num_links = np.linspace(0, 1024, num=num_parts + 1, dtype=int)[1:]
max_iter = 10000
length = 22000
# default protocol case
case_specification = {
    "name": case_name,
    "subcase_name": "default_protocol",
    "num_parts": num_parts,
    "index": num_links,
    "case_args": {part: {"length": length,
                         "max_iter": max_iter,
                         "params": base_params,
                         "num_links": num_links[part],
                         "protocol_class": DefaultManylinkProtocol,
                  } for part in range(num_parts)
    }
}
cases.update({len(cases): case_specification})
# custom protocol case
case_specification = {
    "name": case_name,
    "subcase_name": "custom_protocol",
    "num_parts": num_parts,
    "index": num_links,
    "case_args": {part: {"length": length,
                         "max_iter": max_iter,
                         "params": base_params,
                         "num_links": num_links[part],
                         "protocol_class": CustomManylinkProtocol,
                  } for part in range(num_parts)
    }
}
cases.update({len(cases): case_specification})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a manylink_benchmarking case.")
    parser.add_argument("result_path", help="Path to case specific result directory.")
    parser.add_argument("case", type=int, help="The case number as defined in case_definition.")
    parser.add_argument("part", type=int, nargs="?", help="Which part of the case to run.")
    parser.add_argument("--collect", default=False, action="store_const", const=True, help="Set this flag to collect results instead of running.")
    parser.add_argument("--runexisting", default=False, action="store_const", const=True, help="Set this flag to run parts regardless of whether they already exists.")
    args = parser.parse_args()

    if args.collect:
        base_index = cases[args.case]["index"]
        index = []
        times = []
        results = pd.DataFrame()
        for part in range(cases[args.case]["num_parts"]):
            try:
                results = pd.concat([results, pd.read_csv(os.path.join(args.result_path, "parts", f"part{part}.csv"))])
            except FileNotFoundError:
                print(f"part{part} not found in collect")
                continue
            index.append(base_index[part])
            time = np.loadtxt(os.path.join(args.result_path, "parts", f"part{part}.time"))
            times += [time]
        results.index = index
        results.to_csv(os.path.join(args.result_path, "result.csv"))
        np.savetxt(os.path.join(args.result_path, "times.txt"), times)
    else:
        if args.part is None:
            raise ValueError("If not in --collect mode, `part` must be specified.")
        output_path = os.path.join(args.result_path, "parts")
        if not args.runexisting and os.path.exists(os.path.join(output_path, f"part{args.part}.csv")):
            print(f"Skipping part{args.part} because it already exists. Use option --runexisting to run anyway.")
        else:
            run_args = cases[args.case]["case_args"][args.part]
            start_time = time()
            with open(os.path.join(output_path, f"part{args.part}.log"), "w") as f:
                print(run_args, file=f)
            res = run(**run_args).data
            end_time = time()
            np.savetxt(os.path.join(output_path, f"part{args.part}.time"), [end_time - start_time])
            res.to_pickle(os.path.join(output_path, f"part{args.part}.bz2"))
            evaluated_res = pd.DataFrame([standard_bipartite_evaluation(res)],
                                         columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std"]
                                         )
            evaluated_res.to_csv(os.path.join(output_path, f"part{args.part}.csv"), index=False)
