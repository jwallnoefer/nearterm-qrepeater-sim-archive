import os, sys; sys.path.insert(0, os.path.abspath("."))
import argparse
import scenarios.twolink_epp.case_definition as case_definition
from scenarios.whitepaper.NSP_QR_cell import run as run_nsp
from scenarios.twolink_epp.two_link_epp import run as run_epp
from libs.aux_functions import assert_dir, standard_bipartite_evaluation
import numpy as np
from consts import SPEED_OF_LIGHT_IN_OPTICAL_FIBER as C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an epp_twolink case.")
    parser.add_argument("result_path", help="Path to case specific result directory.")
    parser.add_argument("case", type=int, help="The case number as defined in case_defition.")
    parser.add_argument("part", type=int, nargs="?", help="Which part of the case to run.")
    parser.add_argument("--collect", default=False, action="store_const", const=True, help="Set this flag to collect results instead of running.")
    args = parser.parse_args()

    if args.collect:
        # TODO: COLLATE results
        pass
    else:
        if args.part is None:
            raise ValueError("If not in --collect mode, `part` must be specified.")
        output_path = os.path.join(args.result_path, "parts", f"part{part}.bz2")
        run_args = case_definition.case(case=args.case, part=args.part)
        res = run_epp(**run_args).data
        res.to_pickle(out_path)
