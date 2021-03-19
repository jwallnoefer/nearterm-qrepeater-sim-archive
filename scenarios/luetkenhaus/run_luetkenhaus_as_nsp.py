import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.whitepaper.NSP_QR_cell import run
from libs.aux_functions import assert_dir, standard_bipartite_evaluation
import numpy as np
from multiprocessing import Pool
import pandas as pd
from time import time

ETA_P = 0.66  # preparation efficiency
T_P = 2 * 10**-6  # preparation time
ETA_C = 0.04 * 0.3  # phton-fiber coupling efficiency * wavelength conversion
T_2 = 1  # dephasing time
E_M_A = 0.01  # misalignment error
P_D = 10**-8  # dark count probability per detector
ETA_D = 0.3  # detector efficiency
P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
LAMBDA_BSM = 0.97  # BSM ideality parameter
F = 1.16  # error correction inefficiency
ETA_TOT = ETA_P * ETA_C * ETA_D

C = 2 * 10**8  # speed of light in optical fiber
result_path = os.path.join("results", "luetkenhaus", "as_nsp")

luetkenhaus_params = {"P_LINK": ETA_TOT,
                      "T_P": T_P,
                      "T_DP": T_2,
                      "E_MA": E_M_A,
                      "P_D": P_D,
                      "LAMBDA_BSM": LAMBDA_BSM}


def do_the_thing(length, max_iter, params, cutoff_time, mode="sim"):
    p = run(length=length, max_iter=max_iter, params=params, cutoff_time=cutoff_time, mode=mode)
    return p.data


if __name__ == "__main__":
    length_list = np.linspace(0, 70e3, num=128)
    mode = "seq"
    num_processes = 32
    max_iter = 1e5
    start_time = time()
    with Pool(num_processes) as pool:
        num_calls = len(length_list)
        aux_list = zip(length_list, [max_iter] * num_calls, [luetkenhaus_params] * num_calls, [None] * num_calls, [mode] * num_calls)
        res = pool.starmap(do_the_thing, aux_list)
        pool.close()
        pool.join()

    data_series = pd.Series(data=res, index=length_list)
    output_path = result_path
    assert_dir(output_path)
    try:
        existing_series = pd.read_pickle(os.path.join(output_path, "raw_data.bz2"))
        combined_series = existing_series.append(data_series)
        combined_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
    except FileNotFoundError:
        data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
    result_list = [standard_bipartite_evaluation(data_frame=df, err_corr_ineff=F) for df in data_series]
    output_data = pd.DataFrame(data=result_list, index=length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
    try:
        existing_data = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
        combined_data = pd.concat([existing_data, output_data])
        combined_data.to_csv(os.path.join(output_path, "result.csv"))
    except FileNotFoundError:
        output_data.to_csv(os.path.join(output_path, "result.csv"))

    print("The whole run took %.2f minutes." % ((time() - start_time) / 60.0))
