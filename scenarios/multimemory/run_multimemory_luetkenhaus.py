import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.multimemory.multi_memory_luetkenhaus import run
from libs.aux_functions import assert_dir, standard_bipartite_evaluation
import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool
import pandas as pd

C = 2 * 10**8  # speed of light in optical fiber
result_path = os.path.join("results", "multimemory_luetkenhaus")

# # # values taken from Róbert Trényi, Norbert Lütkenhaus https://arxiv.org/abs/1910.10962
T_P = 2 * 10**-6  # preparation time
E_M_A = 0.01  # misalignment error
P_D = 1.8 * 10**-11  # dark count probability per detector
P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
LAMBDA_BSM = 0.98  # BSM ideality parameter
F = 1.16  # error correction inefficiency

T_2 = 2  # dephasing time
ETA_P = 0.66  # preparation efficiency
ETA_C = 0.05 * 0.5  # phton-fiber coupling efficiency * wavelength conversion
ETA_D = 0.7  # detector efficiency

ETA_TOT = ETA_P * ETA_C * ETA_D  # = 0.0115
params = {"P_LINK": ETA_TOT,
          "T_P": T_P,
          "T_DP": T_2,
          "E_MA": E_M_A,
          "P_D": P_D,
          "LAMBDA_BSM": LAMBDA_BSM}


def do_the_thing(length, max_iter, params, num_memories, mode="seq"):
    np.random.seed()
    p = run(length=length, max_iter=max_iter, params=params, num_memories=num_memories, mode=mode)
    return p.data


if __name__ == "__main__":
    case = int(sys.argv[1])
    assert case == 0
    num_processes = int(sys.argv[2])
    length_list = np.linspace(0, 500e3, num=128)
    memories = [1, 10, 100, 400, 1000]
    mode = "seq"
    max_iter = 1e5
    res = {}
    start_time = time()
    with Pool(num_processes) as pool:
        for num_memories in memories:
            num_calls = len(length_list)
            aux_list = zip(length_list, [max_iter] * num_calls,
                           [params] * num_calls, [num_memories] * num_calls,
                           [mode] * num_calls
                           )
            res[num_memories] = pool.starmap_async(do_the_thing, aux_list)
        pool.close()

        for num_memories in memories:
            data_series = pd.Series(data=res[num_memories].get(), index=length_list)
            print(f"num_memories={num_memories} finished after {(time()-start_time)/60:.2f} minutes")
            output_path = os.path.join(result_path, f"{num_memories}_memories")
            assert_dir(output_path)
            data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
            result_list = [standard_bipartite_evaluation(data_frame=df, err_corr_ineff=F) for df in data_series]
            output_data = pd.DataFrame(data=result_list, index=length_list,
                                       columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"]
                                       )
            output_data.to_csv(os.path.join(output_path, "result.csv"))
