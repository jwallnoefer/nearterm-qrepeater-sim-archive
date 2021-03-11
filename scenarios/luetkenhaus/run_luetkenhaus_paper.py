import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from time import time
from scenarios.luetkenhaus.luetkenhaus import run, F  # F is the error correction inefficiency
from multiprocessing import Pool
from libs.aux_functions import assert_dir, standard_bipartite_evaluation


def do_the_thing(length, max_iter, mode):
    p = run(L_TOT=length, max_iter=max_iter, mode=mode)
    return p.data


if __name__ == "__main__":
    result_path = os.path.join("results", "luetkenhaus")
    length_list = np.linspace(0, 70e3, num=128)
    modes = ["seq", "sim"]
    num_processes = 32
    max_iter = 1e5
    res = {}
    start_time = time()
    with Pool(num_processes) as pool:
        for mode in modes:
            num_calls = len(length_list)
            aux_list = zip(length_list, [max_iter] * num_calls, [mode] * num_calls)
            res[mode] = pool.starmap_async(do_the_thing, aux_list)
        pool.close()

        for mode in modes:
            data_series = pd.Series(data=res[mode].get(), index=length_list)
            print("mode=%s finished after %.2f minutes." % (str(mode), (time() - start_time) / 60.0))
            output_path = os.path.join(result_path, f"mode_{mode}")
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
