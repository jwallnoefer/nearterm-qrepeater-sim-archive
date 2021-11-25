import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.whitepaper.NSP_QR_cell_timeslots import run
from libs.aux_functions import assert_dir, standard_bipartite_evaluation
import numpy as np
from multiprocessing import Pool
from time import time
import pandas as pd

C = 2 * 10**8  # speed of light in optical fiber

result_path = os.path.join("results", "whitepaper_timeslots")

params_available_NV = {"P_LINK": 5 * 10**-2,
                       "f_clock": 50 * 10**6,
                       "T_DP": 10 * 10**-3}
params_available_SiV = {"P_LINK": 5 * 10**-2,
                        "f_clock": 30 * 10**6,
                        "T_DP": 1 * 10**-3}
params_available_Qdot = {"P_LINK": 10 * 10**-2,
                         "f_clock": 1000 * 10**6,
                         "T_DP": 0.003 * 10**-3}
params_available_Ca = {"P_LINK": 25 * 10**-2,
                       "f_clock": 0.47 * 10**6,
                       "T_DP": 20 * 10**-3}
params_available_Rb = {"P_LINK": 50 * 10**-2,
                       "f_clock": 5 * 10**6,
                       "T_DP": 100 * 10**-3}
params_future_NV = {"P_LINK": 50 * 10**-2,
                    "f_clock": 250 * 10**6,
                    "T_DP": 10000 * 10**-3}
params_future_SiV = {"P_LINK": 50 * 10**-2,
                     "f_clock": 500 * 10**6,
                     "T_DP": 100 * 10**-3}
params_future_Qdot = {"P_LINK": 60 * 10**-2,
                      "f_clock": 1000 * 10**6,
                      "T_DP": 0.3 * 10**-3}
params_future_Ca = {"P_LINK": 50 * 10**-2,
                    "f_clock": 10 * 10**6,
                    "T_DP": 300 * 10**-3}
params_future_Rb = {"P_LINK": 70 * 10**-2,
                    "f_clock": 10 * 10**6,
                    "T_DP": 1000 * 10**-3}

available_params = [params_available_NV, params_available_SiV, params_available_Qdot, params_available_Ca, params_available_Rb]
future_params = [params_future_NV, params_future_SiV, params_future_Qdot, params_future_Ca, params_future_Rb]
name_list = ["NV", "SiV", "Qdot", "Ca", "Rb"]
ms_available = [25, 10, 0, 20, 100]  # # 25/20/0/100/10 for NV/Ca/Qdot/Rb/SiV (current values on the left) and
ms_future = [5000, 50, 0, 200, 500]  # #5000/200/0/500/50 for NV/Ca/Qdot/Rb/SiV (future values on the right).


def do_the_thing(length, max_iter, params, m, mode="sim"):
    p = run(length=length, max_iter=max_iter, params=params, m=m, mode=mode)
    return p.data


if __name__ == "__main__":
    mode = "sim"
    if int(sys.argv[1]) == 0:  # available_params without cutoff
        length_list = np.linspace(0, 425000, num=128)
        num_processes = int(sys.argv[2])
        max_iter = 1e5
        res = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for name, params in zip(name_list, available_params):
                num_calls = len(length_list)
                aux_list = zip(length_list, [max_iter] * num_calls, [params] * num_calls, [None] * num_calls, [mode] * num_calls)
                res[name] = pool.starmap_async(do_the_thing, aux_list)
            pool.close()

            for name, params in zip(name_list, available_params):
                data_series = pd.Series(data=res[name].get(), index=length_list)
                print("available_%s finished after %.2f minutes." % (str(name), (time() - start_time) / 60.0))
                output_path = os.path.join(result_path, "available", "no_cutoff", name)
                assert_dir(output_path)
                try:
                    existing_series = pd.read_pickle(os.path.join(output_path, "raw_data.bz2"))
                    combined_series = existing_series.append(data_series)
                    combined_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                except FileNotFoundError:
                    data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
                output_data = pd.DataFrame(data=result_list, index=length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
                try:
                    existing_data = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
                    combined_data = pd.concat([existing_data, output_data])
                    combined_data.to_csv(os.path.join(output_path, "result.csv"))
                except FileNotFoundError:
                    output_data.to_csv(os.path.join(output_path, "result.csv"))

        print("The whole run took %.2f minutes." % ((time() - start_time) / 60.0))

    elif int(sys.argv[1]) == 1:  # available_params with cutoff
        length_list = np.linspace(0, 425000, num=128)
        # looks like full range of lengths is not feasible for bad parameters
        length_cutoffs = {"NV": 200e3, "SiV": 125e3, "Qdot": 100e3, "Ca": 300e3, "Rb": 350e3}
        num_processes = int(sys.argv[2])
        max_iter = 1e5
        res = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for name, params, m in zip(name_list, available_params, ms_available):
                shortened_length_list = length_list[length_list <= length_cutoffs[name]]
                trial_times = shortened_length_list / C
                num_calls = len(shortened_length_list)
                aux_list = zip(shortened_length_list, [max_iter] * num_calls, [params] * num_calls, [m] * num_calls, [mode] * num_calls)
                res[name] = pool.starmap_async(do_the_thing, aux_list)
            pool.close()

            for name, params, m in zip(name_list, available_params, ms_available):
                shortened_length_list = length_list[length_list <= length_cutoffs[name]]
                data_series = pd.Series(data=res[name].get(), index=shortened_length_list)
                print("available_%s finished after %.2f minutes." % (str(name), (time() - start_time) / 60.0))
                output_path = os.path.join(result_path, "available", "with_cutoff", name)
                assert_dir(output_path)
                try:
                    existing_series = pd.read_pickle(os.path.join(output_path, "raw_data.bz2"))
                    combined_series = existing_series.append(data_series)
                    combined_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                except FileNotFoundError:
                    data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
                output_data = pd.DataFrame(data=result_list, index=shortened_length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
                try:
                    existing_data = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
                    combined_data = pd.concat([existing_data, output_data])
                    combined_data.to_csv(os.path.join(output_path, "result.csv"))
                except FileNotFoundError:
                    output_data.to_csv(os.path.join(output_path, "result.csv"))

        print("The whole run took %.2f minutes." % ((time() - start_time) / 60.0))

    elif int(sys.argv[1]) == 2:  # future_params without cutoff
        length_list = np.linspace(0, 425000, num=128)
        num_processes = int(sys.argv[2])
        max_iter = 1e5
        res = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for name, params in zip(name_list, future_params):
                num_calls = len(length_list)
                aux_list = zip(length_list, [max_iter] * num_calls, [params] * num_calls, [None] * num_calls, [mode] * num_calls)
                res[name] = pool.starmap_async(do_the_thing, aux_list)
            pool.close()

            for name, params in zip(name_list, future_params):
                data_series = pd.Series(data=res[name].get(), index=length_list)
                print("future_%s finished after %.2f minutes." % (str(name), (time() - start_time) / 60.0))
                output_path = os.path.join(result_path, "future", "no_cutoff", name)
                assert_dir(output_path)
                try:
                    existing_series = pd.read_pickle(os.path.join(output_path, "raw_data.bz2"))
                    combined_series = existing_series.append(data_series)
                    combined_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                except FileNotFoundError:
                    data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
                output_data = pd.DataFrame(data=result_list, index=length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
                try:
                    existing_data = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
                    combined_data = pd.concat([existing_data, output_data])
                    combined_data.to_csv(os.path.join(output_path, "result.csv"))
                except FileNotFoundError:
                    output_data.to_csv(os.path.join(output_path, "result.csv"))

        print("The whole run took %.2f minutes." % ((time() - start_time) / 60.0))

    elif int(sys.argv[1]) == 3:  # future_params with cutoff
        length_list = np.linspace(0, 425000, num=128)
        num_processes = int(sys.argv[2])
        length_cutoffs = {"NV": 425000, "SiV": 425000, "Qdot": 150e3, "Ca": 425000, "Rb": 425000}
        max_iter = 1e5
        res = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for name, params, m in zip(name_list, future_params, ms_future):
                shortened_length_list = length_list[length_list <= length_cutoffs[name]]
                trial_times = shortened_length_list / C
                num_calls = len(shortened_length_list)
                aux_list = zip(shortened_length_list, [max_iter] * num_calls, [params] * num_calls, [m] * num_calls, [mode] * num_calls)
                res[name] = pool.starmap_async(do_the_thing, aux_list)
            pool.close()

            for name, params, m in zip(name_list, future_params, ms_future):
                shortened_length_list = length_list[length_list <= length_cutoffs[name]]
                data_series = pd.Series(data=res[name].get(), index=shortened_length_list)
                print("future_%s finished after %.2f minutes." % (str(name), (time() - start_time) / 60.0))
                output_path = os.path.join(result_path, "future", "with_cutoff", name)
                assert_dir(output_path)
                try:
                    existing_series = pd.read_pickle(os.path.join(output_path, "raw_data.bz2"))
                    combined_series = existing_series.append(data_series)
                    combined_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                except FileNotFoundError:
                    data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
                output_data = pd.DataFrame(data=result_list, index=shortened_length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
                try:
                    existing_data = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
                    combined_data = pd.concat([existing_data, output_data])
                    combined_data.to_csv(os.path.join(output_path, "result.csv"))
                except FileNotFoundError:
                    output_data.to_csv(os.path.join(output_path, "result.csv"))

        print("The whole run took %.2f minutes." % ((time() - start_time) / 60.0))

    elif int(sys.argv[1]) == 4:  # investigate different cutoff times for Rb
        name = "Rb"
        params = params_available_Rb
        length_list = np.linspace(0, 425000, num=128)
        ms = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        num_processes = int(sys.argv[2])
        max_iter = 1e5
        res = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for m in ms:
                trial_times = length_list / C
                num_calls = len(length_list)
                aux_list = zip(length_list, [max_iter] * num_calls, [params] * num_calls, [m] * num_calls, [mode] * num_calls)
                res[m] = pool.starmap_async(do_the_thing, aux_list)
            pool.close()

            for m in ms:
                data_series = pd.Series(data=res[m].get(), index=length_list)
                print("m=%s finished after %.2f minutes." % (str(m), (time() - start_time) / 60.0))
                output_path = os.path.join(result_path, f"{name}_m_test", f"m_{m}")
                assert_dir(output_path)
                try:
                    existing_series = pd.read_pickle(os.path.join(output_path, "raw_data.bz2"))
                    combined_series = existing_series.append(data_series)
                    combined_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                except FileNotFoundError:
                    data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
                output_data = pd.DataFrame(data=result_list, index=length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
                try:
                    existing_data = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
                    combined_data = pd.concat([existing_data, output_data])
                    combined_data.to_csv(os.path.join(output_path, "result.csv"))
                except FileNotFoundError:
                    output_data.to_csv(os.path.join(output_path, "result.csv"))

        print("The whole run took %.2f minutes." % ((time() - start_time) / 60.0))

    elif int(sys.argv[1]) == 5:  # investigate claim that one can set cutoff too low
        # the effect should be most visible if memory quality is high and link quality is low
        test_params = {"P_LINK": 10 * 10**-2,
                       "T_DP": 0.5}
        length = 200e3  # fixed length
        trial_time_manual = length / C
        # m_list = np.arange(1, 258, 2)
        m_list = np.arange(1, 4002, 25)
        cutoff_list = [m * trial_time_manual + 10**-6 * trial_time_manual for m in m_list]
        num_processes = int(sys.argv[2])
        max_iter = 1e5
        start_time = time()
        with Pool(num_processes) as pool:
            num_calls = len(m_list)
            aux_list = zip([length] * num_calls, [max_iter] * num_calls, [test_params] * num_calls, cutoff_list, [mode] * num_calls)
            res = pool.starmap(do_the_thing, aux_list)
            pool.close()
            pool.join()

        data_series = pd.Series(data=res, index=m_list)
        output_path = os.path.join(result_path, "cutoff_test")
        assert_dir(output_path)
        try:
            existing_series = pd.read_pickle(os.path.join(output_path, "raw_data.bz2"))
            combined_series = existing_series.append(data_series)
            combined_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
        except FileNotFoundError:
            data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
        result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
        output_data = pd.DataFrame(data=result_list, index=m_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
        try:
            existing_data = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
            combined_data = pd.concat([existing_data, output_data])
            combined_data.to_csv(os.path.join(output_path, "result.csv"))
        except FileNotFoundError:
            output_data.to_csv(os.path.join(output_path, "result.csv"))

        print("The whole run took %.2f minutes." % ((time() - start_time) / 60.0))
