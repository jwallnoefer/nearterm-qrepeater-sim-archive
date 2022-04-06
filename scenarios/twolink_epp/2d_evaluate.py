
import os, sys; sys.path.insert(0, os.path.abspath("."))
import scenarios.twolink_epp.case_definition as case_definition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_path = os.path.join("results", "twolink_epp")

no_epp_cases = np.arange(6, 69)
epp_cases = np.arange(69, 132)


def is_always_better(no_epp_data, epp_data):
    no_epp_series = no_epp_data["key_per_time"]
    epp_series = epp_data["key_per_time"]
    positive_idx = epp_series > 0
    return np.all(epp_series[positive_idx] > no_epp_series[positive_idx])


def extends_range(no_epp_data, epp_data):
    no_epp_series = no_epp_data["key_per_time"]
    no_epp_idx = no_epp_series > 0
    no_epp_reachable = no_epp_series[no_epp_idx].index[-1]

    epp_series = epp_data["key_per_time"]
    epp_idx = epp_series > 0
    epp_reachable = epp_series[epp_idx].index[-1]

    return epp_reachable > no_epp_reachable


res_better = []
res_extend = []

for no_epp_case, epp_case in zip(no_epp_cases, epp_cases):
    no_epp_path = os.path.join(base_path,
                               case_definition.name(no_epp_case),
                               case_definition.subcase_name(no_epp_case),
                               "result.csv")
    no_epp_data = pd.read_csv(no_epp_path, index_col=0)
    epp_path = os.path.join(base_path,
                            case_definition.name(epp_case),
                            case_definition.subcase_name(epp_case),
                            "result.csv")
    epp_data = pd.read_csv(epp_path, index_col=0)
    res_better += [is_always_better(no_epp_data, epp_data)]
    res_extend += [extends_range(no_epp_data, epp_data)]
    # print(case_definition.subcase_name(no_epp_case),
    #       is_always_better(no_epp_data, epp_data),
    #       extends_range(no_epp_data, epp_data))


fidelities = [0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
memory_times = np.logspace(-3, 0, num=7)  # 1 ms to 1 second
# fidelities = [0.93, 0.96]
# memory_times = [10e-3, 100e-3]

res_better = np.array(res_better, dtype=int).reshape((len(memory_times), len(fidelities)))
res_extend = np.array(res_extend, dtype=int).reshape((len(memory_times), len(fidelities)))

plt.pcolormesh(memory_times, fidelities, res_better, shading="nearest")
# plt.yscale("log")
plt.show()

plt.pcolormesh(memory_times, fidelities, res_extend, shading="nearest")
plt.show()