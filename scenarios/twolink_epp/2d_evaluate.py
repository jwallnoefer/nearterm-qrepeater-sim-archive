
import os, sys; sys.path.insert(0, os.path.abspath("."))
import scenarios.twolink_epp.case_definition as case_definition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_path = os.path.join("results", "twolink_epp")

no_epp_cases = np.arange(6, 227)
epp_cases = np.arange(227, 448)
# no_epp_cases = np.arange(448, 669)
# epp_cases = np.arange(669, 890)
# no_epp_cases = np.arange(890, 1111)
# epp_cases = np.arange(1111, 1332)


def is_always_better(no_epp_data, epp_data):
    no_epp_series = no_epp_data["key_per_time"]
    epp_series = epp_data["key_per_time"]
    positive_idx = epp_series > 0
    return np.all(epp_series[positive_idx] >= no_epp_series[positive_idx])


def extends_range(no_epp_data, epp_data):
    no_epp_series = no_epp_data["key_per_time"]
    no_epp_idx = no_epp_series > 0
    no_epp_reachable = no_epp_series[no_epp_idx].index[-1]

    epp_series = epp_data["key_per_time"]
    epp_idx = epp_series > 0
    epp_reachable = epp_series[epp_idx].index[-1]

    return epp_reachable - no_epp_reachable
    # return epp_reachable / no_epp_reachable


res_better = []
res_extend = []

for no_epp_case, epp_case in zip(no_epp_cases, epp_cases):
    try:
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
    except FileNotFoundError:
        print("something didn't work at", no_epp_case, epp_case)
        res_better += [False]
        res_extend += [False]
    # print(case_definition.subcase_name(no_epp_case),
    #       is_always_better(no_epp_data, epp_data),
    #       extends_range(no_epp_data, epp_data))


fidelities = np.linspace(0.92, 1.00, num=17)
memory_times = np.logspace(-3, 0, num=13)  # 1 ms to 1 second

res_better = np.array(res_better, dtype=int).reshape((len(fidelities), len(memory_times)))
res_extend = np.array(res_extend, dtype=int).reshape((len(fidelities), len(memory_times)))

plt.pcolormesh(memory_times, fidelities, res_better, shading="nearest")
plt.xscale("log")
plt.xlim(10**-3, 1)
plt.xlabel("dephasing time T_DP")
plt.ylabel("initial fidelity F_INIT")
plt.show()

lim = np.max(np.abs(res_extend))
pcm = plt.pcolormesh(memory_times, fidelities, res_extend, shading="nearest", cmap="RdBu", vmin=-lim, vmax=lim)
import itertools
aux_x = []
aux_y = []
for i, j in itertools.product(memory_times, fidelities):
    aux_x += [i]
    aux_y += [j]
plt.scatter(aux_x, aux_y, s=5)
plt.xscale("log")
plt.xlim(0.8*10**-3, 1.2)
plt.xlabel("dephasing time T_DP")
plt.ylabel("initial fidelity F_INIT")
plt.colorbar(pcm)
plt.show()
