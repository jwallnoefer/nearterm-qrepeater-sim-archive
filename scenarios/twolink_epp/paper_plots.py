import os, sys; sys.path.insert(0, os.path.abspath("."))
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scenarios.twolink_epp.case_definition as case_definition
from collections import defaultdict
import rsmf
import numpy as np
import itertools as it

# colorblind friendly color set taken from https://personal.sron.nl/~pault/
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
# make them the standard colors for matplotlib
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
color_list = colors

formatter = rsmf.setup(r"\documentclass[twocolumn]{revtex4-2}")

result_path = os.path.join("results", "twolink_epp")
scenario_prefix = "twolink_epp"

# group all cases by name, so we can pick which we want
case_and_names = [(case, case_definition.name(case)) for case in range(case_definition.num_cases)]
grouped_dict = defaultdict(list)
for case, name in case_and_names:
    grouped_dict[name].append(case)

# in low fidelity, entanglement purification is very good
case_name = "with_pd_f_935"
case_list = grouped_dict[case_name]
print(case_list)
fig = formatter.figure(width_ratio=1.0, wide=False)
for case in case_list:
    subcase_name = case_definition.subcase_name(case)
    subcase_path = os.path.join(result_path, case_name, subcase_name)
    try:
        res = pd.read_csv(os.path.join(subcase_path, "result.csv"), index_col=0)
    except FileNotFoundError as e:
        print(e)
        continue
    plt.scatter(res.index / 1000, res["key_per_time"], marker="o", linewidths=0, s=2, label=subcase_name)
plt.yscale("log")
plt.grid()
plt.xlabel("Total distance [km]")
plt.ylabel("Key / time [Hz]")
plt.ylim(10**-2, 10**6)
plt.tight_layout()
save_path = os.path.join(result_path, scenario_prefix + "_" + case_name + ".pdf")
plt.savefig(save_path)
print(f"Plot saved at {save_path}")

# if not necessary, can still extend a bit
case_name = "with_pd_f_950"
case_list = grouped_dict[case_name]
print(case_list)
fig = formatter.figure(width_ratio=1.0, wide=False)
for case in case_list:
    subcase_name = case_definition.subcase_name(case)
    subcase_path = os.path.join(result_path, case_name, subcase_name)
    try:
        res = pd.read_csv(os.path.join(subcase_path, "result.csv"), index_col=0)
    except FileNotFoundError as e:
        print(e)
        continue
    plt.scatter(res.index / 1000, res["key_per_time"], marker="o", linewidths=0, s=2, label=subcase_name)
plt.yscale("log")
plt.grid()
plt.xlabel("Total distance [km]")
plt.ylabel("Key / time [Hz]")
plt.ylim(10**-2, 10**6)
plt.tight_layout()
save_path = os.path.join(result_path, scenario_prefix + "_" + case_name + ".pdf")
plt.savefig(save_path)
print(f"Plot saved at {save_path}")

# comparison with cutoff
case_name = "compare_cutoff_1000_f925"
case_list = grouped_dict[case_name]
print(case_list)
subcase_selection = ["cutoff50", "cutoffNone", "epp_cutoff650", "epp_cutoffNone"]
fig = formatter.figure(width_ratio=1.0, wide=False)
for case in case_list:
    subcase_name = case_definition.subcase_name(case)
    if subcase_name not in subcase_selection:
        continue
    if subcase_name[0:3] == "epp":
        marker = "X"
    else:
        marker = "o"
    subcase_path = os.path.join(result_path, case_name, subcase_name)
    try:
        res = pd.read_csv(os.path.join(subcase_path, "result.csv"), index_col=0)
    except FileNotFoundError as e:
        print(e)
        continue
    plt.scatter(res.index / 1000, res["key_per_time"], marker=marker, linewidths=0, s=2, label=subcase_name)
plt.yscale("log")
plt.grid()
plt.xlabel("Total distance [km]")
plt.ylabel("Key / time [Hz]")
plt.ylim(10**-5, 10**5)
plt.tight_layout()
save_path = os.path.join(result_path, scenario_prefix + "_" + case_name + ".pdf")
plt.savefig(save_path)
print(f"Plot saved at {save_path}")



# The 2d plot
no_epp_cases = np.arange(62, 283)
epp_cases = np.arange(283, 504)

def extends_range(no_epp_data, epp_data):
    no_epp_series = no_epp_data["key_per_time"]
    no_epp_idx = no_epp_series > 0
    no_epp_reachable = no_epp_series[no_epp_idx].index[-1]

    epp_series = epp_data["key_per_time"]
    epp_idx = epp_series > 0
    epp_reachable = epp_series[epp_idx].index[-1]

    return epp_reachable - no_epp_reachable
    # return epp_reachable > no_epp_reachable
    # return epp_reachable / no_epp_reachable

res_extend = []

for no_epp_case, epp_case in zip(no_epp_cases, epp_cases):
    try:
        no_epp_path = os.path.join(result_path,
                                   case_definition.name(no_epp_case),
                                   case_definition.subcase_name(no_epp_case),
                                   "result.csv")
        no_epp_data = pd.read_csv(no_epp_path, index_col=0)
        epp_path = os.path.join(result_path,
                                case_definition.name(epp_case),
                                case_definition.subcase_name(epp_case),
                                "result.csv")
        epp_data = pd.read_csv(epp_path, index_col=0)
        # print(len(no_epp_data), len(epp_data))
        res_extend += [extends_range(no_epp_data, epp_data)]
    except (FileNotFoundError, pd.core.indexing.IndexingError):
        print("something didn't work at", no_epp_case, epp_case)
        res_extend += [False]

fidelities = np.linspace(0.92, 1.00, num=17)
memory_times = np.logspace(-3, 0, num=13)  # 1 ms to 1 second

res_extend = np.array(res_extend, dtype=int).reshape((len(fidelities), len(memory_times))) / 1e3

fig = formatter.figure(width_ratio=1.0, wide=False)
lim = np.max(np.abs(res_extend))
pcm = plt.pcolormesh(memory_times, fidelities, res_extend, shading="nearest", cmap="RdBu", vmin=-lim, vmax=lim)

plt.xscale("log")
plt.xlim(0.8*10**-3, 1.2)
plt.xlabel("Dephasing time $T_\mathrm{dp}$")
plt.ylabel("Initial fidelity $F_\mathrm{init}$")
cbar = plt.colorbar(pcm)
cbar.ax.set_ylabel("Additional distance [km]")
plt.tight_layout()
save_path = os.path.join(result_path, scenario_prefix + "_" + "2d_plot_4_memories" + ".pdf")
plt.savefig(save_path)
print(f"Plot saved at {save_path}")
