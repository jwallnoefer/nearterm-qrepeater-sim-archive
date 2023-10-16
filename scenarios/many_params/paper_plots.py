import os, sys; sys.path.insert(0, os.path.abspath("."))
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scenarios.many_params.case_definition as case_definition
from collections import defaultdict
import rsmf
import itertools as it

# colorblind friendly color set taken from https://personal.sron.nl/~pault/
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
# make them the standard colors for matplotlib
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
color_list = list(colors)

formatter = rsmf.setup(r"\documentclass[twocolumn]{revtex4-2}")

result_path = os.path.join("results", "many_params")
scenario_prefix = "many_params"

# group all cases by name, so we can pick which we want
case_and_names = [(case, case_definition.name(case)) for case in range(case_definition.num_cases)]
grouped_dict = defaultdict(list)
for case, name in case_and_names:
    grouped_dict[name].append(case)

case_name = "improve_memories"
case_list = grouped_dict[case_name]
fig = formatter.figure(width_ratio=1.0, wide=False)
for (case, color) in zip(grouped_dict[case_name], color_list):
    subcase_name = case_definition.subcase_name(case)
    subcase_path = os.path.join(result_path, case_name, subcase_name)
    res = pd.read_csv(os.path.join(subcase_path, "result.csv"), index_col=0)
    plt.scatter(res.index, res["key_per_time"], c=color, linewidths=0, s=2, label=subcase_name)

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$T_\mathrm{damp}$ [s]")
plt.ylabel("Key / time [Hz]")
plt.xlim(0.1, 10)
plt.ylim(10**-2, 10**2)
plt.grid()
# plt.legend()
plt.tight_layout()
save_path = os.path.join(result_path, scenario_prefix + "_" + case_name + ".pdf")
plt.savefig(save_path)
print(f"Plot saved at {save_path}")


case_name = "improve_gates"
case_list = grouped_dict[case_name]
fig = formatter.figure(width_ratio=1.0, wide=False)
for (case, color) in zip(grouped_dict[case_name], color_list):
    subcase_name = case_definition.subcase_name(case)
    subcase_path = os.path.join(result_path, case_name, subcase_name)
    res = pd.read_csv(os.path.join(subcase_path, "result.csv"), index_col=0)
    plt.scatter(res.index, res["key_per_time"], c=color, linewidths=0, s=2, label=subcase_name)

# plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"2-qubit operation parameter $p$")
plt.ylabel("Key / time [Hz]")
plt.ylim(10**-2, 10**2)
plt.grid()
# plt.legend()
plt.tight_layout()
save_path = os.path.join(result_path, scenario_prefix + "_" + case_name + ".pdf")
plt.savefig(save_path)
print(f"Plot saved at {save_path}")
