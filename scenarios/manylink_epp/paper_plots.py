import os, sys; sys.path.insert(0, os.path.abspath("."))
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scenarios.manylink_epp.case_definition as case_definition
from collections import defaultdict
import rsmf
import itertools as it

# colorblind friendly color set taken from https://personal.sron.nl/~pault/
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
# make them the standard colors for matplotlib
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
color_list = list(colors)

formatter = rsmf.setup(r"\documentclass[twocolumn]{revtex4-2}")

result_path = os.path.join("results", "manylink_epp")
scenario_prefix = "manylink_epp"

# group all cases by name, so we can pick which we want
case_and_names = [(case, case_definition.name(case)) for case in range(case_definition.num_cases)]
grouped_dict = defaultdict(list)
for case, name in case_and_names:
    grouped_dict[name].append(case)


case_name = "compare_num_links_f_990"
case_list = grouped_dict[case_name]
num_links_list = [2, 4, 8, 16, 32]
num_links_colors = color_list[:5]
no_epp_marker = "o"
epp_marker = "X"
fig = formatter.figure(width_ratio=1.0, wide=False)
for (num_links, color) in zip(num_links_list, num_links_colors):
    for case in grouped_dict[case_name]:
        subcase_name = case_definition.subcase_name(case)
        if subcase_name == f"num_link_{num_links}":
            subcase_path = os.path.join(result_path, case_name, subcase_name)
            res = pd.read_csv(os.path.join(subcase_path, "result.csv"), index_col=0)
            plt.scatter(res.index / 1000, res["key_per_time"], marker=no_epp_marker, c=color, linewidths=0, s=2, label=subcase_name)
        elif subcase_name == f"epp_num_link_{num_links}":
            subcase_path = os.path.join(result_path, case_name, subcase_name)
            res = pd.read_csv(os.path.join(subcase_path, "result.csv"), index_col=0)
            plt.scatter(res.index / 1000, res["key_per_time"], marker=epp_marker, c=color, linewidths=0, s=2, label=subcase_name)

plt.yscale("log")
plt.grid()
plt.xlabel("Total distance [km]")
plt.ylabel("Key / time [Hz]")
plt.ylim(10**-2, 10**5)
# plt.legend()
plt.tight_layout()
save_path = os.path.join(result_path, scenario_prefix + "_" + case_name + ".pdf")
plt.savefig(save_path)
print(f"Plot saved at {save_path}")
