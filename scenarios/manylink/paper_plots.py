import os, sys; sys.path.insert(0, os.path.abspath("."))
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scenarios.manylink.case_definition as case_definition
from collections import defaultdict
import rsmf
import itertools as it

# colorblind friendly color set taken from https://personal.sron.nl/~pault/
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
# make them the standard colors for matplotlib
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
color_list = colors

formatter = rsmf.setup(r"\documentclass[twocolumn]{revtex4-2}")

result_path = os.path.join("results", "manylink")
scenario_prefix = "manylink"

# group all cases by name, so we can pick which we want
case_and_names = [(case, case_definition.name(case)) for case in range(case_definition.num_cases)]
grouped_dict = defaultdict(list)
for case, name in case_and_names:
    grouped_dict[name].append(case)


# Plot 1A: Show that for low errors more links is generally favorable.
case_name = "compare_num_links_tdp10"
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
plt.ylim(10**-3, 10**7)
plt.tight_layout()
save_path = os.path.join(result_path, scenario_prefix + "_" + case_name + ".pdf")
plt.savefig(save_path)
print(f"Plot saved at {save_path}")

# Plot 1B: Show that for low errors more links is generally favorable.
case_name = "compare_num_links_tdp10_f998"
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
plt.ylim(10**-3, 10**7)
plt.tight_layout()
save_path = os.path.join(result_path, scenario_prefix + "_" + case_name + ".pdf")
plt.savefig(save_path)
print(f"Plot saved at {save_path}")


# Plot 2:
num_links_list = [8, 16]
num_links_colors = color_list[2:4]
f_inits = [0.996, 0.997, 0.998]
f_markers = ["X", "P", "o"]
iterator = it.product(zip(num_links_list, num_links_colors), zip(f_inits, f_markers))
fig = formatter.figure(width_ratio=1.0, wide=False)
for (num_links, color), (f_init, marker) in iterator:
    case_name = f"compare_num_links_tdp10_f{int(f_init*1e3)}"
    for case in grouped_dict[case_name]:
        if case_definition.subcase_name(case) == f"num_link_{num_links}":
            subcase_name = case_definition.subcase_name(case)
            subcase_path = os.path.join(result_path, case_name, subcase_name)
            res = pd.read_csv(os.path.join(subcase_path, "result.csv"), index_col=0)
            break
    plt.scatter(res.index / 1000, res["key_per_time"], marker=marker, linewidths=0, c=color, s=4)
plt.yscale("log")
plt.grid()
plt.xlabel("Total distance [km]")
plt.xlim(0, 170)
plt.ylim(10**0, 10**6)
plt.ylabel("Key / time [Hz]")
plt.tight_layout()
save_path = os.path.join(result_path, scenario_prefix + "_" + "improve_f.pdf")
plt.savefig(save_path)
print(f"Plot saved at {save_path}")
