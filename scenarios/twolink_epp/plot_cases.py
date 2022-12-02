import os, sys; sys.path.insert(0, os.path.abspath("."))
import matplotlib.pyplot as plt
import pandas as pd
import scenarios.twolink_epp.case_definition as case_definition
from collections import defaultdict

result_path = os.path.join("results", "twolink_epp")

# step through cases by name to group them into plots
case_and_names = [(case, case_definition.name(case)) for case in range(case_definition.num_cases)]
# case_and_names = [(case, case_definition.name(case)) for case in range(508, 520)]

grouped_dict = defaultdict(list)
for case, name in case_and_names:
    grouped_dict[name].append(case)


for name, case_list in grouped_dict.items():
    for case in case_list:
        subcase_name = case_definition.subcase_name(case)
        subcase_path = os.path.join(result_path, name, subcase_name)
        try:
            res = pd.read_csv(os.path.join(subcase_path, "result.csv"), index_col=0)
        except FileNotFoundError:
            continue
        if subcase_name[0:3] == "epp":
            marker = "x"
        else:
            marker = "o"
        plt.scatter(res.index, res["key_per_time"], marker=marker, s=10, label=subcase_name)
        # plt.scatter(res.index, res["fidelity"], marker=marker, s=10, label=subcase_name)
    plt.yscale("log")
    plt.grid()
    plt.title(name)
    plt.ylabel("key_per_time")
    plt.legend(ncol=2, loc="upper center", prop={'size': 6})
    plt.savefig(os.path.join(result_path, f"{name}.pdf"))
    plt.show()
