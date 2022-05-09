"""Define all the cases here."""
import numpy as np
import itertools as it

cases = {}

# # Template
# case_specification = {
#     "name": str,
#     "subcase_name": str,
#     "num_parts": int,
#     "index": array_like,
#     "case_args": {part: {"length": lengths[part],
#                          "max_iter": 1e5,
#                          "params": {"P_LINK": 0.5,
#                                     "T_DP": memory_time,
#                                     "F_INIT": fid,
#                                     "P_D": 1e-6
#                                     },
#                          "num_memories": num_memories,
#                          "epp_steps": 1,
#                          }
#                   for part in range(num_parts)
#                   }
# }

# # first thing that could be interesting: how many links to choose for which
# # distance
case_name = "compare_num_links"
if __name__ == "__main__":
    print(f"Case {case_name} starts at case number", len(cases))
link_iter = [2, 4, 8, 16, 32, 64, 128]
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
for num_links in link_iter:
    case_specification = {
        "name": case_name,
        "subcase_name": f"num_link_{num_links}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": 100e-3,
                                        "F_INIT": 1,
                                        "P_D": 1e-6
                                        },
                             "num_links": num_links,
                             "cutoff_time": None,
                             "num_memories": 1,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({len(cases): case_specification})
if __name__ == "__main__":
    print(f"Case {case_name} ends at case number", len(cases))


# # first thing that could be interesting: how many links to choose for which
# # distance
t_dp = 1e-3
case_name = f"compare_num_links_tdp{int(t_dp * 1e3)}"
if __name__ == "__main__":
    print(f"Case {case_name} starts at case number", len(cases))
link_iter = [2, 4, 8, 16, 32, 64, 128]
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
for num_links in link_iter:
    case_specification = {
        "name": case_name,
        "subcase_name": f"num_link_{num_links}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": t_dp,
                                        "F_INIT": 1,
                                        "P_D": 1e-6
                                        },
                             "num_links": num_links,
                             "cutoff_time": None,
                             "num_memories": 1,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({len(cases): case_specification})
if __name__ == "__main__":
    print(f"Case {case_name} ends at case number", len(cases))


num_cases = len(cases)


def case_args(case, part):
    return cases[case]["case_args"][part]


def name(case):
    return cases[case]["name"]


def num_parts(case):
    return cases[case]["num_parts"]


def subcase_name(case):
    return cases[case]["subcase_name"]


def index(case):
    return cases[case]["index"]
