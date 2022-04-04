"""Define all the cases here."""
import numpy as np
import itertools as it

cases = {}

# CASE 0
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_specification = {
    "name": "with_pd_f_96",
    "subcase_name": "0_epp",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 100e-3,
                                    "F_INIT": 0.96,
                                    "P_D": 1e-6
                                    },
                         "num_memories": 100,
                         "epp_steps": 0,
                         }
                  for part in range(num_parts)
                  }
}
cases.update({len(cases): case_specification})

# CASE 1
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_specification = {
    "name": "with_pd_f_96",
    "subcase_name": "1_epp",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 100e-3,
                                    "F_INIT": 0.96,
                                    "P_D": 1e-6
                                    },
                         "num_memories": 100,
                         "epp_steps": 1,
                         }
                  for part in range(num_parts)
                  }
}
cases.update({len(cases): case_specification})

# CASE 2
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_specification = {
    "name": "with_pd_f_96",
    "subcase_name": "2_epp",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 100e-3,
                                    "F_INIT": 0.96,
                                    "P_D": 1e-6
                                    },
                         "num_memories": 100,
                         "epp_steps": 2,
                         }
                  for part in range(num_parts)
                  }
}
cases.update({len(cases): case_specification})

# CASE 3
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_specification = {
    "name": "with_pd_f_93",
    "subcase_name": "0_epp",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 100e-3,
                                    "F_INIT": 0.93,
                                    "P_D": 1e-6
                                    },
                         "num_memories": 100,
                         "epp_steps": 0,
                         }
                  for part in range(num_parts)
                  }
}
cases.update({len(cases): case_specification})

# CASE 4
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_specification = {
    "name": "with_pd_f_93",
    "subcase_name": "1_epp",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 100e-3,
                                    "F_INIT": 0.93,
                                    "P_D": 1e-6
                                    },
                         "num_memories": 100,
                         "epp_steps": 1,
                         }
                  for part in range(num_parts)
                  }
}
cases.update({len(cases): case_specification})


# CASE 5
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_specification = {
    "name": "with_pd_f_93",
    "subcase_name": "2_epp",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 100e-3,
                                    "F_INIT": 0.93,
                                    "P_D": 1e-6
                                    },
                         "num_memories": 100,
                         "epp_steps": 2,
                         }
                  for part in range(num_parts)
                  }
}
cases.update({len(cases): case_specification})


# # now start developing code for a 2d plot
start_case = 6
# fidelities = [0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
# memory_times = np.logspace(-3, 0, num=7)  # 1 ms to 1 second
fidelities = [0.93, 0.96]
memory_times = [10e-3, 100e-3]
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
for case, (fid, memory_time) in zip(it.count(start_case), it.product(fidelities, memory_times)):
    case_specification = {
        "name": "2d_plot_test",
        "subcase_name": f"f{int(fid * 100)}_tdp{memory_time * 1e3:.2f}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": 100e-3,
                                        "F_INIT": 0.93,
                                        "P_D": 1e-6
                                        },
                             "num_memories": 100,
                             "epp_steps": 0,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({case: case_specification})
print("case_count after 2d", len(cases))

# # now start developing code for a 2d plot
start_case = len(cases)  # currently 10 for the test
# fidelities = [0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
# memory_times = np.logspace(-3, 0, num=7)  # 1 ms to 1 second
fidelities = [0.93, 0.96]
memory_times = [10e-3, 100e-3]
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
for case, (fid, memory_time) in zip(it.count(start_case), it.product(fidelities, memory_times)):
    case_specification = {
        "name": "2d_plot_test",
        "subcase_name": f"epp_f{int(fid * 100)}_tdp{memory_time * 1e3:.2f}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": 100e-3,
                                        "F_INIT": 0.93,
                                        "P_D": 1e-6
                                        },
                             "num_memories": 100,
                             "epp_steps": 1,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({case: case_specification})
print("case_count after epp", len(cases))  # 14 for test case


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
