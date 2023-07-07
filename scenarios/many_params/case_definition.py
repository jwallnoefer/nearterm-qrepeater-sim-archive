"""Define all the cases here."""
import numpy as np
import itertools as it

cases = {}

base_params = {
        "P_LINK": 0.01,
        "T_DAMP": 10e-3,
        "E_MA": 0,
        "P_D": 1e-6,
        "P_GATE": 0.98,
        "F_INIT": 0.99
    }
num_links = 8
num_memories = 8


case_name = "base_case"
max_iter = int(1e5)
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
if __name__ == "__main__":
    print(f"Case {case_name} starts at case number", len(cases))
case_specification = {
    "name": case_name,
    "subcase_name": f"num_link_{num_links}",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": max_iter,
                         "params": base_params,
                         "num_links": num_links,
                         "cutoff_time": None,
                         "num_memories": 8,
                         "lowest_level_epp_steps": 1
                         }
                  for part in range(num_parts)
                  }
    }
cases.update({len(cases): case_specification})
if __name__ == "__main__":
    print(f"Case {case_name} ends at case number", len(cases))



case_name = "improve_memories"
max_iter = int(1e5)
num_parts = 128
damping_times = np.logspace(-3, 0, num=num_parts, base=10)
param_collection = []
for t_damp in damping_times:
    params = dict(base_params)
    params["T_DAMP"] = t_damp
    param_collection.append(params)
if __name__ == "__main__":
    print(f"Case {case_name} starts at case number", len(cases))
lengths = [50e3, 100e3, 200e3]
for length in lengths:
    case_specification = {
        "name": case_name,
        "subcase_name": f"len_{int(length/1e3)}",
        "num_parts": num_parts,
        "index": damping_times,
        "case_args": {part: {"length": length,
                             "max_iter": 1e5,
                             "params": param_collection[part],
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

case_name = "improve_gates"
max_iter = int(1e5)
num_parts = 128
gates = np.linspace(0.97, 1.00, num=num_parts)
param_collection = []
for p_gate in gates:
    params = dict(base_params)
    params["P_GATE"] = p_gate
    param_collection.append(params)
if __name__ == "__main__":
    print(f"Case {case_name} starts at case number", len(cases))
lengths = [50e3, 100e3, 200e3]
for length in lengths:
    case_specification = {
        "name": case_name,
        "subcase_name": f"len_{int(length/1e3)}",
        "num_parts": num_parts,
        "index": damping_times,
        "case_args": {part: {"length": length,
                             "max_iter": 1e5,
                             "params": param_collection[part],
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
