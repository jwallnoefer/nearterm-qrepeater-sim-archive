"""Define all the cases here."""
import numpy as np

# CASE 0
num_parts_0 = 128
lengths = np.linspace(1, 250e3, num=num_parts_0)
case_0_specification = {
    "name": "test_case",
    "subcase_name": "by_length",
    "num_parts": num_parts_0,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 0.2,
                                    "F_INIT": 0.95
                                    }
                         }
                  for part in range(num_parts_0)
                  }
}

# CASE 1
num_parts_1 = 128
lengths = np.linspace(1, 250e3, num=num_parts_1)
case_1_specification = {
    "name": "test_case",
    "subcase_name": "perfect_init",
    "num_parts": num_parts_1,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 0.2,
                                    "F_INIT": 1.0
                                    }
                         }
                  for part in range(num_parts_1)
                  }
}

cases = {
    0: case_0_specification,
    1: case_1_specification
}


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
