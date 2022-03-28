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

# CASE 2
# check if the no-epp option works correctly
num_parts_2 = 128
lengths = lengths = np.linspace(1, 250e3, num=num_parts_2)
case_2_specification = {
    "name": "no_epp",
    "subcase_name": "epp_steps_0",
    "num_parts": num_parts_2,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 0.2,
                                    "F_INIT": 1.0
                                    },
                         "epp_steps": 0
                         }
                  for part in range(num_parts_2)
                  }
}


# # now check the opposite cases where initial fidelity is low, but memories are good
# CASE 3
num_parts_3 = 128
lengths = np.linspace(1, 250e3, num=num_parts_3)
case_3_specification = {
    "name": "low_fid",
    "subcase_name": "with_epp",
    "num_parts": num_parts_3,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 100,
                                    "F_INIT": 0.93,
                                    }
                         }
                  for part in range(num_parts_3)
                  }
}

# CASE 4
num_parts_4 = 128
lengths = np.linspace(1, 250e3, num=num_parts_4)
case_4_specification = {
    "name": "low_fid",
    "subcase_name": "without_epp",
    "num_parts": num_parts_4,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 100,
                                    "F_INIT": 0.93
                                    },
                         "epp_steps": 0
                         }
                  for part in range(num_parts_4)
                  }
}

# CASE 5
num_parts_5 = 128
lengths = np.linspace(1, 250e3, num=num_parts_5)
case_5_specification = {
    "name": "low_fid",
    "subcase_name": "with_2_epp",
    "num_parts": num_parts_5,
    "index": lengths,
    "case_args": {part: {"length": lengths[part],
                         "max_iter": 1e5,
                         "params": {"P_LINK": 0.5,
                                    "T_DP": 100,
                                    "F_INIT": 0.93
                                    },
                         "epp_steps": 0
                         }
                  for part in range(num_parts_5)
                  }
}


cases = {
    0: case_0_specification,
    1: case_1_specification,
    2: case_2_specification,
    3: case_3_specification,
    4: case_4_specification,
    5: case_5_specification,
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
