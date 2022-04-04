"""Define all the cases here."""
import numpy as np

# CASE 0
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_0_specification = {
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

# CASE 1
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_1_specification = {
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

# CASE 2
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_2_specification = {
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

# CASE 3
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_3_specification = {
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

# CASE 4
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_4_specification = {
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

# CASE 5
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
case_5_specification = {
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




cases = {}
case_counter = 0
while True:
    try:
        cases.update({case_counter: eval(f"case_{case_counter}_specification")})
    except NameError:
        break
    case_counter += 1

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
