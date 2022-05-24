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
fidelities = np.linspace(0.92, 1.00, num=17)
memory_times = np.logspace(-3, 0, num=13)  # 1 ms to 1 second
num_parts = 128
lengths = np.linspace(1, 450e3, num=num_parts)
for case, (fid, memory_time) in zip(it.count(start_case), it.product(fidelities, memory_times)):
    case_specification = {
        "name": "2d_plot_100_memories",
        "subcase_name": f"f{int(fid * 1000)}_tdp{memory_time * 1e3:.2f}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": memory_time,
                                        "F_INIT": fid,
                                        "P_D": 1e-6
                                        },
                             "num_memories": 100,
                             "epp_steps": 0,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({case: case_specification})
if __name__ == "__main__":
    print("case_count after 2d", len(cases))

# # now start developing code for a 2d plot
start_case = len(cases)  # currently 227
fidelities = np.linspace(0.92, 1.00, num=17)
memory_times = np.logspace(-3, 0, num=13)  # 1 ms to 1 second
num_parts = 128
lengths = np.linspace(1, 450e3, num=num_parts)
for case, (fid, memory_time) in zip(it.count(start_case), it.product(fidelities, memory_times)):
    case_specification = {
        "name": "2d_plot_100_memories",
        "subcase_name": f"epp_f{int(fid * 1000)}_tdp{memory_time * 1e3:.2f}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": memory_time,
                                        "F_INIT": fid,
                                        "P_D": 1e-6
                                        },
                             "num_memories": 100,
                             "epp_steps": 1,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({case: case_specification})
if __name__ == "__main__":
    print("case_count after epp", len(cases))  # 448


# now do the same thing with fewer memories, which may be easier for the paper
start_case = len(cases)
num_memories = 2
fidelities = np.linspace(0.92, 1.00, num=17)
memory_times = np.logspace(-3, 0, num=13)  # 1 ms to 1 second
num_parts = 128
lengths = np.linspace(1, 350e3, num=num_parts)
for case, (fid, memory_time) in zip(it.count(start_case), it.product(fidelities, memory_times)):
    case_specification = {
        "name": f"2d_plot_{num_memories}_memories",
        "subcase_name": f"f{int(fid * 1000)}_tdp{memory_time * 1e3:.2f}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": memory_time,
                                        "F_INIT": fid,
                                        "P_D": 1e-6
                                        },
                             "num_memories": num_memories,
                             "epp_steps": 0,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({case: case_specification})
if __name__ == "__main__":
    print(f"case_count after 2d_plot_{num_memories}_memories", len(cases))


# now do the same thing with fewer memories, which may be easier for the paper
start_case = len(cases)
num_memories = 2
fidelities = np.linspace(0.92, 1.00, num=17)
memory_times = np.logspace(-3, 0, num=13)  # 1 ms to 1 second
num_parts = 128
lengths = np.linspace(1, 350e3, num=num_parts)
for case, (fid, memory_time) in zip(it.count(start_case), it.product(fidelities, memory_times)):
    case_specification = {
        "name": f"2d_plot_{num_memories}_memories",
        "subcase_name": f"epp_f{int(fid * 1000)}_tdp{memory_time * 1e3:.2f}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": memory_time,
                                        "F_INIT": fid,
                                        "P_D": 1e-6
                                        },
                             "num_memories": num_memories,
                             "epp_steps": 1,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({case: case_specification})
if __name__ == "__main__":
    print(f"case_count after epp_2d_plot_{num_memories}_memories", len(cases))


# try four memories
start_case = len(cases)
num_memories = 4
fidelities = np.linspace(0.92, 1.00, num=17)
memory_times = np.logspace(-3, 0, num=13)  # 1 ms to 1 second
num_parts = 128
lengths = np.linspace(1, 400e3, num=num_parts)
for case, (fid, memory_time) in zip(it.count(start_case), it.product(fidelities, memory_times)):
    case_specification = {
        "name": f"2d_plot_{num_memories}_memories",
        "subcase_name": f"f{int(fid * 1000)}_tdp{memory_time * 1e3:.2f}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": memory_time,
                                        "F_INIT": fid,
                                        "P_D": 1e-6
                                        },
                             "num_memories": num_memories,
                             "epp_steps": 0,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({case: case_specification})
if __name__ == "__main__":
    print(f"case_count after 2d_plot_{num_memories}_memories", len(cases))


# try four memories
start_case = len(cases)
num_memories = 4
fidelities = np.linspace(0.92, 1.00, num=17)
memory_times = np.logspace(-3, 0, num=13)  # 1 ms to 1 second
num_parts = 128
lengths = np.linspace(1, 400e3, num=num_parts)
for case, (fid, memory_time) in zip(it.count(start_case), it.product(fidelities, memory_times)):
    case_specification = {
        "name": f"2d_plot_{num_memories}_memories",
        "subcase_name": f"epp_f{int(fid * 1000)}_tdp{memory_time * 1e3:.2f}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": memory_time,
                                        "F_INIT": fid,
                                        "P_D": 1e-6
                                        },
                             "num_memories": num_memories,
                             "epp_steps": 1,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({case: case_specification})
if __name__ == "__main__":
    print(f"case_count after epp_2d_plot_{num_memories}_memories", len(cases))


# let's look at cutoff times
f_init = 0.93
t_dp = 100e-3
case_name = f"compare_cutoff_{int(t_dp * 1e3)}_f{int(f_init * 1e3)}"

cutoff_multipliers = [0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, None]
num_memories = 2
num_parts = 128
lengths = np.linspace(1, 400e3, num=num_parts)
if __name__ == "__main__":
    print(f"Case {case_name} starts at case number", len(cases))
for cutoff_multiplier in cutoff_multipliers:
    try:
        label = str(int(cutoff_multiplier * 1e3))
    except TypeError:
        label = "None"
    try:
        cutoff_time = cutoff_multiplier * t_dp
    except TypeError:
        cutoff_time = None
    case_specification = {
        "name": case_name,
        "subcase_name": f"cutoff{label}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": t_dp,
                                        "F_INIT": f_init,
                                        "P_D": 1e-6
                                        },
                             "cutoff_time": cutoff_time,
                             "num_memories": num_memories,
                             "epp_steps": 0,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({len(cases): case_specification})
for cutoff_multiplier in cutoff_multipliers:
    try:
        label = str(int(cutoff_multiplier * 1e3))
    except TypeError:
        label = "None"
    try:
        cutoff_time = cutoff_multiplier * t_dp
    except TypeError:
        cutoff_time = None
    case_specification = {
        "name": case_name,
        "subcase_name": f"epp_cutoff{label}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": t_dp,
                                        "F_INIT": f_init,
                                        "P_D": 1e-6
                                        },
                             "cutoff_time": cutoff_time,
                             "num_memories": num_memories,
                             "epp_steps": 1,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({len(cases): case_specification})
if __name__ == "__main__":
    print(f"Case {case_name} ends at case number", len(cases))


# how about with even worse states?
f_init = 0.925
t_dp = 100e-3
case_name = f"compare_cutoff_{int(t_dp * 1e3)}_f{int(f_init * 1e3)}"

cutoff_multipliers = [0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, None]
num_memories = 2
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
if __name__ == "__main__":
    print(f"Case {case_name} starts at case number", len(cases))
for cutoff_multiplier in cutoff_multipliers:
    try:
        label = str(int(cutoff_multiplier * 1e3))
    except TypeError:
        label = "None"
    try:
        cutoff_time = cutoff_multiplier * t_dp
    except TypeError:
        cutoff_time = None
    case_specification = {
        "name": case_name,
        "subcase_name": f"cutoff{label}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": t_dp,
                                        "F_INIT": f_init,
                                        "P_D": 1e-6
                                        },
                             "cutoff_time": cutoff_time,
                             "num_memories": num_memories,
                             "epp_steps": 0,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({len(cases): case_specification})
for cutoff_multiplier in cutoff_multipliers:
    try:
        label = str(int(cutoff_multiplier * 1e3))
    except TypeError:
        label = "None"
    try:
        cutoff_time = cutoff_multiplier * t_dp
    except TypeError:
        cutoff_time = None
    case_specification = {
        "name": case_name,
        "subcase_name": f"epp_cutoff{label}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": t_dp,
                                        "F_INIT": f_init,
                                        "P_D": 1e-6
                                        },
                             "cutoff_time": cutoff_time,
                             "num_memories": num_memories,
                             "epp_steps": 1,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({len(cases): case_specification})
if __name__ == "__main__":
    print(f"Case {case_name} ends at case number", len(cases))


# can a much better memory help the epp strategy?
f_init = 0.925
t_dp = 1000e-3
case_name = f"compare_cutoff_{int(t_dp * 1e3)}_f{int(f_init * 1e3)}"

cutoff_multipliers = [0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, None]
num_memories = 2
num_parts = 128
lengths = np.linspace(1, 300e3, num=num_parts)
if __name__ == "__main__":
    print(f"Case {case_name} starts at case number", len(cases))
for cutoff_multiplier in cutoff_multipliers:
    try:
        label = str(int(cutoff_multiplier * 1e3))
    except TypeError:
        label = "None"
    try:
        cutoff_time = cutoff_multiplier * t_dp
    except TypeError:
        cutoff_time = None
    case_specification = {
        "name": case_name,
        "subcase_name": f"cutoff{label}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": t_dp,
                                        "F_INIT": f_init,
                                        "P_D": 1e-6
                                        },
                             "cutoff_time": cutoff_time,
                             "num_memories": num_memories,
                             "epp_steps": 0,
                             }
                      for part in range(num_parts)
                      }
    }
    cases.update({len(cases): case_specification})
for cutoff_multiplier in cutoff_multipliers:
    try:
        label = str(int(cutoff_multiplier * 1e3))
    except TypeError:
        label = "None"
    try:
        cutoff_time = cutoff_multiplier * t_dp
    except TypeError:
        cutoff_time = None
    case_specification = {
        "name": case_name,
        "subcase_name": f"epp_cutoff{label}",
        "num_parts": num_parts,
        "index": lengths,
        "case_args": {part: {"length": lengths[part],
                             "max_iter": 1e5,
                             "params": {"P_LINK": 0.5,
                                        "T_DP": t_dp,
                                        "F_INIT": f_init,
                                        "P_D": 1e-6
                                        },
                             "cutoff_time": cutoff_time,
                             "num_memories": num_memories,
                             "epp_steps": 1,
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
