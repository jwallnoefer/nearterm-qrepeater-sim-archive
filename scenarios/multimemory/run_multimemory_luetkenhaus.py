import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.multimemory.multi_memory_luetkenhaus import run
from libs.aux_functions import assert_dir, binary_entropy, calculate_keyrate_time, calculate_keyrate_channel_use
import numpy as np
import matplotlib.pyplot as plt

C = 2 * 10**8 # speed of light in optical fiber
result_path = os.path.join("results", "multimemory_luetkenhaus")

# # # values taken from Róbert Trényi, Norbert Lütkenhaus https://arxiv.org/abs/1910.10962
T_P = 2 * 10**-6  # preparation time
E_M_A = 0.01  # misalignment error
P_D = 1.8 * 10**-11  # dark count probability per detector
P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
LAMBDA_BSM = 0.98  # BSM ideality parameter
F = 1.16  # error correction inefficiency

T_2 = 2  # dephasing time
ETA_P = 0.66  # preparation efficiency
ETA_C = 0.05 * 0.5  # phton-fiber coupling efficiency * wavelength conversion
ETA_D = 0.7  # detector efficiency

ETA_TOT = ETA_P * ETA_C * ETA_D  # = 0.0115
params = {"P_LINK": ETA_TOT,
          "T_P": T_P,
          "T_DP": T_2,
          "E_MA": E_M_A,
          "P_D": P_D,
          "LAMBDA_BSM": LAMBDA_BSM}

if __name__ == "__main__":
    length_list = np.arange(20000, 400000, 20000)
    num_memories = 400
    mode = "seq"
    key_per_time_list = []
    key_per_resource_list = []
    from time import time
    for l in length_list:
        print(l)
        start_time = time()
        p = run(length=l, max_iter=1000, params=params, num_memories=num_memories, mode=mode)
        key_per_time = calculate_keyrate_time(p.correlations_z_list, p.correlations_x_list, F, p.world.event_queue.current_time + 2 * l / C)
        key_per_resource = calculate_keyrate_channel_use(p.correlations_z_list, p.correlations_x_list, F, p.resource_cost_max_list)

        key_per_time_list += [key_per_time]
        key_per_resource_list += [key_per_resource]
        print("l=%d took %s seconds" % (l, str(time() - start_time)))

    output_path = os.path.join(result_path, "%d_memories" % num_memories)
    assert_dir(output_path)

    np.savetxt(os.path.join(output_path, "length_list_%s.txt" % mode), length_list)
    np.savetxt(os.path.join(output_path, "key_per_time_list_%s.txt" % mode), key_per_time_list)
    np.savetxt(os.path.join(output_path, "key_per_resource_list_%s.txt" % mode), key_per_resource_list)

    plt.plot(length_list, key_per_time_list)
    plt.yscale("log")
    plt.xlabel("total length")
    plt.ylabel("key_rate_per_time")
    plt.show()

    plt.plot(length_list, key_per_resource_list)
    plt.yscale("log")
    plt.xlabel("total length")
    plt.ylabel("key rate per channel use")
    plt.show()
