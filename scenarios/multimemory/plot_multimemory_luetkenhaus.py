import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
from scenarios.multimemory.run_multimemory_luetkenhaus import params, F, P_BSM
# from scipy.stats import binom
from scipy.special import binom as binomial_coefficient
from libs.aux_functions import binary_entropy

# copying the formulas in the paper

L_ATT = 22 * 10**3
C = 2 * 10**8
result_path = os.path.join("results", "multimemory_luetkenhaus")

# B = binom.pmf
# def B(*args, **kwargs):
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('error')
    #     try:
    #         return binom.pmf(*args, **kwargs)
    #     except Warning as e:
    #         print("Warning was raised", e)
    #         print(args, kwargs)
    #         raise e

def B(k, n, p):
    assert 0 <= k <= n
    assert 0 <= p <= 1
    if p == 0:
        if k == 0:
            return 1
        else:
            return 0
    elif p == 1:
        if k == n:
            return 1
        else:
            return 0
    else:
        return binomial_coefficient(n, k) * p**k * (1-p)**(n-k)


def R(L, num_memories):
    m = num_memories
    eta = params["P_LINK"] * np.exp(-L / 2 / L_ATT)
    eta_effective = 1 - (1 - eta) * (1 - params["P_D"])**2
    p_s = 1 - (1 - eta_effective)**m
    m_E_max = m * (2 / p_s - 1 / (2 * p_s - p_s**2))

    def alpha_of_eta(eta):
        return eta * (1 - params["P_D"]) / (1 - (1 - eta) * (1 - params["P_D"])**2)

    def g_m(eta_prime):
        return eta_prime * (1 - eta_prime) * (
            np.sum([B(i, m - 1, eta_prime)**2 for i in range(m)])
            + np.sum([B(i, m - 1, eta_prime) * B(i - 1, m - 1, eta_prime) for i in range(1, m)])
        )

    def k_p_kbits_sum(eta_prime):
        return m * P_BSM * (eta_prime - g_m(eta_prime))

    def lambda_dp(t):
        return (1 - np.exp(-t / params["T_DP"])) / 2

    Y = 1 / p_s**2 * k_p_kbits_sum(eta_effective) / m_E_max

    lambda_bsm = params["LAMBDA_BSM"]
    epsilon_mis = 2 * params["E_MA"] * (1 - params["E_MA"])
    e_Z = lambda_bsm * alpha_of_eta(eta)**2 * epsilon_mis + (1 - lambda_bsm * alpha_of_eta(eta)**2) / 2
    tau = params["T_P"] + L / C
    epsilon_dp = (1 - 2 * lambda_dp(L / C)) * (1 / 2 - 1 / 2 * p_s * np.exp(-L / C / params["T_DP"])
                                               / (np.exp(tau / params["T_DP"]) + p_s - 1)
                                               ) + lambda_dp(L / C)
    e_X = lambda_bsm * alpha_of_eta(eta)**2 * (epsilon_mis * (1 - epsilon_dp) + epsilon_dp * (1 - epsilon_mis)) \
        + (1 - lambda_bsm * alpha_of_eta(eta)**2) / 2
    return Y / 2 * (1 - binary_entropy(e_X) - F * binary_entropy(e_Z))

def inverse_channel_use(L, num_memories):
    m = num_memories
    eta = params["P_LINK"] * np.exp(-L / 2 / L_ATT)
    eta_effective = 1 - (1 - eta) * (1 - params["P_D"])**2
    p_s = 1 - (1 - eta_effective)**m
    m_E_max = m * (2 / p_s - 1 / (2 * p_s - p_s**2))

    def alpha_of_eta(eta):
        return eta * (1 - params["P_D"]) / (1 - (1 - eta) * (1 - params["P_D"])**2)

    def g_m(eta_prime):
        return eta_prime * (1 - eta_prime) * (
            np.sum([B(i, m - 1, eta_prime)**2 for i in range(m)])
            + np.sum([B(i, m - 1, eta_prime) * B(i - 1, m - 1, eta_prime) for i in range(1, m)])
        )

    def k_p_kbits_sum(eta_prime):
        return m * P_BSM * (eta_prime - g_m(eta_prime))

    def lambda_dp(t):
        return (1 - np.exp(-t / params["T_DP"])) / 2

    return 1 / p_s**2 * k_p_kbits_sum(eta_effective) / m_E_max

def entropy_term(L, num_memories):
    m = num_memories
    eta = params["P_LINK"] * np.exp(-L / 2 / L_ATT)
    eta_effective = 1 - (1 - eta) * (1 - params["P_D"])**2
    p_s = 1 - (1 - eta_effective)**m
    m_E_max = m * (2 / p_s - 1 / (2 * p_s - p_s**2))

    def alpha_of_eta(eta):
        return eta * (1 - params["P_D"]) / (1 - (1 - eta) * (1 - params["P_D"])**2)

    def g_m(eta_prime):
        return eta_prime * (1 - eta_prime) * (
            np.sum([B(i, m - 1, eta_prime)**2 for i in range(m)])
            + np.sum([B(i, m - 1, eta_prime) * B(i - 1, m - 1, eta_prime) for i in range(1, m)])
        )

    def k_p_kbits_sum(eta_prime):
        return m * P_BSM * (eta_prime - g_m(eta_prime))

    def lambda_dp(t):
        return (1 - np.exp(-t / params["T_DP"])) / 2

    # Y = 1 / p_s**2 * k_p_kbits_sum(eta_effective) / m_E_max

    lambda_bsm = params["LAMBDA_BSM"]
    epsilon_mis = 2 * params["E_MA"] * (1 - params["E_MA"])
    e_Z = lambda_bsm * alpha_of_eta(eta)**2 * epsilon_mis + (1 - lambda_bsm * alpha_of_eta(eta)**2) / 2
    tau = params["T_P"] + L / C
    epsilon_dp = (1 - 2 * lambda_dp(L / C)) * (1 / 2 - 1 / 2 * p_s * np.exp(-L / C / params["T_DP"])
                                               / (np.exp(tau / params["T_DP"]) + p_s - 1)
                                               ) + lambda_dp(L / C)
    e_X = lambda_bsm * alpha_of_eta(eta)**2 * (epsilon_mis * (1 - epsilon_dp) + epsilon_dp * (1 - epsilon_mis)) \
        + (1 - lambda_bsm * alpha_of_eta(eta)**2) / 2
    return (1 - binary_entropy(e_X) - F * binary_entropy(e_Z))

num_memories = 400
output_path = os.path.join(result_path, "%d_memories" % num_memories)
x = np.loadtxt(os.path.join(output_path, "length_list_seq.txt"))
y = np.loadtxt(os.path.join(output_path, "key_per_resource_list_seq.txt")) / 2
plt.scatter(x / 1000, y)

x = np.linspace(0, 800e3, num=200)
memories = [1, 400, 1000]
for memory in memories:
    print(memory)
    y = [R(L=length, num_memories=memory) for length in x]
    plt.plot(x / 1000, y, label=f"num_memories={memory}")
plt.yscale("log")
plt.ylim(1e-12, 1e-2)
plt.grid()
plt.legend(loc="upper right")
plt.show()

x = np.linspace(0, 800e3, num=200)
for memory in memories:
    print(memory)
    y = [inverse_channel_use(length, memory) for length in x]
    plt.plot(x / 1000, y, label=f"num_memories={memory}")
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()

x = np.linspace(0, 800e3, num=200)
for memory in memories:
    print(memory)
    y = [entropy_term(length, memory) for length in x]
    plt.plot(x / 1000, y, label=f"num_memories={memory}")
plt.grid()
plt.legend()
plt.show()
