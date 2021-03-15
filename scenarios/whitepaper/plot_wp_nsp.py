import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
from scenarios.whitepaper.run_whitepaper_nsp import available_params, future_params, ms_available, ms_future
from libs.aux_functions import binary_entropy
import pandas as pd

result_path = os.path.join("results", "whitepaper")

L_ATT = 22 * 10**3 / 1000  # attenuation length


def db(x):
    """Convert to decibel."""
    return 10 * np.log10(x)


x_base = np.arange(1000, 401000, 1000) / 1000
eta = np.exp(-x_base / L_ATT)
y_repeaterless = 10 * np.log10(-np.log2(1 - eta))
y_optimal = 10 * np.log10(np.sqrt(eta))
y_realistic_repeaterless1 = 10 * np.log10(0.7 * eta / 2)
y_realistic_repeaterless2 = 10 * np.log10(0.1 * eta / 2)


def skr_whitepaper(L, m, params):
    c = 2 * 10**8
    p = params["P_LINK"] * np.exp(-L / 2 / (L_ATT * 1000))
    q = 1 - p
    R = p * (2 - p - 2 * q**(m + 1)) / (3 - 2 * p - 2 * q**(m + 1))
    t_coh = params["T_DP"]
    T_0 = L / c

    def PofMisj(j):
        if j == 0:
            return p / (2 - p)
        else:
            return 2 * p * q**j / (2 - p)
    E = np.sum([PofMisj(j) * np.exp(-(j + 2) * T_0 / t_coh) for j in range(m + 1)]) / np.sum([PofMisj(j) for j in range(m + 1)])
    # E = np.sum([PofMisj(j) * np.exp(-(j) * T_0 / t_coh) for j in range(m+1)]) / np.sum([PofMisj(j) for j in range(m+1)])
    # print(np.sum([PofMisj(j) * np.exp(-(j + 2) * T_0 / t_coh) for j in range(m+1)]))
    # print(np.sum([PofMisj(j) for j in range(m+1)]))
    return R * (1 - binary_entropy(1 / 2 * (1 - E)))


# name_list = ["NV", "SiV", "Qdot", "Ca", "Rb"]
name_list = ["NV", "SiV", "Ca", "Rb"]
color_list = ["blue", "green", "orange", "red"]
available_params = available_params[0:2] + available_params[3:]
future_params = future_params[0:2] + future_params[3:]
ms_available = ms_available[0:2] + ms_available[3:]
ms_future = ms_future[0:2] + ms_future[3:]

# first plot available values
plt.plot(x_base, y_repeaterless, color="black")
plt.plot(x_base, y_optimal, color="gray")
plt.fill_between(x_base, y_repeaterless, y_optimal, facecolor="lightgray")
plt.plot(x_base, y_realistic_repeaterless1, color="black", linestyle="dashed")
plt.plot(x_base, y_realistic_repeaterless2, color="black", linestyle="dashed")
for name, color in zip(name_list, color_list):
    path = os.path.join(result_path, "available", name)
    df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
    x = df.index / 1000
    y = db(df["key_per_resource"] / 2)
    if name == "Ca":
        name = "Ca/Yb"
    plt.scatter(x, y, marker="o", s=5, label=name, color=color)
plt.xlim((0, 400))
plt.ylim((-60, 0))
plt.legend()
plt.grid()
plt.xlabel("L [km]")
plt.ylabel("secret key rate per channel use [dB]")
plt.title("Available parameters")
plt.savefig(os.path.join(result_path, "available", "simulation.png"))
plt.savefig(os.path.join(result_path, "available", "simulation.pdf"))
plt.show()

# now plot future values
plt.plot(x_base, y_repeaterless, color="black")
plt.plot(x_base, y_optimal, color="gray")
plt.fill_between(x_base, y_repeaterless, y_optimal, facecolor="lightgray")
plt.plot(x_base, y_realistic_repeaterless1, color="black", linestyle="dashed")
plt.plot(x_base, y_realistic_repeaterless2, color="black", linestyle="dashed")
for name, color in zip(name_list, color_list):
    path = os.path.join(result_path, "future", name)
    df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
    x = df.index / 1000
    y = db(df["key_per_resource"] / 2)
    if name == "Ca":
        name = "Ca/Yb"
    plt.scatter(x, y, marker="o", s=5, label=name, color=color)
plt.xlim((0, 400))
plt.ylim((-60, 0))
plt.legend()
plt.grid()
plt.xlabel("L [km]")
plt.ylabel("secret key rate per channel use [dB]")
plt.title("Future parameters")
plt.savefig(os.path.join(result_path, "future", "simulation.png"))
plt.savefig(os.path.join(result_path, "future", "simulation.pdf"))
plt.show()


## now compare with what we THINK is the line in the whitepaper

# first plot available values
plt.plot(x_base, y_repeaterless, color="black")
plt.plot(x_base, y_optimal, color="gray")
plt.fill_between(x_base, y_repeaterless, y_optimal, facecolor="lightgray")
plt.plot(x_base, y_realistic_repeaterless1, color="black", linestyle="dashed")
plt.plot(x_base, y_realistic_repeaterless2, color="black", linestyle="dashed")
for name, color, params, m in zip(name_list, color_list, available_params, ms_available):
    path = os.path.join(result_path, "available", name)
    df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
    x = df.index / 1000
    y = db(df["key_per_resource"] / 2)
    if name == "Ca":
        name = "Ca/Yb"
    plt.scatter(x, y, marker="o", s=5, label=name, color=color)
    y_whitepaper = db([skr_whitepaper(l, m, params) / 2 for l in (x_base * 1000)])
    plt.plot(x_base, y_whitepaper, color=color)
plt.xlim((0, 400))
plt.ylim((-60, 0))
plt.legend()
plt.grid()
plt.xlabel("L [km]")
plt.ylabel("secret key rate per channel use [dB]")
plt.title("Available parameters")
plt.savefig(os.path.join(result_path, "available", "compare.png"))
plt.savefig(os.path.join(result_path, "available", "compare.pdf"))
plt.show()

# now plot future values
plt.plot(x_base, y_repeaterless, color="black")
plt.plot(x_base, y_optimal, color="gray")
plt.fill_between(x_base, y_repeaterless, y_optimal, facecolor="lightgray")
plt.plot(x_base, y_realistic_repeaterless1, color="black", linestyle="dashed")
plt.plot(x_base, y_realistic_repeaterless2, color="black", linestyle="dashed")
for name, color, params, m in zip(name_list, color_list, future_params, ms_future):
    path = os.path.join(result_path, "future", name)
    df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
    x = df.index / 1000
    y = db(df["key_per_resource"] / 2)
    if name == "Ca":
        name = "Ca/Yb"
    plt.scatter(x, y, marker="o", s=5, label=name, color=color)
    y_whitepaper = db([skr_whitepaper(l, m, params) / 2 for l in (x_base * 1000)])
    plt.plot(x_base, y_whitepaper, color=color)
plt.xlim((0, 400))
plt.ylim((-60, 0))
plt.legend()
plt.grid()
plt.xlabel("L [km]")
plt.ylabel("secret key rate per channel use [dB]")
plt.title("Future parameters")
plt.savefig(os.path.join(result_path, "future", "compare.png"))
plt.savefig(os.path.join(result_path, "future", "compare.pdf"))
plt.show()
