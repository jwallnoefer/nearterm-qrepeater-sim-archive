import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
from scenarios.whitepaper.run_whitepaper_nsp import params_available_Rb
from libs.aux_functions import binary_entropy
import pandas as pd

result_path = os.path.join("results", "whitepaper")


def db(x):
    """Convert to decibel."""
    return 10 * np.log10(x)


L_ATT = 22 * 10**3 / 1000  # attenuation length

x_base = np.arange(1000, 401000, 1000) / 1000
eta = np.exp(-x_base / L_ATT)
y_repeaterless = 10 * np.log10(-np.log2(1 - eta))
y_optimal = 10 * np.log10(np.sqrt(eta))
y_realistic_repeaterless1 = 10 * np.log10(0.7 * eta / 2)
y_realistic_repeaterless2 = 10 * np.log10(0.1 * eta / 2)


def skr_whitepaper(L, m):
    c = 2 * 10**8
    p = params_available_Rb["P_LINK"] * np.exp(-L / 2 / (L_ATT * 1000))
    q = 1 - p
    R = p * (2 - p - 2 * q**(m + 1)) / (3 - 2 * p - 2 * q**(m + 1))
    t_coh = params_available_Rb["T_DP"]
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


plt.plot(x_base, y_repeaterless, color="black")
plt.plot(x_base, y_optimal, color="gray")
plt.fill_between(x_base, y_repeaterless, y_optimal, facecolor="lightgray")
plt.plot(x_base, y_realistic_repeaterless1, color="black", linestyle="dashed")
plt.plot(x_base, y_realistic_repeaterless2, color="black", linestyle="dashed")
ms = [50, 200, 500, 1000, 2000, 5000, 10000, 20000]
for m in ms:
    path = os.path.join(result_path, "Rb_m_test", f"m_{m}")
    df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
    x = df.index / 1000
    y = db(df["key_per_resource"] / 2)
    plt.scatter(x, y, marker="o", s=5, label="m=%d" % m)
    y_whitepaper = db([skr_whitepaper(l, m) / 2 for l in (x_base * 1000)])
    plt.plot(x_base, y_whitepaper)
plt.xlim((0, 400))
plt.ylim((-60, 0))
plt.legend()
plt.grid()
plt.xlabel("L [km]")
plt.ylabel("secret key rate per channel use [dB]")
plt.savefig(os.path.join(result_path, "Rb_m_test", "m_compare.png"))
plt.show()

# now do the plot where we see that one can set cutoff too low
path = os.path.join(result_path, "cutoff_test")
df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
x = df.index
y = db(df["key_per_resource"] / 2)
plt.scatter(x, y, marker="o", s=5)
plt.grid()
plt.xlabel("cutoff_m")
plt.ylabel("secret key rate per channel use [dB]")
plt.savefig(os.path.join(path, "cutoff_test.png"))
plt.show()
