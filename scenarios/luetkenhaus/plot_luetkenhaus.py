import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

result_path = os.path.join("results", "luetkenhaus")

for mode in ["seq", "sim"]:
    df = pd.read_csv(os.path.join(result_path, f"mode_{mode}", "result.csv"), index_col=0)
    x = df.index / 1000
    y = df["key_per_resource"] / 2
    plt.scatter(x, y, marker="o", s=5, label=f"simulation_{mode}")

# compare to analytical results
xx = np.loadtxt(os.path.join(result_path, "onerep_length.txt"), dtype=np.complex) / 1000
y1 = np.loadtxt(os.path.join(result_path, "onerep_sequential.txt"), dtype=np.complex)
y2 = np.loadtxt(os.path.join(result_path, "onerep_simultaneous.txt"), dtype=np.complex)

plt.plot(xx, y1, label="analytical_seq")
plt.plot(xx, y2, label="analytical_sim")
plt.yscale("log")
plt.xlabel("total distance [km]")
plt.ylabel("key rate per channel use")
plt.grid()
plt.legend()
plt.savefig(os.path.join(result_path, "key_per_resource.png"))
plt.show()


for mode in ["seq", "sim"]:
    df = pd.read_csv(os.path.join(result_path, f"mode_{mode}", "result.csv"), index_col=0)
    x = df.index / 1000
    y = df["key_per_time"] / 2
    plt.scatter(x, y, marker="o", s=5, label=f"simulation_{mode}")

plt.yscale("log")
plt.ylim(1e-4, 1e3)
plt.xlabel("total distance [km]")
plt.ylabel("key rate per time")
plt.grid()
plt.legend()
plt.savefig(os.path.join(result_path, "key_per_time.png"))
plt.show()


# compare to alternative calculation: NSP with the parameters of the Luetkenhaus paper
path_alternative = os.path.join("results", "luetkenhaus", "as_nsp")

df = pd.read_csv(os.path.join(result_path, "mode_seq", "result.csv"), index_col=0)
x = df.index / 1000
y = df["key_per_resource"] / 2
plt.scatter(x, y, marker="o", s=5, label="luetkenhaus")

df = pd.read_csv(os.path.join(path_alternative, "result.csv"), index_col=0)
x = df.index / 1000
y = df["key_per_resource"] / 2
plt.scatter(x, y, marker="o", s=5, label="luetkenhaus_as_nsp")
plt.yscale("log")
plt.ylim(1e-7, 1e-3)
plt.xlabel("total distance [km]")
plt.ylabel("key rate per channel use")
plt.grid()
plt.legend()
plt.savefig(os.path.join(result_path, "compare_to_nsp_resource.png"))
plt.show()


df = pd.read_csv(os.path.join(result_path, "mode_seq", "result.csv"), index_col=0)
x = df.index / 1000
y = df["key_per_time"] / 2
plt.scatter(x, y, marker="o", s=5, label="luetkenhaus")

df = pd.read_csv(os.path.join(path_alternative, "result.csv"), index_col=0)
x = df.index / 1000
y = df["key_per_time"] / 2
plt.scatter(x, y, marker="o", s=5, label="luetkenhaus_as_nsp")
plt.yscale("log")
plt.ylim(1e-4, 1e3)
plt.xlabel("total distance [km]")
plt.ylabel("key rate per time")
plt.grid()
plt.legend()
plt.savefig(os.path.join(result_path, "compare_to_nsp_time.png"))
plt.show()
