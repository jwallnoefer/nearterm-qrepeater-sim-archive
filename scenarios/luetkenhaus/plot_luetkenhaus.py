import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import rsmf

# colorblind friendly color set taken from https://personal.sron.nl/~pault/
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
# make them the standard colors for matplotlib
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

formatter = rsmf.setup(r"\documentclass{revtex4-2}")



result_path = os.path.join("results", "luetkenhaus")

fig = formatter.figure()
for mode in ["seq", "sim"]:
    df = pd.read_csv(os.path.join(result_path, f"mode_{mode}", "result.csv"), index_col=0)
    x = df.index / 1000
    y = df["key_per_resource"] / 2
    plt.scatter(x, y, marker="o", edgecolors="none", s=1.5)#, label=f"simulation_{mode}")

# compare to analytical results
xx = np.loadtxt(os.path.join(result_path, "onerep_length.txt"), dtype=complex) / 1000
y1 = np.loadtxt(os.path.join(result_path, "onerep_sequential.txt"), dtype=complex)
y2 = np.loadtxt(os.path.join(result_path, "onerep_simultaneous.txt"), dtype=complex)

plt.plot(xx, y1, lw=0.4, label="sequential mode")
plt.plot(xx, y2, lw=0.4, label="simultaneous mode")
plt.yscale("log")
plt.xlabel("total distance [km]")
plt.ylabel("key rate per channel use")
plt.grid()
plt.legend(fontsize=formatter.fontsizes.footnotesize)
ax = plt.gca()
axins = ax.inset_axes([0.1, 0.33, 0.3, 0.44])
for mode in ["seq", "sim"]:
    df = pd.read_csv(os.path.join(result_path, f"mode_{mode}", "result.csv"), index_col=0)
    x = df.index / 1000
    y = df["key_per_resource"] / 2
    axins.scatter(x, y, marker="o", edgecolors="none", s=1.5)#, label=f"simulation_{mode}")
axins.plot(xx, y1, lw=0.4)
axins.plot(xx, y2, lw=0.4)
axins.set_yscale("log")
# sub region of the original image
x1, x2, y1, y2 = 45.1, 45.8, 5e-5, 5.4e-5
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticks([])
axins.set_yticks([])
axins.set_xticks([], minor=True)
axins.set_yticks([], minor=True)
rect, connectors = ax.indicate_inset_zoom(axins, lw=0.1, edgecolor="black")
rect.set(lw=0.4)
connectors[0].set(visible=False)
connectors[1].set(visible=False)
connectors[2].set(lw=0.4, ls="dashed", visible=True)
connectors[3].set(lw=0.4, ls="dashed", visible=True)
plt.ylim(1e-7, 1e-3)
plt.tight_layout()
plt.savefig(os.path.join(result_path, "key_per_resource.png"))
plt.savefig(os.path.join(result_path, "key_per_resource.pdf"))
plt.cla()


fig = formatter.figure()
for mode in ["seq", "sim"]:
    df = pd.read_csv(os.path.join(result_path, f"mode_{mode}", "result.csv"), index_col=0)
    x = df.index / 1000
    y = df["key_per_time"] / 2
    plt.scatter(x, y, marker="o", s=1.5, label=f"simulation_{mode}")

plt.yscale("log")
plt.ylim(1e-4, 1e3)
plt.xlabel("total distance [km]")
plt.ylabel("key rate per time")
plt.grid()
plt.legend()
plt.savefig(os.path.join(result_path, "key_per_time.png"))
plt.savefig(os.path.join(result_path, "key_per_time.pdf"))
plt.cla()


# compare to alternative calculation: NSP with the parameters of the Luetkenhaus paper
path_alternative = os.path.join("results", "luetkenhaus", "as_nsp")

df = pd.read_csv(os.path.join(result_path, "mode_seq", "result.csv"), index_col=0)
x = df.index / 1000
y = df["key_per_resource"] / 2
plt.scatter(x, y, marker="o", s=1.5, label="luetkenhaus")

df = pd.read_csv(os.path.join(path_alternative, "result.csv"), index_col=0)
x = df.index / 1000
y = df["key_per_resource"] / 2
plt.scatter(x, y, marker="o", s=1.5, label="luetkenhaus_as_nsp")
plt.yscale("log")
plt.ylim(1e-7, 1e-3)
plt.xlabel("total distance [km]")
plt.ylabel("key rate per channel use")
plt.grid()
plt.legend()
plt.savefig(os.path.join(result_path, "compare_to_nsp_resource.png"))
plt.savefig(os.path.join(result_path, "compare_to_nsp_resource.pdf"))
plt.cla()


df = pd.read_csv(os.path.join(result_path, "mode_seq", "result.csv"), index_col=0)
x = df.index / 1000
y = df["key_per_time"] / 2
plt.scatter(x, y, marker="o", s=1.5, label="luetkenhaus")

df = pd.read_csv(os.path.join(path_alternative, "result.csv"), index_col=0)
x = df.index / 1000
y = df["key_per_time"] / 2
plt.scatter(x, y, marker="o", s=1.5, label="luetkenhaus_as_nsp")
plt.yscale("log")
plt.ylim(1e-4, 1e3)
plt.xlabel("total distance [km]")
plt.ylabel("key rate per time")
plt.grid()
plt.legend()
plt.savefig(os.path.join(result_path, "compare_to_nsp_time.png"))
plt.savefig(os.path.join(result_path, "compare_to_nsp_time.pdf"))
plt.cla()
