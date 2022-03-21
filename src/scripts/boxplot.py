import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

sns.set_context("paper")

KEYS = ("Volumetric Strain",
        "Normalized Surface Area Change",
        "Deformed Surface Area to Volume")

TITLES = ("Volumetric Strain",
          "Normalized Surface Area Change",
          "Surface Area to Volume Ratio")


def main(filename):
    file = pathlib.Path(filename)
    dataframe = pd.read_excel(file)
    fig, axs = plt.subplots(len(KEYS), 1, sharex=True, figsize=(4, 6))
    for i, key in enumerate(KEYS):
        dataframe.boxplot(column=[key], by="Time", notch=True, showfliers=False,
                          grid=False, ax=axs[i],
                          color=dict(boxes='k', whiskers='k',
                                     medians='k'))
        axs[i].set_title(TITLES[i])
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
        if i < len(KEYS) - 1:
            axs[i].set_xlabel("")
        else:
            axs[i].set_xlabel("Time (min)")
    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(filename.replace(".xlsx", "_boxplot.svg"))


if __name__ == "__main__":
    main(sys.argv[-1])
