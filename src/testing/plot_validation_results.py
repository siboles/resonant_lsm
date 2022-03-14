import sys
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
sns.set_context("paper")

FILENAMES = (
    'gn_0p1_sn_0p0.xlsx',
    'gn_0p1_sn_0p1.xlsx',
    'gn_0p1_sn_0p2.xlsx',
    'gn_0p2_sn_0p0.xlsx',
    'gn_0p2_sn_0p1.xlsx',
    'gn_0p2_sn_0p2.xlsx',
)

KEYS = (
    'Volume Percentage Error',
    'Surface Area Percentage Error',
    'Root Mean Square Offset Error',
)


def main(base_directory):
    dataframes = []
    titles = []
    for f in FILENAMES:
        file = os.path.join(base_directory, f)
        gaussian_noise = f[3:6].replace('p', '.')
        speckle_noise = f[10:13].replace('p', '.')
        titles.append('({}, {})'.format(gaussian_noise, speckle_noise))
        dataframes.append(pd.read_excel(file))
    fig, axs = plt.subplots(len(KEYS), 1, sharex=True, figsize=(4, 7))
    for i, key in enumerate(KEYS):
        data = []
        for dataframe in dataframes:
            data.append(dataframe[key].values)
        axs[i].boxplot(data, notch=True, bootstrap=1000, labels=titles, showfliers=False,
                       medianprops=dict(color='k'))
        axs[i].set_title(key)
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    plt.xticks(rotation=30)
    plt.savefig(base_directory + '_segmentation_error.svg')

if __name__ == '__main__':
    main(sys.argv[-1])