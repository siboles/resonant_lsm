import sys
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

sns.set_context("paper")

FILENAMES = (
    'gn_0p1_sn_0p0_strain_errors.xlsx',
    'gn_0p1_sn_0p1_strain_errors.xlsx',
    'gn_0p1_sn_0p2_strain_errors.xlsx',
    'gn_0p2_sn_0p0_strain_errors.xlsx',
    'gn_0p2_sn_0p1_strain_errors.xlsx',
    'gn_0p2_sn_0p2_strain_errors.xlsx',
)

KEYS = (
    'Volumetric Strain',
    'Normalized Surface Area Change',
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
    fig, axs = plt.subplots(len(KEYS), 1, sharex=True, figsize=(4, 5))
    for i, key in enumerate(KEYS):
        data = []
        for dataframe in dataframes:
            data.append(dataframe[key].values)
        axs[i].boxplot(data, notch=True, bootstrap=1000, labels=titles, showfliers=False)
        axs[i].set_title(key + " Error")
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.xticks(rotation=30)
    plt.savefig(base_directory + '_strain_error.svg')

if __name__ == '__main__':
    main(sys.argv[-1])