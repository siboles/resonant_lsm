import sys
import os
import pandas as pd

ROOT_DIRECTORY = os.path.abspath(sys.argv[-1])
if not os.path.exists(ROOT_DIRECTORY):
    raise FileNotFoundError(ROOT_DIRECTORY)

REPEATS = 5
DEFORMATIONS = 10

basenames = ["gn_0p1_sn_0p0",
             "gn_0p1_sn_0p1",
             "gn_0p1_sn_0p2",
             "gn_0p2_sn_0p0",
             "gn_0p2_sn_0p1",
             "gn_0p2_sn_0p2"]
for name in basenames:
    aggregate_dataframe = pd.DataFrame()
    for i in range(REPEATS):
        filename = os.path.join(ROOT_DIRECTORY, "_".join([name, "{:03d}".format(i + 1)]), 
                "reference_results", "results.xlsx")
        excel_data = pd.read_excel(filename, sheet_name=None, index_col=0)
        results_dataframe = pd.concat(excel_data)
        results_dataframe.reset_index(inplace=True)
        results_dataframe.drop(["level_0"], inplace=True, axis=1)
        segmentation_dataframe = results_dataframe[results_dataframe["Cell ID"] == "cell_001.vtp"]
        segmentation_dataframe.set_index("Cell ID", inplace=True)
        ground_truth_dataframe = results_dataframe[results_dataframe["Cell ID"] == "comparison.vtk"]
        ground_truth_dataframe.set_index("Cell ID", inplace=True)
        difference = ground_truth_dataframe.values - segmentation_dataframe.values
        difference = pd.DataFrame(data=difference, columns=segmentation_dataframe.columns)
        aggregate_dataframe = pd.concat([aggregate_dataframe, difference], axis=0)
    aggregate_dataframe.to_excel(os.path.join(ROOT_DIRECTORY, '_'.join([name, 'strain_errors.xlsx'])))






