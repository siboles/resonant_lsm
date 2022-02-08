import sys
from typing import List
import os
import pandas as pd


def aggregate_strain_results(filelist, time_map, output_file=None):
    """Concatenates data from excel files into a single file"""
    aggregate_dataframe = pd.DataFrame()
    for specimen_id, filename in enumerate(filelist):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        excel_data = pd.read_excel(filename, sheet_name=None, index_col=0)
        results_dataframes = []
        for sheetname, data in excel_data.items():
            data["Time"] = [time_map[sheetname]] * data.shape[0]
            data["Specimen ID"] = [specimen_id + 1] * data.shape[0]
            results_dataframes.append(data)
        results_dataframe = pd.concat(results_dataframes)
        aggregate_dataframe = pd.concat([aggregate_dataframe, results_dataframe], axis=0)
    aggregate_dataframe.to_excel(os.path.abspath('.'.join([output_file, "xlsx"])))

